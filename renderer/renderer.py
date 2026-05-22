"""
Grammar → CuTile Python callable compiler.

Compilation pipeline
--------------------
1. KernelPartitioner  – assigns each ComputeNode to a kernel index
                        (its top-level LoopLevel in ProgramNode.children).
2. LifetimeAnalyzer   – marks each DDG tensor as "global" (crosses a kernel
                        boundary → must materialise to device memory) or
                        "local" (confined to one kernel, stays in smem/rmem).
3. CodeGen            – walks the Grammar tree and emits cuda.tile source.
4. Exec               – compiles the source string and returns the callable.

CuTile code model (mirrors example.cpp)
-----------------------------------------
  parallel LoopLevel  → ct.bid() grid axis; one block per tile of that dim
  serial LoopLevel    → for-loop inside the kernel body
  carried_dims op     → ct.rmem() accumulator initialised before the loop,
                        updated on every iteration, stored afterwards
  global tensor       → materialised to device memory between launches;
                        loaded into ct.smem() at the top of the loop that
                        needs it, or once before the op for parallel context
  local tensor        → stays as a ct.rmem() tile inside the kernel

Tile sizes
----------
tile_q  (default 64): query-sequence tile size per block  (TILE_Q  constant)
tile_kv (default 64): key-value tile size per block       (TILE_KV constant)
D (head_dim) is never tiled at the loop level; ct.gemm handles the D
reduction internally ("HEAD_DIM … is handled inside the CuTile matmul
primitive").
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from grammar.ast import ComputeNode, Grammar, LoopLevel, ProgramNode
from grammar.utils import all_compute_nodes, tree_hash


# ---------------------------------------------------------------------------
# Op registry
# ---------------------------------------------------------------------------

# External tensors consumed by ops that have no DDG producer for that input.
# Values are the Python variable names in the generated wrapper, after
# permuting q/k/v to the (B, H, S, D) layout.
_OP_EXT: dict[str, dict[str, str]] = {
    "MatMul_QK": {"q": "_q", "k": "_k"},
    "MatMul_PV": {"v": "_v"},
}

# Tile shape of each op's output (as Python source, TILE_Q/TILE_KV/D in scope).
_OP_TILE_SHAPE: dict[str, str] = {
    "MatMul_QK": "(TILE_Q, TILE_KV)",
    "Scale":     "(TILE_Q, TILE_KV)",
    "RowMax":    "(TILE_Q, 1)",
    "Subtract":  "(TILE_Q, TILE_KV)",
    "Exp":       "(TILE_Q, TILE_KV)",
    "RowSum":    "(TILE_Q, 1)",
    "Divide":    "(TILE_Q, TILE_KV)",
    "MatMul_PV": "(TILE_Q, D)",
}

# Global-memory allocation shape (Bsz/H/Sq/Skv/D available as locals).
_OP_ALLOC_SHAPE: dict[str, str] = {
    "MatMul_QK": "(Bsz, H, Sq, Skv)",
    "Scale":     "(Bsz, H, Sq, Skv)",
    "RowMax":    "(Bsz, H, Sq, 1)",
    "Subtract":  "(Bsz, H, Sq, Skv)",
    "Exp":       "(Bsz, H, Sq, Skv)",
    "RowSum":    "(Bsz, H, Sq, 1)",
    "Divide":    "(Bsz, H, Sq, Skv)",
    "MatMul_PV": "(Bsz, H, Sq, D)",
}

# Accumulator initial value for ops with carried_dims.
# Ops that use ct.gemm (MatMul_PV) also need a fill=0.0 for the rmem tile.
_OP_ACC_INIT: dict[str, str] = {
    "RowMax":    "float('-inf')",
    "RowSum":    "0.0",
    "MatMul_PV": "0.0",
}

# True if the op uses ct.gemm (accumulates in-place into its output tile).
_GEMM_OPS: frozenset[str] = frozenset({"MatMul_QK", "MatMul_PV"})


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def _kernel_of(grammar: Grammar) -> dict[int, int]:
    """node_id → 0-based index of its top-level LoopLevel."""
    result: dict[int, int] = {}
    for idx, top in enumerate(grammar.program.children):
        for node in all_compute_nodes(top):
            result[node.node_id] = idx
    return result


def _global_tensors(grammar: Grammar, kernel_of: dict[int, int]) -> set[str]:
    """
    Tensor names that cross a kernel boundary → must live in device memory.
    'out' is always global (it is the function return value).
    """
    cross: set[str] = {"out"}
    for edge in grammar.ddg:
        pk = kernel_of.get(edge.producer_id)
        ck = kernel_of.get(edge.consumer_id)
        if pk is not None and ck is not None and pk != ck:
            cross.add(edge.tensor)
    return cross


def _out_tensor(grammar: Grammar, node: ComputeNode) -> str:
    """DDG tensor name produced by node (or 'out' for the sink)."""
    edges = grammar.consumers_of(node.node_id)
    return edges[0].tensor if edges else "out"


def _parallel_loops(top: LoopLevel) -> list[LoopLevel]:
    """
    Outermost contiguous run of parallel LoopLevels in the nesting.
    These become the CUDA grid axes (beyond the implicit Bsz*H axis).
    """
    result: list[LoopLevel] = []
    node: LoopLevel | ComputeNode = top
    while isinstance(node, LoopLevel) and node.parallel:
        result.append(node)
        if node.children and isinstance(node.children[0], LoopLevel):
            node = node.children[0]
        else:
            break
    return result


def _all_loops(node: LoopLevel | ComputeNode) -> list[LoopLevel]:
    """All LoopLevels in the subtree (pre-order)."""
    if isinstance(node, ComputeNode):
        return []
    result = [node]
    for child in node.children:
        result.extend(_all_loops(child))
    return result


def _tile_const(dim: str) -> str:
    """Tile-size constant name for a loop dimension."""
    return "TILE_KV" if "KV" in dim else "TILE_Q"


def _idx_var(dim: str) -> str:
    """Loop-index variable name for a loop dimension."""
    return dim.lower() + "_i"


# ---------------------------------------------------------------------------
# Slice generation
# ---------------------------------------------------------------------------

def _global_load_slice(tensor: str, grammar: Grammar) -> str:
    """
    Return the Python subscript expression for the per-block tile of *tensor*.
    Uses q_i / kv_i variables that are always in scope inside a kernel
    (either from ct.bid() for parallel dims or from the serial loop variable).
    Includes the b, h indices as the first two axes.
    """
    prod_id = next(
        (e.producer_id for e in grammar.ddg if e.tensor == tensor), None
    )
    if prod_id is None:
        # Should not happen for a valid grammar
        return f"{tensor}[b, h, ...]"

    prod_op = grammar.node_index[prod_id].op
    shape = _OP_TILE_SHAPE.get(prod_op, "(TILE_Q, TILE_KV)")

    if shape == "(TILE_Q, 1)":
        return f"{tensor}[b, h, seq_q_i*TILE_Q:(seq_q_i+1)*TILE_Q, :1]"
    elif shape == "(TILE_Q, D)":
        return f"{tensor}[b, h, seq_q_i*TILE_Q:(seq_q_i+1)*TILE_Q, :D]"
    elif shape == "(TILE_KV, D)":
        return f"{tensor}[b, h, seq_kv_i*TILE_KV:(seq_kv_i+1)*TILE_KV, :D]"
    else:  # (TILE_Q, TILE_KV)
        return (
            f"{tensor}[b, h, "
            f"seq_q_i*TILE_Q:(seq_q_i+1)*TILE_Q, "
            f"seq_kv_i*TILE_KV:(seq_kv_i+1)*TILE_KV]"
        )


def _global_store_slice(op: str) -> str:
    """
    Return the Python subscript expression for storing the output tile of *op*
    back to global memory.
    """
    shape = _OP_TILE_SHAPE.get(op, "(TILE_Q, TILE_KV)")
    if shape == "(TILE_Q, 1)":
        return "[b, h, seq_q_i*TILE_Q:(seq_q_i+1)*TILE_Q, :1]"
    elif shape == "(TILE_Q, D)":
        return "[b, h, seq_q_i*TILE_Q:(seq_q_i+1)*TILE_Q, :D]"
    elif shape == "(TILE_KV, D)":
        return "[b, h, seq_kv_i*TILE_KV:(seq_kv_i+1)*TILE_KV, :D]"
    else:  # (TILE_Q, TILE_KV)
        return (
            "[b, h, "
            "seq_q_i*TILE_Q:(seq_q_i+1)*TILE_Q, "
            "seq_kv_i*TILE_KV:(seq_kv_i+1)*TILE_KV]"
        )


# ---------------------------------------------------------------------------
# Per-op tile code emission
# ---------------------------------------------------------------------------

def _emit_op(
    node: ComputeNode,
    grammar: Grammar,
    global_tensors: set[str],
    local_node_ids: set[int],
    indent: str,
    in_serial_loop: bool,
) -> list[str]:
    """
    Emit all CuTile lines for one ComputeNode:
      - smem allocations and ct.copy loads for inputs
      - ct.sync() after loads
      - the compute expression
      - ct.copy store to global memory (for non-accumulated outputs)

    local_node_ids: set of node_ids that belong to the same kernel.  Used to
    decide whether a global tensor is arriving from another kernel (→ load via
    smem) or was produced earlier in THIS kernel (→ reference the _tile var).

    For carried-dim ops (RowMax / RowSum / MatMul_PV) the accumulator rmem
    is allocated and pre-initialised by _emit_kernel() before the serial loop;
    here we only emit the update expression.

    For gemm ops (MatMul_QK, MatMul_PV) we allocate a local rmem tile as the
    ct.gemm accumulator even when the op has no carried_dims (MatMul_QK),
    because ct.gemm accumulates in-place.
    """
    I = indent
    L: list[str] = []
    t_out = _out_tensor(grammar, node)
    op    = node.op

    # ── 1. Load global DDG inputs into smem ──────────────────────────────
    # Only load from global memory when the producer lives in a DIFFERENT
    # kernel.  If the producer is in this kernel the tensor is already
    # available as a local _tile / _acc variable.
    global_ddg_inputs: list[str] = [
        e.tensor for e in grammar.producers_of(node.node_id)
        if e.tensor in global_tensors and e.producer_id not in local_node_ids
    ]
    for tensor in global_ddg_inputs:
        smem_var  = tensor + "_smem"
        prod_id   = next(e.producer_id for e in grammar.ddg if e.tensor == tensor)
        prod_op   = grammar.node_index[prod_id].op
        tile_sh   = _OP_TILE_SHAPE.get(prod_op, "(TILE_Q, TILE_KV)")
        src_slice = _global_load_slice(tensor, grammar)
        L += [
            f"{I}{smem_var} = ct.smem({tile_sh})",
            f"{I}ct.copy({src_slice}, {smem_var})",
        ]

    # ── 2. Load external inputs (q, k, v) into smem ─────────────────────
    # For external inputs the index variable (seq_q_i / seq_kv_i) is
    # always in scope: parallel loops expose it via ct.bid(); serial loops
    # define it as the for-loop variable.
    ext = _OP_EXT.get(op, {})
    for role, pyvar in ext.items():
        smem_var = role + "_smem"   # q_smem, k_smem, v_smem
        if role == "q":
            L += [
                f"{I}{smem_var} = ct.smem((TILE_Q, D))",
                f"{I}ct.copy({pyvar}[b, h, seq_q_i*TILE_Q:(seq_q_i+1)*TILE_Q, :D], {smem_var})",
            ]
        elif role == "k":
            L += [
                f"{I}{smem_var} = ct.smem((TILE_KV, D))",
                f"{I}ct.copy({pyvar}[b, h, seq_kv_i*TILE_KV:(seq_kv_i+1)*TILE_KV, :D], {smem_var})",
            ]
        elif role == "v":
            # v is always consumed inside a serial SEQ_KV loop
            L += [
                f"{I}{smem_var} = ct.smem((TILE_KV, D))",
                f"{I}ct.copy({pyvar}[b, h, seq_kv_i*TILE_KV:(seq_kv_i+1)*TILE_KV, :D], {smem_var})",
            ]

    if global_ddg_inputs or ext:
        L.append(f"{I}ct.sync()")
        L.append("")

    # ── 3. Build substitution map for the expression template ────────────
    # Each placeholder {name} maps to the concrete variable name in scope.
    subs: dict[str, str] = {}
    # DDG inputs
    for edge in grammar.producers_of(node.node_id):
        tensor = edge.tensor
        if tensor in global_tensors and edge.producer_id not in local_node_ids:
            # Tensor crosses a kernel boundary → load from smem staging area
            subs[tensor] = tensor + "_smem"
        else:
            # Local tile produced by an earlier op in this kernel.
            # The producing op always stores the result in `tensor + "_tile"`.
            prod_node = grammar.node_index[edge.producer_id]
            if prod_node.carried_dims:
                subs[tensor] = tensor + "_acc"   # carried-dim op → _acc
            else:
                subs[tensor] = tensor + "_tile"  # all other ops → _tile
    # External inputs
    for role in ext:
        subs[role] = role + "_smem"

    # ── 4. Output tile variable name ─────────────────────────────────────
    # Always use `t_out + "_tile"` for non-accumulated outputs so that
    # a downstream op in the same kernel can reference it as `tensor_tile`.
    if node.carried_dims:
        # Accumulator was pre-allocated; use it directly
        out_var = t_out + "_acc"
    elif op in _GEMM_OPS:
        # Gemm ops need a local rmem tile even without carried_dims
        tile_sh = _OP_TILE_SHAPE.get(op, "(TILE_Q, TILE_KV)")
        out_var = t_out + "_tile"
        L.append(f"{I}{out_var} = ct.rmem({tile_sh}, fill=0.0)")
    else:
        out_var = t_out + "_tile"   # consistent: always _tile
    subs["out"] = out_var

    # ── 5. Emit the CuTile expression ────────────────────────────────────
    L.append(f"{I}# {op}")
    L.append(f"{I}{_cutile_expr(op, subs)}")
    L.append("")

    # ── 6. Store non-accumulated global outputs ──────────────────────────
    if not node.carried_dims and t_out in global_tensors:
        store_slice = _global_store_slice(op)
        L.append(f"{I}ct.copy({out_var}, {t_out}{store_slice})")
        L.append("")

    return L


def _cutile_expr(op: str, subs: dict[str, str]) -> str:
    """Return the CuTile expression for *op* with all {placeholders} filled."""
    templates: dict[str, str] = {
        # ct.gemm accumulates into {out}; {out} must be pre-initialised to 0.
        "MatMul_QK": "ct.gemm({q}, {k}, {out}, transpose_b=True)",
        # Elementwise scale (1/sqrt(D) is _inv_scale, computed in kernel header)
        "Scale":     "{out} = {S} * _inv_scale",
        # Running maximum; {out} is the accumulator (fill=-inf before loop)
        "RowMax":    "{out} = ct.maximum({out}, ct.max({S_scaled}, axis=1, keepdim=True))",
        # Elementwise subtract; {m} must be broadcast-compatible (TILE_Q, 1)
        "Subtract":  "{out} = {S_scaled} - {m}",
        # Elementwise exp
        "Exp":       "{out} = ct.exp({S_shifted})",
        # Running sum; {out} is the accumulator (fill=0 before loop)
        "RowSum":    "{out} = {out} + ct.sum({P}, axis=1, keepdim=True)",
        # Elementwise divide; {l} must be broadcast-compatible (TILE_Q, 1)
        "Divide":    "{out} = {P} / {l}",
        # ct.gemm accumulates partial A @ V tiles into {out}
        "MatMul_PV": "ct.gemm({A}, {v}, {out})",
    }
    tmpl = templates.get(op)
    if tmpl is None:
        raise NotImplementedError(f"No CuTile expression for op '{op}'")
    result = tmpl
    for ph, val in subs.items():
        result = result.replace("{" + ph + "}", val)
    return result


# ---------------------------------------------------------------------------
# Kernel body (recursive tree walk)
# ---------------------------------------------------------------------------

def _emit_body(
    node: LoopLevel | ComputeNode,
    kernel_nodes: list[ComputeNode],
    grammar: Grammar,
    global_tensors: set[str],
    local_node_ids: set[int],
    indent: str,
    in_serial_loop: bool,
) -> list[str]:
    """Recursively emit the kernel body for a subtree."""
    I = indent
    L: list[str] = []

    if isinstance(node, ComputeNode):
        return _emit_op(node, grammar, global_tensors, local_node_ids, I, in_serial_loop)

    loop: LoopLevel = node  # type: ignore[assignment]

    if loop.parallel:
        # Parallel → already mapped to ct.bid(); recurse without a loop wrapper
        for child in loop.children:
            L += _emit_body(child, kernel_nodes, grammar, global_tensors,
                            local_node_ids, I, False)
        return L

    # ── Serial loop ───────────────────────────────────────────────────────
    loop_var   = _idx_var(loop.dim)
    tile_const = _tile_const(loop.dim)

    L.append(f"{I}# Serial loop over {loop.dim}  (bound={loop.bound}, tile={tile_const})")
    L.append(f"{I}for {loop_var} in range({loop.bound} // {tile_const}):")
    II = I + "    "

    for child in loop.children:
        L += _emit_body(child, kernel_nodes, grammar, global_tensors,
                        local_node_ids, II, True)

    L.append(f"{II}ct.sync()")
    L.append("")

    return L


# ---------------------------------------------------------------------------
# Full kernel emitter
# ---------------------------------------------------------------------------

def _emit_kernel(
    kid: int,
    top: LoopLevel,
    grammar: Grammar,
    global_tensors: set[str],
    tile_q: int,
    tile_kv: int,
) -> list[str]:
    """Return source lines for one @ct.kernel function."""
    nodes    = all_compute_nodes(top)
    par_loops = _parallel_loops(top)

    fn_name = "_stage_{}_{}".format(kid, "_".join(n.op for n in nodes))

    # ── Parameters ────────────────────────────────────────────────────────
    # External tensor pointers
    ext_params: list[str] = []
    for n in nodes:
        for pv in _OP_EXT.get(n.op, {}).values():
            if pv not in ext_params:
                ext_params.append(pv)
    # Global input tensors produced by an EARLIER kernel.
    # If the producer is in THIS kernel the consumer reads from the local
    # _tile variable; we must not also pass the tensor as a function param.
    node_ids_in_kernel: set[int] = {n.node_id for n in nodes}
    global_in: list[str] = []
    for n in nodes:
        for edge in grammar.producers_of(n.node_id):
            if (edge.tensor in global_tensors
                    and edge.producer_id not in node_ids_in_kernel
                    and edge.tensor not in global_in):
                global_in.append(edge.tensor)
    # Global output tensors produced by this kernel
    global_out: list[str] = []
    for n in nodes:
        t = _out_tensor(grammar, n)
        if t in global_tensors and t not in global_out:
            global_out.append(t)

    params = ext_params + global_in + global_out + ["H: int", "D: int", "Sq: int", "Skv: int"]

    L: list[str] = []

    # ── Header comment ────────────────────────────────────────────────────
    loop_summary = " → ".join(
        f"{lp.dim}({'par' if lp.parallel else 'ser'})"
        for lp in _all_loops(top)
    )
    grid_axes = ["Bsz*H"] + [
        f"{lp.dim}//{_tile_const(lp.dim)}" for lp in par_loops
    ]
    L += [
        f"# ── Kernel {kid}: {', '.join(n.op for n in nodes)} ──",
        f"# Loops: {loop_summary}",
        f"# Grid:  ({', '.join(grid_axes)})",
        "@ct.kernel",
        f"def {fn_name}({', '.join(params)}):",
    ]
    I = "    "

    L += [
        f"{I}TILE_Q  = {tile_q}",
        f"{I}TILE_KV = {tile_kv}",
        f"{I}_inv_scale = 1.0 / math.sqrt(D)",
        "",
    ]

    # ── Block-ID unpacking ────────────────────────────────────────────────
    L += [
        f"{I}# Block IDs → (batch, head, tile indices)",
        f"{I}bh_i     = ct.bid(0)  # batch * head",
        f"{I}b        = bh_i // H",
        f"{I}h        = bh_i  % H",
    ]
    for axis, lp in enumerate(par_loops, start=1):
        var = _idx_var(lp.dim)
        L.append(f"{I}{var:10s} = ct.bid({axis})  # {lp.dim} tile index")
    L.append("")

    # ── Accumulators for carried-dim ops ──────────────────────────────────
    for n in nodes:
        if n.carried_dims:
            t_out   = _out_tensor(grammar, n)
            tile_sh = _OP_TILE_SHAPE.get(n.op, "(TILE_Q, 1)")
            init    = _OP_ACC_INIT.get(n.op, "0.0")
            L.append(
                f"{I}{t_out}_acc = ct.rmem({tile_sh}, fill={init})"
                f"  # {n.op} accumulator"
            )
    if any(n.carried_dims for n in nodes):
        L.append("")

    # ── Kernel body (recursive walk) ──────────────────────────────────────
    local_ids = {n.node_id for n in nodes}
    L += _emit_body(top, nodes, grammar, global_tensors, local_ids, I, False)

    # ── Store accumulated global outputs (after the serial loop) ─────────
    for n in nodes:
        if not n.carried_dims:
            continue
        t_out = _out_tensor(grammar, n)
        if t_out not in global_tensors:
            continue
        acc_var     = t_out + "_acc"
        store_slice = _global_store_slice(n.op)
        L.append(f"{I}ct.copy({acc_var}, {t_out}{store_slice})")
    L.append("")

    return L


# ---------------------------------------------------------------------------
# Wrapper emitter
# ---------------------------------------------------------------------------

def _emit_wrapper(
    grammar: Grammar,
    kernel_names: list[str],
    global_tensors: set[str],
    tile_q: int,
    tile_kv: int,
) -> list[str]:
    """
    Emit attention(q, k, v) → out wrapper that:
      1. Permutes inputs to (B, H, S, D)
      2. Allocates global intermediate tensors with torch.empty
      3. Launches each @ct.kernel with the correct grid tuple
      4. Permutes output back to (B, S, H, D) and returns it
    """
    ordered = all_compute_nodes(grammar.program)

    L: list[str] = [
        "def attention(q, k, v):",
        "    # q, k, v: (B, S, H, D)  float CUDA tensors",
        "    Bsz, Sq, H, D = q.shape",
        "    Skv = k.shape[1]",
        "    dtype  = q.dtype",
        "    device = q.device",
        "    _stream = cupy.cuda.get_current_stream()",
        "",
        "    # Permute to (B, H, S, D) for tiled computation",
        "    _q = q.permute(0, 2, 1, 3).contiguous()",
        "    _k = k.permute(0, 2, 1, 3).contiguous()",
        "    _v = v.permute(0, 2, 1, 3).contiguous()",
        "",
        "    # Allocate global (cross-kernel) tensors",
    ]

    for n in ordered:
        t = _out_tensor(grammar, n)
        if t in global_tensors:
            shape = _OP_ALLOC_SHAPE.get(n.op, "(Bsz, H, Sq, D)")
            L.append(f"    {t} = torch.empty({shape}, dtype=dtype, device=device)")
    L.append("")

    for kid, (fn_name, top) in enumerate(
        zip(kernel_names, grammar.program.children)
    ):
        nodes_k   = all_compute_nodes(top)
        par_loops = _parallel_loops(top)

        # Grid tuple: (Bsz*H, [Sq//tile_q], [Skv//tile_kv])
        grid_dims = ["Bsz * H"]
        for lp in par_loops:
            if "KV" in lp.dim:
                grid_dims.append(f"Skv // {tile_kv}")
            else:
                grid_dims.append(f"Sq  // {tile_q}")
        grid_str = "(" + ", ".join(grid_dims) + ",)"

        # Positional arguments
        ext_args: list[str] = []
        for n in nodes_k:
            for pv in _OP_EXT.get(n.op, {}).values():
                if pv not in ext_args:
                    ext_args.append(pv)
        node_ids_k = {n.node_id for n in nodes_k}
        global_in_args: list[str] = []
        for n in nodes_k:
            for edge in grammar.producers_of(n.node_id):
                if (edge.tensor in global_tensors
                        and edge.producer_id not in node_ids_k
                        and edge.tensor not in global_in_args):
                    global_in_args.append(edge.tensor)
        global_out_args: list[str] = []
        for n in nodes_k:
            t = _out_tensor(grammar, n)
            if t in global_tensors and t not in global_out_args:
                global_out_args.append(t)

        all_pos  = ext_args + global_in_args + global_out_args

        all_args     = all_pos + ["H", "D", "Sq", "Skv"]
        args_tup_str = "(" + ", ".join(all_args) + ",)"

        ops_label = "+".join(n.op for n in nodes_k)
        L.append(f"    # kernel {kid}: {ops_label}")
        L.append(f"    ct.launch(_stream, {grid_str}, {fn_name}, {args_tup_str})")

    L += [
        "",
        "    # Permute output back to (B, S, H, D)",
        "    return out.permute(0, 2, 1, 3).contiguous()",
    ]
    return L


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render_source(
    grammar: Grammar,
    tile_q: int = 64,
    tile_kv: int = 64,
    output_path: str | None = None,
) -> str:
    """
    Return the generated CuTile Python source for *grammar*.

    Use for inspection and debugging::

        print(render_source(grammar))

    Parameters
    ----------
    grammar     : Grammar
    tile_q      : query-sequence tile size per block (default 64)
    tile_kv     : key-value tile size per block (default 64)
    output_path : if provided, write the source to this file path in addition
                  to returning it (creates or overwrites the file)
    """
    k_of    = _kernel_of(grammar)
    globals_= _global_tensors(grammar, k_of)

    lines: list[str] = [
        "# CuTile attention kernel — generated by renderer",
        f"# Grammar hash  : {tree_hash(grammar)}",
        f"# Kernels        : {len(grammar.program.children)}",
        f"# Global tensors : {sorted(globals_)}",
        "",
        "import math",
        "import cupy",
        "import torch",
        "import cuda.tile as ct",
        "",
    ]

    kernel_names: list[str] = []
    for kid, top in enumerate(grammar.program.children):
        nodes   = all_compute_nodes(top)
        fn_name = "_stage_{}_{}".format(kid, "_".join(n.op for n in nodes))
        kernel_names.append(fn_name)
        lines += _emit_kernel(kid, top, grammar, globals_, tile_q, tile_kv)

    lines += _emit_wrapper(grammar, kernel_names, globals_, tile_q, tile_kv)
    source = "\n".join(lines)

    if output_path is not None:
        import os
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w") as f:
            f.write(source)

    return source


def render(
    grammar: Grammar,
    tile_q: int = 64,
    tile_kv: int = 64,
    output_path: str | None = None,
) -> Callable:
    """
    Compile *grammar* to an ``attention(q, k, v) → out`` callable.

    All tensors have shape ``(B, S, H, D)`` and must be float CUDA tensors.
    The returned callable launches one CuTile kernel per top-level LoopLevel
    in the grammar's ProgramNode.

    The source is written to *output_path* (if provided) and then loaded as a
    module via importlib.  This avoids exec() so that missing runtime
    dependencies (torch, cuda.tile) fail with a clear ImportError at module
    load time rather than inside an opaque exec call.  A static check on the
    source text verifies that an ``attention`` function was produced before any
    import is attempted.

    Parameters
    ----------
    grammar     : Grammar
    tile_q      : query-sequence tile size per block (default 64)
    tile_kv     : key-value tile size per block (default 64)
    output_path : path where the generated source is written; if None a
                  temporary file is used and deleted after loading

    Raises
    ------
    NotImplementedError
        If the grammar contains an op with no CuTile expression.
    RuntimeError
        If the generated source does not contain an ``attention`` definition
        (static check — indicates a renderer bug).
    ImportError
        If the module cannot be loaded because runtime dependencies are absent
        (torch, cuda.tile).  Run inside the ``attn_bench`` conda environment
        on a GCE VM.
    """
    import importlib.util
    import os
    import tempfile

    source = render_source(grammar, tile_q=tile_q, tile_kv=tile_kv,
                           output_path=output_path)

    # ── Static verification ───────────────────────────────────────────────
    # Check that the source contains an attention function definition before
    # attempting to import it, so renderer bugs surface clearly.
    if "def attention(" not in source:
        raise RuntimeError(
            "Renderer did not produce an 'attention' function definition.\n"
            "This is a renderer bug — inspect render_source() output."
        )

    # ── Load the module from a file (no exec) ────────────────────────────
    # importlib.util.spec_from_file_location gives Python a real module with
    # a proper file path, which means tracebacks and debuggers can locate
    # source lines correctly.
    delete_after = output_path is None
    if delete_after:
        fd, file_path = tempfile.mkstemp(suffix=".py", prefix="cutile_kernel_")
        os.close(fd)
        with open(file_path, "w") as f:
            f.write(source)
    else:
        file_path = output_path

    try:
        spec   = importlib.util.spec_from_file_location("cutile_kernel", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        if delete_after:
            os.unlink(file_path)

    fn = getattr(module, "attention", None)
    if fn is None:
        raise RuntimeError(
            "Module loaded successfully but 'attention' attribute is missing."
        )
    return fn
