"""
Atomic grammar for the attention kernel.

This is the starting state of the optimizer: a single GPU kernel containing
eight sequential stages, one per compute node, with all intermediates in global
memory and no fusion. This is the "compute all at root" state from the Halide
autoscheduler paper — the maximally unfused, obviously-correct baseline from
which all rewrites proceed.

The entire ProgramNode is one kernel launch. Each top-level LoopLevel child is
one stage within that kernel; an implicit barrier synchronizes threads between
stages. The loop structure for each stage is derived from its declared dims:
  output_dims  → parallel loops (parallel=True)
  carried_dims → serial loops  (parallel=False)

See AST-spec.md § "The Atomic Grammar (Initial State)" for the full spec.
"""

from __future__ import annotations

from .ast import ComputeNode, DDGEdge, Grammar, LoopLevel, ProgramNode


def attention_atomic_grammar(
    seq_q: int = 2048,
    seq_kv: int = 2048,
) -> Grammar:
    """
    Build and return the atomic attention grammar.

    Parameters:
        seq_q:  number of query positions (SEQ_Q loop bound).
        seq_kv: number of key/value positions (SEQ_KV loop bound).

    Both values must be powers of 2 to satisfy LoopLevel's bound invariant.

    Tree structure (single GPU kernel, 8 sequential stages with barriers):

        ProgramNode                                          # one kernel launch
        ├── LoopLevel(SEQ_Q, seq_q, parallel=True)          # stage 1: MatMul_QK
        │   └── LoopLevel(SEQ_KV, seq_kv, parallel=True)
        │       └── Compute(MatMul_QK)
        ├── <barrier>
        ├── LoopLevel(SEQ_Q, seq_q, parallel=True)          # stage 2: Scale
        │   └── LoopLevel(SEQ_KV, seq_kv, parallel=True)
        │       └── Compute(Scale)
        ├── <barrier>
        ├── LoopLevel(SEQ_Q, seq_q, parallel=True)          # stage 3: RowMax
        │   └── LoopLevel(SEQ_KV, seq_kv, parallel=False)
        │       └── Compute(RowMax)
        ├── <barrier>
        ├── LoopLevel(SEQ_Q, seq_q, parallel=True)          # stage 4: Subtract
        │   └── LoopLevel(SEQ_KV, seq_kv, parallel=True)
        │       └── Compute(Subtract)
        ├── <barrier>
        ├── LoopLevel(SEQ_Q, seq_q, parallel=True)          # stage 5: Exp
        │   └── LoopLevel(SEQ_KV, seq_kv, parallel=True)
        │       └── Compute(Exp)
        ├── <barrier>
        ├── LoopLevel(SEQ_Q, seq_q, parallel=True)          # stage 6: RowSum
        │   └── LoopLevel(SEQ_KV, seq_kv, parallel=False)
        │       └── Compute(RowSum)
        ├── <barrier>
        ├── LoopLevel(SEQ_Q, seq_q, parallel=True)          # stage 7: Divide
        │   └── LoopLevel(SEQ_KV, seq_kv, parallel=True)
        │       └── Compute(Divide)
        ├── <barrier>
        └── LoopLevel(SEQ_Q, seq_q, parallel=True)          # stage 8: MatMul_PV
            └── LoopLevel(SEQ_KV, seq_kv, parallel=False)
                └── Compute(MatMul_PV)

    DDG:
        MatMul_QK -> Scale       [S]
        Scale     -> RowMax      [S_scaled]
        Scale     -> Subtract    [S_scaled]
        RowMax    -> Subtract    [m]
        Subtract  -> Exp         [S_shifted]
        Exp       -> RowSum      [P]
        Exp       -> Divide      [P]
        RowSum    -> Divide      [l]
        Divide    -> MatMul_PV   [A]
    """
    # ------------------------------------------------------------------
    # Kernel 1: MatMul_QK
    # output_dims={SEQ_Q, SEQ_KV}, carried_dims={}
    # → SEQ_Q(parallel) → SEQ_KV(parallel)
    # ------------------------------------------------------------------
    matmul_qk = ComputeNode.make(
        "MatMul_QK",
        output_dims={"SEQ_Q", "SEQ_KV"},
        carried_dims={},
    )
    k1 = LoopLevel(
        dim="SEQ_Q",
        bound=seq_q,
        parallel=True,
        children=(
            LoopLevel(
                dim="SEQ_KV",
                bound=seq_kv,
                parallel=True,
                children=(matmul_qk,),
            ),
        ),
    )

    # ------------------------------------------------------------------
    # Kernel 2: Scale
    # output_dims={SEQ_Q, SEQ_KV}, carried_dims={}
    # → SEQ_Q(parallel) → SEQ_KV(parallel)
    # ------------------------------------------------------------------
    scale = ComputeNode.make(
        "Scale",
        output_dims={"SEQ_Q", "SEQ_KV"},
        carried_dims={},
    )
    k2 = LoopLevel(
        dim="SEQ_Q",
        bound=seq_q,
        parallel=True,
        children=(
            LoopLevel(
                dim="SEQ_KV",
                bound=seq_kv,
                parallel=True,
                children=(scale,),
            ),
        ),
    )

    # ------------------------------------------------------------------
    # Kernel 3: RowMax
    # output_dims={SEQ_Q}, carried_dims={SEQ_KV}
    # → SEQ_Q(parallel) → SEQ_KV(serial)
    # ------------------------------------------------------------------
    row_max = ComputeNode.make(
        "RowMax",
        output_dims={"SEQ_Q"},
        carried_dims={"SEQ_KV"},
    )
    k3 = LoopLevel(
        dim="SEQ_Q",
        bound=seq_q,
        parallel=True,
        children=(
            LoopLevel(
                dim="SEQ_KV",
                bound=seq_kv,
                parallel=False,
                children=(row_max,),
            ),
        ),
    )

    # ------------------------------------------------------------------
    # Kernel 4: Subtract
    # output_dims={SEQ_Q, SEQ_KV}, carried_dims={}
    # → SEQ_Q(parallel) → SEQ_KV(parallel)
    # ------------------------------------------------------------------
    subtract = ComputeNode.make(
        "Subtract",
        output_dims={"SEQ_Q", "SEQ_KV"},
        carried_dims={},
    )
    k4 = LoopLevel(
        dim="SEQ_Q",
        bound=seq_q,
        parallel=True,
        children=(
            LoopLevel(
                dim="SEQ_KV",
                bound=seq_kv,
                parallel=True,
                children=(subtract,),
            ),
        ),
    )

    # ------------------------------------------------------------------
    # Kernel 5: Exp
    # output_dims={SEQ_Q, SEQ_KV}, carried_dims={}
    # → SEQ_Q(parallel) → SEQ_KV(parallel)
    # ------------------------------------------------------------------
    exp = ComputeNode.make(
        "Exp",
        output_dims={"SEQ_Q", "SEQ_KV"},
        carried_dims={},
    )
    k5 = LoopLevel(
        dim="SEQ_Q",
        bound=seq_q,
        parallel=True,
        children=(
            LoopLevel(
                dim="SEQ_KV",
                bound=seq_kv,
                parallel=True,
                children=(exp,),
            ),
        ),
    )

    # ------------------------------------------------------------------
    # Kernel 6: RowSum
    # output_dims={SEQ_Q}, carried_dims={SEQ_KV}
    # → SEQ_Q(parallel) → SEQ_KV(serial)
    # ------------------------------------------------------------------
    row_sum = ComputeNode.make(
        "RowSum",
        output_dims={"SEQ_Q"},
        carried_dims={"SEQ_KV"},
    )
    k6 = LoopLevel(
        dim="SEQ_Q",
        bound=seq_q,
        parallel=True,
        children=(
            LoopLevel(
                dim="SEQ_KV",
                bound=seq_kv,
                parallel=False,
                children=(row_sum,),
            ),
        ),
    )

    # ------------------------------------------------------------------
    # Kernel 7: Divide
    # output_dims={SEQ_Q, SEQ_KV}, carried_dims={}
    # → SEQ_Q(parallel) → SEQ_KV(parallel)
    # ------------------------------------------------------------------
    divide = ComputeNode.make(
        "Divide",
        output_dims={"SEQ_Q", "SEQ_KV"},
        carried_dims={},
    )
    k7 = LoopLevel(
        dim="SEQ_Q",
        bound=seq_q,
        parallel=True,
        children=(
            LoopLevel(
                dim="SEQ_KV",
                bound=seq_kv,
                parallel=True,
                children=(divide,),
            ),
        ),
    )

    # ------------------------------------------------------------------
    # Kernel 8: MatMul_PV
    # output_dims={SEQ_Q}, carried_dims={SEQ_KV}
    # → SEQ_Q(parallel) → SEQ_KV(serial)
    # ------------------------------------------------------------------
    matmul_pv = ComputeNode.make(
        "MatMul_PV",
        output_dims={"SEQ_Q"},
        carried_dims={"SEQ_KV"},
    )
    k8 = LoopLevel(
        dim="SEQ_Q",
        bound=seq_q,
        parallel=True,
        children=(
            LoopLevel(
                dim="SEQ_KV",
                bound=seq_kv,
                parallel=False,
                children=(matmul_pv,),
            ),
        ),
    )

    program = ProgramNode(children=(k1, k2, k3, k4, k5, k6, k7, k8))

    # ------------------------------------------------------------------
    # DDG
    # ------------------------------------------------------------------
    ddg = (
        DDGEdge(matmul_qk.node_id, scale.node_id,    "S"),
        DDGEdge(scale.node_id,     row_max.node_id,   "S_scaled"),
        DDGEdge(scale.node_id,     subtract.node_id,  "S_scaled"),
        DDGEdge(row_max.node_id,   subtract.node_id,  "m"),
        DDGEdge(subtract.node_id,  exp.node_id,       "S_shifted"),
        DDGEdge(exp.node_id,       row_sum.node_id,   "P"),
        DDGEdge(exp.node_id,       divide.node_id,    "P"),
        DDGEdge(row_sum.node_id,   divide.node_id,    "l"),
        DDGEdge(divide.node_id,    matmul_pv.node_id, "A"),
    )

    return Grammar(program=program, ddg=ddg)
