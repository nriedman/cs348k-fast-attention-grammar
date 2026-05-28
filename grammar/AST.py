"""Barebones kernel AST: nodes, backward tile-shape inference, a cuTile
renderer (external visitor), and a structural cache for equivalent ASTs.

Scope of this first cut:
  - nodes: Program, ParallelLoop, Load, Store, Compute
  - no rewrite rules yet
  - Load/Store tile sizes are DERIVED at render time from the dataflow DAG,
    seeded by the ParallelLoop's output tiling (Halide-style bounds flow).
"""

from __future__ import annotations
from dataclasses import dataclass, field
import math

Shape = tuple[int, ...]


# --------------------------------------------------------------------------
# Nodes
# --------------------------------------------------------------------------
# Marker base classes let us (later) type-check what may appear where.
class Node: ...
class Stmt(Node): ...     # may appear in a scope body
class Value(Node): ...    # produces a tile; usable as a Compute/Store operand


@dataclass
class Program(Node):
    # Global shapes of every tensor the kernels read or write. This is the
    # "signature" inference uses to recover dims the output tile can't reveal
    # (e.g. a matmul's contraction dim K).
    tensors: dict[str, Shape]
    body: list["ParallelLoop"] = field(default_factory=list)


@dataclass
class ParallelLoop(Stmt):
    out: str                       # output tensor this kernel writes
    tile_shape: Shape              # shape of the output tile one instance computes
    index_vars: tuple[str, ...]    # grid index var per tiled output dim, e.g. ("bm","bn")
    body: list[Stmt] = field(default_factory=list)


@dataclass
class Load(Stmt, Value):
    source: str                    # global tensor to read from
    index: list[str]               # one index NAME per dim of `source`, e.g. ["bm", "k"]
    # No shape here on purpose -- it is inferred at render time. Each name is
    # the logical axis it rides on; whether that axis is currently tiled (gets a
    # live coordinate) or whole (collapses to tile 0) is derived from the shape.


@dataclass
class Compute(Stmt, Value):
    op: str                        # "matmul", "add", "relu", ...
    inputs: list[Value] = field(default_factory=list)


@dataclass
class Store(Stmt):
    dest: str                      # global tensor to write to
    src: Value                     # tile to store
    index: list[str]               # one index NAME per dim of `dest`, e.g. ["bm", "bn"]


@dataclass
class ReductionLoop(Stmt, Value):
    """A sequential loop that tiles an existing inner (contraction) dimension.

    It is a Value: its result is the final accumulator. `axis` is the index
    NAME of the dimension being reduced (already present in the body's loads,
    e.g. "k"); the loop binds it in scope and flips it from whole -> tiled.
    `tile` is the tile size ALONG that dimension (TS_<axis>, a tunable). The
    accumulator's init and its per-iteration update are derived from
    `partial.op` via REDUCE_RULES (e.g. matmul -> zeros + ct.mma)."""
    axis: str                      # reduced dimension's index name, e.g. "k"
    tile: int                      # tile size along `axis` (TS_<axis>, tunable)
    body: list[Stmt] = field(default_factory=list)
    partial: Value = None          # the body Compute accumulated each iteration


# --------------------------------------------------------------------------
# Backward shape rules: given an op's OUTPUT tile shape (and the global shapes
# of its operands, for dims the output projects away), return the required
# INPUT tile shapes -- one per input, in order. `reduce_tile`, when set, is the
# tile size of a contraction dim being reduced (so the input tile uses TS_k
# instead of the full extent K).
# --------------------------------------------------------------------------
def _elementwise(out_shape: Shape, operand_globals: list[Shape | None],
                 reduce_tile: int | None = None) -> list[Shape]:
    # add / mul / relu / exp: every input tile matches the output tile.
    return [out_shape for _ in operand_globals]


def _matmul(out_shape: Shape, operand_globals: list[Shape | None],
            reduce_tile: int | None = None) -> list[Shape]:
    # out (M, N) <- A (M, K) @ B (K, N). The contraction dim is the full extent
    # K (recovered from operand A's global shape) unless we're inside a
    # ReductionLoop, in which case it is the reduction tile TS_k.
    m, n = out_shape
    k = reduce_tile if reduce_tile is not None else operand_globals[0][1]
    return [(m, k), (k, n)]


SHAPE_RULES = {
    "matmul": _matmul,
    "add": _elementwise,
    "mul": _elementwise,
    "relu": _elementwise,
    "exp": _elementwise,
}


# --------------------------------------------------------------------------
# Emission rules: map a Compute op to the cuTile expression that computes it,
# given the already-emitted variable names of its inputs. The op name in the
# AST is logical; the cuTile (1.3.0) spelling can differ -- e.g. relu maps to
# ct.maximum(0, x). Adding an op = one entry here + one in SHAPE_RULES above.
# Unknown ops are rejected rather than emitted blindly.
# --------------------------------------------------------------------------
EMIT_RULES = {
    # all confirmed against cuTile 1.3.0
    "matmul": lambda a: f"ct.matmul({a[0]}, {a[1]})",
    "add":    lambda a: f"ct.add({a[0]}, {a[1]})",
    "relu":   lambda a: f"ct.maximum(0, {a[0]})",
    "mul":    lambda a: f"{a[0]} * {a[1]}",
    "exp":    lambda a: f"ct.exp({a[0]})",
}


# --------------------------------------------------------------------------
# Reduction rules: how an op behaves when it is the `partial` of a
# ReductionLoop. Each entry is (init, accumulate):
#   init(out_shape_expr, dtype_expr) -> str   # accumulator initializer
#   accumulate(arg_names, acc) -> str         # per-iteration update RHS
# The accumulate form takes the accumulator as an operand (fused), which is
# what unlocks tensor cores -- ct.mma(t1, t2, acc) rather than acc + matmul.
# Only ops listed here may sit in a ReductionLoop's `partial` position.
# --------------------------------------------------------------------------
REDUCE_RULES = {
    "matmul": (
        lambda shp, dtype: f"ct.zeros({shp}, dtype={dtype})",
        lambda a, acc:     f"ct.mma({a[0]}, {a[1]}, {acc})",
    ),
    # later, e.g.:
    # "rowmax": (lambda shp, dtype: f"ct.full({shp}, -inf, dtype={dtype})",
    #            lambda a, acc: f"ct.maximum({acc}, ct.row_max({a[0]}))"),
}


def _resolve_index(index: list[str], tile_shape: Shape, global_shape: Shape,
                   scope: set[str], what: str) -> list[str]:
    """Turn per-dim index NAMES into emitted tile coordinates.

    A dim is *tiled* iff its inferred tile is smaller than the global dim; then
    its name must name a live index in `scope` and we emit that name. Otherwise
    the dim is loaded/stored whole and collapses to tile index 0 -- the name is
    then a latent axis (e.g. a reduction index not yet wrapped in a loop)."""
    assert len(index) == len(tile_shape) == len(global_shape), (
        f"{what}: index rank {len(index)} must match tensor rank {len(global_shape)}"
    )
    out: list[str] = []
    for d, (name, t, g) in enumerate(zip(index, tile_shape, global_shape)):
        if t < g:                                   # subdivided -> needs a live index
            if name not in scope:
                raise ValueError(
                    f"{what}: dim {d} is tiled ({t} of {g}) but index name "
                    f"{name!r} is not in scope {sorted(scope)}")
            out.append(name)
        else:                                       # whole dim -> single tile
            out.append("0")
    return out


# --------------------------------------------------------------------------
# Renderer: external visitor. Runs backward shape inference per kernel, then
# emits cuTile-ish code. (The ct.* strings are illustrative placeholders --
# swap them for the real cuTile API surface you target.)
# --------------------------------------------------------------------------
class CuTileRenderer:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.indent = 0
        self.shapes: dict[int, Shape] = {}   # id(value) -> inferred tile shape
        self.names: dict[int, str] = {}      # id(value) -> emitted var name
        self._n = 0

    # ---- shape inference (backward from each Store) ----
    def _infer(self, value: Value, out_shape: Shape, tensors: dict[str, Shape],
               reduce_tile: int | None = None) -> None:
        prev = self.shapes.get(id(value))
        if prev is not None:
            assert prev == out_shape, (
                f"conflicting tile shapes for {value!r}: {prev} vs {out_shape}"
            )
            return
        self.shapes[id(value)] = out_shape
        if isinstance(value, ReductionLoop):
            # the loop's result is the accumulator (same shape as the output
            # tile); infer the partial with the contraction tiled by `tile`.
            self._infer(value.partial, out_shape, tensors, reduce_tile=value.tile)
        elif isinstance(value, Compute):
            operand_globals = [
                tensors.get(i.source) if isinstance(i, Load) else None
                for i in value.inputs
            ]
            in_shapes = SHAPE_RULES[value.op](out_shape, operand_globals, reduce_tile)
            for inp, shp in zip(value.inputs, in_shapes):
                self._infer(inp, shp, tensors)
        # Load: nothing further -- its tile shape is now fixed.

    # ---- emission helpers ----
    def _emit(self, s: str) -> None:
        self.lines.append("    " * self.indent + s)

    def _fresh(self, prefix: str = "t") -> str:
        self._n += 1
        return f"{prefix}{self._n}"

    def _name(self, v: Value) -> str:
        return self.names[id(v)]

    # ---- visitor dispatch ----
    def render(self, program: Program) -> str:
        assert isinstance(program, Program)
        for loop in program.body:
            self._render_kernel(loop, program.tensors)
        return "\n".join(self.lines).rstrip() + "\n"

    def _render_kernel(self, loop: ParallelLoop, tensors: dict[str, Shape]) -> None:
        # index resolution context for this kernel
        self._tensors = tensors
        self._scope = set(loop.index_vars)   # live tile indices (grid vars for now)
        self._cur_out = loop.out             # output tensor (for accumulator dtype)
        # 1. infer every tile shape, seeded by the output tile at each Store.
        for stmt in loop.body:
            if isinstance(stmt, Store):
                self._infer(stmt.src, tuple(loop.tile_shape), tensors)
        # 2. emit the kernel.
        args = ", ".join(_loop_tensors(loop))
        grid = tuple(
            tensors[loop.out][d] // loop.tile_shape[d]
            for d in range(len(loop.tile_shape))
        )
        self._emit("@ct.kernel")
        self._emit(f"def {loop.out}_kernel({args}):  # launch grid {self._launch_grid(grid)}")
        self.indent += 1
        self._emit_grid_decode(loop.index_vars, grid)
        for stmt in loop.body:
            self._visit(stmt)
        self.indent -= 1
        self.lines.append("")

    def _visit(self, node: Node) -> None:
        getattr(self, f"_visit_{type(node).__name__}")(node)

    @staticmethod
    def _launch_grid(grid: Shape) -> Shape:
        # CUDA grids are at most 3-D. With <=3 tiled dims we map one per axis.
        # With more, the leading dims are linearized onto grid.x -- the axis
        # with the large 2**31-1 extent limit -- and the last two stay on
        # grid.y / grid.z (capped at 65535 each, which tile-grid dims fit).
        if len(grid) <= 3:
            return grid
        lead = 1
        for g in grid[:-2]:
            lead *= g
        return (lead, grid[-2], grid[-1])

    def _emit_grid_decode(self, index_vars: tuple[str, ...], grid: Shape) -> None:
        n = len(index_vars)
        if n <= 3:
            for axis, var in enumerate(index_vars):
                self._emit(f"{var} = ct.bid({axis})")
            return
        # last two tiled dims keep dedicated grid axes
        self._emit(f"{index_vars[-2]} = ct.bid(1)")
        self._emit(f"{index_vars[-1]} = ct.bid(2)")
        # leading dims are row-major linearized on grid.x; peel innermost first
        self._emit("_flat = ct.bid(0)")
        lead_vars, lead_grid = index_vars[:-2], grid[:-2]
        for d in range(len(lead_vars) - 1, -1, -1):
            if d == 0:
                self._emit(f"{lead_vars[0]} = _flat")          # outermost remainder
            else:
                self._emit(f"{lead_vars[d]} = _flat % {lead_grid[d]}")
                self._emit(f"_flat //= {lead_grid[d]}")

    def _visit_Load(self, n: Load) -> None:
        nm = self._fresh()
        self.names[id(n)] = nm
        shp = self.shapes[id(n)]
        idx = _resolve_index(n.index, shp, self._tensors[n.source],
                             self._scope, f"load {n.source}")
        self._emit(f"{nm} = ct.load({n.source}, ({', '.join(idx)}), {shp})")

    def _visit_Compute(self, n: Compute) -> None:
        if n.op not in EMIT_RULES:
            raise ValueError(
                f"no cuTile emission rule for compute op {n.op!r}; "
                f"supported: {sorted(EMIT_RULES)}")
        nm = self._fresh()
        arg_names = [self._name(i) for i in n.inputs]
        rhs = EMIT_RULES[n.op](arg_names)
        self.names[id(n)] = nm
        self._emit(f"{nm} = {rhs}  # tile {self.shapes[id(n)]}")

    def _visit_Store(self, n: Store) -> None:
        shp = self.shapes[id(n.src)]
        idx = _resolve_index(n.index, shp, self._tensors[n.dest],
                             self._scope, f"store {n.dest}")
        self._emit(
            f"ct.store({n.dest}, ({', '.join(idx)}), {self._name(n.src)})"
            f"  # tile {shp}"
        )

    def _reduction_extent(self, n: "ReductionLoop") -> int:
        """Global size of the reduced dimension (from a body load on `axis`)."""
        for s in n.body:
            if isinstance(s, Load) and n.axis in s.index:
                return self._tensors[s.source][s.index.index(n.axis)]
        raise ValueError(f"reduction axis {n.axis!r} not found in any body load")

    def _visit_ReductionLoop(self, n: "ReductionLoop") -> None:
        if n.partial.op not in REDUCE_RULES:
            raise ValueError(
                f"op {n.partial.op!r} cannot be a reduction partial; "
                f"reducible: {sorted(REDUCE_RULES)}")
        init_fn, acc_fn = REDUCE_RULES[n.partial.op]
        acc = self._fresh("acc")                      # unique name, avoids collisions
        self.names[id(n)] = acc                       # the loop's result
        self._emit(f"{acc} = {init_fn(self.shapes[id(n)], f'{self._cur_out}.dtype')}")
        trips = -(-self._reduction_extent(n) // n.tile)   # cdiv(K, TS_k), literal
        self._emit(f"for {n.axis} in range({trips}):")
        self.indent += 1
        self._scope.add(n.axis)
        for s in n.body:
            if s is n.partial:
                args = [self._name(i) for i in s.inputs]
                self.names[id(s)] = acc
                self._emit(f"{acc} = {acc_fn(args, acc)}")
            else:
                self._visit(s)
        self._scope.discard(n.axis)
        self.indent -= 1


# --------------------------------------------------------------------------
# Structural key + cache: equivalent ASTs collapse to the same key, so we
# never re-evaluate (here: re-render) a shape we've already seen. Memoized so
# shared DAG nodes are folded once.
# --------------------------------------------------------------------------
def structural_key(node: Node, _memo: dict[int, tuple] | None = None) -> tuple:
    if _memo is None:
        _memo = {}
    if id(node) in _memo:
        return _memo[id(node)]

    if isinstance(node, Program):
        key = ("Program",
               tuple(sorted(node.tensors.items())),
               tuple(structural_key(c, _memo) for c in node.body))
    elif isinstance(node, ParallelLoop):
        key = ("ParallelLoop", node.out, tuple(node.tile_shape),
               tuple(node.index_vars),
               tuple(structural_key(c, _memo) for c in node.body))
    elif isinstance(node, Load):
        key = ("Load", node.source, tuple(node.index))
    elif isinstance(node, Compute):
        key = ("Compute", node.op,
               tuple(structural_key(i, _memo) for i in node.inputs))
    elif isinstance(node, Store):
        key = ("Store", node.dest, tuple(node.index), structural_key(node.src, _memo))
    elif isinstance(node, ReductionLoop):
        key = ("ReductionLoop", node.axis, node.tile,
               tuple(structural_key(c, _memo) for c in node.body),
               structural_key(node.partial, _memo))
    else:
        raise TypeError(f"unhashable node {type(node).__name__}")

    _memo[id(node)] = key
    return key


class RenderCache:
    """Structural-key cache that fronts module generation. Two facets, both
    keyed by structural_key so equivalent ASTs collapse:

      module(program)  -> (source, hit)   memoizes emit_module; a hit means an
                                           identical program was already emitted,
                                           so the caller can skip recompilation.
      result/record    -> measurement     the caller stashes the eval result
                                           (e.g. mean_gpu_ms) per key, so a repeat
                                           configuration skips compile+benchmark.

    Source dedup alone saves re-emission; pairing it with the result store is
    what actually skips the expensive compile + GPU benchmark in the search."""

    def __init__(self) -> None:
        self._src: dict[tuple, str] = {}      # structural key -> module source
        self._result: dict[tuple, object] = {}  # structural key -> eval result

    def module(self, program: Program, fn_name: str = "fn") -> tuple[str, bool]:
        key = structural_key(program)
        if key in self._src:
            return self._src[key], True            # cache hit -> skip emit + recompile
        src = emit_module(program, fn_name)
        self._src[key] = src
        return src, False

    def result(self, program: Program):
        """Prior evaluation result for this program, or None if never run."""
        return self._result.get(structural_key(program))

    def record(self, program: Program, measurement) -> None:
        self._result[structural_key(program)] = measurement


# --------------------------------------------------------------------------
# Module emission: turn a Program into a full, runnable cuTile module --
# imports, @ct.kernel defs with ConstInt tile/extent params, and a host-side
# callable `fn(*inputs) -> output` that allocates intermediates and launches
# each stage. Mirrors the cuTile 1.3.0 runtime idiom (cuda.tile + cupy).
# --------------------------------------------------------------------------
def _tup(parts: list[str]) -> str:
    inner = ", ".join(parts)
    if len(parts) == 1:
        inner += ","          # 1-tuple needs a trailing comma
    return f"({inner})"


def _infer_program(program: Program) -> dict[int, Shape]:
    """Run backward shape inference over every kernel; return id->tile shape."""
    r = CuTileRenderer()
    r._tensors = program.tensors
    for loop in program.body:
        r._scope = set(loop.index_vars)
        for stmt in loop.body:
            if isinstance(stmt, Store):
                r._infer(stmt.src, tuple(loop.tile_shape), program.tensors)
    return r.shapes


def _iter_loads(stmts):
    """Yield every Load, descending into ReductionLoop bodies."""
    for s in stmts:
        if isinstance(s, Load):
            yield s
        elif isinstance(s, ReductionLoop):
            yield from _iter_loads(s.body)


def _program_io(program: Program) -> tuple[list[str], list[str], list[str]]:
    """Classify tensors as inputs (read, never written), intermediates
    (written then read by a later stage), and outputs (written, never read)."""
    written, read = set(), set()
    for loop in program.body:
        written.add(loop.out)
        for ld in _iter_loads(loop.body):
            read.add(ld.source)
    return (sorted(read - written),         # inputs
            sorted(written & read),          # intermediates
            sorted(written - read))          # outputs


def _loop_tensors(loop: ParallelLoop) -> list[str]:
    """Tensors this kernel touches, ordered: load sources (first-seen), dest."""
    seen: list[str] = []
    for ld in _iter_loads(loop.body):
        if ld.source not in seen:
            seen.append(ld.source)
    seen.append(loop.out)
    return seen


def _emit_kernel(loop: ParallelLoop, tensors: dict[str, Shape],
                 shapes: dict[int, Shape]):
    """Return (name, tensor_args, const_params, body_lines) for one kernel.

    Tile dims resolve to ConstInt params: a subdivided dim -> TS_<axis> (a tile
    size), a whole dim -> <AXIS> (an extent like K). The same machinery that
    picks the coordinate (_resolve_index) decides which is which from the shape.
    A ReductionLoop additionally needs both: TS_<axis> for its loads and <AXIS>
    for the ct.cdiv(K, TS_k) bound.
    """
    name = f"{loop.out}_kernel"
    targs = _loop_tensors(loop)
    scope = set(loop.index_vars)
    tiles: dict[str, int] = {}      # TS_<axis> -> tile size
    extents: dict[str, int] = {}    # <AXIS>    -> full extent
    red_tiles: list[str] = []       # reduction TS_<axis> params, in encounter order
    names: dict[int, str] = {}
    counter = 0

    def fresh(prefix: str = "t") -> str:
        nonlocal counter
        counter += 1
        return f"{prefix}{counter}"

    def sym_shape(index, tile_shape, global_shape) -> list[str]:
        out = []
        for nm, t, g in zip(index, tile_shape, global_shape):
            if t < g:
                s = f"TS_{nm}"; tiles[s] = t; out.append(s)
            else:
                s = nm.upper(); extents[s] = g; out.append(s)
        return out

    def reduction_extent(rl: ReductionLoop):
        for s in rl.body:
            if isinstance(s, Load) and rl.axis in s.index:
                return rl.axis.upper(), tensors[s.source][s.index.index(rl.axis)]
        raise ValueError(f"reduction axis {rl.axis!r} not found in any body load")

    def emit_stmts(stmts, ind, acc_for=None) -> list[str]:
        out: list[str] = []
        for stmt in stmts:
            if isinstance(stmt, Load):
                v = fresh(); names[id(stmt)] = v
                ts, g = shapes[id(stmt)], tensors[stmt.source]
                idx = _resolve_index(stmt.index, ts, g, scope, f"load {stmt.source}")
                shp = sym_shape(stmt.index, ts, g)
                out.append(f"{ind}{v} = ct.load({stmt.source}, {_tup(idx)}, {_tup(shp)})")
            elif isinstance(stmt, Compute):
                args = [names[id(i)] for i in stmt.inputs]
                if acc_for is not None and stmt is acc_for[0]:   # the accumulated partial
                    acc = acc_for[1]
                    if stmt.op not in REDUCE_RULES:
                        raise ValueError(
                            f"op {stmt.op!r} cannot be a reduction partial; "
                            f"reducible: {sorted(REDUCE_RULES)}")
                    names[id(stmt)] = acc
                    out.append(f"{ind}{acc} = {REDUCE_RULES[stmt.op][1](args, acc)}")
                else:
                    v = fresh(); names[id(stmt)] = v
                    out.append(f"{ind}{v} = {EMIT_RULES[stmt.op](args)}")
            elif isinstance(stmt, ReductionLoop):
                acc = fresh("acc")                # unique name, avoids collisions
                # accumulator shape == output tile (also registers TS_<out dims>)
                acc_shape = []
                for d, vv in enumerate(loop.index_vars):
                    s = f"TS_{vv}"; tiles[s] = loop.tile_shape[d]; acc_shape.append(s)
                init_fn = REDUCE_RULES[stmt.partial.op][0]
                out.append(f"{ind}{acc} = {init_fn(_tup(acc_shape), f'{loop.out}.dtype')}")
                k_sym, k_val = reduction_extent(stmt)
                extents[k_sym] = k_val
                ts_sym = f"TS_{stmt.axis}"; tiles[ts_sym] = stmt.tile
                if ts_sym not in red_tiles:
                    red_tiles.append(ts_sym)
                out.append(f"{ind}for {stmt.axis} in range(ct.cdiv({k_sym}, {ts_sym})):")
                scope.add(stmt.axis)
                out += emit_stmts(stmt.body, ind + "    ", acc_for=(stmt.partial, acc))
                scope.discard(stmt.axis)
                names[id(stmt)] = acc                 # the loop's result
            elif isinstance(stmt, Store):
                ts, g = shapes[id(stmt.src)], tensors[stmt.dest]
                idx = _resolve_index(stmt.index, ts, g, scope, f"store {stmt.dest}")
                out.append(f"{ind}ct.store({stmt.dest}, {_tup(idx)}, {names[id(stmt.src)]})")
        return out

    body: list[str] = []
    # prologue: bind block indices (reuse the grid-decode logic)
    grid = tuple(-(-tensors[loop.out][d] // loop.tile_shape[d])
                 for d in range(len(loop.tile_shape)))
    tmp = CuTileRenderer(); tmp.indent = 1
    tmp._emit_grid_decode(loop.index_vars, grid)
    body.extend(tmp.lines)
    body += emit_stmts(loop.body, "    ")

    # params: extents (sorted), then output tiles (output-dim order), then
    # reduction tiles (encounter order). Matches kernel signatures like
    #   (x, y, a, K, TS_n, TS_m, TS_k)
    tile_order = [f"TS_{v}" for v in loop.index_vars if f"TS_{v}" in tiles]
    const_params = ([(k, extents[k]) for k in sorted(extents)]
                    + [(k, tiles[k]) for k in tile_order]
                    + [(k, tiles[k]) for k in red_tiles])
    return name, targs, const_params, body


def _iter_computes(stmts):
    """Yield every Compute, descending into ReductionLoop bodies."""
    for s in stmts:
        if isinstance(s, Compute):
            yield s
        elif isinstance(s, ReductionLoop):
            yield from _iter_computes(s.body)


def program_flops(program: Program, shapes: dict[int, Shape] | None = None):
    """Total FLOPs from the AST. Exact for matmul/elementwise; if any op has
    no rule, returns exact=False so callers can suppress TFLOP/s. Tiling the
    contraction (a ReductionLoop) does not change the total -- a matmul is
    always 2*M*N*K -- so we use the global contraction extent, not TS_k."""
    if shapes is None:
        shapes = _infer_program(program)
    total, exact = 0, True
    for loop in program.body:
        gout = math.prod(program.tensors[loop.out])
        for stmt in _iter_computes(loop.body):
            if stmt.op == "matmul":
                inp0 = stmt.inputs[0]
                k = (program.tensors[inp0.source][-1] if isinstance(inp0, Load)
                     else shapes[id(inp0)][-1])      # global K, not the tile TS_k
                total += 2 * gout * k                # 2*M*N*K
            elif stmt.op in ("add", "mul", "relu", "exp"):
                total += gout
            else:
                exact = False
    return total, exact


def emit_module(program: Program, fn_name: str = "fn") -> str:
    shapes = _infer_program(program)
    inputs, intermediates, outputs = _program_io(program)
    if not outputs:
        raise ValueError("program has no output tensor (written but never read)")

    L = ["import cuda.tile as ct", "import cupy as cp", "import numpy as np", "",
         "ConstInt = ct.Constant[int]", "",
         "# --- tunable tile sizes (vary these to autotune) ---"]
    for loop in program.body:
        L.append(f"TILE_{loop.out} = {tuple(loop.tile_shape)}")
        red = next((s for s in loop.body if isinstance(s, ReductionLoop)), None)
        if red is not None:
            L.append(f"RTILE_{loop.out} = {red.tile}   # reduction tile along {red.axis}")
    L.append("")

    launches = []
    for loop in program.body:
        name, targs, const_params, body = _emit_kernel(loop, program.tensors, shapes)
        sig = ", ".join(targs + [f"{p}: ConstInt" for p, _ in const_params])
        L += ["@ct.kernel", f"def {name}({sig}):"] + body + [""]

        rank = len(loop.tile_shape)
        cdivs = [f"ct.cdiv({loop.out}.shape[{d}], TILE_{loop.out}[{d}])"
                 for d in range(rank)]
        if rank <= 3:
            grid_parts = cdivs + ["1"] * (3 - rank)
        else:                                  # collapse leading dims onto grid.x
            grid_parts = [" * ".join(cdivs[:-2]), cdivs[-2], cdivs[-1]]
        vals = []
        for p, v in const_params:
            if p.startswith("TS_"):
                axis = p[3:]
                if axis in loop.index_vars:                    # output tile dim
                    vals.append(f"TILE_{loop.out}[{list(loop.index_vars).index(axis)}]")
                else:                                          # reduction tile
                    vals.append(f"RTILE_{loop.out}")
            else:                                              # extent (e.g. K)
                vals.append(str(v))
        launches.append((f"({', '.join(grid_parts)})", name, _tup(targs + vals)))

    # host callable
    L.append(f"def {fn_name}({', '.join(inputs)}):")
    L.append(f"    dtype = {inputs[0]}.dtype")
    L.append("    stream = cp.cuda.get_current_stream()")
    for t in intermediates + outputs:
        L.append(f"    {t} = cp.zeros({tuple(program.tensors[t])}, dtype=dtype)")
    for grid_expr, kname, args in launches:
        L.append(f"    grid = {grid_expr}")
        L.append(f"    ct.launch(stream, grid, {kname}, {args})")
    L.append(f"    return {outputs[0]}" if len(outputs) == 1
             else f"    return ({', '.join(outputs)})")
    L.append("")

    flops, exact = program_flops(program, shapes)
    meta = {
        "inputs": [[t, list(program.tensors[t])] for t in inputs],
        "output": [outputs[0], list(program.tensors[outputs[0]])],
        "flops": flops,
        "flops_exact": exact,
    }
    L += [f"KERNEL_META = {meta!r}", "",
          'if __name__ == "__main__":',
          '    args = [cp.random.randn(*s, dtype=cp.float32) for _, s in KERNEL_META["inputs"]]',
          f"    out = {fn_name}(*args)",
          "    cp.cuda.runtime.deviceSynchronize()",
          '    print("ran:", KERNEL_META["inputs"], "->", list(out.shape))']
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------
# Demo
# --------------------------------------------------------------------------
if __name__ == "__main__":
    M, K, N = 1024, 512, 768
    Mt, Nt = 128, 128

    # ---- C = A @ B  (no ReductionLoop yet, so the full K is loaded) ----
    a = Load("A", ["bm", "k"])      # 'k' is whole today -> renders as 0
    b = Load("B", ["k", "bn"])
    c = Compute("matmul", [a, b])
    matmul = Program(
        tensors={"A": (M, K), "B": (K, N), "C": (M, N)},
        body=[ParallelLoop(
            out="C", tile_shape=(Mt, Nt), index_vars=("bm", "bn"),
            body=[a, b, c, Store("C", c, ["bm", "bn"])],
        )],
    )

    # ---- D = relu(A + B)  (elementwise: every tile == output tile) ----
    pa = Load("A", ["bm", "bn"])
    pb = Load("B", ["bm", "bn"])
    s = Compute("add", [pa, pb])
    r = Compute("relu", [s])
    ew = Program(
        tensors={"A": (M, N), "B": (M, N), "D": (M, N)},
        body=[ParallelLoop(
            out="D", tile_shape=(Mt, Nt), index_vars=("bm", "bn"),
            body=[pa, pb, s, r, Store("D", r, ["bm", "bn"])],
        )],
    )

    # ---- Y = relu(X), 4-D output -> needs grid collapse/decode ----
    x = Load("X", ["b0", "b1", "bm", "bn"])
    ry = Compute("relu", [x])
    batched = Program(
        tensors={"X": (4, 8, 256, 256), "Y": (4, 8, 256, 256)},
        body=[ParallelLoop(
            out="Y", tile_shape=(1, 1, 128, 128),
            index_vars=("b0", "b1", "bm", "bn"),
            body=[x, ry, Store("Y", ry, ["b0", "b1", "bm", "bn"])],
        )],
    )

    # Inline snippet display (uncached -- cheap, just for reading the kernel).
    for label, prog in [("matmul", matmul), ("elementwise", ew), ("4-D batched", batched)]:
        print(f"# {label}")
        print(CuTileRenderer().render(prog))

    # RenderCache fronts emit_module: a structurally identical AST skips both
    # re-emission and -- via the recorded result -- recompilation + benchmarking.
    cache = RenderCache()
    _, hit = cache.module(matmul)
    print(f"# emit matmul module        (cache hit: {hit})")    # False -- first time
    cache.record(matmul, 1.23)                                  # pretend we benchmarked it (ms)

    a2 = Load("A", ["bm", "k"]); b2 = Load("B", ["k", "bn"]); c2 = Compute("matmul", [a2, b2])
    matmul_dup = Program(                                        # independently rebuilt, identical
        tensors={"A": (M, K), "B": (K, N), "C": (M, N)},
        body=[ParallelLoop(
            out="C", tile_shape=(Mt, Nt), index_vars=("bm", "bn"),
            body=[a2, b2, c2, Store("C", c2, ["bm", "bn"])],
        )],
    )
    _, hit = cache.module(matmul_dup)
    print(f"# rebuilt-identical matmul  (cache hit: {hit})")     # True -- emit skipped
    print(f"# reused measurement        : {cache.result(matmul_dup)} ms")  # 1.23 -- compile+bench skipped