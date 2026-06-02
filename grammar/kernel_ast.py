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
    op: str                        # "matmul", "add", "relu", "rowmax", ...
    inputs: list[Value] = field(default_factory=list)
    axis: str = None               # for unary reductions (rowmax/rowsum): the
                                   # input axis collapsed away. None otherwise --
                                   # matmul's reduced axis is inferred from its
                                   # two inputs' shared index; elementwise has none.
    const: float = None            # compile-time scalar for const ops (mulc/addc),
                                   # e.g. 1/D for a mean or eps for a variance floor.


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
    partial: Value = None          # single-accumulator form (kept for back-compat)
    partials: list = None          # multi-accumulator form: each is a Compute
                                   # accumulated into its own accumulator. Fused
                                   # reduction loops carry >1. If None, derived
                                   # from `partial`.

    def __post_init__(self):
        # normalize: `partials` is the source of truth; `partial` mirrors the
        # first (the loop's Value result for the single-accumulator case).
        if self.partials is None:
            self.partials = [self.partial] if self.partial is not None else []
        if self.partial is None and self.partials:
            self.partial = self.partials[0]


@dataclass
class SpatialLoop(Stmt):
    """REMOVED from the active grammar. Spatial (output-axis) subtiling is a
    tile-size choice (ParallelLoop.tile_shape + a finer grid), not a structural
    node -- it was only non-redundant for flash-style intra-block reuse, which
    is out of scope. The class is retained as a stub so any stale reference
    fails loudly rather than silently; nothing in the renderer handles it."""
    axis: str
    tile: int
    body: list[Stmt] = field(default_factory=list)
    def __post_init__(self):
        raise NotImplementedError(
            "SpatialLoop has been removed from the grammar; output-axis "
            "subtiling is expressed via ParallelLoop.tile_shape instead.")


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


def _row_reduce(out_shape: Shape, operand_globals: list[Shape | None],
                reduce_tile: int | None = None, *, axis_pos: int = None,
                axis_extent: int = None) -> list[Shape]:
    # rowmax / rowsum: out has a 1 on the reduced axis; the single input restores
    # that axis to its full extent (or TS_<axis> inside a ReductionLoop).
    full = reduce_tile if reduce_tile is not None else axis_extent
    in_shape = tuple(full if d == axis_pos else s for d, s in enumerate(out_shape))
    return [in_shape]


SHAPE_RULES = {
    "matmul": _matmul,
    "add": _elementwise,
    "mul": _elementwise,
    "sub": _elementwise,
    "div": _elementwise,
    "relu": _elementwise,
    "exp": _elementwise,
    "sqrt": _elementwise,
    "mulc": _elementwise,          # x * const
    "addc": _elementwise,          # x + const
    "rowmax": _row_reduce,
    "rowsum": _row_reduce,
}

# ops whose output collapses one input axis to size 1 (a per-row reduction).
ROW_REDUCE_OPS = {"rowmax", "rowsum"}

# unary ops carrying a compile-time scalar in Compute.const (mean's 1/D, eps).
CONST_OPS = {"mulc": "*", "addc": "+"}


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
    "sub":    lambda a: f"({a[0]} - {a[1]})",
    "div":    lambda a: f"({a[0]} / {a[1]})",
    "relu":   lambda a: f"ct.maximum(0, {a[0]})",
    "mul":    lambda a: f"{a[0]} * {a[1]}",
    "exp":    lambda a: f"ct.exp({a[0]})",
    "sqrt":   lambda a: f"ct.sqrt({a[0]})",
}

# row-reduction emission needs the collapsed-axis position; keyed separately so
# the rest of EMIT_RULES keeps the simple arg-only signature.
ROW_EMIT_RULES = {
    "rowmax": lambda a, ax: f"ct.max({a[0]}, axis={ax}, keepdims=True)",
    "rowsum": lambda a, ax: f"ct.sum({a[0]}, axis={ax}, keepdims=True)",
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
    # per-row reductions tiled over the key axis: each iteration reduces the
    # loaded key-tile to a per-row partial, then folds it into the accumulator.
    # The collapsed-axis position is supplied at the call site (acc is [.., 1]).
    "rowmax": (
        lambda shp, dtype: f"ct.full({shp}, -float('inf'), dtype={dtype})",
        lambda a, acc, ax: f"ct.maximum({acc}, ct.max({a[0]}, axis={ax}, keepdims=True))",
    ),
    "rowsum": (
        lambda shp, dtype: f"ct.zeros({shp}, dtype={dtype})",
        lambda a, acc, ax: f"({acc} + ct.sum({a[0]}, axis={ax}, keepdims=True))",
    ),
}


def _resolve_index(index: list[str], tile_shape: Shape, global_shape: Shape,
                   scope: set[str], what: str,
                   coords: dict[str, str] | None = None) -> list[str]:
    """Turn per-dim index NAMES into emitted tile coordinates.

    A dim is *tiled* iff its inferred tile is smaller than the global dim; then
    its name must name a live index in `scope` and we emit that name's coordinate
    (an axis subtiled by a SpatialLoop resolves to a composed expression via
    `coords`, e.g. n -> sn; otherwise the name is its own coordinate, e.g. a grid
    var). Otherwise the dim is loaded/stored whole and collapses to tile index 0
    -- the name is then a latent axis (e.g. a reduction index not yet looped)."""
    coords = coords or {}
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
            out.append(coords.get(name, name))
        else:                                       # whole dim -> single tile
            out.append("0")
    return out


# --------------------------------------------------------------------------
# Renderer: external visitor. Runs backward shape inference per kernel, then
# emits cuTile-ish code.
# --------------------------------------------------------------------------
class CuTileRenderer:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.indent = 0
        self.shapes: dict[int, Shape] = {}   # id(value) -> inferred tile shape
        self.names: dict[int, str] = {}      # id(value) -> emitted var name
        self._n = 0
        self._coords: dict[str, str] = {}    # axis -> composed coordinate expr

    # ---- shape inference (each value resolved at its OWN lexical scope) ----
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
            # Each partial shares the loop's scope (no SpatialLoop between them).
            # partials[0] is the loop's Value (seeded by out_shape); others are
            # seeded through their own consumers -- here we ensure every partial
            # is descended into with the contraction tiled by `tile`, using each
            # partial's already-demanded shape when known.
            for i, p in enumerate(value.partials):
                pshape = self.shapes.get(id(p), out_shape if i == 0 else None)
                if pshape is not None:
                    self._infer(p, pshape, tensors, reduce_tile=value.tile)
        elif isinstance(value, Compute):
            operand_globals = [
                tensors.get(i.source) if isinstance(i, Load) else None
                for i in value.inputs
            ]
            # if this Compute is a reduction partial, tile its contraction even
            # when reached directly via its own consumer (not via the loop).
            if reduce_tile is None:
                reduce_tile = getattr(self, "_partial_tile", {}).get(id(value))
            if value.op in ROW_REDUCE_OPS:
                # unary per-row reduction: out has a 1 on `value.axis`; the input
                # restores that axis to full extent (or TS_<axis> if reducing).
                out_axes = _value_axes(value)
                axis_pos = out_axes.index(value.axis)
                axis_extent = _axis_global_extent(value, value.axis, tensors)
                in_shapes = _row_reduce(out_shape, operand_globals, reduce_tile,
                                        axis_pos=axis_pos, axis_extent=axis_extent)
            else:
                in_shapes = SHAPE_RULES[value.op](out_shape, operand_globals, reduce_tile)
                # broadcasting: an input that is itself a row-reduction (or whose
                # axes collapse a dim the output spans) keeps its 1 on that dim.
                in_shapes = [_broadcast_in(value, inp, shp, out_shape)
                             for inp, shp in zip(value.inputs, in_shapes)]
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
        self._coords = {}
        self._parent_tile = {v: loop.tile_shape[d]     # TS_<axis> per output axis
                             for d, v in enumerate(loop.index_vars)}
        # 1. infer every tile shape, seeded by each Store. The seed is the output
        #    tile, narrowed on any axis subtiled by an enclosing SpatialLoop.
        self._seed_shapes(loop.body, loop, {})
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

    def _seed_shapes(self, stmts, loop: ParallelLoop, subtiles: dict[str, int]) -> None:
        """Walk a body, seeding _infer at each Store with the output tile shape
        narrowed by `subtiles` (axis -> SUB_TS for enclosing SpatialLoops)."""
        for s in stmts:
            if isinstance(s, Store):
                seed = tuple(subtiles.get(v, loop.tile_shape[d])
                             for d, v in enumerate(loop.index_vars))
                self._infer(s.src, seed, self._tensors)
            elif isinstance(s, SpatialLoop):
                self._seed_shapes(s.body, loop, {**subtiles, s.axis: s.tile})
            elif isinstance(s, ReductionLoop):
                self._seed_shapes(s.body, loop, subtiles)

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
                             self._scope, f"load {n.source}", self._coords)
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
                             self._scope, f"store {n.dest}", self._coords)
        self._emit(
            f"ct.store({n.dest}, ({', '.join(idx)}), {self._name(n.src)})"
            f"  # tile {shp}"
        )

    def _reduction_extent(self, n: "ReductionLoop") -> int:
        """Global size of the reduced dimension (from a body load on `axis`)."""
        for s in _iter_loads(n.body):
            if n.axis in s.index:
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

    def _visit_SpatialLoop(self, n: "SpatialLoop") -> None:
        ts = self._parent_tile[n.axis]          # parent tile size TS_<axis>
        sub = n.tile                            # subtile size SUB_TS_<axis>
        it = f"st_{n.axis}"                     # loop iterator
        coord = f"s{n.axis}"                    # composed global subtile index
        trips = -(-ts // sub)                   # cdiv(TS, SUB_TS), literal
        self._emit(f"for {it} in range({trips}):")
        self.indent += 1
        self._emit(f"{coord} = {n.axis} * {ts // sub} + {it}")
        saved = self._coords.get(n.axis)
        self._coords[n.axis] = coord            # axis resolves to composed coord
        for s in n.body:
            self._visit(s)
        if saved is None:
            self._coords.pop(n.axis, None)
        else:
            self._coords[n.axis] = saved
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
        key = ("Compute", node.op, node.axis, node.const,
               tuple(structural_key(i, _memo) for i in node.inputs))
    elif isinstance(node, Store):
        key = ("Store", node.dest, tuple(node.index), structural_key(node.src, _memo))
    elif isinstance(node, ReductionLoop):
        key = ("ReductionLoop", node.axis, node.tile,
               tuple(structural_key(c, _memo) for c in node.body),
               tuple(structural_key(p, _memo) for p in node.partials))
    elif isinstance(node, SpatialLoop):
        key = ("SpatialLoop", node.axis, node.tile,
               tuple(structural_key(c, _memo) for c in node.body))
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
    """Run shape inference over every kernel; return id->tile shape.

    Backward demand (op footprint) gives the per-axis extent a value's
    consumers need; then each Load is re-resolved at its OWN lexical scope, so a
    Load placed outside a loop materialises the full (un-narrowed) extent on that
    loop's axis and the loop bridges with ct.extract. Without this, a Load lifted
    out of a reduction/spatial loop would be wrongly narrowed to the subtile."""
    r = CuTileRenderer()
    r._tensors = program.tensors
    # map every reduction partial -> its loop's tile, so a partial reached via
    # its own consumer (not via the loop) still tiles the contraction. Covers
    # multi-accumulator fused loops where only partials[0] is the loop's Value.
    r._partial_tile = {}
    def collect(stmts):
        for s in stmts:
            if isinstance(s, (ReductionLoop, SpatialLoop)):
                if isinstance(s, ReductionLoop):
                    for p in s.partials:
                        r._partial_tile[id(p)] = s.tile
                collect(s.body)
    for loop in program.body:
        collect(loop.body)
    for loop in program.body:
        r._scope = set(loop.index_vars)
        r._seed_shapes(loop.body, loop, {})

    # A top-level Load feeding a row-reduction (rowsum/rowmax) over axis `a`
    # must keep its FULL extent on `a` even though `a` is an output index var:
    # the reduction needs the whole row, not just this block's output slice.
    # Collect, per kernel, the (id(load), axis) pairs that must stay full.
    def reduced_axis_loads(stmts, acc):
        for s in stmts:
            if isinstance(s, Compute) and s.op in ROW_REDUCE_OPS:
                for ld in _loads_feeding(s):
                    acc.setdefault(id(ld), set()).add(s.axis)
            if isinstance(s, (ReductionLoop, SpatialLoop)):
                reduced_axis_loads(s.body, acc)
        return acc

    # re-resolve each Load at its lexical scope: an axis subtiled only by a loop
    # the Load is NOT inside stays at full extent.
    def fix(stmts, loop, sub_axes: set[str], red_axes: set[str], rax: dict):
        for s in stmts:
            if isinstance(s, Load):
                g = program.tensors[s.source]
                full_on = rax.get(id(s), set())     # axes this load reduces over
                shp = []
                for d, a in enumerate(s.index):
                    if g[d] == 1:                      # tensor is size-1 on this axis:
                        shp.append(1)                  # load width-1 and BROADCAST. A
                                                       # stored row-reduction result
                                                       # ([N,1]) reloaded for a broadcast
                                                       # sub/div must load its 1-wide axis
                                                       # as 1, not the output-tile width.
                    elif a in full_on:                 # feeds a row-reduction over `a`:
                        shp.append(g[d])               # keep the FULL reduced extent
                    elif a in loop.index_vars:         # output axis: TS, or SUB if inside
                        base = loop.tile_shape[loop.index_vars.index(a)]
                        shp.append(r.shapes[id(s)][d] if a in sub_axes else base)
                    else:                              # contraction axis
                        shp.append(r.shapes[id(s)][d] if a in red_axes else g[d])
                r.shapes[id(s)] = tuple(shp)
            elif isinstance(s, SpatialLoop):
                fix(s.body, loop, sub_axes | {s.axis}, red_axes, rax)
            elif isinstance(s, ReductionLoop):
                fix(s.body, loop, sub_axes, red_axes | {s.axis}, rax)

    for loop in program.body:
        fix(loop.body, loop, set(), set(), reduced_axis_loads(loop.body, {}))
    return r.shapes


def _loads_feeding(v: Value):
    """Yield Load nodes in the dataflow that feeds value `v` (transitively)."""
    if isinstance(v, Load):
        yield v
    elif isinstance(v, Compute):
        for i in v.inputs:
            yield from _loads_feeding(i)
    elif isinstance(v, ReductionLoop):
        for p in v.partials:
            yield from _loads_feeding(p)


def _iter_loads(stmts):
    """Yield every Load, descending into ReductionLoop and SpatialLoop bodies."""
    for s in stmts:
        if isinstance(s, Load):
            yield s
        elif isinstance(s, (ReductionLoop, SpatialLoop)):
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
    coords: dict[str, str] = {}     # axis -> composed coordinate expr (spatial)
    tiles: dict[str, int] = {}      # TS_<axis> -> tile size
    extents: dict[str, int] = {}    # <AXIS>    -> full extent
    red_tiles: list[str] = []       # reduction TS_<axis> params, in encounter order
    sub_tiles: list[str] = []       # spatial SUB_TS_<axis> params, in encounter order
    names: dict[int, str] = {}
    mat_frames: dict[int, dict[str, list]] = {}   # id(value) -> active frames at its scope
    counter = 0

    def fresh(prefix: str = "t") -> str:
        nonlocal counter
        # never collide with a tensor parameter name (e.g. an intermediate named
        # "t1" would shadow a freshly-named load tile "t1", corrupting stores).
        while True:
            counter += 1
            cand = f"{prefix}{counter}"
            if cand not in targs:
                return cand

    # active subtiling frames per axis (outer->inner), pushed by SpatialLoops and
    # the ReductionLoop. Each frame is (subtile_index_var, subtile_shape_sym).
    # A value materialized when fewer frames were active is reached from a deeper
    # scope by a ct.extract that names the per-frame subtile index.
    active_sub: dict[str, str] = {}   # axis -> SUB_TS_<axis> while inside its SpatialLoop
    frames: dict[str, list[tuple[str, str]]] = {}   # axis -> [(index_var, shape_sym)]

    def sym_shape(index, tile_shape, global_shape) -> list[str]:
        out = []
        for nm, t, g in zip(index, tile_shape, global_shape):
            if g == 1:                        # size-1 (broadcast) axis: literal 1.
                out.append("1")               # a stored [N,1] reduction result loaded
                                              # for broadcast -- never a shared extent
                                              # symbol (which other tensors set to N).
            elif t < g:
                s = active_sub.get(nm, f"TS_{nm}"); tiles[s] = t
                out.append(s)
            else:
                s = nm.upper(); extents[s] = g; out.append(s)
        return out

    def reduction_extent(rl: ReductionLoop):
        # the k-extent comes from any load feeding a partial that indexes `axis`
        # -- that load may sit OUTSIDE the loop (extracted in), so search inputs.
        for ld in _loads_feeding(rl):
            if rl.axis in ld.index:
                return rl.axis.upper(), tensors[ld.source][ld.index.index(rl.axis)]
        for s in _iter_loads(rl.body):
            if rl.axis in s.index:
                return rl.axis.upper(), tensors[s.source][s.index.index(rl.axis)]
        raise ValueError(f"reduction axis {rl.axis!r} not found in any body load")

    def snapshot() -> dict[str, list]:
        return {a: list(f) for a, f in frames.items() if f}

    def read(v: Value, ind: str, out: list) -> str:
        """Tile `v` as seen from the CURRENT scope. If `v` was materialized at a
        shallower scope (an axis was subtiled in between), emit a ct.extract to
        produce the subtile -- tiles are immutable, so this is a new tile, never
        a view. Otherwise the value is used directly. (One subtiling level per
        axis between materialization and use; deeper nesting on one axis would
        compose the index, which validate() can require not to occur for now.)"""
        nm = names[id(v)]
        axes = _value_axes(v)
        mat = mat_frames.get(id(v), {})
        idx, shp, need = [], [], False
        for a in axes:
            cur, m = frames.get(a, []), mat.get(a, [])
            if len(cur) > len(m):                       # narrowed since materialization
                need = True
                ivar, sh = cur[len(m)]                  # first frame deeper than mat
                idx.append(ivar)
                shp.append(sh)
            else:
                idx.append("0")
                shp.append(active_sub.get(a, f"TS_{a}") if a in scope else a.upper())
        if not need:
            return nm
        s = fresh()
        out.append(f"{ind}{s} = ct.extract({nm}, {_tup(idx)}, {_tup(shp)})")
        return s

    def emit_stmts(stmts, ind, acc_for=None) -> list[str]:
        out: list[str] = []
        for stmt in stmts:
            if isinstance(stmt, Load):
                v = fresh(); names[id(stmt)] = v
                mat_frames[id(stmt)] = snapshot()
                ts, g = shapes[id(stmt)], tensors[stmt.source]
                idx = _resolve_index(stmt.index, ts, g, scope, f"load {stmt.source}", coords)
                shp = sym_shape(stmt.index, ts, g)
                out.append(f"{ind}{v} = ct.load({stmt.source}, {_tup(idx)}, {_tup(shp)})")
            elif isinstance(stmt, Compute):
                args = [read(i, ind, out) for i in stmt.inputs]
                if acc_for is not None and id(stmt) in acc_for:   # an accumulated partial
                    acc = acc_for[id(stmt)]
                    if stmt.op not in REDUCE_RULES:
                        raise ValueError(
                            f"op {stmt.op!r} cannot be a reduction partial; "
                            f"reducible: {sorted(REDUCE_RULES)}")
                    names[id(stmt)] = acc                # whole-tile rebind (immutable)
                    acc_fn = REDUCE_RULES[stmt.op][1]
                    if stmt.op in ROW_REDUCE_OPS:
                        ax = _value_axes(stmt).index(stmt.axis)
                        out.append(f"{ind}{acc} = {acc_fn(args, acc, ax)}")
                    else:
                        out.append(f"{ind}{acc} = {acc_fn(args, acc)}")
                else:
                    v = fresh(); names[id(stmt)] = v
                    mat_frames[id(stmt)] = snapshot()
                    if stmt.op in ROW_REDUCE_OPS:
                        ax = _value_axes(stmt).index(stmt.axis)
                        out.append(f"{ind}{v} = {ROW_EMIT_RULES[stmt.op](args, ax)}")
                    elif stmt.op in CONST_OPS:
                        out.append(f"{ind}{v} = ({args[0]} {CONST_OPS[stmt.op]} {stmt.const!r})")
                    else:
                        out.append(f"{ind}{v} = {EMIT_RULES[stmt.op](args)}")
            elif isinstance(stmt, ReductionLoop):
                # accumulator: full output tile at THIS scope (narrowed by any
                # SpatialLoop enclosing the reduction). No SpatialLoop may live
                # INSIDE -- enforced by validate() -- so it is never partially
                # written; every iteration rebinds the whole tile.
                # one accumulator per partial; independent partials share the
                # loop's iteration. Each accumulator is the full output tile at
                # this scope (narrowed by any enclosing SpatialLoop -- forbidden
                # inside -- so never partially written; whole-tile rebind only).
                acc_map = {}
                for p in stmt.partials:
                    acc = fresh("acc")
                    acc_shape = []
                    for d, vv in enumerate(loop.index_vars):
                        dim = shapes[id(p)][d]
                        if dim == 1:                     # collapsed axis (row-reduction)
                            acc_shape.append("1")
                        else:
                            s = active_sub.get(vv, f"TS_{vv}")
                            tiles[s] = dim; acc_shape.append(s)
                    init_fn = REDUCE_RULES[p.op][0]
                    out.append(f"{ind}{acc} = {init_fn(_tup(acc_shape), f'{loop.out}.dtype')}")
                    acc_map[id(p)] = acc
                k_sym, k_val = reduction_extent(stmt)
                extents[k_sym] = k_val
                ts_sym = f"TS_{stmt.axis}"; tiles[ts_sym] = stmt.tile
                if ts_sym not in red_tiles:
                    red_tiles.append(ts_sym)
                out.append(f"{ind}for {stmt.axis} in range(ct.cdiv({k_sym}, {ts_sym})):")
                scope.add(stmt.axis)
                frames.setdefault(stmt.axis, []).append((stmt.axis, ts_sym))
                out += emit_stmts(stmt.body, ind + "    ", acc_for=acc_map)
                frames[stmt.axis].pop()
                scope.discard(stmt.axis)
                names[id(stmt)] = acc_map[id(stmt.partials[0])]   # loop Value = first acc
                mat_frames[id(stmt)] = snapshot()
            elif isinstance(stmt, SpatialLoop):
                ts_sym = f"TS_{stmt.axis}"
                par = active_sub.get(stmt.axis, f"TS_{stmt.axis}")
                tiles[ts_sym] = loop.tile_shape[list(loop.index_vars).index(stmt.axis)]
                sub_sym = f"SUB_TS_{stmt.axis}"; tiles[sub_sym] = stmt.tile
                if sub_sym not in sub_tiles:
                    sub_tiles.append(sub_sym)
                it, coord = fresh(f"st_{stmt.axis}_"), fresh(f"s{stmt.axis}_")
                out.append(f"{ind}for {it} in range(ct.cdiv({par}, {sub_sym})):")
                base = coords.get(stmt.axis, stmt.axis)
                out.append(f"{ind}    {coord} = {base} * ({par} // {sub_sym}) + {it}")
                saved, saved_sub = coords.get(stmt.axis), active_sub.get(stmt.axis)
                coords[stmt.axis] = coord
                active_sub[stmt.axis] = sub_sym
                frames.setdefault(stmt.axis, []).append((it, sub_sym))
                out += emit_stmts(stmt.body, ind + "    ", acc_for=acc_for)
                frames[stmt.axis].pop()
                if saved is None: coords.pop(stmt.axis, None)
                else: coords[stmt.axis] = saved
                if saved_sub is None: active_sub.pop(stmt.axis, None)
                else: active_sub[stmt.axis] = saved_sub
            elif isinstance(stmt, Store):
                src = read(stmt.src, ind, out)           # extract if produced shallower
                ts, g = shapes[id(stmt.src)], tensors[stmt.dest]
                idx = _resolve_index(stmt.index, ts, g, scope, f"store {stmt.dest}", coords)
                out.append(f"{ind}ct.store({stmt.dest}, {_tup(idx)}, {src})")
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
    # reduction tiles, then spatial subtiles (encounter order). Dedup by name so
    # an axis that is both an output dim and a reduction axis (a row-reduction's
    # collapsed axis) contributes a single TS_ param.
    tile_order = [f"TS_{v}" for v in loop.index_vars if f"TS_{v}" in tiles]
    seen = set()
    ordered = []
    for k in tile_order + red_tiles + sub_tiles:
        if k not in seen:
            seen.add(k); ordered.append(k)
    const_params = ([(k, extents[k]) for k in sorted(extents)]
                    + [(k, tiles[k]) for k in ordered])
    return name, targs, const_params, body, set(red_tiles)


def _iter_computes(stmts):
    """Yield every Compute, descending into ReductionLoop and SpatialLoop bodies."""
    for s in stmts:
        if isinstance(s, Compute):
            yield s
        elif isinstance(s, (ReductionLoop, SpatialLoop)):
            yield from _iter_computes(s.body)


def _iter_spatial(stmts):
    """Yield every SpatialLoop, descending into nested loop bodies."""
    for s in stmts:
        if isinstance(s, SpatialLoop):
            yield s
            yield from _iter_spatial(s.body)
        elif isinstance(s, ReductionLoop):
            yield from _iter_spatial(s.body)


def _axis_global_extent(v: Value, axis: str, tensors: dict) -> int:
    """Full (global) extent of `axis`, found from a Load feeding `v` that
    indexes it."""
    for ld in _loads_feeding(v):
        if axis in ld.index:
            return tensors[ld.source][ld.index.index(axis)]
    raise ValueError(f"axis {axis!r} not found in any load feeding {v!r}")


def _broadcast_in(compute: Compute, inp: Value, shp: Shape, out_shape: Shape) -> Shape:
    """For an elementwise op, narrow `shp` to a 1 on any dim where `inp` is a
    per-row reduction (its output collapses that axis). The operand then
    broadcasts against the full output tile at emission."""
    if isinstance(inp, Compute) and inp.op in ROW_REDUCE_OPS:
        in_axes = _value_axes(inp)
        pos = in_axes.index(inp.axis)
        return tuple(1 if d == pos else s for d, s in enumerate(shp))
    if isinstance(inp, ReductionLoop) and isinstance(inp.partial, Compute) \
            and inp.partial.op in ROW_REDUCE_OPS:
        in_axes = _value_axes(inp)
        pos = in_axes.index(inp.partial.axis)
        return tuple(1 if d == pos else s for d, s in enumerate(shp))
    return shp


def _value_axes(v: Value) -> list[str]:
    """Logical axes a value spans, derived forward. Load: its index. Compute:
    elementwise keeps input axes; matmul drops the shared (contraction) axis.
    ReductionLoop: the axes of its accumulator == its partial's output axes."""
    if isinstance(v, Load):
        return list(v.index)
    if isinstance(v, ReductionLoop):
        return _value_axes(v.partial)
    if isinstance(v, Compute):
        ins = [_value_axes(i) for i in v.inputs]
        if v.op == "matmul":
            a0, a1 = ins
            shared = set(a0) & set(a1)
            return [x for x in a0 if x not in shared] + [x for x in a1 if x not in shared]
        if v.op in ROW_REDUCE_OPS:
            return list(ins[0])             # same axes; the reduced one is now size 1
        # elementwise: the broadest operand's axes (handles broadcast operands,
        # whose collapsed dim is size 1 but still named).
        return max(ins, key=len)
    raise TypeError(f"no axes for {type(v).__name__}")


def _iter_reductions(stmts):
    """Yield every ReductionLoop, descending into nested loop bodies."""
    for s in stmts:
        if isinstance(s, ReductionLoop):
            yield s
            yield from _iter_reductions(s.body)
        elif isinstance(s, SpatialLoop):
            yield from _iter_reductions(s.body)


def _tile_producing_nodes(body):
    """Every node that materialises a tile in a kernel body: Loads, Compute
    results, and reduction accumulators (partials). Descends into loops."""
    for s in body:
        if isinstance(s, (Load, Compute)):
            yield s
        if isinstance(s, (ReductionLoop, SpatialLoop)):
            yield from _tile_producing_nodes(s.body)
            if isinstance(s, ReductionLoop):
                yield from s.partials


def peak_tile_bytes(program: Program, shapes: dict[int, Shape] | None = None,
                    dtype_bytes: int = 4) -> int:
    """Largest single per-block tile across the whole program, in bytes.

    Each tile's shape (from inference) is already the PER-BLOCK shape -- grid
    dimensions are tiled, contractions are TS_k or full-extent depending on the
    loop nesting -- so `prod(shape) * dtype_bytes` is the bytes one thread block
    holds for that tile. We take the maximum over every materialised tile (loads,
    compute outputs, reduction accumulators) in every kernel.

    This is a cheap (no-compile) lower bound on a block's shared-memory demand,
    grounded in the hardware: an L4 SM gives a thread block up to ~99 KB of SMEM,
    so a single tile near or above that cannot be scheduled and makes cuTile/ptxas
    struggle. A full-K matmul's (TS, K) operand or a row-reduction's full (TILE, D)
    row shows up here as one oversized tile. We bound the LARGEST single tile
    rather than the sum of all tiles, because the sum over-counts (not all tiles
    are live simultaneously) whereas the largest tile is a true per-block floor on
    demand and maps directly onto the SMEM limit. The search penalises (does not
    compile) programs whose largest tile exceeds the budget."""
    if shapes is None:
        shapes = _infer_program(program)
    peak = 0
    for loop in program.body:
        for n in _tile_producing_nodes(loop.body):
            shp = shapes.get(id(n))
            if shp is not None:
                peak = max(peak, math.prod(shp) * dtype_bytes)
    return peak


def kernel_peak_tile_bytes(program: Program, shapes: dict[int, Shape] | None = None,
                           dtype_bytes: int = 4) -> list[int]:
    """Largest single per-block tile in EACH kernel, in bytes (one entry per
    stage, in order). `peak_tile_bytes` is the max of this list -- the hardware
    feasibility bound. This per-kernel breakdown is what lets the search penalise
    by the SUM of per-stage over-budget excess: with a single max-based penalty,
    two equally-oversized stages deadlock coordinate descent (shrinking either
    alone doesn't change the max, so neither shows gradient); summing each
    stage's excess gives every over-budget stage its own gradient."""
    if shapes is None:
        shapes = _infer_program(program)
    out = []
    for loop in program.body:
        big = 0
        for n in _tile_producing_nodes(loop.body):
            shp = shapes.get(id(n))
            if shp is not None:
                big = max(big, math.prod(shp) * dtype_bytes)
        out.append(big)
    return out


def kernel_compile_risk(program: Program, shapes: dict[int, Shape] | None = None
                        ) -> list[int]:
    """Cheap, no-compile estimate of ptxas compile difficulty for EACH kernel.

    The observed slow compiles (minutes vs the usual few seconds) all share one
    shape: a kernel does a reduction/contraction over a LARGE axis as STRAIGHT-LINE
    work -- not folded into a sub-tiling ReductionLoop runtime loop -- so ptxas has
    to schedule a deep unrolled body. Two forms:

      * a matmul NOT inside a ReductionLoop: its full contraction K is materialised
        in one shot (the (TS, K) operand). Cost ~ K. (e.g. atomic O=P@V over j=512.)
      * a row-reduce (rowsum/rowmax) NOT inside a ReductionLoop whose result is
        broadcast back into a WIDE output in the SAME kernel -- the fused
        reduction+broadcast keeps the wide (TS, D) tile live across both, which
        blows up register allocation. Cost ~ the reduced extent D. (e.g.
        merge(sm,P): rowsum over j=512 fused with the e/sm division.)

    The crucial discriminations, which a raw tile-volume or FLOP proxy gets wrong:
      - a row-reduce with a NARROW (collapsed [N,1]) output -- a plain rowmax/rowsum
        stage -- is NOT risky (ptxas handles a reduction-to-scalar fine); only the
        fused-into-wide-output form is. So we require the kernel's output tensor to
        be a full matrix (no collapsed dim), which mx/sm are not.
      - a matmul over a SMALL contraction (e.g. the QK matmul over d=64) is NOT
        risky, so the cost is the contraction extent itself, thresholded by caller.
      - once a matmul/reduce IS sub-tiled (wrapped in a ReductionLoop), the heavy
        axis becomes a runtime loop of width TS_k, so it stops counting here --
        which is exactly the repair the search should make.

    Returns the per-kernel risk (largest straight-line reduction/contraction
    extent), one entry per stage. The caller penalises a program whose max exceeds
    a threshold, without compiling it."""
    if shapes is None:
        shapes = _infer_program(program)

    def _straightline_computes(loop):
        # Computes in the kernel body that are NOT inside a ReductionLoop (i.e.
        # their heavy axis is materialised, not folded into a runtime loop).
        out = []
        for s in loop.body:
            if isinstance(s, Compute):
                out.append(s)
            elif isinstance(s, SpatialLoop):
                out.extend(c for c in s.body if isinstance(c, Compute))
            # ReductionLoop bodies are deliberately skipped: their axis is a loop.
        return out

    risks = []
    for loop in program.body:
        out_shape = program.tensors[loop.out]
        wide_output = sum(1 for d in out_shape if d > 1) >= 2   # full matrix, not [N,1]
        risk = 0
        for c in _straightline_computes(loop):
            if c.op == "matmul":
                inp0 = c.inputs[0]
                k = (program.tensors[inp0.source][-1] if isinstance(inp0, Load)
                     else shapes.get(id(inp0), (1,))[-1])
                risk = max(risk, k)
            elif c.op in ROW_REDUCE_OPS and wide_output:
                # fused reduction broadcast back into a wide output
                red = _axis_global_extent(c, c.axis, program.tensors)
                risk = max(risk, red)
        risks.append(risk)
    return risks


def program_flops(program: Program, shapes: dict[int, Shape] | None = None):
    """Total FLOPs from the AST. Exact for matmul, pointwise ops, and row
    reductions; if any op has no rule, returns exact=False so callers can
    suppress TFLOP/s. Tiling the contraction (a ReductionLoop) does not change
    the total -- a matmul is always 2*M*N*K -- so we use the global contraction
    extent, not TS_k."""
    if shapes is None:
        shapes = _infer_program(program)
    # one logical FLOP per output element (div/sqrt cost more cycles in hardware,
    # but a logical FLOP count treats them as one op per element).
    POINTWISE = {"add", "mul", "sub", "div", "relu", "exp", "sqrt", "mulc", "addc"}
    total, exact = 0, True
    for loop in program.body:
        gout = math.prod(program.tensors[loop.out])
        for stmt in _iter_computes(loop.body):
            if stmt.op == "matmul":
                inp0 = stmt.inputs[0]
                k = (program.tensors[inp0.source][-1] if isinstance(inp0, Load)
                     else shapes[id(inp0)][-1])      # global K, not the tile TS_k
                total += 2 * gout * k                # 2*M*N*K
            elif stmt.op in ROW_REDUCE_OPS:
                # a per-row reduction reads its full input row and accumulates:
                # ~one add/compare per input element. Count from GLOBAL extents
                # (number of output rows x the full reduced-axis extent), not the
                # tile, so the total is tiling-invariant like the matmul.
                axes = _value_axes(stmt)
                red_ext = _axis_global_extent(stmt, stmt.axis, program.tensors)
                rows = 1
                for a in axes:
                    if a == stmt.axis:
                        continue
                    rows *= _axis_global_extent(stmt, a, program.tensors)
                total += rows * red_ext
            elif stmt.op in POINTWISE:
                total += gout
            else:
                exact = False
    return total, exact


def validate(program: Program) -> None:
    """Enforce the invariants that keep emission within cuTile's tile model:

      1. A SpatialLoop may not be nested inside a ReductionLoop. The reverse
         (ReductionLoop in SpatialLoop) is fine. Spatial-in-reduction would
         require partially writing an accumulator (or N accumulators), which
         cuTile's immutable tiles don't support.
      2. No value is consumed at a scope SHALLOWER than where it is produced.
         Tiles are immutable and flow outer->inner via ct.extract; there is no
         inner->outer flow (a Store consuming a subtile must sit inside the
         loop, the accumulator is reduced and consumed at its own scope).

    Raises ValueError on violation."""
    # 1. structural: no SpatialLoop under any ReductionLoop
    def check_no_spatial(stmts, in_reduction: bool):
        for s in stmts:
            if isinstance(s, SpatialLoop):
                if in_reduction:
                    raise ValueError(
                        "SpatialLoop nested inside a ReductionLoop is not "
                        "supported (would require partial accumulator writes); "
                        "put the ReductionLoop inside the SpatialLoop instead")
                check_no_spatial(s.body, in_reduction)
            elif isinstance(s, ReductionLoop):
                check_no_spatial(s.body, True)

    # 2. dataflow: a value's producer scope must enclose (or equal) every
    #    consumer scope -- never the other way around.
    def depth_map(stmts, path, out):
        for s in stmts:
            out[id(s)] = path
            if isinstance(s, (SpatialLoop, ReductionLoop)):
                depth_map(s.body, path + (s,), out)

    for loop in program.body:
        check_no_spatial(loop.body, False)
        scopes: dict[int, tuple] = {}
        depth_map(loop.body, (), scopes)
        # partials are accumulators: their RESULT is available at their loop's
        # scope (the renderer binds the accumulator name there), so a partial
        # feeding a consumer outside its loop is the legal accumulator handoff.
        partial_ids = set()
        for s in _all_nodes(loop.body):
            if isinstance(s, ReductionLoop):
                partial_ids |= {id(p) for p in s.partials}
        for s in _all_nodes(loop.body):
            if isinstance(s, ReductionLoop):
                continue          # partial lives inside by design (accumulator rebind)
            consumers = (s.inputs if isinstance(s, Compute)
                         else [s.src] if isinstance(s, Store) else [])
            for src in consumers:
                if id(src) in partial_ids:
                    continue       # accumulator handoff -- result is at loop scope
                psc, csc = scopes.get(id(src)), scopes.get(id(s))
                if psc is None or csc is None:
                    continue
                # producer scope must be a prefix of (enclose) consumer scope
                if len(psc) > len(csc) or psc != csc[:len(psc)]:
                    raise ValueError(
                        f"value produced in a deeper scope than its consumer "
                        f"({type(src).__name__} -> {type(s).__name__}); a tile "
                        f"cannot flow inner->outer (move the consumer inside)")


def _all_nodes(stmts):
    for s in stmts:
        yield s
        if isinstance(s, (SpatialLoop, ReductionLoop)):
            yield from _all_nodes(s.body)


def emit_module(program: Program, fn_name: str = "fn") -> str:
    validate(program)
    shapes = _infer_program(program)
    inputs, intermediates, outputs = _program_io(program)
    if not outputs:
        raise ValueError("program has no output tensor (written but never read)")

    L = ["import cuda.tile as ct", "import cupy as cp", "import numpy as np", "",
         "ConstInt = ct.Constant[int]", "",
         "# --- tunable tile sizes (vary these to autotune) ---"]
    for loop in program.body:
        L.append(f"TILE_{loop.out} = {tuple(loop.tile_shape)}")
        red = next(_iter_reductions(loop.body), None)
        if red is not None:
            L.append(f"RTILE_{loop.out} = {red.tile}   # reduction tile along {red.axis}")
        for sp in _iter_spatial(loop.body):
            L.append(f"STILE_{loop.out}_{sp.axis} = {sp.tile}   # subtile along {sp.axis}")
    L.append("")

    launches = []
    for loop in program.body:
        name, targs, const_params, body, red_set = _emit_kernel(loop, program.tensors, shapes)
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
            if p.startswith("SUB_TS_"):                        # spatial subtile
                vals.append(f"STILE_{loop.out}_{p[len('SUB_TS_'):]}")
            elif p in red_set:                                 # reduction tile
                vals.append(f"RTILE_{loop.out}")
            elif p.startswith("TS_"):
                axis = p[3:]
                idx = list(loop.index_vars).index(axis)        # output tile dim
                vals.append(f"TILE_{loop.out}[{idx}]")
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