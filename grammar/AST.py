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


# --------------------------------------------------------------------------
# Backward shape rules: given an op's OUTPUT tile shape (and the global shapes
# of its operands, for dims the output projects away), return the required
# INPUT tile shapes -- one per input, in order.
# --------------------------------------------------------------------------
def _elementwise(out_shape: Shape, operand_globals: list[Shape | None]) -> list[Shape]:
    # add / mul / relu / exp: every input tile matches the output tile.
    return [out_shape for _ in operand_globals]


def _matmul(out_shape: Shape, operand_globals: list[Shape | None]) -> list[Shape]:
    # out (M, N) <- A (M, K) @ B (K, N). K is projected away by the product,
    # so we recover it from operand A's global shape.
    m, n = out_shape
    k = operand_globals[0][1]      # A is (M, K)
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
    # confirmed against cuTile 1.3.0
    "matmul": lambda a: f"ct.matmul({a[0]}, {a[1]})",
    "add":    lambda a: f"ct.add({a[0]}, {a[1]})",
    "relu":   lambda a: f"ct.maximum(0, {a[0]})",
    "mul":    lambda a: f"{a[0]} * {a[1]}",
    "exp":    lambda a: f"ct.exp({a[0]})",
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
    def _infer(self, value: Value, out_shape: Shape, tensors: dict[str, Shape]) -> None:
        prev = self.shapes.get(id(value))
        if prev is not None:
            assert prev == out_shape, (
                f"conflicting tile shapes for {value!r}: {prev} vs {out_shape}"
            )
            return
        self.shapes[id(value)] = out_shape
        if isinstance(value, Compute):
            operand_globals = [
                tensors.get(i.source) if isinstance(i, Load) else None
                for i in value.inputs
            ]
            in_shapes = SHAPE_RULES[value.op](out_shape, operand_globals)
            for inp, shp in zip(value.inputs, in_shapes):
                self._infer(inp, shp, tensors)
        # Load: nothing further -- its tile shape is now fixed.

    # ---- emission helpers ----
    def _emit(self, s: str) -> None:
        self.lines.append("    " * self.indent + s)

    def _fresh(self) -> str:
        self._n += 1
        return f"t{self._n}"

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
        # 1. infer every tile shape, seeded by the output tile at each Store.
        for stmt in loop.body:
            if isinstance(stmt, Store):
                self._infer(stmt.src, tuple(loop.tile_shape), tensors)
        # 2. emit the kernel.
        args = ", ".join(sorted(tensors))
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
    else:
        raise TypeError(f"unhashable node {type(node).__name__}")

    _memo[id(node)] = key
    return key


class RenderCache:
    """Maps a structural key -> rendered kernel. Later this can just as easily
    cache a benchmark measurement instead of (or alongside) the source."""

    def __init__(self) -> None:
        self._cache: dict[tuple, str] = {}

    def render(self, program: Program) -> tuple[str, bool]:
        key = structural_key(program)
        if key in self._cache:
            return self._cache[key], True          # cache hit
        src = CuTileRenderer().render(program)      # fresh renderer per call
        self._cache[key] = src
        return src, False


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

    cache = RenderCache()

    src, hit = cache.render(matmul)
    print(f"# matmul  (cache hit: {hit})")
    print(src)

    src, hit = cache.render(ew)
    print(f"# elementwise  (cache hit: {hit})")
    print(src)

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
    src, hit = cache.render(batched)
    print(f"# 4-D batched  (cache hit: {hit})")
    print(src)

    # An independently built but structurally identical matmul -> cache hit,
    # renderer is never invoked.
    a2 = Load("A", ["bm", "k"]); b2 = Load("B", ["k", "bn"]); c2 = Compute("matmul", [a2, b2])
    matmul_dup = Program(
        tensors={"A": (M, K), "B": (K, N), "C": (M, N)},
        body=[ParallelLoop(
            out="C", tile_shape=(Mt, Nt), index_vars=("bm", "bn"),
            body=[a2, b2, c2, Store("C", c2, ["bm", "bn"])],
        )],
    )
    _, hit = cache.render(matmul_dup)
    print(f"# rebuilt-identical matmul  (cache hit: {hit})")