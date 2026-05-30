"""
rewrites.py -- structure-preserving rewrites over the kernel AST.

Every rewrite returns a NEW program (deep-copied, references remapped); the
input tree is never mutated, so a search can hold many candidates and backtrack
freely. The first two rules are Hoist and Sink, which relocate a `Load` across
one loop boundary. They are the only memory-op placement rewrites: a `Store` is
pinned to the scope where its (immutable) tile is produced, so it has no freedom.

The AST wires dataflow by reference -- `Compute.inputs`, `Store.src`, and
`ReductionLoop.partial` hold the actual producing nodes. A correct clone must
therefore remap every such reference to the corresponding cloned node, which
`clone_program` does via an id->clone table.
"""

from __future__ import annotations

from kernel_ast import (
    Program, ParallelLoop, Load, Store, Compute, ReductionLoop, SpatialLoop,
    Value, Stmt, REDUCE_RULES, _value_axes, _loads_feeding,
)


# --------------------------------------------------------------------------
# Deep clone with reference remapping
# --------------------------------------------------------------------------
def clone_program(program: Program) -> Program:
    """Deep-copy a Program, preserving the dataflow DAG. Producing nodes
    (Load/Compute/ReductionLoop) are cloned once; every reference to them
    (Compute.inputs, Store.src, ReductionLoop.partial) is rewired to the clone.

    Returns the cloned Program. `clone_map` (id(original) -> clone) is attached
    as `._clone_map` so callers (the rewrites below) can locate the clone of a
    node they identified in the original tree."""
    cmap: dict[int, object] = {}

    def clone_stmt(s: Stmt) -> Stmt:
        if isinstance(s, Load):
            c = Load(s.source, list(s.index))
        elif isinstance(s, Compute):
            # inputs are remapped after all producers in scope are cloned; we
            # clone the node now and fill inputs via the map (producers always
            # appear as body statements, so they are cloned by the body walk).
            c = Compute(s.op, [])
            c._orig_inputs = s.inputs            # stash; rewired in second pass
        elif isinstance(s, Store):
            c = Store(s.dest, None, list(s.index))
            c._orig_src = s.src
        elif isinstance(s, ReductionLoop):
            c = ReductionLoop(s.axis, s.tile, [clone_stmt(b) for b in s.body], None)
            c._orig_partial = s.partial
        elif isinstance(s, SpatialLoop):
            c = SpatialLoop(s.axis, s.tile, [clone_stmt(b) for b in s.body])
        else:
            raise TypeError(f"cannot clone statement {type(s).__name__}")
        cmap[id(s)] = c
        return c

    def clone_loop(loop: ParallelLoop) -> ParallelLoop:
        c = ParallelLoop(loop.out, tuple(loop.tile_shape), tuple(loop.index_vars),
                         [clone_stmt(b) for b in loop.body])
        cmap[id(loop)] = c
        return c

    new_body = [clone_loop(l) for l in program.body]

    # second pass: rewire all references through the clone map.
    def rewire(s: Stmt) -> None:
        if isinstance(s, Compute):
            s.inputs = [cmap[id(v)] for v in s._orig_inputs]
            del s._orig_inputs
        elif isinstance(s, Store):
            s.src = cmap[id(s._orig_src)]
            del s._orig_src
        elif isinstance(s, ReductionLoop):
            s.partial = cmap[id(s._orig_partial)]
            del s._orig_partial
            for b in s.body:
                rewire(b)
        elif isinstance(s, SpatialLoop):
            for b in s.body:
                rewire(b)

    for loop in new_body:
        for s in loop.body:
            rewire(s)

    cloned = Program(dict(program.tensors), new_body)
    cloned._clone_map = cmap
    return cloned


# --------------------------------------------------------------------------
# Tree-walk helpers (parent/scope lookup)
# --------------------------------------------------------------------------
def _all_stmts(body):
    for s in body:
        yield s
        if isinstance(s, (ReductionLoop, SpatialLoop)):
            yield from _all_stmts(s.body)


def _consumers(program: Program, value) -> list:
    """Every node that consumes `value` as a tile (Compute input, Store src,
    or ReductionLoop partial)."""
    out = []
    for loop in program.body:
        for s in _all_stmts(loop.body):
            if isinstance(s, Compute) and any(i is value for i in s.inputs):
                out.append(s)
            elif isinstance(s, Store) and s.src is value:
                out.append(s)
            elif isinstance(s, ReductionLoop) and s.partial is value:
                out.append(s)
    return out


def _scope_of(program: Program, node) -> list | None:
    """The chain of enclosing loops for `node`, outermost first (the ParallelLoop
    is index 0). None if not found."""
    def walk(body, chain):
        for s in body:
            if s is node:
                return chain
            if isinstance(s, (ReductionLoop, SpatialLoop)):
                hit = walk(s.body, chain + [s])
                if hit is not None:
                    return hit
        return None
    for loop in program.body:
        hit = walk(loop.body, [loop])
        if hit is not None:
            return hit
    return None


# --------------------------------------------------------------------------
# Hoist: move a Load from inside a loop to the loop's parent body
# --------------------------------------------------------------------------
def can_hoist(program: Program, load: Load) -> tuple[bool, str]:
    """Hoist is legal iff `load` is a Load nested in at least one loop. There is
    no dataflow precondition: a Load inside a loop already has all consumers at
    its scope or deeper, so lifting it only ever creates outer->inner flow
    (handled by ct.extract)."""
    if not isinstance(load, Load):
        return False, "hoist target is not a Load"
    scope = _scope_of(program, load)
    if scope is None:
        return False, "load not found in program"
    if len(scope) < 2:                  # [ParallelLoop] only -> already at top
        return False, "load is already at the kernel top (no enclosing loop)"
    return True, ""


def hoist(program: Program, load: Load) -> Program:
    """Return a new program with `load` moved out of its immediate enclosing
    loop into that loop's parent body, just before the loop."""
    ok, why = can_hoist(program, load)
    if not ok:
        raise ValueError(f"cannot hoist: {why}")
    new = clone_program(program)
    cload = new._clone_map[id(load)]
    scope = _scope_of(new, cload)       # enclosing loops, outermost first
    loop = scope[-1]                    # immediate enclosing loop
    parent_body = scope[-2].body        # the loop's parent (len>=2 guaranteed)
    loop.body.remove(cload)
    parent_body.insert(parent_body.index(loop), cload)
    _strip_clone_meta(new)
    return new


# --------------------------------------------------------------------------
# Sink: move a Load from a body into a sibling loop's body
# --------------------------------------------------------------------------
def can_sink(program: Program, load: Load, loop) -> tuple[bool, str]:
    """Sink is legal iff `load` is a Load, `loop` is a sibling loop in the same
    body, and EVERY consumer of `load` lives inside `loop`. Otherwise sinking
    would strand a consumer at a shallower scope than the relocated producer."""
    if not isinstance(load, Load):
        return False, "sink target is not a Load"
    if not isinstance(loop, (ReductionLoop, SpatialLoop)):
        return False, "sink destination is not a loop"
    lscope = _scope_of(program, load)
    kscope = _scope_of(program, loop)
    if lscope is None or kscope is None:
        return False, "load or loop not found"
    # siblings: load and loop sit in the SAME body, i.e. share the enclosing
    # loop chain (_scope_of returns enclosing loops, not the node itself).
    if lscope != kscope:
        return False, "load and loop are not siblings in the same body"
    inside = set(id(s) for s in _all_stmts(loop.body)) | {id(loop)}
    for c in _consumers(program, load):
        if id(c) not in inside:
            return False, "a consumer of the load lives outside the target loop"
    if not _consumers(program, load):
        return False, "load has no consumers (dead); nothing to sink toward"
    return True, ""


def sink(program: Program, load: Load, loop) -> Program:
    """Return a new program with `load` moved into `loop.body` (at the front)."""
    ok, why = can_sink(program, load, loop)
    if not ok:
        raise ValueError(f"cannot sink: {why}")
    new = clone_program(program)
    cload = new._clone_map[id(load)]
    cloop = new._clone_map[id(loop)]
    lscope = _scope_of(new, cload)      # enclosing loops, outermost first
    cur_body = lscope[-1].body          # body that currently holds the load
    cur_body.remove(cload)
    cloop.body.insert(0, cload)
    _strip_clone_meta(new)
    return new


# --------------------------------------------------------------------------
# small internal helpers
# --------------------------------------------------------------------------
def _strip_clone_meta(program: Program) -> None:
    if hasattr(program, "_clone_map"):
        del program._clone_map


# --------------------------------------------------------------------------
# SubtileReduction: wrap a reducible Compute in a ReductionLoop over a
# contraction axis. Tiles the contraction by `tile` (TS_<axis>) and accumulates
# with the reduce op (e.g. ct.mma). Consumers of the Compute are UNTOUCHED --
# they already reference the Compute, which becomes the loop's accumulator value
# (the loop is a Value whose result is the Compute). Load placement is left to
# Hoist/Sink: loads feeding the Compute that live in its own body move into the
# loop (loading TS_<axis> per iteration); loads in an outer scope stay there and
# the loop extracts from them.
# --------------------------------------------------------------------------
DEFAULT_REDUCTION_TILE = 32


def _output_axes(c: Compute) -> list[str]:
    """Axes that survive into the Compute's output tile (forward-derived)."""
    return _value_axes(c)


def _contraction_axes(c: Compute) -> set[str]:
    """Axes present in the inputs but reduced away (not in the output)."""
    ins = set()
    for ld in _loads_feeding(c):
        ins.update(ld.index)
    return ins - set(_output_axes(c))


def can_subtile_reduction(program: Program, compute: Compute, axis: str
                          ) -> tuple[bool, str]:
    """Legal iff `compute` is a reducible Compute, `axis` is one of its
    contraction axes (in its inputs, absent from its output), and `axis` is not
    already reduced at the compute's scope (no enclosing ReductionLoop on it)."""
    if not isinstance(compute, Compute):
        return False, "target is not a Compute"
    if compute.op not in REDUCE_RULES:
        return False, f"op {compute.op!r} is not reducible (not in REDUCE_RULES)"
    scope = _scope_of(program, compute)
    if scope is None:
        return False, "compute not found in program"
    if axis not in _contraction_axes(compute):
        return False, (f"axis {axis!r} is not a contraction axis of this compute "
                       f"(contraction axes: {sorted(_contraction_axes(compute))})")
    # not already reduced on this axis anywhere enclosing the compute
    for loop in scope:
        if isinstance(loop, ReductionLoop) and loop.axis == axis:
            return False, f"axis {axis!r} is already reduced at this scope"
    return True, ""


def subtile_reduction(program: Program, compute: Compute, axis: str,
                      tile: int = DEFAULT_REDUCTION_TILE) -> Program:
    """Return a new program with `compute` wrapped in a ReductionLoop over
    `axis`. ONLY the Compute is wrapped -- the loop body is exactly `[compute]`.
    Loads stay where they are; since they now sit in a scope shallower than the
    Compute, the renderer extracts the per-iteration subtile from them. Moving
    those loads into (or back out of) the loop is left entirely to Hoist/Sink.
    Consumers are untouched -- they reference the Compute, which becomes the
    loop's accumulator value."""
    ok, why = can_subtile_reduction(program, compute, axis)
    if not ok:
        raise ValueError(f"cannot subtile_reduction: {why}")
    new = clone_program(program)
    ccomp = new._clone_map[id(compute)]
    scope = _scope_of(new, ccomp)
    body = scope[-1].body                       # body that directly holds ccomp

    # wrap ONLY the compute; loads remain in place and are extracted from.
    red = ReductionLoop(axis, tile, [ccomp], ccomp)
    body[body.index(ccomp)] = red

    # every consumer that referenced the bare Compute must now reference the
    # ReductionLoop -- the loop is the Value whose result is the accumulator.
    for c in _consumers(new, ccomp):
        if c is red:
            continue                            # the loop's own partial edge
        if isinstance(c, Compute):
            c.inputs = [red if i is ccomp else i for i in c.inputs]
        elif isinstance(c, Store):
            if c.src is ccomp:
                c.src = red
        elif isinstance(c, ReductionLoop):
            if c.partial is ccomp:
                c.partial = red
    _strip_clone_meta(new)
    return new


# --------------------------------------------------------------------------
# UnwrapReduction: the exact inverse of SubtileReduction. Collapse a
# ReductionLoop whose body is exactly its bare Compute back into that Compute
# (full-extent, no accumulator), and repoint every consumer of the loop's
# accumulator value to the Compute. Load placement is untouched -- whatever
# Hoist/Sink left the feeders as, the now-unwrapped Compute consumes them.
# --------------------------------------------------------------------------
def can_unwrap_reduction(program: Program, loop: ReductionLoop) -> tuple[bool, str]:
    """Legal iff `loop` is a ReductionLoop whose body is exactly `[partial]` --
    a single Compute, which is the partial. (Strict on purpose: this is the
    precise post-state of SubtileReduction, so the two are clean inverses. A
    body with sunk-in loads is rejected; Hoist them out first.)"""
    if not isinstance(loop, ReductionLoop):
        return False, "target is not a ReductionLoop"
    if _scope_of(program, loop) is None:
        return False, "loop not found in program"
    if len(loop.body) != 1 or loop.body[0] is not loop.partial:
        return False, ("reduction body is not exactly its bare Compute "
                       "(hoist any loads/other statements out first)")
    if not isinstance(loop.partial, Compute):
        return False, "reduction partial is not a Compute"
    return True, ""


def unwrap_reduction(program: Program, loop: ReductionLoop) -> Program:
    """Return a new program with `loop` replaced by its bare Compute, and every
    consumer of the loop's accumulator repointed to that Compute."""
    ok, why = can_unwrap_reduction(program, loop)
    if not ok:
        raise ValueError(f"cannot unwrap_reduction: {why}")
    new = clone_program(program)
    cloop = new._clone_map[id(loop)]
    ccomp = cloop.partial
    scope = _scope_of(new, cloop)
    body = scope[-1].body                       # body that directly holds cloop

    # replace the loop with its Compute in the same slot
    body[body.index(cloop)] = ccomp

    # every consumer that referenced the ReductionLoop now references the Compute
    for c in _consumers(new, cloop):
        if isinstance(c, Compute):
            c.inputs = [ccomp if i is cloop else i for i in c.inputs]
        elif isinstance(c, Store):
            if c.src is cloop:
                c.src = ccomp
        elif isinstance(c, ReductionLoop):
            if c.partial is cloop:
                c.partial = ccomp
    _strip_clone_meta(new)
    return new