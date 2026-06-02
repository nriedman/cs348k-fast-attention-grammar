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
    Value, Stmt, REDUCE_RULES, _value_axes, _loads_feeding, _iter_loads,
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
            c = Compute(s.op, [], axis=s.axis)
            c._orig_inputs = s.inputs            # stash; rewired in second pass
        elif isinstance(s, Store):
            c = Store(s.dest, None, list(s.index))
            c._orig_src = s.src
        elif isinstance(s, ReductionLoop):
            c = ReductionLoop(s.axis, s.tile, [clone_stmt(b) for b in s.body])
            c._orig_partials = list(s.partials)
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
            s.partials = [cmap[id(p)] for p in s._orig_partials]
            s.partial = s.partials[0] if s.partials else None
            del s._orig_partials
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
    loop.body[:] = [s for s in loop.body if s is not cload]
    parent_body.insert(_index_of(parent_body, loop), cload)
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
    cur_body[:] = [s for s in cur_body if s is not cload]
    cloop.body.insert(0, cload)
    _strip_clone_meta(new)
    return new


# --------------------------------------------------------------------------
# small internal helpers
# --------------------------------------------------------------------------
def _strip_clone_meta(program: Program) -> None:
    if hasattr(program, "_clone_map"):
        del program._clone_map


def _index_of(body: list, node) -> int:
    """Position of `node` in `body` BY IDENTITY. dataclass __eq__ makes
    structurally identical nodes (e.g. two Load('x',['n','k'])) compare equal,
    so list.index would return the wrong position."""
    for i, s in enumerate(body):
        if s is node:
            return i
    raise ValueError("node not found in body")


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
    """Axes present in the inputs but reduced away (not spanned by the output).
    For matmul this is the shared input axis; for a row-reduction (rowmax/rowsum)
    the op already names its reduced axis, and that axis is still subtileable
    even though it remains (as size 1) in the output's axis list."""
    from kernel_ast import ROW_REDUCE_OPS
    if c.op in ROW_REDUCE_OPS:
        return {c.axis}
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
    body[_index_of(body, ccomp)] = red

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
    body[_index_of(body, cloop)] = ccomp

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


# ==========================================================================
# STEP 1 -- explicit cross-stage data dependencies.
#
# The dependency between ParallelLoop stages is otherwise implicit: stage B
# "depends on" stage A only because B contains a Load whose source equals A's
# `out`. These helpers compute that graph as a first-class object the fusion
# rewrites query, without adding stored fields to the AST (consistent with the
# "derive the tensor/tile contract, don't store it" decision).
# ==========================================================================
def stage_inputs(loop: ParallelLoop) -> set:
    """Tensor names this stage reads (its Load sources)."""
    return {ld.source for ld in _iter_loads(loop.body)}


def stage_output(loop: ParallelLoop) -> str:
    """Tensor name this stage writes."""
    return loop.out


def stage_deps(program: Program) -> dict:
    """id(stage) -> set of ids of EARLIER stages whose output this stage reads.
    The explicit cross-stage edges: an edge A->B exists iff A.out in B's
    inputs and A precedes B."""
    deps: dict = {}
    for i, b in enumerate(program.body):
        ins = stage_inputs(b)
        deps[id(b)] = {id(a) for a in program.body[:i] if stage_output(a) in ins}
    return deps


def _stage_index(program: Program, loop: ParallelLoop) -> int:
    for i, b in enumerate(program.body):
        if b is loop:
            return i
    return -1


def _tensor_consumers(program: Program, name: str) -> list:
    """Stages that read tensor `name` (it appears among their Load sources)."""
    return [b for b in program.body if name in stage_inputs(b)]


# ==========================================================================
# STEP 2 -- Merge two ParallelLoop stages.
#
# Fuse producer A (writes tensor `d`) and consumer B (reads `d`) into one
# kernel, eliminating the global round-trip of `d`: the tile A's Store would
# have written is handed directly to where B's Load would have read it. This is
# the "demote a TENSOR edge to a TILE edge" operation -- the named cross-stage
# tensor `d` becomes an unnamed in-scope tile reference.
#
# Step 3's EliminateRoundTrip is folded in here: dropping the Store->Load pair
# that crossed global memory IS the core of this transform.
# ==========================================================================
def can_merge(program: Program, producer: ParallelLoop,
              consumer: ParallelLoop) -> tuple:
    """Legal iff: producer immediately precedes consumer; consumer reads the
    producer's output tensor `d`; `d` is consumed by no other stage and is not a
    program output; and the two stages share identical index_vars and tile_shape
    (one grid covers both). Returns (ok, reason)."""
    ia, ib = _stage_index(program, producer), _stage_index(program, consumer)
    if ia < 0 or ib < 0:
        return False, "producer or consumer not found in program"
    if ib != ia + 1:
        return False, "stages are not adjacent (producer immediately before consumer)"
    d = producer.out
    if d not in stage_inputs(consumer):
        return False, f"consumer does not read producer's output {d!r}"
    # `d` must be a pure intermediate consumed only by this consumer.
    others = [b for b in _tensor_consumers(program, d) if b is not consumer]
    if others:
        return False, f"tensor {d!r} is also read by another stage"
    _, _, outputs = _program_io_names(program)
    if d in outputs:
        return False, f"tensor {d!r} is a program output (cannot be dropped)"
    # iteration structure must match so one grid covers both stages
    if tuple(producer.index_vars) != tuple(consumer.index_vars):
        return False, "stages have different index_vars"
    if tuple(producer.tile_shape) != tuple(consumer.tile_shape):
        return False, "stages have different tile_shape"
    # producer must write `d` exactly once, as a top-level Store of a single tile
    pstores = [s for s in producer.body if isinstance(s, Store) and s.dest == d]
    if len(pstores) != 1:
        return False, f"producer does not write {d!r} with a single top-level Store"
    # consumer must read `d` with loads whose index matches the store's index
    cloads = [ld for ld in _iter_loads(consumer.body) if ld.source == d]
    if not cloads:
        return False, f"consumer has no Load of {d!r}"
    store_idx = pstores[0].index
    for ld in cloads:
        if ld.index != store_idx:
            return False, (f"consumer Load index {ld.index} != producer Store "
                           f"index {store_idx} for {d!r} (mismatched tiling)")
    return True, ""


def merge(program: Program, producer: ParallelLoop,
          consumer: ParallelLoop) -> Program:
    """Return a new program with `producer` and `consumer` fused into one
    ParallelLoop, the intermediate tensor `d` eliminated, and `d`'s loads in the
    consumer rebound to the producer's stored tile."""
    ok, why = can_merge(program, producer, consumer)
    if not ok:
        raise ValueError(f"cannot merge: {why}")
    new = clone_program(program)
    A = new._clone_map[id(producer)]
    B = new._clone_map[id(consumer)]
    d = A.out

    # the producer's Store of `d` and the tile it stored
    a_store = next(s for s in A.body if isinstance(s, Store) and s.dest == d)
    handoff = a_store.src                       # the tile to pipe directly into B

    # consumer's Load(s) of `d` -> rebind their consumers to `handoff`
    d_loads = [ld for ld in _iter_loads(B.body) if ld.source == d]
    for ld in d_loads:
        for c in _consumers(new, ld):
            if isinstance(c, Compute):
                c.inputs = [handoff if i is ld else i for i in c.inputs]
            elif isinstance(c, Store):
                if c.src is ld:
                    c.src = handoff
            elif isinstance(c, ReductionLoop):
                if c.partial is ld:
                    c.partial = handoff

    # build the fused body: producer's body minus its Store of `d`, then
    # consumer's body minus its now-dead Load(s) of `d`.
    a_body = [s for s in A.body if s is not a_store]
    b_body = [s for s in B.body if not (isinstance(s, Load) and s in d_loads)]
    fused = ParallelLoop(B.out, tuple(B.tile_shape), tuple(B.index_vars),
                         a_body + b_body)

    # splice: replace the A..B pair with the fused stage; drop `d` from tensors.
    ia = _stage_index(new, A)
    new.body[ia:ia + 2] = [fused]
    new.tensors.pop(d, None)
    _strip_clone_meta(new)
    return new


def _program_io_names(program: Program):
    """(inputs, intermediates, outputs) tensor-name lists -- mirror of
    kernel_ast._program_io but local to the rewrite layer."""
    written, read = set(), set()
    for loop in program.body:
        written.add(loop.out)
        for ld in _iter_loads(loop.body):
            read.add(ld.source)
    return (sorted(read - written), sorted(written & read), sorted(written - read))


# ==========================================================================
# STEP 4a -- dedup_loads: rewrite-time CSE on Load nodes.
#
# After fusing two reduction loops, their bodies concatenate; if both contained
# a Load of the same tensor at the same index, those are two distinct Load nodes
# emitting two ct.load calls. This pass collapses structurally identical Loads
# (same source AND index) that live in the SAME body to a single node, rebinding
# every consumer of the duplicates to the survivor. Standalone and reusable --
# run it after any fusion to recover load reuse.
# ==========================================================================
def dedup_loads(program: Program) -> Program:
    """Return a new program where structurally identical sibling Loads (same
    source + index, same body) are collapsed to one, consumers rebound."""
    new = clone_program(program)

    def dedup_body(body):
        survivors: dict[tuple, Load] = {}
        removed: list[Load] = []
        for s in body:
            if isinstance(s, Load):
                key = (s.source, tuple(s.index))
                if key in survivors:
                    keep = survivors[key]
                    for c in _consumers(new, s):        # rebind to the survivor
                        if isinstance(c, Compute):
                            c.inputs = [keep if i is s else i for i in c.inputs]
                        elif isinstance(c, Store):
                            if c.src is s:
                                c.src = keep
                        elif isinstance(c, ReductionLoop):
                            c.partials = [keep if p is s else p for p in c.partials]
                            c.partial = c.partials[0] if c.partials else None
                    removed.append(s)
                else:
                    survivors[key] = s
            elif isinstance(s, (ReductionLoop, SpatialLoop)):
                dedup_body(s.body)
        # rebuild the body excluding removed nodes BY IDENTITY -- Load is a
        # dataclass, so list.remove (which uses ==) would drop the wrong one.
        rid = {id(r) for r in removed}
        body[:] = [s for s in body if id(s) not in rid]

    for loop in new.body:
        dedup_body(loop.body)
    _strip_clone_meta(new)
    return new


# ==========================================================================
# STEP 4b -- merge_reductions: fuse two adjacent sibling ReductionLoops.
#
# Two reduction loops in the same body fuse into one (sharing the iteration)
# iff: same axis, same tile, immediately adjacent, and the second references
# NEITHER the first loop NOR any of the first's partials (independence -- shared
# *inputs* are allowed and are the point). The fused loop carries both loops'
# partials; consumers of each partial already reference it directly, so the only
# rewiring is redirecting consumers of the SECOND loop's Value to its partial.
# dedup_loads is then run to collapse shared input loads.
# ==========================================================================
def _refs_in(node, targets: set) -> bool:
    """True if any node in the dataflow under `node` is one of `targets` (by id)."""
    seen = set()
    def walk(v):
        if id(v) in targets:
            return True
        if id(v) in seen:
            return False
        seen.add(id(v))
        if isinstance(v, Compute):
            return any(walk(i) for i in v.inputs)
        if isinstance(v, ReductionLoop):
            return any(walk(p) for p in v.partials)
        return False
    return walk(node)


def can_merge_reductions(program: Program, first: ReductionLoop,
                         second: ReductionLoop) -> tuple:
    """Legal iff first and second are adjacent siblings in the same body, same
    axis and tile, and `second` references neither `first` nor first's partials
    (independence; shared inputs allowed)."""
    if not isinstance(first, ReductionLoop) or not isinstance(second, ReductionLoop):
        return False, "both targets must be ReductionLoops"
    fscope, sscope = _scope_of(program, first), _scope_of(program, second)
    if fscope is None or sscope is None:
        return False, "loop not found in program"
    if fscope != sscope:
        return False, "loops are not siblings in the same body"
    body = fscope[-1].body
    fi, si = _index_of(body, first), _index_of(body, second)
    if si != fi + 1:
        return False, "loops are not immediately adjacent (first then second)"
    if first.axis != second.axis:
        return False, f"different reduction axes ({first.axis!r} vs {second.axis!r})"
    if first.tile != second.tile:
        return False, f"different tile sizes ({first.tile} vs {second.tile})"
    # independence: second must not consume first's result or any of its partials
    forbidden = {id(first)} | {id(p) for p in first.partials}
    for p in second.partials:
        if _refs_in(p, forbidden):
            return False, "second reduction depends on the first's result/partials"
    return True, ""


def merge_reductions(program: Program, first: ReductionLoop,
                     second: ReductionLoop) -> Program:
    """Return a new program with `first` and `second` fused into one
    ReductionLoop carrying both loops' partials, then load-deduplicated."""
    ok, why = can_merge_reductions(program, first, second)
    if not ok:
        raise ValueError(f"cannot merge_reductions: {why}")
    new = clone_program(program)
    A = new._clone_map[id(first)]
    B = new._clone_map[id(second)]
    scope = _scope_of(new, A)
    body = scope[-1].body

    fused = ReductionLoop(A.axis, A.tile, A.body + B.body,
                          partials=A.partials + B.partials)

    # both loop nodes are replaced by `fused`, so consumers of EITHER loop's
    # Value must rebind to that loop's first partial (its Value result).
    for loop_node in (A, B):
        val = loop_node.partials[0]
        for c in _consumers(new, loop_node):
            if isinstance(c, Compute):
                c.inputs = [val if i is loop_node else i for i in c.inputs]
            elif isinstance(c, Store):
                if c.src is loop_node:
                    c.src = val
            elif isinstance(c, ReductionLoop):
                c.partials = [val if p is loop_node else p for p in c.partials]
                c.partial = c.partials[0] if c.partials else None

    fi = _index_of(body, A)
    body[fi:fi + 2] = [fused]
    _strip_clone_meta(new)
    # collapse loads now shared between the two reductions' bodies
    return dedup_loads(new)


# ==========================================================================
# STEP 5 -- Reorder: swap two adjacent sibling nodes.
#
# A pure, orthogonal primitive: it swaps the execution order of two adjacent
# siblings (any kinds -- ParallelLoop/ParallelLoop, Load/ReductionLoop, etc.)
# whenever doing so respects data flow. Its purpose is to bring a producer and
# consumer adjacent so Merge / merge_reductions can fire; the search prunes the
# many legal-but-inert swaps (e.g. Load/Load).
#
# Legal iff (a) neither node depends on the other through tile references
# (no read-after-write either direction -- the swap must not invert a real
# edge), and (b) they do not both write the same tensor (write-write hazard,
# which pure dataflow reachability cannot see). For sibling statements the edge
# is tile dataflow; for ParallelLoop stages it is the tensor edge (stage_deps).
# ==========================================================================
def _depends_on(consumer, producer) -> bool:
    """True if `consumer` reads (transitively) a tile produced by `producer`.
    Covers every reference kind: Compute.inputs, Store.src, and a ReductionLoop's
    whole body (its partials AND any other body statement's inputs)."""
    target = {id(producer)}

    # nodes whose dataflow `consumer` depends on
    def surface(node):
        if isinstance(node, Compute):
            return list(node.inputs)
        if isinstance(node, Store):
            return [node.src]
        if isinstance(node, ReductionLoop):
            # a reduction depends on everything its body reads from outside:
            # gather all inputs referenced anywhere in the body.
            refs = []
            for s in _all_stmts(node.body):
                if isinstance(s, Compute):
                    refs.extend(s.inputs)
                elif isinstance(s, Store):
                    refs.append(s.src)
            refs.extend(node.partials)
            return refs
        return []

    seen = set()
    stack = list(surface(consumer))
    while stack:
        v = stack.pop()
        if id(v) in target:
            return True
        if id(v) in seen:
            continue
        seen.add(id(v))
        stack.extend(surface(v))
    return False


def _writes(node) -> set:
    """Tensor names this node writes (Store.dest, or a stage's output)."""
    if isinstance(node, Store):
        return {node.dest}
    if isinstance(node, ParallelLoop):
        return {node.out}
    return set()


def can_reorder(program: Program, first, second) -> tuple:
    """Legal iff first and second are immediately-adjacent siblings, neither
    depends on the other (no tile/tensor RAW either direction), and they do not
    both write the same tensor."""
    # ParallelLoop stages: siblings in program.body
    if isinstance(first, ParallelLoop) and isinstance(second, ParallelLoop):
        ia, ib = _stage_index(program, first), _stage_index(program, second)
        if ia < 0 or ib < 0:
            return False, "stage not found in program"
        if abs(ia - ib) != 1:
            return False, "stages are not adjacent"
        a, b = (first, second) if ia < ib else (second, first)
        # RAW: later stage reads earlier stage's output
        if stage_output(a) in stage_inputs(b):
            return False, "later stage consumes the earlier stage's output"
        # WAW: both write the same tensor
        if _writes(a) & _writes(b):
            return False, "stages write the same tensor (write-write hazard)"
        return True, ""

    # statements within one kernel body: must be adjacent siblings
    fscope, sscope = _scope_of(program, first), _scope_of(program, second)
    if fscope is None or sscope is None:
        return False, "node not found in program"
    if fscope != sscope:
        return False, "nodes are not siblings in the same body"
    body = fscope[-1].body
    fi, si = _index_of(body, first), _index_of(body, second)
    if abs(fi - si) != 1:
        return False, "nodes are not immediately adjacent"
    a, b = (first, second) if fi < si else (second, first)
    # neither may depend on the other (a real edge would be inverted by the swap)
    if _depends_on(b, a):
        return False, "second node depends on the first (tile dataflow)"
    if _depends_on(a, b):
        return False, "first node depends on the second (tile dataflow)"
    # write-write hazard (dataflow reachability cannot see this)
    if _writes(a) & _writes(b):
        return False, "nodes write the same tensor (write-write hazard)"
    return True, ""


def reorder(program: Program, first, second) -> Program:
    """Return a new program with the two adjacent siblings `first` and `second`
    swapped in their body."""
    ok, why = can_reorder(program, first, second)
    if not ok:
        raise ValueError(f"cannot reorder: {why}")
    new = clone_program(program)
    A = new._clone_map[id(first)]
    B = new._clone_map[id(second)]
    if isinstance(A, ParallelLoop) and isinstance(B, ParallelLoop):
        body = new.body
    else:
        body = _scope_of(new, A)[-1].body
    ia, ib = _index_of(body, A), _index_of(body, B)
    body[ia], body[ib] = body[ib], body[ia]
    _strip_clone_meta(new)
    return new