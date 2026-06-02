"""
optimization.py -- Stochastic Rewrite Descent (SRD) autotuner over the kernel
grammar, after Kodnongbua et al., "Design for Descent" (SIGGRAPH Asia 2025).

The search interleaves two kinds of step on a program that all compute the same
result:

  Stage 1 (parameters): discrete coordinate descent over tile sizes -- the
    gradient-descent analogue for our discrete parameter space. One full sweep
    moves each tunable tile dimension to its best-improving neighbor on a size
    ladder.

  Stage 2 (structure): the stochastic rewrite step. Sample K applicable rewrites,
    trial each (apply on a clone + one parameter step), accept all that don't
    worsen the loss, and apply them best-first, re-checking preconditions so
    interacting rewrites stay safe.

The objective is real GPU runtime: compile to cuTile, run on the benchmark
inputs, and time it (warmup + repeats, failures -> inf). A trivial proxy is
provided only as a dev convenience for exercising the loop off-GPU.

The set of enabled rewrites is a parameter (`rules=`), so the central experiment
-- search quality as a function of grammar design (reversibility,
overparameterization) -- is a one-line ablation.
"""

from __future__ import annotations

import math
import random
import sys
import time
import tempfile
import os
import importlib.util
from dataclasses import dataclass, field

from .kernel_ast import (
    Program, ParallelLoop, ReductionLoop, SpatialLoop, Load, Store, Compute,
    emit_module, structural_key, peak_tile_bytes,
)
from . import rewrites as R


# A penalised (over-memory-budget) program scores MEM_PENALTY_BASE + KB. The base
# is far above any plausible kernel runtime in ms, so any program that actually
# compiles and runs always ranks better than one rejected for memory -- while the
# +KB term gives a gradient among penalised programs toward smaller footprints.
MEM_PENALTY_BASE = 1e9


# ==========================================================================
# Objective: compile -> run -> time. Failures (e.g. out-of-SMEM tilings) -> inf
# so the search treats infeasible points as maximally bad and moves on. This is
# what lets the descent start from the atomic grammar and *discover* that tiling
# is necessary.
# ==========================================================================
TILE_LADDER = (16, 32, 64, 128)        # ORIGINAL_DIM (full extent) is added per-axis


@dataclass
class EvalConfig:
    warmup: int = 1                     # lean: JIT-compile + cache warm; min-of-reps denoises
    iters: int = 3                      # relative ranking only needs a few timed reps
    proxy: bool = False                 # dev-only: skip GPU, use a cheap fake score
    seed: int = 0
    log: bool = False                   # per-eval phase timing (emit / compile / run)
    mem_budget: int = 64 * 1024         # bytes; the LARGEST single per-block tile
                                        # may not exceed this. An L4 SM gives a
                                        # block up to ~99 KB of shared memory, so a
                                        # tile near/over that can't be scheduled and
                                        # makes cuTile/ptxas struggle. 64 KB leaves
                                        # headroom for register pressure + overhead;
                                        # raise toward 99 KB if the search runs clean.
                                        # Over-budget programs are NOT compiled; they
                                        # score a large, size-proportional penalty so
                                        # the search still descends toward feasibility.
                                        # 0 disables.


class Evaluator:
    """Compiles and times programs, memoized by structural key so identical
    programs are never recompiled or re-timed."""

    def __init__(self, cfg: EvalConfig = EvalConfig()):
        self.cfg = cfg
        self._cache: dict = {}
        self._compiles = 0
        self._runs = 0

    def __call__(self, program: Program) -> float:
        key = structural_key(program)
        if key in self._cache:
            return self._cache[key]
        ms = self._measure(program)
        self._cache[key] = ms
        return ms

    def _measure(self, program: Program) -> float:
        t0 = time.perf_counter()
        try:
            src = emit_module(program)
        except AssertionError:
            # Inference found a value demanded at two irreconcilable tile widths.
            # This is an ILLEGAL FUSION (e.g. a stage that both tiles an axis and
            # reduces, over that axis, a value computed within it -- the flash-style
            # boundary). It is correctly infeasible; report it cleanly rather than
            # as a raw assertion traceback.
            if self.cfg.log:
                print("[eval] illegal fusion (incompatible tile widths) -> inf",
                      flush=True)
            return math.inf
        except Exception as e:
            if self.cfg.log:
                print(f"[eval] emit failed ({type(e).__name__}) -> inf", flush=True)
            return math.inf                 # invalid / unrenderable structure
        if self.cfg.log:
            tiles = " ".join(f"{l.out}{tuple(l.tile_shape)}" for l in program.body)
            print(f"[eval] emit {1e3*(time.perf_counter()-t0):.0f}ms  "
                  f"{len(program.body)} stage(s): {tiles}", flush=True)
        if self.cfg.proxy:
            return _proxy_score(program, src)
        # Memory pre-check: a kernel whose resident tile footprint exceeds the
        # budget will blow up the compiler (and can exhaust host RAM), so we do
        # NOT compile it. Instead of inf -- which gives the search no direction --
        # we return a large penalty PROPORTIONAL to the footprint, so descent
        # still feels a gradient pulling tiles smaller until a program fits and
        # actually compiles. The MEM_PENALTY_BASE offset keeps every penalised
        # score far above any real runtime (ms), so a compilable program always
        # beats an over-budget one.
        if self.cfg.mem_budget:
            footprint = peak_tile_bytes(program)
            if footprint > self.cfg.mem_budget:
                penalty = MEM_PENALTY_BASE + footprint / 1024.0   # +1 per KB
                if self.cfg.log:
                    print(f"[eval] OVER BUDGET {footprint/1024:.0f}KB > "
                          f"{self.cfg.mem_budget/1024:.0f}KB -> penalty {penalty:.0f} "
                          f"(not compiled)", flush=True)
                return penalty
        self._compiles += 1
        try:
            return self._run_gpu(src)
        except Exception as e:
            if self.cfg.log:
                print(f"[eval] run FAILED ({type(e).__name__}: {e}) -> inf", flush=True)
            return math.inf                 # compile or launch failure (e.g. SMEM)

    def _run_gpu(self, src: str) -> float:
        import cupy as cp
        log = self.cfg.log
        t0 = time.perf_counter()
        mod = _import_source(src)
        meta = mod.KERNEL_META
        rng = cp.random.RandomState(self.cfg.seed)
        args = [rng.randn(*s).astype(cp.float32) for _, s in meta["inputs"]]
        fn = mod.fn
        if log:
            print(f"[eval]   import+alloc {1e3*(time.perf_counter()-t0):.0f}ms", flush=True)
        # First launch triggers cuTile JIT compilation of every kernel in the
        # module -- time it separately, since it dominates and is the usual hang.
        t1 = time.perf_counter()
        fn(*args)
        cp.cuda.runtime.deviceSynchronize()
        if log:
            print(f"[eval]   JIT compile + first launch "
                  f"{1e3*(time.perf_counter()-t1):.0f}ms", flush=True)
        for _ in range(max(0, self.cfg.warmup - 1)):
            fn(*args)
        cp.cuda.runtime.deviceSynchronize()
        start = cp.cuda.Event(); end = cp.cuda.Event()
        times = []
        for _ in range(self.cfg.iters):
            start.record()
            fn(*args)
            end.record(); end.synchronize()
            times.append(cp.cuda.get_elapsed_time(start, end))   # ms
        self._runs += 1
        # minimum is the least scheduling-noise-contaminated sample for a
        # throughput-bound kernel; median still drifts with system jitter.
        best = min(times)
        if log:
            print(f"[eval]   {self.cfg.iters} timed reps -> min {best:.4g}ms", flush=True)
        return best


_MODULE_SEQ = [0]


def _import_source(src: str):
    """Write to a temp .py and import as a fresh module -- cuTile's JIT needs
    real source (inspect.getsource) and constant resolution, so exec-into-dict
    does not work."""
    _MODULE_SEQ[0] += 1
    name = f"_srd_kernel_{_MODULE_SEQ[0]}"
    fd, path = tempfile.mkstemp(suffix=".py", prefix=name + "_")
    os.write(fd, src.encode()); os.close(fd)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _proxy_score(program: Program, src: str) -> float:
    """Extremely cheap, deliberately fake objective for exercising the search
    loop without a GPU. Rewards smaller tiles (more parallelism) and tensor-core
    use (presence of ct.mma), penalizes huge tiles. NOT a real cost model."""
    score = 0.0
    for loop in _all_loops(program):
        score += math.prod(loop.tile_shape)            # bigger tiles -> worse
        for rl in _reduction_loops(loop):
            score += rl.tile * 4
    if "ct.mma" in src:
        score *= 0.4                                   # tensor cores: reward
    if "ct.extract" in src:
        score *= 0.95
    return score


# ==========================================================================
# Parameters: every tunable tile dimension, keyed by the EMITTED ConstInt name
# so coupled knobs (an axis that is both an output dim and a reduction axis, or
# shared across fused stages) move together rather than fighting.
# ==========================================================================
@dataclass
class TileParam:
    """One tunable tile dimension. `setters` apply a chosen size to every AST
    slot that emits under the same ConstInt name (so coupled knobs stay in sync).
    `extent` is the axis's global size (its ORIGINAL_DIM)."""
    name: str
    extent: int
    current: int
    setters: list = field(default_factory=list)        # callables: size -> None

    def ladder(self) -> list:
        rungs = sorted({s for s in TILE_LADDER if s <= self.extent} | {self.extent})
        return rungs

    def set(self, size: int) -> None:
        for s in self.setters:
            s(size)
        self.current = size


def _all_loops(program: Program):
    return list(program.body)


def _reduction_loops(loop: ParallelLoop):
    out = []
    def walk(body):
        for s in body:
            if isinstance(s, ReductionLoop):
                out.append(s); walk(s.body)
            elif isinstance(s, SpatialLoop):
                walk(s.body)
    walk(loop.body)
    return out


def parameter_vector(program: Program) -> list:
    """Build the tunable tile parameters of `program`, keyed by emitted name.

    Output-tile dims emit as TILE_<out>[d]; reduction tiles as RTILE_<out>. An
    axis that is both an output dim and a reduction axis shares one ConstInt, so
    we group setters under a single name and drive them together."""
    params: dict[str, TileParam] = {}

    def add(name: str, extent: int, current: int, setter) -> None:
        if name in params:
            params[name].setters.append(setter)
            # extent/current should agree across coupled slots; keep the first
        else:
            params[name] = TileParam(name, extent, current, [setter])

    for loop in program.body:
        out = loop.out
        gshape = program.tensors[out]
        # output tile dims
        for d, (sz, ext) in enumerate(zip(loop.tile_shape, gshape)):
            if ext <= 1:
                continue                                # collapsed/degenerate dim
            def make_setter(L=loop, dd=d):
                def s(v):
                    t = list(L.tile_shape); t[dd] = v; L.tile_shape = tuple(t)
                return s
            add(f"TILE_{out}[{d}]", ext, sz, make_setter())
        # reduction tiles
        for rl in _reduction_loops(loop):
            ext = R._reduction_axis_extent(program, rl) if hasattr(R, "_reduction_axis_extent") \
                else _reduction_extent(program, rl)
            def make_rsetter(RL=rl):
                def s(v):
                    RL.tile = v
                return s
            add(f"RTILE_{out}", ext, rl.tile, make_rsetter())
    return list(params.values())


def _reduction_extent(program: Program, rl: ReductionLoop) -> int:
    """Global extent of a reduction loop's axis, from a load feeding a partial."""
    for ld in R._loads_feeding(rl):
        if rl.axis in ld.index:
            return program.tensors[ld.source][ld.index.index(rl.axis)]
    from .kernel_ast import _iter_loads
    for ld in _iter_loads(rl.body):
        if rl.axis in ld.index:
            return program.tensors[ld.source][ld.index.index(rl.axis)]
    return rl.tile


# ==========================================================================
# Stage 1 -- parameter optimization: one full coordinate sweep. Each tunable
# tile dim is moved to its best-improving neighbor on the ladder (adjacent rungs,
# to stay gradient-flavored). Tile dims interact, so we sweep all coordinates.
# ==========================================================================
def stage1_sweep(program: Program, ev: Evaluator) -> tuple[Program, float]:
    """One full coordinate sweep over tile parameters. For each tunable tile
    dimension, evaluate EVERY rung on its ladder and keep the best. All-rungs
    (not just adjacent) because the ladder is short (4-5 values), so it is cheap
    and far more thorough: it escapes local bumps and an infeasible (inf) start
    where the adjacent rung is also infeasible. Mutates a CLONE in place."""
    prog = R.clone_program(program)
    R._strip_clone_meta(prog)
    params = parameter_vector(prog)
    best = ev(prog)
    for p in params:
        best_size, best_loss = p.current, best
        for size in p.ladder():
            if size == p.current:
                continue
            p.set(size)
            loss = ev(prog)
            if loss < best_loss:
                best_loss, best_size = loss, size
        p.set(best_size)
        best = best_loss
    return prog, best


# ==========================================================================
# Stage 2 -- structure optimization: the stochastic rewrite step.
# ==========================================================================
# Each entry yields (apply_fn, can_fn, args) candidates for the current program.
DEFAULT_RULES = ("hoist", "sink", "subtile_reduction", "unwrap_reduction",
                 "merge", "merge_reductions", "reorder")


def _iter_stmts(body):
    for s in body:
        yield s
        if isinstance(s, (ReductionLoop, SpatialLoop)):
            yield from _iter_stmts(s.body)


# A rewrite is identified by a stable, tree-INDEPENDENT descriptor: the path(s)
# to its target node(s) plus any scalar args. `resolve(prog)` finds the matching
# nodes in `prog`, checks the precondition there, and applies -- returning the
# new program, or None if it no longer applies. This is what lets the apply
# phase land several rewrites per step: each is re-resolved against the evolving
# clone rather than replayed via stale-node closures.
def _path_of(program: Program, node) -> tuple | None:
    """Stable address of `node`: (stage_index, (body_index, ...)) walking into
    nested loop bodies. A ParallelLoop stage is (stage_index, ())."""
    for si, loop in enumerate(program.body):
        if loop is node:
            return (si, ())
        hit = _walk_path(loop.body, node, ())
        if hit is not None:
            return (si, hit)
    return None


def _walk_path(body, node, prefix):
    for i, s in enumerate(body):
        if s is node:
            return prefix + (i,)
        if isinstance(s, (ReductionLoop, SpatialLoop)):
            hit = _walk_path(s.body, node, prefix + (i,))
            if hit is not None:
                return hit
    return None


def _at_path(program: Program, path: tuple):
    """Inverse of _path_of: the node at (stage_index, (body_index, ...))."""
    si, idxs = path
    if si >= len(program.body):
        return None
    node = program.body[si]
    body = node.body
    for j, i in enumerate(idxs):
        if i >= len(body):
            return None
        node = body[i]
        if j < len(idxs) - 1:
            if not isinstance(node, (ReductionLoop, SpatialLoop)):
                return None
            body = node.body
    return node


@dataclass
class Rewrite:
    """A re-resolvable structural move. `label` is for logging; `resolve` applies
    it to a given program, returning the new program or None if inapplicable."""
    label: str
    resolve: object                      # callable: program -> program | None


def _mk(kind, label, program, *nodes, **kw):
    """Build a Rewrite whose targets are addressed by path (captured now from
    `program`) and re-resolved at apply time against whatever tree is passed."""
    paths = [_path_of(program, n) for n in nodes]

    def resolve(prog, kind=kind, paths=paths, kw=kw):
        targets = [_at_path(prog, p) for p in paths]
        if any(t is None for t in targets):
            return None
        try:
            if kind == "hoist":
                ld, = targets
                return R.hoist(prog, ld) if R.can_hoist(prog, ld)[0] else None
            if kind == "sink":
                ld, k = targets
                return R.sink(prog, ld, k) if R.can_sink(prog, ld, k)[0] else None
            if kind == "subtile":
                c, = targets
                ax = kw["axis"]
                return (R.subtile_reduction(prog, c, ax)
                        if R.can_subtile_reduction(prog, c, ax)[0] else None)
            if kind == "unwrap":
                rl, = targets
                return (R.unwrap_reduction(prog, rl)
                        if R.can_unwrap_reduction(prog, rl)[0] else None)
            if kind == "merge":
                a, b = targets
                return R.merge(prog, a, b) if R.can_merge(prog, a, b)[0] else None
            if kind == "merge_red":
                a, b = targets
                return (R.merge_reductions(prog, a, b)
                        if R.can_merge_reductions(prog, a, b)[0] else None)
            if kind == "reorder":
                a, b = targets
                return R.reorder(prog, a, b) if R.can_reorder(prog, a, b)[0] else None
        except Exception:
            return None
        return None
    return Rewrite(label, resolve)


def enumerate_rewrites(program: Program, rules) -> list:
    """Every applicable rewrite as a path-addressed Rewrite descriptor (its
    precondition currently holds)."""
    out = []
    all_stmts = [s for loop in program.body for s in _iter_stmts(loop.body)]

    if "hoist" in rules:
        for s in all_stmts:
            if isinstance(s, Load) and R.can_hoist(program, s)[0]:
                out.append(_mk("hoist", f"hoist({s.source})", program, s))
    if "sink" in rules:
        for loop in program.body:
            loops_in = [x for x in _iter_stmts(loop.body)
                        if isinstance(x, (ReductionLoop, SpatialLoop))]
            for s in _iter_stmts(loop.body):
                if isinstance(s, Load):
                    for k in loops_in:
                        if R.can_sink(program, s, k)[0]:
                            out.append(_mk("sink", f"sink({s.source})", program, s, k))
    if "subtile_reduction" in rules:
        for s in all_stmts:
            if isinstance(s, Compute):
                for ax in R._contraction_axes(s):
                    if R.can_subtile_reduction(program, s, ax)[0]:
                        out.append(_mk("subtile", f"subtile({s.op},{ax})",
                                       program, s, axis=ax))
    if "unwrap_reduction" in rules:
        for s in all_stmts:
            if isinstance(s, ReductionLoop) and R.can_unwrap_reduction(program, s)[0]:
                out.append(_mk("unwrap", f"unwrap({s.axis})", program, s))
    if "merge" in rules:
        for a, b in zip(program.body, program.body[1:]):
            if R.can_merge(program, a, b)[0]:
                out.append(_mk("merge", f"merge({a.out},{b.out})", program, a, b))
    if "merge_reductions" in rules:
        for loop in program.body:
            sibs = loop.body
            for a, b in zip(sibs, sibs[1:]):
                if isinstance(a, ReductionLoop) and isinstance(b, ReductionLoop) \
                        and R.can_merge_reductions(program, a, b)[0]:
                    out.append(_mk("merge_red", f"merge_red({a.axis})", program, a, b))
    if "reorder" in rules:
        for a, b in zip(program.body, program.body[1:]):
            if R.can_reorder(program, a, b)[0]:
                out.append(_mk("reorder", f"reorder({a.out},{b.out})", program, a, b))
        for loop in program.body:
            sibs = loop.body
            for a, b in zip(sibs, sibs[1:]):
                if R.can_reorder(program, a, b)[0]:
                    out.append(_mk("reorder", "reorder(stmt)", program, a, b))
    return out


def stage2_step(program: Program, ev: Evaluator, K: int, rules, rng: random.Random
                ) -> tuple[Program, float, list]:
    """Sample K applicable rewrites, trial each (apply + one Stage-1 sweep),
    then apply all non-worsening best-first -- re-resolving each against the
    evolving program and skipping any whose precondition no longer holds. Because
    descriptors re-resolve (rather than replay stale-node closures), several
    rewrites can land in one step."""
    base_loss = stage1_sweep(program, ev)[1]
    menu = enumerate_rewrites(program, rules)
    if not menu:
        return program, base_loss, []
    sample = rng.sample(menu, min(K, len(menu)))

    trials = []                                  # (delta, rewrite)
    for rw in sample:
        cand = rw.resolve(program)               # apply to a fresh clone
        if cand is None:
            continue
        _, loss = stage1_sweep(cand, ev)         # judge the move WITH a re-tune
        if loss <= base_loss:                    # accept non-worsening (enabling moves)
            trials.append((base_loss - loss, rw))

    trials.sort(key=lambda t: -t[0])             # best improvement first
    applied = []
    prog = program
    for delta, rw in trials:
        cand = rw.resolve(prog)                  # re-resolve against current tree
        if cand is None:
            continue                             # precondition no longer holds here
        prog = cand
        applied.append((rw.label, delta))
    return prog, base_loss, applied


# ==========================================================================
# Outer loop
# ==========================================================================
@dataclass
class SearchConfig:
    iters: int = 30
    N: int = 2                                   # run Stage 2 every N iterations
    K: int = 6                                   # rewrites sampled per Stage-2 step
    rules: tuple = DEFAULT_RULES
    seed: int = 0
    verbose: bool = True


def autotune(program: Program, ev: Evaluator | None = None,
             cfg: SearchConfig = SearchConfig()) -> Program:
    """Stochastic Rewrite Descent. Each iteration optionally takes a structural
    step (every N iters) then a single parameter sweep -- with the all-rungs
    greedy sweep that one sweep already reaches the coordinate-descent optimum,
    so N is kept small (a few extra param-only iterations between rewrites would
    just re-evaluate cached neighbors). Returns the best program seen, since the
    descent can wander under the <=-acceptance rule."""
    ev = ev or Evaluator()
    rng = random.Random(cfg.seed)

    program, loss = stage1_sweep(program, ev)        # initial tune
    best_prog, best_loss = R.clone_program(program), loss
    R._strip_clone_meta(best_prog)
    if cfg.verbose:
        print(f"[init] loss={loss:.4g}")

    for it in range(1, cfg.iters + 1):
        if it % cfg.N == 0:
            program, _, applied = stage2_step(program, ev, cfg.K, cfg.rules, rng)
            if cfg.verbose and applied:
                print(f"[it {it}] applied {[a for a, _ in applied]}")
        program, loss = stage1_sweep(program, ev)    # tune (after rewrite, or refine)
        if loss < best_loss:
            best_loss, best_prog = loss, R.clone_program(program)
            R._strip_clone_meta(best_prog)
        if cfg.verbose:
            print(f"[it {it}] loss={loss:.4g}  best={best_loss:.4g}")

    if cfg.verbose:
        print(f"[done] best={best_loss:.4g}  compiles={ev._compiles} runs={ev._runs}")
    return best_prog