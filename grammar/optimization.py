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
import json
import pickle
import signal
import threading
import importlib.util
from dataclasses import dataclass, field

from .kernel_ast import (
    Program, ParallelLoop, ReductionLoop, SpatialLoop, Load, Store, Compute,
    emit_module, structural_key, peak_tile_bytes, kernel_peak_tile_bytes,
    program_flops,
)
from . import rewrites as R


# A penalised (over-memory-budget) program scores MEM_PENALTY_BASE + KB. The base
# is far above any plausible kernel runtime in ms, so any program that actually
# compiles and runs always ranks better than one rejected for memory -- while the
# +KB term gives a gradient among penalised programs toward smaller footprints.
MEM_PENALTY_BASE = 1e9


class _CompileTimeout(Exception):
    """Raised when a single compile+first-launch exceeds cfg.compile_timeout."""


def _timeout_handler(signum, frame):
    raise _CompileTimeout()


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
    compile_timeout: float = 0          # seconds; if a single compile+first-launch
                                        # exceeds this, abandon it and score a penalty
                                        # (some fusions emit ptxas-hostile kernels that
                                        # take minutes; healthy compiles are a few
                                        # seconds, so a 45-60s cap separates them
                                        # cleanly). The penalised key is cached (and
                                        # persisted in the snapshot), so the search
                                        # never retries it. 0 disables. Best-effort:
                                        # relies on SIGALRM, which interrupts at the
                                        # next Python/syscall boundary -- it reliably
                                        # catches subprocess-ptxas stalls but cannot
                                        # preempt a fully in-process compiler call.


class Evaluator:
    """Compiles and times programs, memoized by structural key so identical
    programs are never recompiled or re-timed."""

    def __init__(self, cfg: EvalConfig = EvalConfig()):
        self.cfg = cfg
        self._cache: dict = {}
        self._compiles = 0
        self._runs = 0
        self.ctx = ""                   # search context prefix for log lines, set by
                                        # autotune/stage2 (e.g. "it 4 | S2 subtile(matmul,j)")

    @property
    def _tag(self) -> str:
        """Per-log-line prefix: '[it 4 | S2 ...]' if context is set, else '[eval]'."""
        return f"[{self.ctx}]" if self.ctx else "[eval]"

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
                print(f"{self._tag} illegal fusion (incompatible tile widths) -> inf",
                      flush=True)
            return math.inf
        except Exception as e:
            if self.cfg.log:
                print(f"{self._tag} emit failed ({type(e).__name__}) -> inf", flush=True)
            return math.inf                 # invalid / unrenderable structure
        if self.cfg.log:
            tiles = " ".join(f"{l.out}{tuple(l.tile_shape)}" for l in program.body)
            print(f"{self._tag} emit {1e3*(time.perf_counter()-t0):.0f}ms  "
                  f"{len(program.body)} stage(s): {tiles}", flush=True)
        if self.cfg.proxy:
            return _proxy_score(program, src)
        # Memory pre-check: a kernel whose largest single per-block tile exceeds
        # the budget can't be scheduled (SMEM) and blows up the compiler, so we do
        # NOT compile it. The FEASIBILITY gate is max-based (the hardware limit is
        # per-tile). But the PENALTY MAGNITUDE is the SUM of every stage's
        # over-budget excess -- not the global max -- so each over-budget stage
        # gets its own descent gradient. With a max-based penalty, two equally
        # oversized stages deadlock: shrinking either alone leaves the max (and the
        # loss) unchanged, so coordinate descent sees no improvement on either and
        # never escapes. Summing the excess makes shrinking ANY oversized stage
        # lower the loss. MEM_PENALTY_BASE keeps every penalised score far above
        # any real runtime, so a compilable program always wins.
        if self.cfg.mem_budget:
            per_kernel = kernel_peak_tile_bytes(program)
            gmax = max(per_kernel) if per_kernel else 0
            if gmax > self.cfg.mem_budget:
                excess = sum(max(0, k - self.cfg.mem_budget) for k in per_kernel)
                penalty = MEM_PENALTY_BASE + excess / 1024.0   # +1 per KB over budget
                if self.cfg.log:
                    n_over = sum(1 for k in per_kernel if k > self.cfg.mem_budget)
                    print(f"{self._tag} OVER BUDGET max {gmax/1024:.0f}KB > "
                          f"{self.cfg.mem_budget/1024:.0f}KB in {n_over} stage(s), "
                          f"excess {excess/1024:.0f}KB -> penalty {penalty:.0f} "
                          f"(not compiled)", flush=True)
                return penalty
        self._compiles += 1
        try:
            return self._run_gpu(src)
        except _CompileTimeout:
            # a ptxas-hostile kernel (e.g. a full-width unsubtiled reduction) that
            # exceeded the compile budget. Score it as a penalty -- worse than any
            # real kernel, so the search abandons this rewrite -- and let it cache,
            # so the config is never recompiled (this run or a resumed one).
            if self.cfg.log:
                print(f"{self._tag} COMPILE TIMEOUT (>{self.cfg.compile_timeout:.0f}s) "
                      f"-> penalty (abandoned, not retried)", flush=True)
            return MEM_PENALTY_BASE
        except Exception as e:
            if self.cfg.log:
                print(f"{self._tag} run FAILED ({type(e).__name__}: {e}) -> inf", flush=True)
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
            print(f"{self._tag}   import+alloc {1e3*(time.perf_counter()-t0):.0f}ms", flush=True)
        # First launch triggers cuTile JIT compilation of every kernel in the
        # module -- time it separately, since it dominates and is the usual hang.
        # Guard it with a wall-clock alarm: a pathological fusion can take minutes
        # to compile, so cap it and abandon (-> _CompileTimeout). Only on the main
        # thread (setitimer requirement); the autotune loop runs there.
        timeout = self.cfg.compile_timeout
        armed = (timeout and hasattr(signal, "SIGALRM")
                 and threading.current_thread() is threading.main_thread())
        t1 = time.perf_counter()
        if armed:
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.setitimer(signal.ITIMER_REAL, timeout)
        try:
            fn(*args)
            cp.cuda.runtime.deviceSynchronize()
        finally:
            if armed:
                signal.setitimer(signal.ITIMER_REAL, 0)   # disarm
                signal.signal(signal.SIGALRM, old_handler)
        if log:
            print(f"{self._tag}   JIT compile + first launch "
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
            print(f"{self._tag}   {self.cfg.iters} timed reps -> min {best:.4g}ms", flush=True)
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
    `extent` is the axis's global size (its ORIGINAL_DIM). `rungs_src` is the
    base ladder of candidate sizes (default TILE_LADDER, or an override for the
    jump-continuity ablation)."""
    name: str
    extent: int
    current: int
    setters: list = field(default_factory=list)        # callables: size -> None
    rungs_src: tuple = TILE_LADDER

    def ladder(self) -> list:
        rungs = sorted({s for s in self.rungs_src if s <= self.extent} | {self.extent})
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


def parameter_vector(program: Program, ladder: tuple | None = None,
                     couple: bool = False) -> list:
    """Build the tunable tile parameters of `program`, keyed by emitted name.

    Output-tile dims emit as TILE_<out>[d]; reduction tiles as RTILE_<out>. An
    axis that is both an output dim and a reduction axis shares one ConstInt, so
    we group setters under a single name and drive them together.

    Ablation knobs: `ladder` overrides the per-param candidate sizes (a coarse
    ladder ablates JUMP CONTINUITY). `couple` collapses every tile knob into ONE
    global parameter whose setters drive all dims together (ablates LOCAL
    GEOMETRIC CONTROL -- no stage can be tiled independently of the others)."""
    rungs_src = ladder if ladder is not None else TILE_LADDER
    params: dict[str, TileParam] = {}

    def add(name: str, extent: int, current: int, setter) -> None:
        if name in params:
            params[name].setters.append(setter)
            # extent/current should agree across coupled slots; keep the first
        else:
            params[name] = TileParam(name, extent, current, [setter], rungs_src)

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

    plist = list(params.values())
    if couple and plist:
        # LOCAL-CONTROL ABLATION: one global knob drives every dim's setter. Its
        # extent is the largest axis (so the ladder spans all sizes); setting it
        # clamps each underlying dim to min(size, that dim's extent) so we never
        # tile a dim beyond its axis. No stage can be sized independently.
        all_setters = []
        for p in plist:
            ext_p = p.extent
            for st in p.setters:
                def clamped(v, _st=st, _e=ext_p):
                    _st(min(v, _e))
                all_setters.append(clamped)
        gext = max(p.extent for p in plist)
        cur = max(p.current for p in plist)
        return [TileParam("TILE_GLOBAL", gext, cur, all_setters, rungs_src)]
    return plist


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
def stage1_sweep(program: Program, ev: Evaluator, ladder: tuple | None = None,
                 couple: bool = False, neighbor: bool = False) -> tuple[Program, float]:
    """One full coordinate sweep over tile parameters. For each tunable tile
    dimension, evaluate candidate rungs and keep the best. Mutates a CLONE.

    `neighbor=False`: ALL-RUNGS -- try every rung on the ladder for every param.
    Thorough but ~ladder-size compiles per param.

    `neighbor=True`: ADAPTIVE. While the program is INFEASIBLE (over the memory
    budget -> penalty), fall back to all-rungs, because a single adjacent-rung step
    usually shows NO gradient there: the penalty is the binding stage's excess, and
    moving one param that doesn't feed that stage leaves the loss flat, so neighbor
    descent stalls at penalty and never reaches feasibility (verified). Big jumps
    are needed to find a feasible tiling. Once FEASIBLE, switch to adjacent-rung
    only (current +- one step): the runtime landscape is smooth enough that cheap
    neighbor refinement suffices, and over several sweeps (Stage 2 runs every N
    iterations) the params still traverse the ladder. This confines the expensive
    all-rungs cost to the brief infeasible phase at the start.

    `ladder`/`couple` carry the jump-continuity / local-control ablation knobs
    through to parameter_vector."""
    prog = R.clone_program(program)
    R._strip_clone_meta(prog)
    params = parameter_vector(prog, ladder=ladder, couple=couple)
    best = ev(prog)
    # adaptive: neighbor steps only once the program is feasible; otherwise the
    # penalty plateau traps single-rung coordinate moves, so use all-rungs.
    use_neighbor = neighbor and best < MEM_PENALTY_BASE
    for p in params:
        rungs = p.ladder()
        if use_neighbor and p.current in rungs:
            i = rungs.index(p.current)
            cands = [rungs[j] for j in (i - 1, i + 1) if 0 <= j < len(rungs)]
        else:
            cands = rungs                            # all-rungs (infeasible or off-ladder)
        best_size, best_loss = p.current, best
        for size in cands:
            if size == p.current:
                continue
            p.set(size)
            loss = ev(prog)
            if loss < best_loss:
                best_loss, best_size = loss, size
        p.set(best_size)
        best = best_loss
        # a param move can flip the program from infeasible to feasible; from then
        # on within this sweep, switch to cheap neighbor steps.
        if neighbor and not use_neighbor and best < MEM_PENALTY_BASE:
            use_neighbor = True
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


def stage2_step(program: Program, ev: Evaluator, K: int, rules, rng: random.Random,
                ladder: tuple | None = None, couple: bool = False, depth: int = 1,
                it: int = 0) -> tuple[Program, float, list]:
    """A depth-bounded greedy rewrite chain. For each of `depth` links: sample K
    applicable rewrites, NEIGHBOR-tune each to rank it cheaply, and commit the
    single best non-worsening one. Stop early if no rewrite is non-worsening.

    `depth=1` (default, fast): commit one rewrite per Stage-2 step. `depth=2`:
    explore a second rewrite ON TOP of the first (a 2-step lookahead) -- finds
    pairs where the first move only pays off after the second, at depth*K trials.

    Everything is neighbor-tuned (matching Stage 1), so the cost per step is
    depth*K cheap adjacent-rung sweeps rather than K full all-rungs sweeps. The
    committed program is exactly the best accepted trial, so the chain is
    monotonically non-worsening and 'if any trial compiled, the commit compiles'
    holds by construction (no separate combination/fallback needed)."""
    verbose_log = ev.cfg.log
    prog = program
    applied = []
    ev.ctx = f"it {it} | S2 base-tune"
    cur_loss = stage1_sweep(prog, ev, ladder, couple, neighbor=True)[1]
    for d in range(max(1, depth)):
        menu = enumerate_rewrites(prog, rules)
        if not menu:
            break
        sample = rng.sample(menu, min(K, len(menu)))
        if verbose_log:
            link = f" (chain link {d + 1}/{depth})" if depth > 1 else ""
            print(f"\n{'=' * 8} it {it} | STAGE 2{link}: ranking {len(sample)} "
                  f"rewrite(s), base loss {cur_loss:.4g} {'=' * 8}", flush=True)
        best = None                              # (loss, tuned_prog, label, delta)
        for rw in sample:
            cand = rw.resolve(prog)
            if cand is None:
                if verbose_log:
                    print(f"  -- rewrite {rw.label}: precondition no longer holds, "
                          f"skipped", flush=True)
                continue
            if verbose_log:
                print(f"  -- trying rewrite: {rw.label}", flush=True)
            ev.ctx = f"it {it} | S2 try {rw.label}"
            tuned, loss = stage1_sweep(cand, ev, ladder, couple, neighbor=True)
            if verbose_log:
                verdict = "ACCEPT" if loss <= cur_loss else "reject"
                print(f"     {rw.label} -> {loss:.4g} ({verdict}, base {cur_loss:.4g})",
                      flush=True)
            if loss <= cur_loss and (best is None or loss < best[0]):
                best = (loss, tuned, rw.label, cur_loss - loss)
        if best is None:
            if verbose_log:
                print(f"  no non-worsening rewrite found -> chain stops", flush=True)
            break                                # no non-worsening rewrite -> stop chain
        if verbose_log:
            print(f"  >> committing {best[2]}  (loss {cur_loss:.4g} -> {best[0]:.4g})",
                  flush=True)
        prog = best[1]                           # commit the best accepted (neighbor-tuned)
        applied.append((best[2], best[3]))
        cur_loss = best[0]
    return prog, cur_loss, applied


# ==========================================================================
# Persistence: analysis log (append-only JSONL) + resume snapshot (pickle).
# Two artifacts with different jobs --
#   * ANALYSIS (run_dir/log.jsonl, run_dir/kernels/best_it{N}.py): an append-only
#     per-iteration record plus the emitted cuTile SOURCE of every new best. Source
#     is durable -- re-benchmarkable later regardless of code changes -- so the
#     publication graphs never depend on un-pickling a live AST.
#   * RESUME (run_dir/snapshot.pkl): the full mutable search state (current + best
#     program, iteration, RNG state, evaluator cache, config) pickled every
#     iteration so an interrupted run continues deterministically. Pickle is used
#     for fidelity; it must be unpickled with compatible code.
# ==========================================================================
def loss_kind(loss: float) -> str:
    """Classify a score so analysis can separate the three regimes the search
    moves through: a real compiled runtime, an over-memory-budget penalty, or an
    infeasible program (illegal fusion / compile failure)."""
    if loss == math.inf:
        return "infeasible"
    if loss >= MEM_PENALTY_BASE:
        return "penalty"
    return "feasible"


class RunLogger:
    """Writes the analysis log + resume snapshots for one search run."""

    def __init__(self, run_dir: str, cfg, start_program: Program):
        self.dir = run_dir
        self.kernels_dir = os.path.join(run_dir, "kernels")
        os.makedirs(self.kernels_dir, exist_ok=True)
        self.log_path = os.path.join(run_dir, "log.jsonl")
        self.snap_path = os.path.join(run_dir, "snapshot.pkl")
        # config.json: the run settings + the starting program's source, so a
        # later reader knows exactly what produced this run.
        try:
            start_src = emit_module(start_program)
        except Exception:
            start_src = None
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump({
                "search": {k: _jsonable(v) for k, v in vars(cfg).items()},
                "start_stages": [[s.out, list(s.tile_shape)] for s in start_program.body],
                "start_source": start_src,
            }, f, indent=2)

    # ---- analysis ----
    def log_iter(self, *, iteration, loss, best_loss, applied, program, best_key):
        """Append one JSONL record describing this iteration."""
        try:
            flops, flops_exact = program_flops(program)
        except Exception:
            flops, flops_exact = None, False
        try:
            footprint = peak_tile_bytes(program)
        except Exception:
            footprint = None
        rec = {
            "iter": iteration,
            "loss": _num(loss),
            "loss_kind": loss_kind(loss),
            "best_loss": _num(best_loss),
            "best_loss_kind": loss_kind(best_loss),
            "applied": [a for a, _ in applied],
            "stages": [[s.out, list(s.tile_shape)] for s in program.body],
            "n_stages": len(program.body),
            "flops": flops,
            "flops_exact": flops_exact,
            "peak_tile_bytes": footprint,
            "best_key": best_key,
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def save_best_source(self, iteration: int, best_program: Program):
        """Write the emitted source of a new best program (durable artifact)."""
        try:
            src = emit_module(best_program)
        except Exception:
            return
        with open(os.path.join(self.kernels_dir, f"best_it{iteration}.py"), "w") as f:
            f.write(src)

    # ---- resume ----
    def snapshot(self, state: dict):
        """Pickle the full search state. Write to a temp file then rename, so a
        crash mid-write can't corrupt the existing snapshot (atomic on POSIX)."""
        tmp = self.snap_path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, self.snap_path)


def load_snapshot(run_dir: str) -> dict:
    """Load a resume snapshot from a run directory."""
    with open(os.path.join(run_dir, "snapshot.pkl"), "rb") as f:
        return pickle.load(f)


def _num(x):
    """JSON-safe number: inf -> a sentinel string (JSON has no inf)."""
    if x == math.inf:
        return "inf"
    return x


def _jsonable(v):
    """Best-effort conversion of a config value to something JSON can hold."""
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (tuple, list)):
        return list(v)
    return str(v)


# ==========================================================================
# Outer loop
# ==========================================================================
@dataclass
class SearchConfig:
    iters: int = 30
    N: int = 4                                   # run Stage 2 every N iterations.
                                                 # With neighbor Stage-1 sweeps, the
                                                 # N-1 param iterations between rewrites
                                                 # let tiles cross the whole ladder.
    K: int = 6                                   # rewrites sampled per Stage-2 step
    depth: int = 1                               # rewrites chained per Stage-2 step
                                                 # (1 = one move; 2 = a 2-step lookahead)
    rules: tuple = DEFAULT_RULES
    seed: int = 0
    verbose: bool = True
    run_dir: str | None = None                   # if set, write log + snapshots here
    resume: bool = False                          # resume from run_dir/snapshot.pkl
    # --- Design-for-Descent ablation knobs (parameter-space) ---
    tile_ladder: tuple | None = None              # override the tile-size ladder.
                                                  # None => default fine ladder. A coarse
                                                  # ladder (e.g. (16,)) makes every move a
                                                  # large jump -> ablates JUMP CONTINUITY.
    couple_tiles: bool = False                    # if True, ALL tile knobs collapse into
                                                  # one global size -> no independent
                                                  # per-stage control -> ablates LOCAL
                                                  # GEOMETRIC CONTROL.


def _better(loss: float, best: float) -> bool:
    """Is `loss` a better result to record as best than `best`? A feasible (real
    ms) result always beats a penalty, which always beats infeasible; within the
    same regime, lower is better. This fixes the bug where an early penalty score
    (1e9) was recorded as 'best' and a later real-millisecond compile -- numerically
    smaller, but the FIRST feasible point -- still needs to replace it. Because
    feasible losses are < MEM_PENALTY_BASE < inf, a plain `loss < best` already
    orders the regimes correctly; this helper makes that ordering explicit and is
    the single place best-tracking is decided."""
    return loss < best


def autotune(program: Program, ev: Evaluator | None = None,
             cfg: SearchConfig = SearchConfig()) -> Program:
    """Stochastic Rewrite Descent. Each iteration optionally takes a structural
    step (every N iters) then a single parameter sweep -- with the all-rungs
    greedy sweep that one sweep already reaches the coordinate-descent optimum,
    so N is kept small (a few extra param-only iterations between rewrites would
    just re-evaluate cached neighbors). Returns the best program seen, since the
    descent can wander under the <=-acceptance rule.

    If cfg.run_dir is set, an analysis log (log.jsonl), best-program sources
    (kernels/best_it{N}.py) and a resume snapshot (snapshot.pkl) are written each
    iteration. With cfg.resume, the run continues from an existing snapshot."""
    ev = ev or Evaluator()
    logger = RunLogger(cfg.run_dir, cfg, program) if cfg.run_dir else None

    if cfg.resume and cfg.run_dir:
        # restore full state: programs, iteration, RNG stream, evaluator cache.
        st = load_snapshot(cfg.run_dir)
        program = st["program"]
        best_prog, best_loss = st["best_prog"], st["best_loss"]
        start_it = st["iteration"] + 1
        rng = random.Random()
        rng.setstate(st["rng_state"])
        ev._cache = st.get("eval_cache", ev._cache)
        if cfg.verbose:
            print(f"[resume] from iter {st['iteration']} best={best_loss:.4g}")
    else:
        rng = random.Random(cfg.seed)
        ev.ctx = "init | S1"
        program, loss = stage1_sweep(program, ev, cfg.tile_ladder, cfg.couple_tiles,
                                       neighbor=True)  # initial tune (neighbor)
        best_prog, best_loss = R.clone_program(program), loss
        R._strip_clone_meta(best_prog)
        start_it = 1
        if cfg.verbose:
            print(f"[init] loss={loss:.4g}")
        if logger:
            if loss_kind(loss) == "feasible":
                logger.save_best_source(0, best_prog)
            logger.log_iter(iteration=0, loss=loss, best_loss=best_loss,
                            applied=[], program=program,
                            best_key=structural_key(best_prog))
            logger.snapshot(_search_state(program, best_prog, best_loss, 0, rng, ev, cfg))

    for it in range(start_it, cfg.iters + 1):
        applied = []
        is_s2 = it % cfg.N == 0
        if cfg.verbose:
            kind = "STAGE 2 (rewrite) + STAGE 1 (tune)" if is_s2 else "STAGE 1 (tune only)"
            print(f"\n{'#' * 60}\n# ITERATION {it}/{cfg.iters} -- {kind}\n{'#' * 60}",
                  flush=True)
        if is_s2:
            program, _, applied = stage2_step(program, ev, cfg.K, cfg.rules, rng,
                                              cfg.tile_ladder, cfg.couple_tiles,
                                              cfg.depth, it)
            if cfg.verbose and applied:
                print(f"[it {it}] applied {[a for a, _ in applied]}")
        if cfg.verbose:
            print(f"\n----- it {it} | STAGE 1: refine tiles "
                  f"(neighbor sweep) -----", flush=True)
        ev.ctx = f"it {it} | S1"
        program, loss = stage1_sweep(program, ev, cfg.tile_ladder, cfg.couple_tiles,
                                       neighbor=True)  # tune (neighbor)
        # best-tracking is over the ACCEPTED path only: the program the search
        # commits to each iteration. A program tuned inside a rejected Stage-2
        # trial is deliberately NOT recorded -- the search walked away from it.
        improved = _better(loss, best_loss)
        if improved:
            best_loss, best_prog = loss, R.clone_program(program)
            R._strip_clone_meta(best_prog)
        if cfg.verbose:
            print(f"[it {it}] loss={loss:.4g}  best={best_loss:.4g}")
        if logger:
            # save the durable best-source whenever best improves to a FEASIBLE
            # program (penalties/infeasible aren't worth re-benchmarking).
            if improved and loss_kind(best_loss) == "feasible":
                logger.save_best_source(it, best_prog)
            logger.log_iter(iteration=it, loss=loss, best_loss=best_loss,
                            applied=applied, program=program,
                            best_key=structural_key(best_prog))
            logger.snapshot(_search_state(program, best_prog, best_loss, it, rng, ev, cfg))

    if cfg.verbose:
        print(f"[done] best={best_loss:.4g}  compiles={ev._compiles} runs={ev._runs}")
    return best_prog


def _search_state(program, best_prog, best_loss, iteration, rng, ev, cfg) -> dict:
    """Bundle the full mutable search state for a resume snapshot. The evaluator
    cache (structural_key -> ms) is included because every entry cost a real GPU
    compile and is expensive to rebuild; the RNG state (not just the seed) is
    included so a resumed run reproduces the exact stochastic stream."""
    return {
        "program": program,
        "best_prog": best_prog,
        "best_loss": best_loss,
        "iteration": iteration,
        "rng_state": rng.getstate(),
        "eval_cache": ev._cache,
        "config": {k: _jsonable(v) for k, v in vars(cfg).items()},
    }