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
import tempfile
import os
import importlib.util
from dataclasses import dataclass, field

from kernel_ast import (
    Program, ParallelLoop, ReductionLoop, SpatialLoop, Load, Store, Compute,
    emit_module, structural_key,
)
import rewrites as R


# ==========================================================================
# Objective: compile -> run -> time. Failures (e.g. out-of-SMEM tilings) -> inf
# so the search treats infeasible points as maximally bad and moves on. This is
# what lets the descent start from the atomic grammar and *discover* that tiling
# is necessary.
# ==========================================================================
TILE_LADDER = (16, 32, 64, 128)        # ORIGINAL_DIM (full extent) is added per-axis


@dataclass
class EvalConfig:
    warmup: int = 3
    iters: int = 10                     # fewer than the standalone bench, for speed
    proxy: bool = False                 # dev-only: skip GPU, use a cheap fake score
    seed: int = 0


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
        try:
            src = emit_module(program)
        except Exception:
            return math.inf                 # invalid / unrenderable structure
        if self.cfg.proxy:
            return _proxy_score(program, src)
        self._compiles += 1
        try:
            return self._run_gpu(src)
        except Exception:
            return math.inf                 # compile or launch failure (e.g. SMEM)

    def _run_gpu(self, src: str) -> float:
        import cupy as cp
        mod = _import_source(src)
        meta = mod.KERNEL_META
        rng = cp.random.RandomState(self.cfg.seed)
        args = [rng.randn(*s).astype(cp.float32) for _, s in meta["inputs"]]
        fn = mod.fn
        for _ in range(self.cfg.warmup):
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
        times.sort()
        return times[len(times) // 2]       # median, to denoise


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
    from kernel_ast import _iter_loads
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
    """Mutates a CLONE of `program` in place via its parameter vector. Returns
    (tuned_program, loss)."""
    prog = R.clone_program(program)
    R._strip_clone_meta(prog)
    params = parameter_vector(prog)
    best = ev(prog)
    for p in params:
        rungs = p.ladder()
        i = rungs.index(p.current) if p.current in rungs else len(rungs) - 1
        # adjacent-rung neighbors
        cand = []
        if i > 0:
            cand.append(rungs[i - 1])
        if i < len(rungs) - 1:
            cand.append(rungs[i + 1])
        best_size, best_loss = p.current, best
        for size in cand:
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


def enumerate_rewrites(program: Program, rules) -> list:
    """Every (label, apply_callable) whose precondition currently holds. Each
    apply_callable takes a program and returns a new program."""
    out = []
    bodies = [(loop, loop.body) for loop in program.body]
    all_stmts = [(loop, s) for loop in program.body for s in _iter_stmts(loop.body)]

    if "hoist" in rules:
        for loop, s in all_stmts:
            if isinstance(s, Load) and R.can_hoist(program, s)[0]:
                out.append((f"hoist({s.source})", lambda p, s=s: R.hoist(p, s)))
    if "sink" in rules:
        for loop in program.body:
            loops_in = [x for x in _iter_stmts(loop.body)
                        if isinstance(x, (ReductionLoop, SpatialLoop))]
            for s in _iter_stmts(loop.body):
                if isinstance(s, Load):
                    for k in loops_in:
                        if R.can_sink(program, s, k)[0]:
                            out.append((f"sink({s.source})",
                                        lambda p, s=s, k=k: R.sink(p, s, k)))
    if "subtile_reduction" in rules:
        for loop, s in all_stmts:
            if isinstance(s, Compute):
                for ax in R._contraction_axes(s):
                    if R.can_subtile_reduction(program, s, ax)[0]:
                        out.append((f"subtile({s.op},{ax})",
                                    lambda p, s=s, ax=ax: R.subtile_reduction(p, s, ax)))
    if "unwrap_reduction" in rules:
        for loop, s in all_stmts:
            if isinstance(s, ReductionLoop) and R.can_unwrap_reduction(program, s)[0]:
                out.append((f"unwrap({s.axis})", lambda p, s=s: R.unwrap_reduction(p, s)))
    if "merge" in rules:
        for a, b in zip(program.body, program.body[1:]):
            if R.can_merge(program, a, b)[0]:
                out.append((f"merge({a.out},{b.out})", lambda p, a=a, b=b: R.merge(p, a, b)))
    if "merge_reductions" in rules:
        for loop in program.body:
            sibs = loop.body
            for a, b in zip(sibs, sibs[1:]):
                if isinstance(a, ReductionLoop) and isinstance(b, ReductionLoop) \
                        and R.can_merge_reductions(program, a, b)[0]:
                    out.append((f"merge_red({a.axis})",
                                lambda p, a=a, b=b: R.merge_reductions(p, a, b)))
    if "reorder" in rules:
        # adjacent stage pairs and adjacent statement pairs
        for a, b in zip(program.body, program.body[1:]):
            if R.can_reorder(program, a, b)[0]:
                out.append((f"reorder({a.out},{b.out})",
                            lambda p, a=a, b=b: R.reorder(p, a, b)))
        for loop in program.body:
            sibs = loop.body
            for a, b in zip(sibs, sibs[1:]):
                if R.can_reorder(program, a, b)[0]:
                    out.append(("reorder(stmt)", lambda p, a=a, b=b: R.reorder(p, a, b)))
    return out


def stage2_step(program: Program, ev: Evaluator, K: int, rules, rng: random.Random
                ) -> tuple[Program, float, list]:
    """Sample K applicable rewrites, trial each (apply + one Stage-1 sweep),
    accept all non-worsening, apply best-first re-checking preconditions."""
    base_loss = stage1_sweep(program, ev)[1]
    menu = enumerate_rewrites(program, rules)
    if not menu:
        return program, base_loss, []
    sample = rng.sample(menu, min(K, len(menu)))

    trials = []                                  # (delta, label, apply_fn)
    for label, apply_fn in sample:
        try:
            cand = apply_fn(program)
        except Exception:
            continue
        _, loss = stage1_sweep(cand, ev)         # judge a structural move WITH a re-tune
        if loss <= base_loss:                    # accept non-worsening (enabling moves)
            trials.append((base_loss - loss, label, apply_fn))

    trials.sort(key=lambda t: -t[0])             # best improvement first
    applied = []
    prog = program
    for delta, label, apply_fn in trials:
        try:
            cand = apply_fn(prog)                # re-check by re-applying on current prog
        except Exception:
            continue                             # precondition no longer holds
        prog = cand
        applied.append((label, delta))
    return prog, base_loss, applied


# ==========================================================================
# Outer loop
# ==========================================================================
@dataclass
class SearchConfig:
    iters: int = 40
    N: int = 4                                   # run Stage 2 every N iterations
    K: int = 6                                   # rewrites sampled per Stage-2 step
    rules: tuple = DEFAULT_RULES
    seed: int = 0
    verbose: bool = True


def autotune(program: Program, ev: Evaluator | None = None,
             cfg: SearchConfig = SearchConfig()) -> Program:
    """Stochastic Rewrite Descent. Returns the best program seen (the descent can
    wander, so we track and return the best, not the last)."""
    ev = ev or Evaluator()
    rng = random.Random(cfg.seed)

    program, loss = stage1_sweep(program, ev)
    best_prog, best_loss = program, loss
    if cfg.verbose:
        print(f"[init] loss={loss:.4g}")

    for it in range(1, cfg.iters + 1):
        program, loss = stage1_sweep(program, ev)
        if it % cfg.N == 0:
            program, _, applied = stage2_step(program, ev, cfg.K, cfg.rules, rng)
            program, loss = stage1_sweep(program, ev)   # fully tune after structure change
            if cfg.verbose and applied:
                print(f"[it {it}] applied {[a for a, _ in applied]} -> loss={loss:.4g}")
        if loss < best_loss:
            best_loss, best_prog = loss, R.clone_program(program)
            R._strip_clone_meta(best_prog)
        if cfg.verbose:
            print(f"[it {it}] loss={loss:.4g}  best={best_loss:.4g}")

    if cfg.verbose:
        print(f"[done] best={best_loss:.4g}  compiles={ev._compiles} runs={ev._runs}")
    return best_prog