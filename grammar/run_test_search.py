"""
run_elementwise_search.py -- run the SRD autotuner on a small real-GPU kernel.

Task:  d = a + b      (elementwise add)
       e = d * c      (elementwise multiply)

Two stages, so the interesting structural move the search can discover is Merge:
fuse the add and the multiply into one kernel, eliminating the intermediate `d`
round-trip through global memory. On top of that, tile sizes are the continuous
(here: discrete-ladder) knobs Stage 1 tunes. Elementwise + small => compiles and
runs fast, good for a first real-GPU loop.

Run on the L4:
    python run_elementwise_search.py                 # real GPU timing
    python run_elementwise_search.py --proxy         # off-GPU plumbing check
    python run_elementwise_search.py --n 2048 --tile 64 --iters 30
"""

from __future__ import annotations

import argparse

from kernel_ast import (
    Program, ParallelLoop, Load, Store, Compute, emit_module,
)
from optimization import autotune, Evaluator, EvalConfig, SearchConfig


def build_add_mul(n: int, m: int, tile: int) -> Program:
    """d = a + b ; e = d * c. Starts at a uniform (tile x tile) output tile on
    both stages -- a feasible, un-fused starting point for the search to tune
    and fuse from."""
    a, b = Load("a", ["n", "m"]), Load("b", ["n", "m"])
    add = Compute("add", [a, b])
    s1 = ParallelLoop("d", (tile, tile), ("n", "m"),
                      [a, b, add, Store("d", add, ["n", "m"])])
    d, c = Load("d", ["n", "m"]), Load("c", ["n", "m"])
    mul = Compute("mul", [d, c])
    s2 = ParallelLoop("e", (tile, tile), ("n", "m"),
                      [d, c, mul, Store("e", mul, ["n", "m"])])
    return Program({"a": (n, m), "b": (n, m), "c": (n, m),
                    "d": (n, m), "e": (n, m)}, [s1, s2])


def describe(program: Program) -> str:
    parts = []
    for loop in program.body:
        kinds = "+".join(type(s).__name__ for s in loop.body)
        parts.append(f"{loop.out}{tuple(loop.tile_shape)}[{kinds}]")
    return "  ->  ".join(parts)


def verify(program: Program, n: int, m: int) -> bool:
    """Run the tuned kernel on the GPU and check it still computes (a+b)*c."""
    import cupy as cp
    import numpy as np
    from optimization import _import_source
    mod = _import_source(emit_module(program))
    rng = cp.random.RandomState(1)
    ins = {nm: rng.randn(*s).astype(cp.float32) for nm, s in mod.KERNEL_META["inputs"]}
    out = mod.fn(*[ins[nm] for nm, _ in mod.KERNEL_META["inputs"]])
    ref = (ins["a"] + ins["b"]) * ins["c"]
    return bool(cp.allclose(out, ref, rtol=1e-3, atol=1e-4))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--m", type=int, default=1024)
    ap.add_argument("--tile", type=int, default=64, help="starting tile size")
    ap.add_argument("--iters", type=int, default=24)
    ap.add_argument("--every", type=int, default=4, help="Stage-2 every N iters")
    ap.add_argument("--k", type=int, default=6, help="rewrites sampled per Stage-2")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--reps", type=int, default=8, help="timed iters per eval")
    ap.add_argument("--proxy", action="store_true", help="off-GPU plumbing check")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    prog = build_add_mul(args.n, args.m, args.tile)
    ev = Evaluator(EvalConfig(warmup=args.warmup, iters=args.reps,
                              proxy=args.proxy, seed=args.seed))

    print(f"task: (a+b)*c  on {args.n}x{args.m}  fp32")
    print(f"start: {describe(prog)}")
    init_loss = ev(prog)
    print(f"start loss: {init_loss:.4g}{' (proxy)' if args.proxy else ' ms'}\n")

    best = autotune(prog, ev, SearchConfig(
        iters=args.iters, N=args.every, K=args.k, seed=args.seed, verbose=True))

    best_loss = ev(best)
    print(f"\nfinal: {describe(best)}")
    print(f"final loss: {best_loss:.4g}{' (proxy)' if args.proxy else ' ms'}")
    if init_loss not in (0, float('inf')) and best_loss not in (float('inf'),):
        print(f"speedup vs start: {init_loss / best_loss:.2f}x")
    print(f"evals: compiles={ev._compiles} runs={ev._runs} cached={len(ev._cache)}")

    if not args.proxy:
        ok = verify(best, args.n, args.m)
        print(f"correctness (tuned kernel == (a+b)*c): {ok}")

    print("\n--- tuned kernel source ---")
    print(emit_module(best))


if __name__ == "__main__":
    main()