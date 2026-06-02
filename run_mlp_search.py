"""
run_mlp_search.py -- run the SRD autotuner on the final benchmark: an MLP block
with a residual connection and LayerNorm.

    h   = X @ W1            [B, H]   matmul, reduces feature dim Dm
    a   = relu(h)           [B, H]   activation
    y   = a @ W2            [B, Dm]  matmul, reduces hidden dim H
    r   = y + X             [B, Dm]  residual add
    out = LayerNorm(r)      [B, Dm]  row-reduction over the feature dim

This is the smallest pipeline that gives the whole rewrite family something to
do, with a genuinely contested optimum:

  * subtile_reduction -> both matmuls become tiled ct.mma reductions (tensor
    cores; the ~7x structural win measured earlier),
  * merge -> fuse relu into matmul-1's epilogue, fuse the residual into
    matmul-2's epilogue,
  * merge_reductions + dedup_loads -> LayerNorm's mean (sum) and variance
    (sum-of-squares) are two independent same-axis reductions over r sharing the
    same load -- the canonical reduction-merge case,
  * reorder -> shuffle stages adjacent so the above merges can fire.

The program is built ATOMICALLY (full-extent tiles, every op its own stage), so
the search must discover tiling, tensor-core reductions, and fusion from scratch.

Run on the L4:
    python run_mlp_search.py                       # real GPU timing
    python run_mlp_search.py --proxy               # off-GPU plumbing check
    python run_mlp_search.py --b 512 --dm 1024 --h 4096 --iters 30
"""

from __future__ import annotations

import argparse

from grammar.kernel_ast import (
    Program, ParallelLoop, Load, Store, Compute, ReductionLoop, emit_module,
)
from grammar.optimization import autotune, Evaluator, EvalConfig, SearchConfig


def build_mlp_ln(B: int, Dm: int, H: int, eps: float = 1e-5,
                 tile: int | None = None, subtile_k: int | None = None) -> Program:
    """MLP + residual + LayerNorm as separate stages. `tile` sets a uniform
    starting output-tile size (None => atomic: full-extent tiles, so the search
    must discover tiling). `subtile_k` (if set) pre-wraps each matmul in a
    reduction loop tiling its contraction by that amount -- this makes the
    matmuls emit as ct.mma K-loops from the start, avoiding the pathologically
    slow cuTile compile of a full-K single-tile ct.matmul. The search can still
    retune the K-tile and apply every other rewrite. Axis names: b (batch),
    dm (feature), hid (hidden) -- distinct from tensor names to avoid clashes."""
    def t2(d0, d1):
        return (d0, d1) if tile is None else (min(tile, d0), min(tile, d1))

    from grammar.rewrites import subtile_reduction
    import grammar.rewrites as R

    # h = X @ W1  -> [B, H], reduces dm
    X = Load("X", ["b", "dm"]); W1 = Load("W1", ["dm", "hid"])
    h = Compute("matmul", [X, W1])
    s_h = ParallelLoop("h", t2(B, H), ("b", "hid"), [X, W1, h, Store("h", h, ["b", "hid"])])

    # a = relu(h) -> [B, H]
    hL = Load("h", ["b", "hid"]); a = Compute("relu", [hL])
    s_a = ParallelLoop("a", t2(B, H), ("b", "hid"), [hL, a, Store("a", a, ["b", "hid"])])

    # y = a @ W2  -> [B, Dm], reduces hid
    aL = Load("a", ["b", "hid"]); W2 = Load("W2", ["hid", "dm"])
    y = Compute("matmul", [aL, W2])
    s_y = ParallelLoop("y", t2(B, Dm), ("b", "dm"), [aL, W2, y, Store("y", y, ["b", "dm"])])

    # r = y + X  -> [B, Dm]  (residual)
    yL = Load("y", ["b", "dm"]); XL = Load("X", ["b", "dm"]); r = Compute("add", [yL, XL])
    s_r = ParallelLoop("r", t2(B, Dm), ("b", "dm"), [yL, XL, r, Store("r", r, ["b", "dm"])])

    # out = LayerNorm(r) over dm:
    #   s1 = sum(r); s2 = sum(r*r); mean = s1/Dm; var = s2/Dm - mean^2;
    #   out = (r - mean) / sqrt(var + eps)
    rL = Load("r", ["b", "dm"]); s1 = Compute("rowsum", [rL], axis="dm")
    r2a, r2b = Load("r", ["b", "dm"]), Load("r", ["b", "dm"])
    rr = Compute("mul", [r2a, r2b]); s2 = Compute("rowsum", [rr], axis="dm")
    mean = Compute("mulc", [s1], const=1.0 / Dm)
    meansq = Compute("mulc", [s2], const=1.0 / Dm)
    mean2 = Compute("mul", [mean, mean])
    var = Compute("sub", [meansq, mean2])
    vareps = Compute("addc", [var], const=eps)
    std = Compute("sqrt", [vareps])
    r3 = Load("r", ["b", "dm"])
    centered = Compute("sub", [r3, mean])
    out = Compute("div", [centered, std])
    s_ln = ParallelLoop("out", (B, Dm), ("b", "dm"),
                        [rL, s1, r2a, r2b, rr, s2, mean, meansq, mean2, var,
                         vareps, std, r3, centered, out, Store("out", out, ["b", "dm"])])

    tens = {"X": (B, Dm), "W1": (Dm, H), "W2": (H, Dm),
            "h": (B, H), "a": (B, H), "y": (B, Dm), "r": (B, Dm), "out": (B, Dm)}
    prog = Program(tens, [s_h, s_a, s_y, s_r, s_ln])

    if subtile_k is not None:
        # wrap each matmul's contraction in a reduction loop so it emits as a
        # ct.mma K-loop (not a full-K single ct.matmul), THEN sink the operand
        # loads into the loop. Sinking is essential: without it the loads stay
        # hoisted and materialize a full-contraction-width tile (e.g. 64x1024),
        # which exhausts memory during cuTile/ptxas compilation. Sunk, each
        # iteration loads only a TS x TS slice.
        prog = subtile_reduction(prog, h, "dm", tile=min(subtile_k, Dm))
        def bare_matmuls(p):
            return [s for loop in p.body for s in loop.body
                    if isinstance(s, Compute) and s.op == "matmul"]
        for mm in bare_matmuls(prog):
            prog = subtile_reduction(prog, mm, "hid", tile=min(subtile_k, H))
        # sink every hoisted load into its sibling reduction loop
        changed = True
        while changed:
            changed = False
            for loop in prog.body:
                redloops = [s for s in loop.body if isinstance(s, ReductionLoop)]
                for s in list(loop.body):
                    if isinstance(s, Load):
                        for rl in redloops:
                            if R.can_sink(prog, s, rl)[0]:
                                prog = R.sink(prog, s, rl); changed = True; break
                        if changed:
                            break
                if changed:
                    break
    return prog


def describe(program: Program) -> str:
    parts = []
    for loop in program.body:
        kinds = "+".join(type(s).__name__[:4] for s in loop.body)
        parts.append(f"{loop.out}{tuple(loop.tile_shape)}")
    return " -> ".join(parts)


def verify(program: Program, B: int, Dm: int, H: int, eps: float = 1e-5) -> bool:
    import cupy as cp
    from grammar.optimization import _import_source
    mod = _import_source(emit_module(program))
    rng = cp.random.RandomState(1)
    ins = {nm: rng.randn(*s).astype(cp.float32) for nm, s in mod.KERNEL_META["inputs"]}
    out = mod.fn(*[ins[nm] for nm, _ in mod.KERNEL_META["inputs"]])
    X, W1, W2 = ins["X"], ins["W1"], ins["W2"]
    h = X @ W1; a = cp.maximum(h, 0); y = a @ W2; r = y + X
    mean = r.mean(1, keepdims=True); var = r.var(1, keepdims=True)
    ref = (r - mean) / cp.sqrt(var + eps)
    return bool(cp.allclose(out, ref, rtol=1e-2, atol=1e-2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b", type=int, default=256, help="batch rows")
    ap.add_argument("--dm", type=int, default=512, help="feature/model dim")
    ap.add_argument("--h", type=int, default=1024, help="hidden dim")
    ap.add_argument("--tile", type=int, default=None,
                    help="uniform starting tile (default: atomic full-extent)")
    ap.add_argument("--subtile-k", type=int, default=None, dest="subtile_k",
                    help="pre-wrap matmuls in a K-loop of this tile (avoids the "
                         "slow full-K compile; recommended for the first run)")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--every", type=int, default=2, help="Stage-2 every N iters")
    ap.add_argument("--k", type=int, default=8, help="rewrites sampled per Stage-2")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--proxy", action="store_true")
    ap.add_argument("--log", action="store_true", help="per-eval phase timing")
    ap.add_argument("--mem-budget-kb", type=int, default=64, dest="mem_budget_kb",
                    help="largest single per-block tile budget (KB); programs whose "
                         "biggest tile exceeds it are penalized, not compiled. The L4 "
                         "SMEM limit is ~99 KB; 64 leaves headroom. 0 disables.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    prog = build_mlp_ln(args.b, args.dm, args.h, tile=args.tile, subtile_k=args.subtile_k)
    ev = Evaluator(EvalConfig(warmup=args.warmup, iters=args.reps,
                              proxy=args.proxy, seed=args.seed, log=args.log,
                              mem_budget=args.mem_budget_kb * 1024))

    print(f"task: MLP + residual + LayerNorm   X[{args.b},{args.dm}] "
          f"W1[{args.dm},{args.h}] W2[{args.h},{args.dm}]  fp32")
    print(f"start: {describe(prog)}")
    if args.tile is None and not args.proxy:
        print("note: atomic start -> the first eval JIT-compiles full-extent "
              "single-tile matmuls, which can be VERY slow to compile. If the "
              "first eval hangs, start feasible with e.g. --tile 64.", flush=True)
    print("timing start program (compiling)...", flush=True)
    init_loss = ev(prog)
    print(f"start loss: {init_loss:.4g}{' (proxy)' if args.proxy else ' ms'}\n", flush=True)

    best = autotune(prog, ev, SearchConfig(
        iters=args.iters, N=args.every, K=args.k, seed=args.seed, verbose=True))

    best_loss = ev(best)
    print(f"\nfinal: {describe(best)}")
    print(f"final loss: {best_loss:.4g}{' (proxy)' if args.proxy else ' ms'}")
    if init_loss not in (0, float('inf')) and best_loss != float('inf'):
        print(f"speedup vs start: {init_loss / best_loss:.2f}x")
    print(f"evals: compiles={ev._compiles} runs={ev._runs} cached={len(ev._cache)}")

    if not args.proxy:
        print(f"correctness (tuned == reference): {verify(best, args.b, args.dm, args.h)}")

    print("\n--- tuned kernel source ---")
    print(emit_module(best))


if __name__ == "__main__":
    main()