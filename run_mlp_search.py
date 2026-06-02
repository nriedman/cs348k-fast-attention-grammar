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
    Program, ParallelLoop, Load, Store, Compute, emit_module,
)
from grammar.optimization import autotune, Evaluator, EvalConfig, SearchConfig


def build_mlp_ln(B: int, Dm: int, H: int, eps: float = 1e-5,
                 tile: int | None = None) -> Program:
    """MLP + residual + LayerNorm as separate stages. `tile` sets a uniform
    starting output-tile size (None => atomic: full-extent tiles, so the search
    must discover tiling). Axis names: b (batch), dm (feature), hid (hidden) --
    deliberately distinct from tensor names (h, a, y) to avoid an axis/tensor
    name clash in the emitted index variables."""
    def t2(d0, d1):
        return (d0, d1) if tile is None else (min(tile, d0), min(tile, d1))

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
    return Program(tens, [s_h, s_a, s_y, s_r, s_ln])


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
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--every", type=int, default=2, help="Stage-2 every N iters")
    ap.add_argument("--k", type=int, default=8, help="rewrites sampled per Stage-2")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--proxy", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    prog = build_mlp_ln(args.b, args.dm, args.h, tile=args.tile)
    ev = Evaluator(EvalConfig(warmup=args.warmup, iters=args.reps,
                              proxy=args.proxy, seed=args.seed))

    print(f"task: MLP + residual + LayerNorm   X[{args.b},{args.dm}] "
          f"W1[{args.dm},{args.h}] W2[{args.h},{args.dm}]  fp32")
    print(f"start: {describe(prog)}")
    init_loss = ev(prog)
    print(f"start loss: {init_loss:.4g}{' (proxy)' if args.proxy else ' ms'}\n")

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