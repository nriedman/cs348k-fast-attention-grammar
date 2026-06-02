"""Run Stochastic Rewrite Descent on a scaled dot-product attention kernel.

Mirror of run_mlp_search.py. The ATOMIC start has one ParallelLoop per compute
node -- the softmax is fully decomposed (rowmax, sub, exp, rowsum, div are each
their own stage), NOT fused. The search must discover the tiling and any fusion
on its own. Attention = two matmuls (QK^T and PV) around a row softmax over the
key axis:

    S  = Q @ K^T          [N,N]   reduces dd (head dim); KT is stored transposed
    Ss = S * (1/sqrt(d))  [N,N]   scale
    mx = rowmax(Ss)       [N,1]   softmax: max over key axis j
    sb = Ss - mx          [N,N]   broadcast subtract (mx reloaded width-1)
    e  = exp(sb)          [N,N]
    sm = rowsum(e)        [N,1]   sum over key axis j
    P  = e / sm           [N,N]   broadcast divide
    O  = P @ V            [N,d]   reduces j (key axis)

The mx/sm stages tile the key axis j to its FULL extent (a row reduction needs
the whole row); their output tensors are [N,1] and are reloaded with broadcast
in the sub/div stages.
"""

from __future__ import annotations

import argparse
import math

from grammar.kernel_ast import (
    Program, ParallelLoop, Load, Store, Compute, ReductionLoop, emit_module,
)
from grammar.optimization import autotune, Evaluator, EvalConfig, SearchConfig


def build_attention(N: int, d: int, tile: int | None = None,
                    subtile_k: int | None = None) -> Program:
    """Scaled dot-product attention as atomic stages (one Compute per stage).
    `tile` sets a uniform starting output tile (None => atomic full-extent, so
    the search must discover tiling). `subtile_k` pre-wraps the two matmuls in
    K-loops AND sinks their loads (see run_mlp_search for why sinking is needed:
    an un-sunk subtiled matmul materialises a full-contraction-width tile that
    blows up compilation). Axis names i (query), j (key), dd (head dim) are kept
    distinct from tensor names to avoid index/tensor name clashes."""
    scale = 1.0 / math.sqrt(d)

    def t2(a, b):
        return (a, b) if tile is None else (min(tile, a), min(tile, b))
    rt = t2(N, N)[0]                         # row tile on the query axis i

    # S = Q @ K^T  -> [N,N], reduces dd
    Q = Load("Q", ["i", "dd"]); KT = Load("KT", ["dd", "j"])
    S = Compute("matmul", [Q, KT])
    s_S = ParallelLoop("S", t2(N, N), ("i", "j"), [Q, KT, S, Store("S", S, ["i", "j"])])

    # Ss = S * (1/sqrt d)
    SL = Load("S", ["i", "j"]); Ss = Compute("mulc", [SL], const=scale)
    s_Ss = ParallelLoop("Ss", t2(N, N), ("i", "j"), [SL, Ss, Store("Ss", Ss, ["i", "j"])])

    # mx = rowmax(Ss) over j -> [N,1]   (key axis tiled to FULL extent for the reduction)
    SsL = Load("Ss", ["i", "j"]); mx = Compute("rowmax", [SsL], axis="j")
    s_mx = ParallelLoop("mx", (rt, N), ("i", "j"), [SsL, mx, Store("mx", mx, ["i", "j"])])

    # sb = Ss - mx   (broadcast: mx is [N,1], reloaded width-1)
    Ss2 = Load("Ss", ["i", "j"]); mxL = Load("mx", ["i", "j"]); sb = Compute("sub", [Ss2, mxL])
    s_sb = ParallelLoop("sb", t2(N, N), ("i", "j"), [Ss2, mxL, sb, Store("sb", sb, ["i", "j"])])

    # e = exp(sb)
    sbL = Load("sb", ["i", "j"]); e = Compute("exp", [sbL])
    s_e = ParallelLoop("e", t2(N, N), ("i", "j"), [sbL, e, Store("e", e, ["i", "j"])])

    # sm = rowsum(e) over j -> [N,1]
    eL = Load("e", ["i", "j"]); sm = Compute("rowsum", [eL], axis="j")
    s_sm = ParallelLoop("sm", (rt, N), ("i", "j"), [eL, sm, Store("sm", sm, ["i", "j"])])

    # P = e / sm   (broadcast)
    e2 = Load("e", ["i", "j"]); smL = Load("sm", ["i", "j"]); P = Compute("div", [e2, smL])
    s_P = ParallelLoop("P", t2(N, N), ("i", "j"), [e2, smL, P, Store("P", P, ["i", "j"])])

    # O = P @ V  -> [N,d], reduces j
    PL = Load("P", ["i", "j"]); V = Load("V", ["j", "dd"]); O = Compute("matmul", [PL, V])
    s_O = ParallelLoop("O", t2(N, d), ("i", "dd"), [PL, V, O, Store("O", O, ["i", "dd"])])

    tens = {"Q": (N, d), "KT": (d, N), "V": (N, d), "S": (N, N), "Ss": (N, N),
            "mx": (N, 1), "sb": (N, N), "e": (N, N), "sm": (N, 1), "P": (N, N),
            "O": (N, d)}
    prog = Program(tens, [s_S, s_Ss, s_mx, s_sb, s_e, s_sm, s_P, s_O])

    if subtile_k is not None:
        from grammar.rewrites import subtile_reduction
        import grammar.rewrites as R
        # S = Q@K^T reduces dd; O = P@V reduces j. Wrap each in a K-loop, then sink.
        prog = subtile_reduction(prog, S, "dd", tile=min(subtile_k, d))

        def bare_matmuls(p):
            return [s for loop in p.body for s in loop.body
                    if isinstance(s, Compute) and s.op == "matmul"]
        for mm in bare_matmuls(prog):
            # the remaining bare matmul is O = P@V, contracting the key axis j
            prog = subtile_reduction(prog, mm, "j", tile=min(subtile_k, N))
        # sink hoisted loads into their reduction loops
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
    return " -> ".join(f"{loop.out}{tuple(loop.tile_shape)}" for loop in program.body)


def verify(program: Program, N: int, d: int) -> bool:
    import cupy as cp
    from grammar.optimization import _import_source
    mod = _import_source(emit_module(program))
    rng = cp.random.RandomState(1)
    ins = {nm: rng.randn(*s).astype(cp.float32) for nm, s in mod.KERNEL_META["inputs"]}
    out = mod.fn(*[ins[nm] for nm, _ in mod.KERNEL_META["inputs"]])
    Q, KT, V = ins["Q"], ins["KT"], ins["V"]
    s = (Q @ KT) / math.sqrt(d)
    e = cp.exp(s - s.max(1, keepdims=True))
    P = e / e.sum(1, keepdims=True)
    ref = P @ V
    return bool(cp.allclose(out, ref, rtol=1e-2, atol=1e-2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512, help="sequence length")
    ap.add_argument("--d", type=int, default=64, help="head dimension")
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
    ap.add_argument("--run-dir", type=str, default=None, dest="run_dir",
                    help="write analysis log + resume snapshots to this directory")
    ap.add_argument("--resume", action="store_true",
                    help="resume the search from <run-dir>/snapshot.pkl")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    prog = build_attention(args.n, args.d, tile=args.tile, subtile_k=args.subtile_k)
    ev = Evaluator(EvalConfig(warmup=args.warmup, iters=args.reps,
                              proxy=args.proxy, seed=args.seed, log=args.log,
                              mem_budget=args.mem_budget_kb * 1024))

    print(f"task: scaled dot-product attention   Q[{args.n},{args.d}] "
          f"K[{args.n},{args.d}] V[{args.n},{args.d}]  fp32")
    print(f"start: {describe(prog)}")
    if args.tile is None and not args.proxy:
        print("note: atomic start -> the first eval JIT-compiles full-extent "
              "single-tile matmuls, which can be VERY slow to compile. If the "
              "first eval hangs, start feasible with e.g. --tile 64.", flush=True)
    print("timing start program (compiling)...", flush=True)
    init_loss = ev(prog)
    print(f"start loss: {init_loss:.4g}{' (proxy)' if args.proxy else ' ms'}\n", flush=True)

    best = autotune(prog, ev, SearchConfig(
        iters=args.iters, N=args.every, K=args.k, seed=args.seed, verbose=True,
        run_dir=args.run_dir, resume=args.resume))

    best_loss = ev(best)
    print(f"\nfinal: {describe(best)}")
    print(f"final loss: {best_loss:.4g}{' (proxy)' if args.proxy else ' ms'}")
    if init_loss not in (0, float('inf')) and best_loss != float('inf'):
        print(f"speedup vs start: {init_loss / best_loss:.2f}x")
    print(f"evals: compiles={ev._compiles} runs={ev._runs} cached={len(ev._cache)}")

    if not args.proxy:
        print(f"correctness (tuned == reference): {verify(best, args.n, args.d)}")

    print("\n--- tuned kernel source ---")
    print(emit_module(best))


if __name__ == "__main__":
    main()