"""Manually crafted fast attention kernel using the grammar.

Three-stage design vs. the six stages in the atomic start / best_it24:

  Stage 1:  Ss = (Q @ KT) * scale       -- matmul fused with scale, tiled over dd
  Stage 2:  P  = softmax(Ss)            -- full-row softmax fused in ONE kernel:
                                             rowmax -> sub -> exp -> rowsum -> div
  Stage 3:  O  = P @ V                  -- matmul tiled over j with ReductionLoop

Key savings vs. the search-discovered best_it24:
  - Eliminates the separate mx (rowmax) and sm (rowsum) stages.
  - Eliminates the intermediate e[N,N] tensor (1 write + 2 reads = ~3 MB for N=512).
  - Reduces from 6 kernel launches to 3.
  - The full softmax is computed in registers with a single pass over Ss.

Tile choices (N=512, d=64) — small/conservative starting point:
  Ss : (16, 16), RTILE_dd = 16  -> 1024 blocks, 4 reduction steps over dd
  P  : (16, N),  no ReductionLoop -> 32 blocks, each loads one full Ss row (32 KB)
                 (full-row load is required for correctness of rowmax/rowsum)
  O  : (16, 16), RTILE_j  = 16  -> 128 blocks, 32 reduction steps over j
"""

import math

from grammar.kernel_ast import (
    Program, ParallelLoop, Load, Store, Compute, ReductionLoop, emit_module,
)


def build_fast_attention(N: int = 512, d: int = 64) -> Program:
    """Return the hand-crafted three-stage attention Program."""
    scale = 1.0 / math.sqrt(d)

    # ------------------------------------------------------------------
    # Stage 1: Ss = (Q @ KT) * scale
    #
    # Tile (16, 16) over output axes (i, j).  A ReductionLoop over dd
    # (tile = 16) tiles the head-dimension contraction.  The scale multiply
    # sits OUTSIDE the loop so it touches the fully-accumulated (16,16) tile
    # once — identical to the fused Ss_kernel in best_it24.
    # ------------------------------------------------------------------
    Q1  = Load("Q",  ["i", "dd"])
    KT1 = Load("KT", ["dd", "j"])
    S1  = Compute("matmul", [Q1, KT1])
    rl_dd = ReductionLoop("dd", tile=16, body=[Q1, KT1, S1], partial=S1)
    Ss1   = Compute("mulc", [rl_dd], const=scale)
    s_Ss  = ParallelLoop(
        "Ss", (16, 16), ("i", "j"),
        [rl_dd, Ss1, Store("Ss", Ss1, ["i", "j"])],
    )

    # ------------------------------------------------------------------
    # Stage 2: P = softmax(Ss) — all ops fused in one kernel.
    #
    # Tile (16, N): j is NOT tiled (tile == global extent N), so each
    # block loads one full row of Ss into registers, then computes:
    #
    #   mx  = rowmax(Ss)          -> (16, 1)
    #   sb  = Ss - mx             -> (16, N)   broadcast
    #   e   = exp(sb)             -> (16, N)
    #   sm  = rowsum(e)           -> (16, 1)
    #   P   = e / sm              -> (16, N)   broadcast
    #
    # This eliminates the mx[N,1], e[N,N], sm[N,1] intermediates and the
    # three extra kernel launches that best_it24 uses for those stages.
    #
    # Peak tile: Ss load (16, N) = 32 KB — safely within the 64 KB budget.
    # ------------------------------------------------------------------
    SsL = Load("Ss", ["i", "j"])                       # (16, N) full row
    mx  = Compute("rowmax", [SsL], axis="j")           # (16,  1)
    sb  = Compute("sub",    [SsL, mx])                 # (16,  N)  broadcasts mx
    e   = Compute("exp",    [sb])                      # (16,  N)
    sm  = Compute("rowsum", [e],   axis="j")           # (16,  1)
    P2  = Compute("div",    [e, sm])                   # (16,  N)  broadcasts sm
    s_P = ParallelLoop(
        "P", (16, N), ("i", "j"),
        [SsL, mx, sb, e, sm, P2, Store("P", P2, ["i", "j"])],
    )

    # ------------------------------------------------------------------
    # Stage 3: O = P @ V
    #
    # Tile (16, 16) over (i, dd).  A ReductionLoop over j (tile = 16)
    # tiles the key-axis contraction.  Both P and V are loaded inside
    # the loop: P(16,16) and V(16,16) per step, 32 steps total.
    #
    #   Grid      : (512/16, 64/16) = (32, 4) = 128 blocks
    #   Peak tile : max(P=1 KB, V=1 KB, O_acc=1 KB) — well within budget
    # ------------------------------------------------------------------
    P3  = Load("P", ["i", "j"])
    V3  = Load("V", ["j", "dd"])
    O3  = Compute("matmul", [P3, V3])
    rl_j = ReductionLoop("j", tile=16, body=[P3, V3, O3], partial=O3)
    s_O  = ParallelLoop(
        "O", (16, 16), ("i", "dd"),
        [rl_j, Store("O", rl_j, ["i", "dd"])],
    )

    tens = {
        "Q":  (N, d), "KT": (d, N), "V": (N, d),
        "Ss": (N, N), "P":  (N, N), "O": (N, d),
    }
    return Program(tens, [s_Ss, s_P, s_O])


# ---------------------------------------------------------------------------
# CLI: --emit / --verify / --bench
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Emit and optionally benchmark the hand-crafted fast attention kernel."
    )
    ap.add_argument("--n",      type=int, default=512, help="sequence length N")
    ap.add_argument("--d",      type=int, default=64,  help="head dimension d")
    ap.add_argument("--emit",   action="store_true",   help="print the emitted kernel source")
    ap.add_argument("--verify", action="store_true",   help="check numerical correctness vs. PyTorch reference")
    ap.add_argument("--bench",  action="store_true",   help="time the kernel on GPU (requires CUDA)")
    ap.add_argument("--warmup", type=int, default=5,   help="warmup iterations for benchmarking")
    ap.add_argument("--reps",   type=int, default=200, help="timed iterations for benchmarking")
    args = ap.parse_args()

    prog = build_fast_attention(args.n, args.d)
    src  = emit_module(prog)

    if args.emit or (not args.verify and not args.bench):
        print(src)

    if args.verify or args.bench:
        import cupy as cp
        from grammar.optimization import _import_source

        mod     = _import_source(src)
        inp_map = {nm: cp.random.randn(*s).astype(cp.float32)
                   for nm, s in mod.KERNEL_META["inputs"]}
        inp_list = [inp_map[nm] for nm, _ in mod.KERNEL_META["inputs"]]

        if args.verify:
            out = mod.fn(*inp_list)
            Q, KT, V = inp_map["Q"], inp_map["KT"], inp_map["V"]
            S_ref = (Q @ KT) / math.sqrt(args.d)
            e_ref = cp.exp(S_ref - S_ref.max(1, keepdims=True))
            ref   = (e_ref / e_ref.sum(1, keepdims=True)) @ V
            ok    = bool(cp.allclose(out, ref, rtol=1e-2, atol=1e-2))
            print(f"correctness: {'PASS' if ok else 'FAIL'}")

        if args.bench:
            import time

            for _ in range(args.warmup):
                mod.fn(*inp_list)
            cp.cuda.runtime.deviceSynchronize()

            t0 = time.perf_counter()
            for _ in range(args.reps):
                mod.fn(*inp_list)
            cp.cuda.runtime.deviceSynchronize()
            ms = (time.perf_counter() - t0) * 1e3 / args.reps

            flops  = mod.KERNEL_META["flops"]
            tflops = flops / (ms * 1e-3) / 1e12
            print(f"mean: {ms:.4f} ms  |  {tflops:.3f} TFLOP/s")
            print(f"(best_it24 reference: 0.1063 ms on NVIDIA L4)")
