import math

from grammar.kernel_ast import (
    Program, ParallelLoop, Load, Store, Compute, ReductionLoop, emit_module,
)

"""
Q:  [N, d]
Kt: [d, N]
V:  [N, d]
"""

def build_fast_attention(N: int = 512, d: int = 64) -> Program:
    s = 1.0 / math.sqrt(d)

    # Q @ K_t
    q_ = Load("Q", ("i", "d"))
    kt_ = Load("Kt", ("d", "j"))
    matmul_qkt = Compute("matmul", [q_, kt_])

    # elementwise divide by scale
    scale = Compute("mulc", [matmul_qkt], const=s)
    store_scale = Store("S", scale, ["i", "j"])

    s1 = ParallelLoop("S", (64, 64), ("i", "j"), [matmul_qkt, scale, store_scale])

    ld_s = Load("S", ("i", "j"))
    cmp_mx = Compute("rowmax", [ld_s], axis="j")
    mx_red = ReductionLoop("j", 64, [ld_s, cmp_mx])
    s2 = ParallelLoop("mx", (16, 1), ("i", "l"), [mx_red, Store("mx", mx_red, ("i", "l"))])

    ld_mx = Load("mx", ("i", "j"))
    ld_S = Load("S", ("i", "j"))
    e_ = Compute("exp", [Compute("sub", [ld_mx, ld_S])])

    s3 = ParallelLoop("e", (64, 64), ("i", "j"), [ld_mx, ld_S, e_, Store("e", e_, ("i", "j"))])

    ld_e = Load("e", ("i", "j"))
    sm_ = Compute("rowsum", [ld_e], axis="j")

    s4 = ParallelLoop("sm", (16, 1), ("i", "l"), [ld_e, sm_, Store("sm", sm_, ("i", "l"))])

    ld_exp = Load("e", ("i", "j"))
    ld_sm = Load("sm", ("i", "j"))
    ld_v = Load("V", ("j", "d"))

    cmp_div = Compute("div", [ld_exp, ld_sm])
    cmp_matmul = Compute("matmul", [cmp_div, ld_v])

    pv_red = ReductionLoop("j", 32, [ld_exp, ld_sm, ld_v, cmp_div, cmp_matmul], cmp_matmul)

    s5 = ParallelLoop("O", (64, 32), ("i", "d"), [pv_red, Store("O", pv_red, ("i", "d"))])

    return Program(
        {
            "Q": (N, d), "Kt": (d, N), "V": (N, d),
            "S": (N, N), "mx": (N, 1), "e": (N, N), "sm": (N, 1), "O": ("N", "d")
        },
        [s1, s2, s3, s4, s5]
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Emit the hand-crafted fast attention kernel."
    )
    ap.add_argument("--n",      type=int, default=512, help="sequence length N")
    ap.add_argument("--d",      type=int, default=64,  help="head dimension d")
    ap.add_argument("--emit",   action="store_true",   help="print the emitted kernel source")
    ap.add_argument("--verify", action="store_true",   help="check numerical correctness vs. PyTorch reference")
    args = ap.parse_args()

    prog = build_fast_attention(args.n, args.d)
    src  = emit_module(prog)

    if args.emit:
        print(src)

    if args.verify:
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
