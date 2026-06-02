import cuda.tile as ct
import cupy as cp
import numpy as np

ConstInt = ct.Constant[int]

# --- tunable tile sizes (vary these to autotune) ---
TILE_S = (64, 64)
RTILE_S = 64   # reduction tile along dd
TILE_Ss = (64, 64)
TILE_mx = (16, 512)
TILE_sb = (64, 64)
TILE_e = (64, 64)
TILE_sm = (32, 512)
TILE_P = (16, 32)
TILE_O = (16, 64)
RTILE_O = 16   # reduction tile along j

@ct.kernel
def S_kernel(KT, Q, S, DD: ConstInt, TS_i: ConstInt, TS_j: ConstInt, TS_dd: ConstInt):
    i = ct.bid(0)
    j = ct.bid(1)
    acc1 = ct.zeros((TS_i, TS_j), dtype=S.dtype)
    for dd in range(ct.cdiv(DD, TS_dd)):
        t2 = ct.load(KT, (0, j), (DD, TS_j))
        t3 = ct.load(Q, (i, 0), (TS_i, DD))
        acc1 = ct.mma(t3, t2, acc1)
    ct.store(S, (i, j), acc1)

@ct.kernel
def Ss_kernel(S, Ss, TS_i: ConstInt, TS_j: ConstInt):
    i = ct.bid(0)
    j = ct.bid(1)
    t1 = ct.load(S, (i, j), (TS_i, TS_j))
    t2 = (t1 * 0.125)
    ct.store(Ss, (i, j), t2)

@ct.kernel
def mx_kernel(Ss, mx, J: ConstInt, TS_i: ConstInt):
    i = ct.bid(0)
    j = ct.bid(1)
    t1 = ct.load(Ss, (i, 0), (TS_i, J))
    t2 = ct.max(t1, axis=1, keepdims=True)
    ct.store(mx, (i, 0), t2)

@ct.kernel
def sb_kernel(Ss, mx, sb, TS_i: ConstInt, TS_j: ConstInt):
    i = ct.bid(0)
    j = ct.bid(1)
    t1 = ct.load(Ss, (i, j), (TS_i, TS_j))
    t2 = ct.load(mx, (i, 0), (TS_i, 1))
    t3 = (t1 - t2)
    ct.store(sb, (i, j), t3)

@ct.kernel
def e_kernel(sb, e, TS_i: ConstInt, TS_j: ConstInt):
    i = ct.bid(0)
    j = ct.bid(1)
    t1 = ct.load(sb, (i, j), (TS_i, TS_j))
    t2 = ct.exp(t1)
    ct.store(e, (i, j), t2)

@ct.kernel
def sm_kernel(e, sm, J: ConstInt, TS_i: ConstInt):
    i = ct.bid(0)
    j = ct.bid(1)
    t1 = ct.load(e, (i, 0), (TS_i, J))
    t2 = ct.sum(t1, axis=1, keepdims=True)
    ct.store(sm, (i, 0), t2)

@ct.kernel
def P_kernel(e, sm, P, TS_i: ConstInt, TS_j: ConstInt):
    i = ct.bid(0)
    j = ct.bid(1)
    t1 = ct.load(e, (i, j), (TS_i, TS_j))
    t2 = ct.load(sm, (i, 0), (TS_i, 1))
    t3 = (t1 / t2)
    ct.store(P, (i, j), t3)

@ct.kernel
def O_kernel(V, P, O, DD: ConstInt, J: ConstInt, TS_i: ConstInt, TS_dd: ConstInt, TS_j: ConstInt):
    i = ct.bid(0)
    dd = ct.bid(1)
    acc1 = ct.zeros((TS_i, TS_dd), dtype=O.dtype)
    for j in range(ct.cdiv(J, TS_j)):
        t2 = ct.load(V, (j, 0), (TS_j, DD))
        t3 = ct.load(P, (i, j), (TS_i, TS_j))
        acc1 = ct.mma(t3, t2, acc1)
    ct.store(O, (i, 0), acc1)

def fn(KT, Q, V):
    dtype = KT.dtype
    stream = cp.cuda.get_current_stream()
    P = cp.zeros((512, 512), dtype=dtype)
    S = cp.zeros((512, 512), dtype=dtype)
    Ss = cp.zeros((512, 512), dtype=dtype)
    e = cp.zeros((512, 512), dtype=dtype)
    mx = cp.zeros((512, 1), dtype=dtype)
    sb = cp.zeros((512, 512), dtype=dtype)
    sm = cp.zeros((512, 1), dtype=dtype)
    O = cp.zeros((512, 64), dtype=dtype)
    grid = (ct.cdiv(S.shape[0], TILE_S[0]), ct.cdiv(S.shape[1], TILE_S[1]), 1)
    ct.launch(stream, grid, S_kernel, (KT, Q, S, 64, TILE_S[0], TILE_S[1], RTILE_S))
    grid = (ct.cdiv(Ss.shape[0], TILE_Ss[0]), ct.cdiv(Ss.shape[1], TILE_Ss[1]), 1)
    ct.launch(stream, grid, Ss_kernel, (S, Ss, TILE_Ss[0], TILE_Ss[1]))
    grid = (ct.cdiv(mx.shape[0], TILE_mx[0]), ct.cdiv(mx.shape[1], TILE_mx[1]), 1)
    ct.launch(stream, grid, mx_kernel, (Ss, mx, 512, TILE_mx[0]))
    grid = (ct.cdiv(sb.shape[0], TILE_sb[0]), ct.cdiv(sb.shape[1], TILE_sb[1]), 1)
    ct.launch(stream, grid, sb_kernel, (Ss, mx, sb, TILE_sb[0], TILE_sb[1]))
    grid = (ct.cdiv(e.shape[0], TILE_e[0]), ct.cdiv(e.shape[1], TILE_e[1]), 1)
    ct.launch(stream, grid, e_kernel, (sb, e, TILE_e[0], TILE_e[1]))
    grid = (ct.cdiv(sm.shape[0], TILE_sm[0]), ct.cdiv(sm.shape[1], TILE_sm[1]), 1)
    ct.launch(stream, grid, sm_kernel, (e, sm, 512, TILE_sm[0]))
    grid = (ct.cdiv(P.shape[0], TILE_P[0]), ct.cdiv(P.shape[1], TILE_P[1]), 1)
    ct.launch(stream, grid, P_kernel, (e, sm, P, TILE_P[0], TILE_P[1]))
    grid = (ct.cdiv(O.shape[0], TILE_O[0]), ct.cdiv(O.shape[1], TILE_O[1]), 1)
    ct.launch(stream, grid, O_kernel, (V, P, O, 64, 512, TILE_O[0], TILE_O[1], RTILE_O))
    return O

KERNEL_META = {'inputs': [['KT', [64, 512]], ['Q', [512, 64]], ['V', [512, 64]]], 'output': ['O', [512, 64]], 'flops': 68681728, 'flops_exact': True}

if __name__ == "__main__":
    args = [cp.random.randn(*s, dtype=cp.float32) for _, s in KERNEL_META["inputs"]]
    out = fn(*args)
    cp.cuda.runtime.deviceSynchronize()
    print("ran:", KERNEL_META["inputs"], "->", list(out.shape))
