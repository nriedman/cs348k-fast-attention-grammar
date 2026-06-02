import cuda.tile as ct
import cupy as cp
import numpy as np

ConstInt = ct.Constant[int]

# --- tunable tile sizes (vary these to autotune) ---
TILE_Ss = (16, 16)
RTILE_Ss = 16   # reduction tile along dd
TILE_P = (16, 512)
TILE_O = (16, 16)
RTILE_O = 16   # reduction tile along j

@ct.kernel
def Ss_kernel(Q, KT, Ss, DD: ConstInt, TS_i: ConstInt, TS_j: ConstInt, TS_dd: ConstInt):
    i = ct.bid(0)
    j = ct.bid(1)
    acc1 = ct.zeros((TS_i, TS_j), dtype=Ss.dtype)
    for dd in range(ct.cdiv(DD, TS_dd)):
        t2 = ct.load(Q, (i, dd), (TS_i, TS_dd))
        t3 = ct.load(KT, (dd, j), (TS_dd, TS_j))
        acc1 = ct.mma(t2, t3, acc1)
    t4 = (acc1 * 0.125)
    ct.store(Ss, (i, j), t4)

@ct.kernel
def P_kernel(Ss, P, J: ConstInt, TS_i: ConstInt):
    i = ct.bid(0)
    j = ct.bid(1)
    t1 = ct.load(Ss, (i, 0), (TS_i, J))
    t2 = ct.max(t1, axis=1, keepdims=True)
    t3 = (t1 - t2)
    t4 = ct.exp(t3)
    t5 = ct.sum(t4, axis=1, keepdims=True)
    t6 = (t4 / t5)
    ct.store(P, (i, 0), t6)

@ct.kernel
def O_kernel(P, V, O, J: ConstInt, TS_i: ConstInt, TS_dd: ConstInt, TS_j: ConstInt):
    i = ct.bid(0)
    dd = ct.bid(1)
    acc1 = ct.zeros((TS_i, TS_dd), dtype=O.dtype)
    for j in range(ct.cdiv(J, TS_j)):
        t2 = ct.load(P, (i, j), (TS_i, TS_j))
        t3 = ct.load(V, (j, dd), (TS_j, TS_dd))
        acc1 = ct.mma(t2, t3, acc1)
    ct.store(O, (i, dd), acc1)

def fn(KT, Q, V):
    dtype = KT.dtype
    stream = cp.cuda.get_current_stream()
    P = cp.zeros((512, 512), dtype=dtype)
    Ss = cp.zeros((512, 512), dtype=dtype)
    O = cp.zeros((512, 64), dtype=dtype)
    grid = (ct.cdiv(Ss.shape[0], TILE_Ss[0]), ct.cdiv(Ss.shape[1], TILE_Ss[1]), 1)
    ct.launch(stream, grid, Ss_kernel, (Q, KT, Ss, 64, TILE_Ss[0], TILE_Ss[1], RTILE_Ss))
    grid = (ct.cdiv(P.shape[0], TILE_P[0]), ct.cdiv(P.shape[1], TILE_P[1]), 1)
    ct.launch(stream, grid, P_kernel, (Ss, P, 512, TILE_P[0]))
    grid = (ct.cdiv(O.shape[0], TILE_O[0]), ct.cdiv(O.shape[1], TILE_O[1]), 1)
    ct.launch(stream, grid, O_kernel, (P, V, O, 512, TILE_O[0], TILE_O[1], RTILE_O))
    return O

KERNEL_META = {'inputs': [['KT', [64, 512]], ['Q', [512, 64]], ['V', [512, 64]]], 'output': ['O', [512, 64]], 'flops': 68681728, 'flops_exact': True}

if __name__ == "__main__":
    args = [cp.random.randn(*s, dtype=cp.float32) for _, s in KERNEL_META["inputs"]]
    out = fn(*args)
    cp.cuda.runtime.deviceSynchronize()
    print("ran:", KERNEL_META["inputs"], "->", list(out.shape))