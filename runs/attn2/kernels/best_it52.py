import cuda.tile as ct
import cupy as cp
import numpy as np

ConstInt = ct.Constant[int]

# --- tunable tile sizes (vary these to autotune) ---
TILE_P = (16, 512)
RTILE_P = 16   # reduction tile along dd
TILE_O = (16, 32)
RTILE_O = 16   # reduction tile along j

@ct.kernel
def P_kernel(Q, KT, P, DD: ConstInt, J: ConstInt, TS_i: ConstInt, TS_j: ConstInt, TS_dd: ConstInt):
    i = ct.bid(0)
    j = ct.bid(1)
    acc1 = ct.zeros((TS_i, TS_j), dtype=P.dtype)
    for dd in range(ct.cdiv(DD, TS_dd)):
        t2 = ct.load(Q, (i, dd), (TS_i, TS_dd))
        t3 = ct.load(KT, (dd, 0), (TS_dd, J))
        acc1 = ct.mma(t2, t3, acc1)
    t4 = (acc1 * 0.125)
    t5 = ct.max(t4, axis=1, keepdims=True)
    t6 = (t4 - t5)
    t7 = ct.exp(t6)
    t8 = ct.sum(t7, axis=1, keepdims=True)
    t9 = (t7 / t8)
    ct.store(P, (i, 0), t9)

@ct.kernel
def O_kernel(P, V, O, J: ConstInt, TS_i: ConstInt, TS_dd: ConstInt, TS_j: ConstInt):
    i = ct.bid(0)
    dd = ct.bid(1)
    t1 = ct.load(P, (i, 0), (TS_i, J))
    t2 = ct.load(V, (0, dd), (J, TS_dd))
    acc3 = ct.zeros((TS_i, TS_dd), dtype=O.dtype)
    for j in range(ct.cdiv(J, TS_j)):
        t4 = ct.extract(t1, (0, j), (TS_i, TS_j))
        t5 = ct.extract(t2, (j, 0), (TS_j, TS_dd))
        acc3 = ct.mma(t4, t5, acc3)
    ct.store(O, (i, dd), acc3)

def fn(KT, Q, V):
    dtype = KT.dtype
    stream = cp.cuda.get_current_stream()
    P = cp.zeros((512, 512), dtype=dtype)
    O = cp.zeros((512, 64), dtype=dtype)
    grid = (ct.cdiv(P.shape[0], TILE_P[0]), ct.cdiv(P.shape[1], TILE_P[1]), 1)
    ct.launch(stream, grid, P_kernel, (Q, KT, P, 64, 512, TILE_P[0], TILE_P[1], RTILE_P))
    grid = (ct.cdiv(O.shape[0], TILE_O[0]), ct.cdiv(O.shape[1], TILE_O[1]), 1)
    ct.launch(stream, grid, O_kernel, (P, V, O, 512, TILE_O[0], TILE_O[1], RTILE_O))
    return O

KERNEL_META = {'inputs': [['KT', [64, 512]], ['Q', [512, 64]], ['V', [512, 64]]], 'output': ['O', [512, 64]], 'flops': 68681728, 'flops_exact': True}

if __name__ == "__main__":
    args = [cp.random.randn(*s, dtype=cp.float32) for _, s in KERNEL_META["inputs"]]
    out = fn(*args)
    cp.cuda.runtime.deviceSynchronize()
    print("ran:", KERNEL_META["inputs"], "->", list(out.shape))
