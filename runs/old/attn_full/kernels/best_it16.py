import cuda.tile as ct
import cupy as cp
import numpy as np

ConstInt = ct.Constant[int]

# --- tunable tile sizes (vary these to autotune) ---
TILE_Ss = (64, 32)
TILE_mx = (16, 512)
TILE_e = (16, 512)
TILE_sm = (32, 512)
TILE_P = (16, 512)
TILE_O = (16, 32)
RTILE_O = 16   # reduction tile along j

@ct.kernel
def Ss_kernel(Q, KT, Ss, DD: ConstInt, TS_i: ConstInt, TS_j: ConstInt):
    i = ct.bid(0)
    j = ct.bid(1)
    t1 = ct.load(Q, (i, 0), (TS_i, DD))
    t2 = ct.load(KT, (0, j), (DD, TS_j))
    t3 = ct.matmul(t1, t2)
    t4 = (t3 * 0.125)
    ct.store(Ss, (i, j), t4)

@ct.kernel
def mx_kernel(Ss, mx, J: ConstInt, TS_i: ConstInt):
    i = ct.bid(0)
    j = ct.bid(1)
    t1 = ct.load(Ss, (i, 0), (TS_i, J))
    t2 = ct.max(t1, axis=1, keepdims=True)
    ct.store(mx, (i, 0), t2)

@ct.kernel
def e_kernel(Ss, mx, e, J: ConstInt, TS_i: ConstInt):
    i = ct.bid(0)
    j = ct.bid(1)
    t1 = ct.load(Ss, (i, 0), (TS_i, J))
    t2 = ct.load(mx, (i, 0), (TS_i, 1))
    t3 = (t1 - t2)
    t4 = ct.exp(t3)
    ct.store(e, (i, 0), t4)

@ct.kernel
def sm_kernel(e, sm, J: ConstInt, TS_i: ConstInt):
    i = ct.bid(0)
    j = ct.bid(1)
    t1 = ct.load(e, (i, 0), (TS_i, J))
    t2 = ct.sum(t1, axis=1, keepdims=True)
    ct.store(sm, (i, 0), t2)

@ct.kernel
def P_kernel(e, sm, P, J: ConstInt, TS_i: ConstInt):
    i = ct.bid(0)
    j = ct.bid(1)
    t1 = ct.load(e, (i, 0), (TS_i, J))
    t2 = ct.load(sm, (i, 0), (TS_i, 1))
    t3 = (t1 / t2)
    ct.store(P, (i, 0), t3)

@ct.kernel
def O_kernel(V, P, O, J: ConstInt, TS_i: ConstInt, TS_dd: ConstInt, TS_j: ConstInt):
    i = ct.bid(0)
    dd = ct.bid(1)
    t1 = ct.load(V, (0, dd), (J, TS_dd))
    t2 = ct.load(P, (i, 0), (TS_i, J))
    acc3 = ct.zeros((TS_i, TS_dd), dtype=O.dtype)
    for j in range(ct.cdiv(J, TS_j)):
        t4 = ct.extract(t2, (0, j), (TS_i, TS_j))
        t5 = ct.extract(t1, (j, 0), (TS_j, TS_dd))
        acc3 = ct.mma(t4, t5, acc3)
    ct.store(O, (i, dd), acc3)

def fn(KT, Q, V):
    dtype = KT.dtype
    stream = cp.cuda.get_current_stream()
    P = cp.zeros((512, 512), dtype=dtype)
    Ss = cp.zeros((512, 512), dtype=dtype)
    e = cp.zeros((512, 512), dtype=dtype)
    mx = cp.zeros((512, 1), dtype=dtype)
    sm = cp.zeros((512, 1), dtype=dtype)
    O = cp.zeros((512, 64), dtype=dtype)
    grid = (ct.cdiv(Ss.shape[0], TILE_Ss[0]), ct.cdiv(Ss.shape[1], TILE_Ss[1]), 1)
    ct.launch(stream, grid, Ss_kernel, (Q, KT, Ss, 64, TILE_Ss[0], TILE_Ss[1]))
    grid = (ct.cdiv(mx.shape[0], TILE_mx[0]), ct.cdiv(mx.shape[1], TILE_mx[1]), 1)
    ct.launch(stream, grid, mx_kernel, (Ss, mx, 512, TILE_mx[0]))
    grid = (ct.cdiv(e.shape[0], TILE_e[0]), ct.cdiv(e.shape[1], TILE_e[1]), 1)
    ct.launch(stream, grid, e_kernel, (Ss, mx, e, 512, TILE_e[0]))
    grid = (ct.cdiv(sm.shape[0], TILE_sm[0]), ct.cdiv(sm.shape[1], TILE_sm[1]), 1)
    ct.launch(stream, grid, sm_kernel, (e, sm, 512, TILE_sm[0]))
    grid = (ct.cdiv(P.shape[0], TILE_P[0]), ct.cdiv(P.shape[1], TILE_P[1]), 1)
    ct.launch(stream, grid, P_kernel, (e, sm, P, 512, TILE_P[0]))
    grid = (ct.cdiv(O.shape[0], TILE_O[0]), ct.cdiv(O.shape[1], TILE_O[1]), 1)
    ct.launch(stream, grid, O_kernel, (V, P, O, 512, TILE_O[0], TILE_O[1], RTILE_O))
    return O

KERNEL_META = {'inputs': [['KT', [64, 512]], ['Q', [512, 64]], ['V', [512, 64]]], 'output': ['O', [512, 64]], 'flops': 68681728, 'flops_exact': True}

if __name__ == "__main__":
    args = [cp.random.randn(*s, dtype=cp.float32) for _, s in KERNEL_META["inputs"]]
    out = fn(*args)
    cp.cuda.runtime.deviceSynchronize()
    print("ran:", KERNEL_META["inputs"], "->", list(out.shape))
