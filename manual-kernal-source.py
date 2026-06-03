import cuda.tile as ct
import cupy as cp
import numpy as np

ConstInt = ct.Constant[int]

# --- tunable tile sizes (vary these to autotune) ---
TILE_S = (64, 512)
TILE_P = (64, 512)
TILE_O = (64, 64)

@ct.kernel
def S_kernel(Q, KT, S, DD: ConstInt, J: ConstInt, TS_i: ConstInt):
    i = ct.bid(0)
    j = ct.bid(1)
    t1 = ct.load(Q, (i, 0), (TS_i, DD))
    t2 = ct.load(KT, (0, 0), (DD, J))
    t3 = ct.matmul(t1, t2)
    ct.store(S, (i, 0), t3)

@ct.kernel
def P_kernel(S, P, J: ConstInt, TS_i: ConstInt):
    i = ct.bid(0)
    j = ct.bid(1)
    t1 = ct.load(S, (i, 0), (TS_i, J))
    t2 = ct.max(t1, axis=1, keepdims=True)
    t3 = (t1 - t2)
    t4 = ct.exp(t3)
    t5 = ct.sum(t4, axis=1, keepdims=True)
    t6 = (t4 / t5)
    ct.store(P, (i, 0), t6)

@ct.kernel
def O_kernel(P, V, O, DD: ConstInt, J: ConstInt, TS_i: ConstInt):
    i = ct.bid(0)
    dd = ct.bid(1)
    t1 = ct.load(P, (i, 0), (TS_i, J))
    t2 = ct.load(V, (0, 0), (J, DD))
    t3 = ct.matmul(t1, t2)
    ct.store(O, (i, 0), t3)

def fn(KT, Q, V):
    dtype = KT.dtype
    stream = cp.cuda.get_current_stream()
    P = cp.zeros((512, 512), dtype=dtype)
    S = cp.zeros((512, 512), dtype=dtype)
    O = cp.zeros((512, 64), dtype=dtype)
    grid = (ct.cdiv(S.shape[0], TILE_S[0]), ct.cdiv(S.shape[1], TILE_S[1]), 1)
    ct.launch(stream, grid, S_kernel, (Q, KT, S, 64, 512, TILE_S[0]))
    grid = (ct.cdiv(P.shape[0], TILE_P[0]), ct.cdiv(P.shape[1], TILE_P[1]), 1)
    ct.launch(stream, grid, P_kernel, (S, P, 512, TILE_P[0]))
    grid = (ct.cdiv(O.shape[0], TILE_O[0]), ct.cdiv(O.shape[1], TILE_O[1]), 1)
    ct.launch(stream, grid, O_kernel, (P, V, O, 64, 512, TILE_O[0]))
    return O

KERNEL_META = {'inputs': [['KT', [64, 512]], ['Q', [512, 64]], ['V', [512, 64]]], 'output': ['O', [512, 64]], 'flops': 68419584, 'flops_exact': True}

if __name__ == "__main__":
    args = [cp.random.randn(*s, dtype=cp.float32) for _, s in KERNEL_META["inputs"]]
    out = fn(*args)
    cp.cuda.runtime.deviceSynchronize()
    print("ran:", KERNEL_META["inputs"], "->", list(out.shape))