import cuda.tile as ct
import cupy as cp
import numpy as np

ConstInt = ct.Constant[int]

# --- tunable tile sizes (vary these to autotune; one tuple per stage) ---
TILE_a = (64, 64)
TILE_o = (32, 32)

@ct.kernel
def a_kernel(x, y, a, K: ConstInt, TS_n: ConstInt, TS_m: ConstInt):
    n = ct.bid(0)
    m = ct.bid(1)
    t1 = ct.load(x, (n, 0), (TS_n, K))
    t2 = ct.load(y, (0, m), (K, TS_m))
    t3 = ct.matmul(t1, t2)
    ct.store(a, (n, m), t3)

@ct.kernel
def o_kernel(a, b, o, TS_n: ConstInt, TS_m: ConstInt):
    n = ct.bid(0)
    m = ct.bid(1)
    t1 = ct.load(a, (n, m), (TS_n, TS_m))
    t2 = ct.load(b, (n, m), (TS_n, TS_m))
    t3 = ct.add(t1, t2)
    ct.store(o, (n, m), t3)

def fn(b, x, y):
    dtype = b.dtype
    stream = cp.cuda.get_current_stream()
    a = cp.zeros((1024, 1024), dtype=dtype)
    o = cp.zeros((1024, 1024), dtype=dtype)
    grid = (ct.cdiv(a.shape[0], TILE_a[0]), ct.cdiv(a.shape[1], TILE_a[1]), 1)
    ct.launch(stream, grid, a_kernel, (x, y, a, 512, TILE_a[0], TILE_a[1]))
    grid = (ct.cdiv(o.shape[0], TILE_o[0]), ct.cdiv(o.shape[1], TILE_o[1]), 1)
    ct.launch(stream, grid, o_kernel, (a, b, o, TILE_o[0], TILE_o[1]))
    return o

KERNEL_META = {'inputs': [['b', [1024, 1024]], ['x', [1024, 512]], ['y', [512, 1024]]], 'output': ['o', [1024, 1024]], 'flops': 1074790400, 'flops_exact': True}

if __name__ == "__main__":
    args = [cp.random.randn(*s, dtype=cp.float32) for _, s in KERNEL_META["inputs"]]
    out = fn(*args)
    cp.cuda.runtime.deviceSynchronize()
    print("ran:", KERNEL_META["inputs"], "->", list(out.shape))
