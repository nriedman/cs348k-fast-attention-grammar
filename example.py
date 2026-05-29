import cuda.tile as ct
import cupy as cp
import numpy as np

ConstInt = ct.Constant[int]

# --- tunable tile sizes (vary these to autotune) ---
TILE_o = (64, 64)
STILE_o_n = 32   # subtile along n

@ct.kernel
def o_kernel(a, o, TS_n: ConstInt, TS_m: ConstInt, SUB_TS_n: ConstInt):
    n = ct.bid(0)
    m = ct.bid(1)
    for st_n in range(ct.cdiv(TS_n, SUB_TS_n)):
        sn = n * (TS_n // SUB_TS_n) + st_n
        t1 = ct.load(a, (sn, m), (SUB_TS_n, TS_m))
        t2 = ct.maximum(0, t1)
        ct.store(o, (sn, m), t2)

def fn(a):
    dtype = a.dtype
    stream = cp.cuda.get_current_stream()
    o = cp.zeros((1024, 1024), dtype=dtype)
    grid = (ct.cdiv(o.shape[0], TILE_o[0]), ct.cdiv(o.shape[1], TILE_o[1]), 1)
    ct.launch(stream, grid, o_kernel, (a, o, TILE_o[0], TILE_o[1], STILE_o_n))
    return o

KERNEL_META = {'inputs': [['a', [1024, 1024]]], 'output': ['o', [1024, 1024]], 'flops': 1048576, 'flops_exact': True}

if __name__ == "__main__":
    args = [cp.random.randn(*s, dtype=cp.float32) for _, s in KERNEL_META["inputs"]]
    out = fn(*args)
    cp.cuda.runtime.deviceSynchronize()
    print("ran:", KERNEL_META["inputs"], "->", list(out.shape))
