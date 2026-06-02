import cuda.tile as ct
import cupy as cp
import numpy as np

ConstInt = ct.Constant[int]

# --- tunable tile sizes (vary these to autotune) ---
TILE_h = (32, 32)
TILE_a = (32, 32)
TILE_y = (32, 32)
TILE_r = (32, 32)
TILE_out = (256, 512)

# @ct.kernel
# def h_kernel(X, W1, h, DM: ConstInt, TS_b: ConstInt, TS_hid: ConstInt):
#     b = ct.bid(0)
#     hid = ct.bid(1)
#     t1 = ct.load(X, (b, 0), (TS_b, DM))
#     t2 = ct.load(W1, (0, hid), (DM, TS_hid))
#     t3 = ct.matmul(t1, t2)
#     ct.store(h, (b, hid), t3)

# @ct.kernel
# def a_kernel(h, a, TS_b: ConstInt, TS_hid: ConstInt):
#     b = ct.bid(0)
#     hid = ct.bid(1)
#     t1 = ct.load(h, (b, hid), (TS_b, TS_hid))
#     t2 = ct.maximum(0, t1)
#     ct.store(a, (b, hid), t2)

@ct.kernel
def y_kernel(a, W2, y, HID: ConstInt, TS_b: ConstInt, TS_dm: ConstInt):
    b = ct.bid(0)
    dm = ct.bid(1)
    acc = ct.zeros((b, dm))
    for h in range(ct.cdiv(HID, 32)):
        t1 = ct.load(a, (b, h), (TS_b, HID))
        t2 = ct.load(W2, (h, dm), (HID, TS_dm))
        ct.mma(t1, t2, acc)
    ct.store(y, (b, dm), acc)

# @ct.kernel
# def r_kernel(y, X, r, TS_b: ConstInt, TS_dm: ConstInt):
#     b = ct.bid(0)
#     dm = ct.bid(1)
#     t1 = ct.load(y, (b, dm), (TS_b, TS_dm))
#     t2 = ct.load(X, (b, dm), (TS_b, TS_dm))
#     t3 = ct.add(t1, t2)
#     ct.store(r, (b, dm), t3)

# @ct.kernel
# def out_kernel(r, out, B: ConstInt, DM: ConstInt):
#     b = ct.bid(0)
#     dm = ct.bid(1)
#     t1 = ct.load(r, (0, 0), (B, DM))
#     t2 = ct.sum(t1, axis=1, keepdims=True)
#     t3 = ct.load(r, (0, 0), (B, DM))
#     t4 = ct.load(r, (0, 0), (B, DM))
#     t5 = t3 * t4
#     t6 = ct.sum(t5, axis=1, keepdims=True)
#     t7 = (t2 * 0.001953125)
#     t8 = (t6 * 0.001953125)
#     t9 = t7 * t7
#     t10 = (t8 - t9)
#     t11 = (t10 + 1e-05)
#     t12 = ct.sqrt(t11)
#     t13 = ct.load(r, (0, 0), (B, DM))
#     t14 = (t13 - t7)
#     t15 = (t14 / t12)
#     ct.store(out, (0, 0), t15)

def fn(W1, W2, X):
    dtype = W1.dtype
    stream = cp.cuda.get_current_stream()
    a = cp.zeros((256, 1024), dtype=dtype)
    h = cp.zeros((256, 1024), dtype=dtype)
    r = cp.zeros((256, 512), dtype=dtype)
    y = cp.zeros((256, 512), dtype=dtype)
    out = cp.zeros((256, 512), dtype=dtype)
    # grid = (ct.cdiv(h.shape[0], TILE_h[0]), ct.cdiv(h.shape[1], TILE_h[1]), 1)
    # ct.launch(stream, grid, h_kernel, (X, W1, h, 512, TILE_h[0], TILE_h[1]))
    # grid = (ct.cdiv(a.shape[0], TILE_a[0]), ct.cdiv(a.shape[1], TILE_a[1]), 1)
    # ct.launch(stream, grid, a_kernel, (h, a, TILE_a[0], TILE_a[1]))
    grid = (ct.cdiv(y.shape[0], TILE_y[0]), ct.cdiv(y.shape[1], TILE_y[1]), 1)
    ct.launch(stream, grid, y_kernel, (a, W2, y, 1024, TILE_y[0], TILE_y[1]))
    # grid = (ct.cdiv(r.shape[0], TILE_r[0]), ct.cdiv(r.shape[1], TILE_r[1]), 1)
    # ct.launch(stream, grid, r_kernel, (y, X, r, TILE_r[0], TILE_r[1]))
    # grid = (ct.cdiv(out.shape[0], TILE_out[0]), ct.cdiv(out.shape[1], TILE_out[1]), 1)
    # ct.launch(stream, grid, out_kernel, (r, out, 256, 512))
    return out

KERNEL_META = {'inputs': [['W1', [512, 1024]], ['W2', [1024, 512]], ['X', [256, 512]]], 'output': ['out', [256, 512]], 'flops': 538705920, 'flops_exact': True}

if __name__ == "__main__":
    args = [cp.random.randn(*s, dtype=cp.float32) for _, s in KERNEL_META["inputs"]]
    out = fn(*args)
    cp.cuda.runtime.deviceSynchronize()
    print("ran:", KERNEL_META["inputs"], "->", list(out.shape))
