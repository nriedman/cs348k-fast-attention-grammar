import cuda.tile as ct
import cupy as cp
import numpy as np

# matmul  (cache hit: False)
@ct.kernel
def C_kernel(A, B, C):  # launch grid (8, 6)
    bm = ct.bid(0)
    bn = ct.bid(1)
    t1 = ct.load(A, (bm, 0), (128, 512))
    t2 = ct.load(B, (0, bn), (512, 128))
    t3 = ct.matmul(t1, t2)  # tile (128, 128)
    ct.store(C, (bm, bn), t3)  # tile (128, 128)

# elementwise  (cache hit: False)
@ct.kernel
def D_kernel(A, B, D):  # launch grid (8, 6)
    bm = ct.bid(0)
    bn = ct.bid(1)
    t1 = ct.load(A, (bm, bn), (128, 128))
    t2 = ct.load(B, (bm, bn), (128, 128))
    t3 = ct.add(t1, t2)  # tile (128, 128)
    t4 = ct.maximum(0, t3)  # tile (128, 128)
    ct.store(D, (bm, bn), t4)  # tile (128, 128)

# 4-D batched  (cache hit: False)
@ct.kernel
def Y_kernel(X, Y):  # launch grid (32, 2, 2)
    bm = ct.bid(1)
    bn = ct.bid(2)
    _flat = ct.bid(0)
    b1 = _flat % 8
    _flat //= 8
    b0 = _flat
    t1 = ct.load(X, (b0, b1, bm, bn), (1, 1, 128, 128))
    t2 = ct.relu(t1)  # tile (1, 1, 128, 128)
    ct.store(Y, (b0, b1, bm, bn), t2)  # tile (1, 1, 128, 128)

def main():
    
    N, M = (1028, 1028)
    K = 512

    # inputs
    x = cp.random.randn(N, K, dtype=cp.float32)
    y = cp.random.randn(K, M, dtype=cp.float32)
    b = cp.random.randn(N, M, dtype=cp.float32)

    # intermedates
    a = cp.zeros((N, M), dtype=cp.float32)

    # output
    o = cp.zeros((N, M), dtype=cp.float32)

    # Stage 1

    print("Stage 1: Matrix Multiply")

    # parameter of the stage!
    s1_tsize = (64, 64)         # N, M
    grid = (
        ct.cdiv(a.shape[0], s1_tsize[0]),
        ct.cdiv(a.shape[1], s1_tsize[1]),
        1       # no third in this case
    )
    ct.launch(cp.cuda.get_current_stream(), grid, stage_1, (x, y, a, K, s1_tsize[0], s1_tsize[1]))

    # Stage 2

    print("Stage 2: Elementwise Addition")

    # parameter of the stage!
    s2_tsize = (32, 32)         # N, M
    grid = (
        ct.cdiv(a.shape[0], s2_tsize[0]),
        ct.cdiv(a.shape[1], s2_tsize[1]),
        1       # no third in this case
    )
    ct.launch(cp.cuda.get_current_stream(), grid, stage_2, (a, b, o, s2_tsize[0], s2_tsize[1]))

    # Sanity check

    ct_out = cp.asnumpy(o)

    # compute on host
    x_host = cp.asnumpy(x)
    y_host = cp.asnumpy(y)
    b_host = cp.asnumpy(b)

    a_host = x_host @ y_host
    o_host = a_host + b_host

    print("Verifying output")

    if np.allclose(ct_out, o_host, rtol=1e-3, atol=1e-3):
        print("SUCCESS")
    else:
        diff = np.abs(ct_out - o_host)
        print(f"ERROR: max diff={diff.max():.6f}, mean diff={diff.mean():.6f}")


if __name__ == "__main__":
    main()

