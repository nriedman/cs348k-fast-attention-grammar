import cuda.tile as ct
import cupy as cp
import numpy as np

ConstInt = ct.Constant[int]

# reduction example: not so easy
@ct.kernel
def stage_1(x, y, a, K: ConstInt, TSIZE_N: ConstInt, TSIZE_M: ConstInt):
    """
        x: (N, K) in global memory
        y: (K, M) in global memory
        a: (N, M) in global memory
    """
    # tile size known at compile time
    # sized relative to output: each thread block handles one tile in output

    # each block handles one tile (again indexed into output)
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)

    # each output tile output needs an entire row / column from the inputs
    # we need to know this at compile time so we can output the correct shape
    x_ = ct.load(x, (bid_n, 0), (TSIZE_N, K))
    y_ = ct.load(y, (0, bid_m), (K, TSIZE_M))

    # compute the output tile (elementwise add in this case)
    a_ = x_ @ y_    # general matrix multiply, not ct.mma (maybe ct.mma if we split the reduction dimension?)

    # store the result back to global memory
    ct.store(a, (bid_n, bid_m), a_)

# elementwise example: easy
@ct.kernel
def stage_2(a, b, o, TSIZE_N: ConstInt, TSIZE_M: ConstInt):
    """
        a: (N, M) in global memory
        b: (N, M) in global memory
        o: (N, M) in global memory
    """

    # tile size known at compile time
    # sized relative to output: each thread block handles one tile in output

    # each block handles one tile
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)

    # load all dependent tiles
    # in this example, just need one tile for an elementwise operation
    a_ = ct.load(a, (bid_n, bid_m), (TSIZE_N, TSIZE_M))
    b_ = ct.load(b, (bid_n, bid_m), (TSIZE_N, TSIZE_M))

    # compute the output tile (elementwise add in this case)
    o_ = a_ + b_

    # store the result back to global memory
    ct.store(o, (bid_n, bid_m), o_)

def main():
    """
        Inputs:
            x: (N, K)
            y: (K, M)
            b: (N, M)

        Intermedates:
            a: (N, M)
        
        Outputs:
            o: (N, M)

        Stages:
        1. Matrix multiply input <x> and <y> to make <a>
            x: (N, K)
            y: (K, M)
            a: (N, M)
        
        2. Elementwise add input <b> and output from S1 <a> to make <o>
            a: (N, M)
            b: (N, M)
            o: (N, M)
    """
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
