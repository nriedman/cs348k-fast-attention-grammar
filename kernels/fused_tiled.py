"""
kernels/fused_tiled.py — Fused, tiled attention (CuTile Python)
================================================================

All four operations (MatMul_QK, Scale, Softmax, MatMul_PV) fused into a
single kernel.  The SEQ_Q dimension is parallelised across blocks; SEQ_KV
is swept serially as a carried reduction.

Softmax here is local to each KV tile (naive).  A globally-correct softmax
requires online rescaling (FlashAttention style), which needs carry-over
support not yet in the grammar.

See grammar/example_fused_tiled.py for the grammar that compiles to this.
"""

import cuda.tile as ct

# Tile size constants — these become compile-time specialisations.
BM = 64   # Q tile rows    (SEQ_Q  tile, maps to block height)
BN = 64   # KV tile rows   (SEQ_KV tile, swept serially)


@ct.kernel
def _fused_attention_kernel(
    Q, K, V, O,
    BM: ct.Constant[int],
    BN: ct.Constant[int],
    D:  ct.Constant[int],
):
    # ------------------------------------------------------------------
    # SEQ_Q loop — parallelised as the grid dim.
    # Each block owns one Q tile; ct.bid(0) selects which one.
    # ------------------------------------------------------------------
    q_blk = ct.bid(0)

    num_kv_tiles = ct.num_tiles(K, axis=0, shape=(BN, D))

    # Load Q tile once — it doesn't change across the KV loop.
    q = ct.load(Q, index=(q_blk, 0), shape=(BM, D))

    # ------------------------------------------------------------------
    # Loop-carried accumulator for the output (MatMul_PV).
    # Initialised before the loop; updated with += each iteration.
    # This is the carried_dims={SEQ_KV} pattern.
    # ------------------------------------------------------------------
    out_acc = ct.full((BM, D), 0.0, dtype=ct.float32)

    # ==================================================================
    # SEQ_KV loop — serial (parallel=False).
    # Sweeps all KV tiles, accumulating into out_acc.
    # ==================================================================
    for kv in range(num_kv_tiles):

        # Load K and V tiles for this iteration.
        k = ct.load(K, index=(kv, 0), shape=(BN, D))
        v = ct.load(V, index=(kv, 0), shape=(BN, D))

        # --------------------------------------------------------------
        # Compute(MatMul_QK) + Compute(Scale)
        # scores[m, n] = dot(q[m], k[n]) / sqrt(D)   shape: (BM, BN)
        # --------------------------------------------------------------
        scores = ct.mma(q, ct.transpose(k), ct.full((BM, BN), 0.0))
        scores = scores * (D ** -0.5)

        # --------------------------------------------------------------
        # Compute(Softmax) — over the BN dim, local to this KV tile.
        # --------------------------------------------------------------
        scores_exp = ct.exp(scores)                          # (BM, BN)
        row_sums   = ct.sum(scores_exp, axis=1, keepdim=True) # (BM,  1)
        weights    = scores_exp / row_sums                   # (BM, BN)

        # --------------------------------------------------------------
        # Compute(MatMul_PV) — accumulate output tile.
        # out_acc is the loop-carried value; ct.mma adds to it.
        # --------------------------------------------------------------
        out_acc = ct.mma(weights, v, out_acc)

    # ------------------------------------------------------------------
    # Write completed output tile back to global memory.
    # ------------------------------------------------------------------
    ct.store(O, index=(q_blk, 0), tile=ct.astype(out_acc, O.dtype))


def attention(q, k, v):
    """
    Entry point matching the benchmark interface: (B, S, H, D) tensors.
    Flattens B and H into the grid so each block handles one (b, h, q_tile)
    combination.  For simplicity this version operates on a single head;
    a production version would fold B*H into the grid.
    """
    B, S, H, D = q.shape
    o = q.new_empty(q.shape)

    # Launch: one block per Q tile.
    grid = (ct.cdiv(S, BM), 1, 1)
    ct.launch(
        q.device.current_stream(),
        grid,
        _fused_attention_kernel,
        (q, k, v, o, BM, BN, D),
    )
    return o
