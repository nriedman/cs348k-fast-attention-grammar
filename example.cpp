// a simple example of an attention kernel written in CuTile C++.
//
// intended to give an idea of what it looks like after compiling
// a grammar -- we can see the structure of the loops and tiles
// and allocating and loading shared memory.

kernel attention_kernel(
    tensor<float> Q,   // [N, D]
    tensor<float> K,   // [N, D]
    tensor<float> V,   // [N, D]
    tensor<float> O    // [N, D]
) {

    // Which output tile this block computes
    int q_tile = blockIdx.x;

    // Shared-memory tiles
    tile<float, BLOCK_M, BLOCK_K> q_smem;
    tile<float, BLOCK_N, BLOCK_K> k_smem;
    tile<float, BLOCK_N, BLOCK_K> v_smem;

    // Attention scores for this tile
    tile<float, BLOCK_M, BLOCK_N> scores;

    // Final output accumulator
    tile<float, BLOCK_M, BLOCK_K> out_accum = 0;

    // --------------------------------------------------
    // Load Q tile
    // --------------------------------------------------

    for (int i = threadIdx.y; i < BLOCK_M; i += blockDim.y) {
        for (int j = threadIdx.x; j < BLOCK_K; j += blockDim.x) {
            q_smem[i][j] =
                Q[q_tile * BLOCK_M + i][j];
        }
    }

    sync_threads();

    // --------------------------------------------------
    // Sweep over all K/V tiles
    // --------------------------------------------------

    for (int kv_tile = 0; kv_tile < N; kv_tile += BLOCK_N) {

        // ----------------------------------------------
        // Load K tile
        // ----------------------------------------------

        for (int i = threadIdx.y; i < BLOCK_N; i += blockDim.y) {
            for (int j = threadIdx.x; j < BLOCK_K; j += blockDim.x) {
                k_smem[i][j] =
                    K[kv_tile + i][j];
            }
        }

        // ----------------------------------------------
        // Load V tile
        // ----------------------------------------------

        for (int i = threadIdx.y; i < BLOCK_N; i += blockDim.y) {
            for (int j = threadIdx.x; j < BLOCK_K; j += blockDim.x) {
                v_smem[i][j] =
                    V[kv_tile + i][j];
            }
        }

        sync_threads();

        // ----------------------------------------------
        // Compute scores = Q @ K^T
        // ----------------------------------------------

        for (int m = 0; m < BLOCK_M; ++m) {
            for (int n = 0; n < BLOCK_N; ++n) {

                float acc = 0.0f;

                for (int k = 0; k < BLOCK_K; ++k) {
                    acc += q_smem[m][k] * k_smem[n][k];
                }

                scores[m][n] = acc / sqrt(float(BLOCK_K));
            }
        }

        // ----------------------------------------------
        // Softmax (naive)
        // ----------------------------------------------

        for (int m = 0; m < BLOCK_M; ++m) {

            float sum = 0.0f;

            for (int n = 0; n < BLOCK_N; ++n) {
                scores[m][n] = exp(scores[m][n]);
                sum += scores[m][n];
            }

            for (int n = 0; n < BLOCK_N; ++n) {
                scores[m][n] /= sum;
            }
        }

        // ----------------------------------------------
        // O += scores @ V
        // ----------------------------------------------

        for (int m = 0; m < BLOCK_M; ++m) {
            for (int d = 0; d < BLOCK_K; ++d) {

                float acc = 0.0f;

                for (int n = 0; n < BLOCK_N; ++n) {
                    acc += scores[m][n] * v_smem[n][d];
                }

                out_accum[m][d] += acc;
            }
        }

        sync_threads();
    }

    // --------------------------------------------------
    // Store output
    // --------------------------------------------------

    for (int i = threadIdx.y; i < BLOCK_M; i += blockDim.y) {
        for (int j = threadIdx.x; j < BLOCK_K; j += blockDim.x) {

            O[q_tile * BLOCK_M + i][j] =
                out_accum[i][j];
        }
    }
}