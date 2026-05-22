"""
grammar/example_fused_tiled.py
================================
The grammar state that should render to kernels/fused_tiled.py.

Tree (single fused kernel):

    ProgramNode
    └── LoopLevel(SEQ_Q,  bound=64, parallel=True)    → ct.bid(0), grid dim
        └── LoopLevel(SEQ_KV, bound=64, parallel=False) → serial for-loop
            ├── Compute(MatMul_QK,  output_dims={SEQ_Q, SEQ_KV}, carried_dims={})
            ├── Compute(Scale,      output_dims={SEQ_Q, SEQ_KV}, carried_dims={})
            ├── Compute(Softmax,    output_dims={SEQ_Q, SEQ_KV}, carried_dims={})
            └── Compute(MatMul_PV,  output_dims={SEQ_Q},         carried_dims={SEQ_KV})

How the renderer reads this tree to produce fused_tiled.py
-----------------------------------------------------------
1.  Single top-level LoopLevel under ProgramNode
        → one GPU kernel (each direct child LoopLevel of ProgramNode is one
          kernel launch; no KernelScope needed)

2.  SEQ_Q, parallel=True
        → ct.bid(0) selects the tile; grid = (N // BM, 1, 1)
        → no explicit loop in kernel body over SEQ_Q

3.  SEQ_KV, parallel=False
        → `for kv in range(num_kv_tiles):` in kernel body

4.  Q is an input tensor consumed by MatMul_QK.
    MatMul_QK lives inside SEQ_KV, but Q doesn't vary with kv
    (Q's output_dims = {SEQ_Q}, no SEQ_KV component).
        → load Q once before the SEQ_KV loop

5.  K, V are inputs to MatMul_QK and MatMul_PV respectively.
    They do vary with kv (K[kv], V[kv]).
        → load K and V inside the SEQ_KV loop

6.  scores is produced by MatMul_QK+Scale and consumed by Softmax
    within the same loop iteration (no SEQ_KV crossing).
        → scores lives as a register tile (ct.full); no global store

7.  weights is produced by Softmax and consumed by MatMul_PV
    within the same iteration.
        → weights lives as a register tile; no global store

8.  MatMul_PV has carried_dims={SEQ_KV}.
        → out_acc initialised with ct.full(..., 0.0) before the loop
        → ct.mma(weights, v, out_acc) accumulates with += each iteration

9.  DDG order inside the loop: MatMul_QK → Scale → Softmax → MatMul_PV
        → emitted in that sequence in the loop body

10. out_acc is written to global memory after the loop with ct.store.

DDG
---
    MatMul_QK  →  Scale      (scores_raw)
    Scale      →  Softmax    (scores_scaled)
    Softmax    →  MatMul_PV  (weights)
"""

from grammar.ast import ComputeNode, DDGEdge, Grammar, LoopLevel, ProgramNode

# ---------------------------------------------------------------------------
# Tile bounds — chosen here; the grammar optimiser will search over these.
# ---------------------------------------------------------------------------
BM = 64   # SEQ_Q  bound (outer loop)
BN = 64   # SEQ_KV bound (inner loop)

# ---------------------------------------------------------------------------
# Compute nodes with output_dims and carried_dims declared.
# ---------------------------------------------------------------------------
matmul_qk = ComputeNode.make(
    "MatMul_QK",
    output_dims={"SEQ_Q", "SEQ_KV"},
    carried_dims={},
)

scale = ComputeNode.make(
    "Scale",
    output_dims={"SEQ_Q", "SEQ_KV"},
    carried_dims={},
)

softmax = ComputeNode.make(
    "Softmax",
    output_dims={"SEQ_Q", "SEQ_KV"},
    carried_dims={},
)

matmul_pv = ComputeNode.make(
    "MatMul_PV",
    output_dims={"SEQ_Q"},
    carried_dims={"SEQ_KV"},
    # carried_dims={SEQ_KV} → renderer emits out_acc with += instead of =
)

# ---------------------------------------------------------------------------
# Tree: one top-level LoopLevel = one GPU kernel
# ---------------------------------------------------------------------------
program = ProgramNode(
    children=(
        LoopLevel(
            dim="SEQ_Q",
            bound=BM,
            parallel=True,          # → ct.bid(0) / grid dim
            children=(
                LoopLevel(
                    dim="SEQ_KV",
                    bound=BN,
                    parallel=False,  # → serial for-loop; carries out_acc
                    children=(
                        matmul_qk,
                        scale,
                        softmax,
                        matmul_pv,
                    ),
                ),
            ),
        ),
    )
)

# ---------------------------------------------------------------------------
# DDG
# ---------------------------------------------------------------------------
ddg = (
    DDGEdge(matmul_qk.node_id, scale.node_id,     "scores_raw"),
    DDGEdge(scale.node_id,     softmax.node_id,    "scores_scaled"),
    DDGEdge(softmax.node_id,   matmul_pv.node_id,  "weights"),
)

grammar = Grammar(program=program, ddg=ddg)
