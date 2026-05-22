"""
Atomic grammar for the attention kernel.

This is the starting state of the optimizer: three unfused kernel scopes
(QK matmul+scale, softmax, PV matmul) with all intermediates in global memory
and minimal tile sizes. It is correct by construction.

See AST-spec.md § "The Atomic Grammar (Initial State)" for the full spec.
"""

from __future__ import annotations

from .ast import ComputeNode, DDGEdge, Grammar, KernelScope, LoopLevel, ProgramNode


def attention_atomic_grammar() -> Grammar:
    """
    Build and return the atomic attention grammar.

    Tree structure (from spec):

        ProgramNode
        ├── KernelScope          # kernel 1: QK matmul + scale
        │   └── LoopLevel(SEQ_Q, 64, parallel)
        │       └── LoopLevel(SEQ_KV, 64, parallel)
        │           ├── Compute(MatMul_QK)
        │           └── Compute(Scale)
        ├── KernelScope          # kernel 2: softmax
        │   └── LoopLevel(SEQ_Q, 64, parallel)
        │       ├── Compute(RowMax)
        │       ├── Compute(Subtract)
        │       ├── Compute(Exp)
        │       ├── Compute(RowSum)
        │       └── Compute(Divide)
        └── KernelScope          # kernel 3: PV matmul
            └── LoopLevel(SEQ_Q, 64, parallel)
                └── LoopLevel(SEQ_KV, 64, parallel)
                    └── Compute(MatMul_PV)

    DDG (from spec):
        MatMul_QK -> Scale       [S]
        Scale     -> RowMax      [S_scaled]
        Scale     -> Subtract    [S_scaled]
        RowMax    -> Subtract    [m]
        Subtract  -> Exp         [S_shifted]
        Exp       -> RowSum      [P]
        Exp       -> Divide      [P]
        RowSum    -> Divide      [l]
        Divide    -> MatMul_PV   [A]
    """
    # ------------------------------------------------------------------
    # Kernel 1: QK matmul + scale
    # ------------------------------------------------------------------
    matmul_qk = ComputeNode.make("MatMul_QK")
    scale = ComputeNode.make("Scale")

    k1 = KernelScope(
        children=(
            LoopLevel(
                dim="SEQ_Q",
                tile_size=64,
                loop_type="parallel",
                children=(
                    LoopLevel(
                        dim="SEQ_KV",
                        tile_size=64,
                        loop_type="parallel",
                        children=(matmul_qk, scale),
                    ),
                ),
            ),
        )
    )

    # ------------------------------------------------------------------
    # Kernel 2: softmax
    # ------------------------------------------------------------------
    row_max = ComputeNode.make("RowMax")
    subtract = ComputeNode.make("Subtract")
    exp = ComputeNode.make("Exp")
    row_sum = ComputeNode.make("RowSum")
    divide = ComputeNode.make("Divide")

    k2 = KernelScope(
        children=(
            LoopLevel(
                dim="SEQ_Q",
                tile_size=64,
                loop_type="parallel",
                children=(row_max, subtract, exp, row_sum, divide),
            ),
        )
    )

    # ------------------------------------------------------------------
    # Kernel 3: PV matmul
    # ------------------------------------------------------------------
    matmul_pv = ComputeNode.make("MatMul_PV")

    k3 = KernelScope(
        children=(
            LoopLevel(
                dim="SEQ_Q",
                tile_size=64,
                loop_type="parallel",
                children=(
                    LoopLevel(
                        dim="SEQ_KV",
                        tile_size=64,
                        loop_type="parallel",
                        children=(matmul_pv,),
                    ),
                ),
            ),
        )
    )

    program = ProgramNode(children=(k1, k2, k3))

    # ------------------------------------------------------------------
    # DDG
    # ------------------------------------------------------------------
    ddg = (
        DDGEdge(matmul_qk.node_id, scale.node_id,    "S"),
        DDGEdge(scale.node_id,     row_max.node_id,   "S_scaled"),
        DDGEdge(scale.node_id,     subtract.node_id,  "S_scaled"),
        DDGEdge(row_max.node_id,   subtract.node_id,  "m"),
        DDGEdge(subtract.node_id,  exp.node_id,       "S_shifted"),
        DDGEdge(exp.node_id,       row_sum.node_id,   "P"),
        DDGEdge(exp.node_id,       divide.node_id,    "P"),
        DDGEdge(row_sum.node_id,   divide.node_id,    "l"),
        DDGEdge(divide.node_id,    matmul_pv.node_id, "A"),
    )

    return Grammar(program=program, ddg=ddg)
