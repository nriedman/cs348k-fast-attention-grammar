"""
kernels/sota.py — SOTA baseline
================================
Invokes FlashAttention-2 as the state-of-the-art reference point.

Structurally this corresponds to the fully-optimised grammar configuration:

    KernelScope
      LoopLevel(seq_len, tile_size=64..128)
        SharedMemLoad(Q_tile)
        SharedMemLoad(K_tile)
        Compute(MatMul)
        Compute(ScaleAdd)
        Compute(Softmax)        # online, numerically stable
        SharedMemLoad(V_tile)
        Compute(MatMul)
        SharedMemStore(O_tile)

i.e. FuseKernels has been applied (QK^T, softmax, and AV share one kernel
launch), SplitLoop has tiled the sequence dimension, and HoistLoad /
SharedMemLoad stage tiles into SRAM so the (S, S) score matrix is never
materialised in global memory.

flash_attn expects (B, S, H, D), which is the calling convention used
throughout this project.
"""

from flash_attn import flash_attn_func
import torch


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    FlashAttention-2 forward pass.

    Args:
        q, k, v: (B, S, H, D) float CUDA tensors (fp16 or bf16)

    Returns:
        out: (B, S, H, D) float CUDA tensor
    """
    return flash_attn_func(q, k, v, causal=False)