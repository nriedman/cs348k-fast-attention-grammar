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

flash_attn expects (B, S, H, D); benchmark.py uses (B, H, S, D).
Transposes are applied here so the calling convention is identical to
every other kernel in this directory.
"""

from flash_attn import flash_attn_func
import torch


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    FlashAttention-2 forward pass.

    Args:
        q, k, v: (B, H, S, D) float CUDA tensors (fp16 or bf16)

    Returns:
        out: (B, H, S, D) float CUDA tensor
    """
    # flash_attn_func expects (B, S, H, D)
    q_ = q.transpose(1, 2).contiguous()
    k_ = k.transpose(1, 2).contiguous()
    v_ = v.transpose(1, 2).contiguous()

    out = flash_attn_func(q_, k_, v_, causal=False)  # (B, S, H, D)

    return out.transpose(1, 2)                        # (B, H, S, D)