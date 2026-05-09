"""
kernels/naive.py — Atomic grammar baseline
===========================================
Implements the degenerate grammar case:

    KernelScope
      Compute(MatMul)       # Q @ K^T
      Compute(ScaleAdd)     # scale by 1/sqrt(D)
      Compute(Softmax)      # row-wise softmax
      Compute(MatMul)       # A @ V

No fusion: each Compute node is a separate operation, writing its full
output back to global memory before the next node reads it.  No tiling
or chunking: every operation sees the full (S, S) or (S, D) tensor at
once.  No shared memory staging: there are no SharedMemLoad /
SharedMemStore nodes, so all intermediates live in global memory (i.e.
on the CUDA device but allocated as ordinary tensors by PyTorch).

This is the worst-case structural configuration the grammar can
represent, and sets the performance floor against which all rewrite
rules are measured.
"""

import torch


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Scaled dot-product attention with every intermediate materialised in
    global memory.

    Args:
        q, k, v: (B, H, S, D) float CUDA tensors

    Returns:
        out: (B, H, S, D) float CUDA tensor
    """
    # (S, S) score matrix written to global memory
    scores = q @ k.transpose(-2, -1)           # (B, H, S, S)

    # elementwise scale, result written to global memory
    scale = q.shape[-1] ** -0.5
    scores = scores * scale                    # (B, H, S, S)

    # row-wise softmax, result written to global memory
    weights = torch.softmax(scores, dim=-1)    # (B, H, S, S)

    # weighted sum, result written to global memory
    out = weights @ v                          # (B, H, S, D)

    return out