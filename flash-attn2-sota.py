"""

    S  = Q @ K^T          [N,N]   reduces dd (head dim); KT is stored transposed
    Ss = S * (1/sqrt(d))  [N,N]   scale
    mx = rowmax(Ss)       [N,1]   softmax: max over key axis j
    sb = Ss - mx          [N,N]   broadcast subtract (mx reloaded width-1)
    e  = exp(sb)          [N,N]
    sm = rowsum(e)        [N,1]   sum over key axis j
    P  = e / sm           [N,N]   broadcast divide
    O  = P @ V            [N,d]   reduces j (key axis)

Kernel module interface
-----------------------
The module must expose:
  - a callable `fn(*inputs) -> output`  (cupy arrays)
  - KERNEL_META = {"inputs": [[name, [dims...]], ...],
                   "output": [name, [dims...]],
                   "flops": int, "flops_exact": bool}
"""

import torch
import torch.nn.functional as F

"""
import torch
import torch.nn.functional as F


q = torch.randn(B, H, L, D, device="cuda", dtype=torch.float16)
k = torch.randn(B, H, L, D, device="cuda", dtype=torch.float16)
v = torch.randn(B, H, L, D, device="cuda", dtype=torch.float16)

out = F.scaled_dot_product_attention(
    q, k, v,
    is_causal=True,
    dropout_p=0.1
)
"""


def fn(K, Q, V):
    """
    FlashAttention-2 forward pass.

    N: 512, D: 64
    q = Q.reshape(1, 1, N, D)
    k = K.reshape(1, 1, N, D)
    v = V.reshape(1, 1, N, D)

    Args:
        q, k, v: (B, S, H, D) float CUDA tensors (fp16 or bf16)

    Returns:
        out: (B, S, H, D) float CUDA tensor
    """
    K = torch.from_dlpack(K)
    Q = torch.from_dlpack(Q)
    V = torch.from_dlpack(V)

    return F.scaled_dot_product_attention(
        Q, K, V,
        is_causal=False
    )

KERNEL_META = {'inputs': [['K', [1, 1, 512, 64]], ['Q', [1, 1, 512, 64]], ['V', [1, 1, 512, 64]]], 'output': ['O', [1, 1, 512, 64]], 'flops': 68681728, 'flops_exact': True}
