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

def fn(KT, Q, V):
    """
    Scaled dot-product attention with every intermediate materialised in
    global memory.

    Args:
        Q, V: (N, D) float CUDA tensors
        KT:   (D, N) float CUDA tensor

    Returns:
        out: (N, D) float CUDA tensor
    """

    KT = torch.from_dlpack(KT)
    Q = torch.from_dlpack(Q)
    V = torch.from_dlpack(V)

    # (S, S) score matrix written to global memory
    scores = Q @ KT

    # elementwise scale, result written to global memory
    scale = Q.shape[-1] ** -0.5
    scores = scores * scale

    # mx = rowmax(Ss) over j -> [N,1]   (key axis tiled to FULL extent for the reduction)
    mx = torch.sum(scores, dim=1)

    # sb = Ss - mx   (broadcast: mx is [N,1], reloaded width-1)
    sb = scores - mx

    # e = exp(sb)
    e = torch.exp(sb)

    # sm = rowsum(e) over j -> [N,1]
    sm = torch.sum(e, dim=1)

    # P = e / sm   (broadcast)
    P = e / sm

    # O = P @ V  -> [N,d], reduces j
    out = P @ V

    return out

KERNEL_META = {'inputs': [['KT', [64, 512]], ['Q', [512, 64]], ['V', [512, 64]]], 'output': ['O', [512, 64]], 'flops': 68681728, 'flops_exact': True}
