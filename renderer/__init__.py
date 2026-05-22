"""
renderer — compiles a Grammar AST to a Python callable.

Public API
----------
    render(grammar, tile_q=64, tile_kv=64, output_path=None) -> Callable
        Compile grammar to attention(q, k, v) -> out.
        Writes the source to output_path if provided.

    render_source(grammar, tile_q=64, tile_kv=64, output_path=None) -> str
        Return the generated CuTile source without executing it.
        Writes the source to output_path if provided.

Backend
-------
cuda.tile — generates @ct.kernel functions; one per top-level LoopLevel.
Requires CUDA 13.1+ and `tileiras` + `ptxas` on PATH.

The generated code mirrors the structure shown in example.cpp:
  - parallel LoopLevels → ct.bid() grid axes
  - serial LoopLevels   → for-loops in the kernel body
  - carried_dims ops     → initialised accumulators before the loop, +=
  - DDG cross-kernel     → global memory round-trips between launches
  - DDG same-kernel      → stays in smem / rmem (register tiles)
"""

from .renderer import render, render_source

__all__ = ["render", "render_source"]
