# Kernel Grammar AST Specification

## Overview

This document specifies the Abstract Syntax Tree (AST) used to represent GPU kernel programs in the grammar-based optimization framework. The framework is inspired by "Design for Descent" (Kodnongbua et al., 2025) and the Halide autoscheduler paper. Here, the "shape" is a GPU kernel, the "render" function compiles and benchmarks the kernel, and the "objective" is kernel runtime on a fixed benchmark.

### Halide Autoscheduler Inspiration

The grammar is structured as a **loop nest tree**, following the Halide autoscheduler's representation of compute schedules. The key insight from Halide is that the structure of a computation (what operations are performed) is separate from its schedule (where, when, and at what granularity each operation is computed). The grammar encodes the schedule as a tree of `LoopLevel` nodes; the computation is encoded in the DDG.

The optimization process begins from the **compute-at-root** atomic state: every `ComputeNode` is placed at the outermost level, each in its own GPU kernel, with no fusion. This is the Halide equivalent of computing every function into a full intermediate buffer before the next function reads it. From this maximally-unfused baseline, rewrite rules progressively fuse operations into shared kernels, tile loops, and reorder computation — traversing the schedule space toward FlashAttention2-level performance.

### Kernel Boundary Convention

**KernelScope is removed.** GPU kernel boundaries are now implicit: each direct child `LoopLevel` of `ProgramNode` is one GPU kernel launch, executed in order. When `ProgramNode` has a single child, the entire computation runs in one kernel. This convention eliminates a redundant layer and makes the tree more uniform.

### Correctness by Construction

Correctness is guaranteed inductively:

1. The atomic grammar is correct by definition — it is the maximally-unfused, obviously-correct implementation of the target computation.
2. Every rewrite rule is **semantics-preserving by design** — it reorganizes the schedule without changing what is computed, provided its data-dependency preconditions are satisfied.
3. Therefore every state reachable from the atomic grammar by valid rewrites is also correct.

No part of the system checks whether the set of `ComputeNode`s is complete or correct. The only validity constraints that matter are those that prevent a rewrite from violating a data dependency.

---

## Node Types

### 1. `ProgramNode` (root)

The root of the AST. Contains one or more `LoopLevel` children. Each direct child `LoopLevel` is compiled into a separate CUDA kernel launch, executed in order.

**Children:** one or more `LoopLevel` nodes, in execution order.

**Parameters:** none.

**Kernel boundary rule:** The number of GPU kernel launches equals the number of direct `LoopLevel` children of `ProgramNode`. Moving a `ComputeNode` from one top-level `LoopLevel` to another (via `LowerCompute`) is the primary mechanism for changing kernel boundaries.

---

### 2. `LoopLevel`

Represents a single tiled loop over one logical dimension of the computation. Nested `LoopLevel` nodes represent nested loops.

**Children:** one or more `LoopLevel` or `ComputeNode` nodes, in execution order.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `dim` | string | The logical dimension this loop iterates over (e.g. `SEQ_Q`, `SEQ_KV`, `HEAD_DIM`). The set of valid dimensions is defined by the atomic grammar. |
| `bound` | int | Number of iterations; must be a power of 2. `SplitLoop` produces two nested `LoopLevel`s where `outer.bound * inner.bound == original.bound`. |
| `parallel` | bool | If `True`, iterations are parallelised across GPU threads/blocks (the renderer maps this to `ct.bid` / grid dimensions). If `False`, this is a serial for-loop in the kernel body. Changed by the `Parallelize` / `Serialize` rewrite rules. |

**Validity:**
- `bound` must be a power of 2.
- If `parallel=True`, no enclosed `ComputeNode` may have `loop.dim` in its `carried_dims` — a loop that carries accumulation state across its iterations cannot be parallelised.

---

### 3. `ComputeNode`

A leaf node representing a single primitive operation. `ComputeNode`s are always leaves — they have no children. Inputs and outputs are defined by the DDG, not by the tree structure.

**Children:** none.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `op` | string | Identifies the primitive operation (e.g. `MatMul_QK`, `RowMax`, `Exp`). Defined by the atomic grammar. The framework does not interpret `op` values; the renderer maps each `op` to the correct CuTile primitive. |
| `node_id` | int | Unique monotonically-increasing integer assigned at creation time. Used for DDG edge identity. |
| `output_dims` | frozenset[str] | The set of dimension names that appear in this op's output tensor. A parallel loop over dimension `D` writes to a different output slice each iteration; `D` must be in `output_dims` for the renderer to correctly address the output. |
| `carried_dims` | frozenset[str] | The set of dimension names over which this op accumulates state across loop iterations. If `D ∈ carried_dims`, the op requires a serial loop (parallel=False) over `D`. The renderer emits an initialised accumulator before the loop and `+=` inside it. |

### Understanding `output_dims` and `carried_dims`

**`output_dims`** declares which loop dimensions index into the op's output. For example:

- `MatMul_QK` with `output_dims={SEQ_Q, SEQ_KV}`: the output attention score matrix `S[q, kv]` depends on both the query tile index `q` and the key-value tile index `kv`. A parallel loop over either dimension writes to a distinct output slice — parallelism is safe.
- `RowMax` with `output_dims={SEQ_Q}`: the output is a per-query row maximum `m[q]`. The output only varies with `q`, not `kv`. The row maximum is the result of accumulating over all `kv` — which brings us to `carried_dims`.

**`carried_dims`** declares which dimensions the op reduces over, requiring a serial accumulation loop. For example:

- `RowMax` with `carried_dims={SEQ_KV}`: computing the row maximum requires seeing all values along the `SEQ_KV` dimension. The renderer emits `m = -inf` before the loop over `SEQ_KV`, then `m = max(m, tile_max)` inside the loop.
- `MatMul_PV` with `carried_dims={SEQ_KV}`: the output accumulator `out_acc[q]` accumulates partial matrix products `A[q, kv] @ V[kv]` across the `SEQ_KV` dimension. The renderer emits `out_acc = 0.0` before the loop, then `out_acc += weights @ v_tile` inside the loop.
- `MatMul_QK` with `carried_dims={}`: the matrix multiply `Q @ K^T` does reduce over the `HEAD_DIM` dimension, but `HEAD_DIM` is handled inside the CuTile matmul primitive — it is not a `LoopLevel` in the grammar. Only dimensions that appear as explicit `LoopLevel` nodes need to be declared in `carried_dims`.

The distinction is essential for correctness: a loop with `parallel=True` over dimension `D` cannot enclose a `ComputeNode` with `D ∈ carried_dims` — the parallel threads would each see only their tile's data, producing incorrect partial accumulations.

---

## Data Dependency Graph

The data dependency graph (DDG) is a **DAG** that runs alongside the tree structure. It encodes data flow between `ComputeNode`s. The tree encodes loop nesting and kernel boundaries; the DDG encodes what data each operation needs and produces.

Each DDG edge has the form:
```
(producer_id: int, consumer_id: int, tensor: str)
```

The renderer guarantees `producer` executes before `consumer` within an iteration. If both nodes are in the same kernel (share the same root-level `LoopLevel` ancestor), the intermediate tensor may live in shared memory or registers depending on the loop structure. If they are in different kernels, the tensor must be materialised to global memory by the producing kernel and loaded by the consuming kernel.

The DDG is **never modified** by rewrite rules — only the tree structure changes.

### Dependency preservation under rewrites

Every rewrite rule must preserve DDG semantics: after the rewrite, every edge `(producer, consumer, tensor)` must still be satisfiable — either both nodes remain in the same kernel, or the tensor is materialised to global memory between kernels. This is the one correctness invariant that rewrite preconditions enforce.

---

## The Atomic Grammar (Initial State)

The optimizer starts from the *atomic grammar*: the **compute-at-root** state where every `ComputeNode` is in its own top-level kernel with no fusion. This is the maximally-unfused baseline — 8 separate GPU kernel launches, one per operation, with all intermediate tensors passing through global memory.

The loop structure of each kernel is derived mechanically from the op's `output_dims` and `carried_dims`:
- `output_dims` → parallel loops (`parallel=True`), one per dimension
- `carried_dims` → serial loops (`parallel=False`), one per dimension

All loops use `bound=64`.

```
ProgramNode
├── LoopLevel(SEQ_Q,  64, parallel=True)      # kernel 1: MatMul_QK
│   └── LoopLevel(SEQ_KV, 64, parallel=True)
│       └── Compute(MatMul_QK,  out={SEQ_Q,SEQ_KV}, car={})
│
├── LoopLevel(SEQ_Q,  64, parallel=True)      # kernel 2: Scale
│   └── LoopLevel(SEQ_KV, 64, parallel=True)
│       └── Compute(Scale,      out={SEQ_Q,SEQ_KV}, car={})
│
├── LoopLevel(SEQ_Q,  64, parallel=True)      # kernel 3: RowMax
│   └── LoopLevel(SEQ_KV, 64, parallel=False)
│       └── Compute(RowMax,     out={SEQ_Q},         car={SEQ_KV})
│
├── LoopLevel(SEQ_Q,  64, parallel=True)      # kernel 4: Subtract
│   └── LoopLevel(SEQ_KV, 64, parallel=True)
│       └── Compute(Subtract,   out={SEQ_Q,SEQ_KV}, car={})
│
├── LoopLevel(SEQ_Q,  64, parallel=True)      # kernel 5: Exp
│   └── LoopLevel(SEQ_KV, 64, parallel=True)
│       └── Compute(Exp,        out={SEQ_Q,SEQ_KV}, car={})
│
├── LoopLevel(SEQ_Q,  64, parallel=True)      # kernel 6: RowSum
│   └── LoopLevel(SEQ_KV, 64, parallel=False)
│       └── Compute(RowSum,     out={SEQ_Q},         car={SEQ_KV})
│
├── LoopLevel(SEQ_Q,  64, parallel=True)      # kernel 7: Divide
│   └── LoopLevel(SEQ_KV, 64, parallel=True)
│       └── Compute(Divide,     out={SEQ_Q,SEQ_KV}, car={})
│
└── LoopLevel(SEQ_Q,  64, parallel=True)      # kernel 8: MatMul_PV
    └── LoopLevel(SEQ_KV, 64, parallel=False)
        └── Compute(MatMul_PV,  out={SEQ_Q},         car={SEQ_KV})
```

DDG for the attention atomic grammar:

| Producer | Consumer | Tensor |
|---|---|---|
| `MatMul_QK` | `Scale` | `S` |
| `Scale` | `RowMax` | `S_scaled` |
| `Scale` | `Subtract` | `S_scaled` |
| `RowMax` | `Subtract` | `m` |
| `Subtract` | `Exp` | `S_shifted` |
| `Exp` | `RowSum` | `P` |
| `Exp` | `Divide` | `P` |
| `RowSum` | `Divide` | `l` |
| `Divide` | `MatMul_PV` | `A` |

Primary inputs (from global memory): `Q`, `K`, `V`.
Primary output (to global memory): `O`.

---

## Rewrite Rules

Each rewrite rule transforms the tree structure while leaving the DDG unchanged. Rules come in inverse pairs where applicable. Preconditions are stated in terms of data dependency preservation and dim/parallel compatibility with `output_dims`/`carried_dims`.

---

### `SplitLoop(loop, outer_bound=1, inner_bound=None)`

Replaces `loop` with two nested `LoopLevel` nodes over the same dimension. The outer iterates `outer_bound` times; the inner iterates `inner_bound` times. `loop.children` become the children of the inner loop.

**Default:** `outer_bound=1`, `inner_bound=loop.bound` (trivial split; useful as a starting point).

**Inverse:** `MergeLoop`

**Preconditions:**
- `loop.parallel == True`. Splitting a serial loop is forbidden because enclosed `ComputeNode`s with `loop.dim ∈ carried_dims` require the full extent of the dimension within a single continuous loop.
- `outer_bound * inner_bound == loop.bound`
- Both `outer_bound` and `inner_bound` are powers of 2.

**Result:** both child loops inherit `loop.dim` and `parallel=True`.

---

### `MergeLoop(outer, inner)`

Collapses two directly nested `LoopLevel` nodes over the same dimension into one with `bound = outer.bound * inner.bound`.

**Inverse:** `SplitLoop`

**Preconditions:**
- `inner` is the only child of `outer`.
- `outer.dim == inner.dim`.
- `outer.parallel == inner.parallel`.

---

### `LowerCompute(compute, target_loop)`

Moves `compute` from its current position in the tree into `target_loop`, appending it as the last child of `target_loop`. This is the **primary fusion mechanism**: moving computes into the same `LoopLevel` subtree fuses them into the same kernel and can eliminate global memory round-trips.

After removing `compute` from its original location, any `LoopLevel` that becomes empty is pruned. If an entire top-level kernel becomes empty, it is removed from `ProgramNode.children`.

**Inverse:** `RaiseCompute`

**Preconditions:**
- `compute` exists in the grammar tree.
- `target_loop` exists in the grammar tree.
- `compute` is not already a child of `target_loop`.
- If `target_loop.parallel == True`: `target_loop.dim` must NOT be in `compute.carried_dims`. A parallel loop cannot carry accumulation state for this op.
- DDG: all DDG producers of `compute` must be in kernels that execute before or at the same kernel as `target_loop` (i.e., their top-level `LoopLevel` appears no later in `ProgramNode.children` than `target_loop`'s top-level `LoopLevel`).

---

### `RaiseCompute(compute)`

Moves `compute` one level up in the tree — out of its current `LoopLevel` parent and into the parent's parent, placed immediately after the parent in the children list.

**Inverse:** `LowerCompute`

**Preconditions:**
- `compute`'s immediate parent is a `LoopLevel` (not `ProgramNode` — cannot raise a compute that is already at the top-level kernel root; use `LowerCompute` into a different kernel for that).
- `parent.dim` must NOT be in `compute.carried_dims`. Raising removes the enclosing loop over `parent.dim`, which is required for carry-over accumulation.

---

### `Parallelize(loop)`

Sets `loop.parallel = True`. Converts a serial loop into a parallel (GPU-threaded) loop.

**Inverse:** `Serialize`

**Preconditions:**
- No `ComputeNode` inside `loop` has `loop.dim` in its `carried_dims`. A loop that carries accumulation state across its iterations cannot be parallelised — each parallel thread would only see one tile's data.

---

### `Serialize(loop)`

Sets `loop.parallel = False`. Converts a parallel loop into a serial for-loop.

**Inverse:** `Parallelize`

**Preconditions:** none. Making a parallel loop serial is always safe — it is strictly more conservative.

---

### `ReorderLoops(loop_a, loop_b)`

Swaps two sibling `LoopLevel` nodes, changing the order in which the corresponding loops execute.

**Inverse:** `ReorderLoops` (self-inverse)

**Preconditions:**
- `loop_a` and `loop_b` are siblings under the same parent node.
- No `ComputeNode` inside `loop_a` has a DDG dependency (in either direction) on a `ComputeNode` inside `loop_b`.

---

### `HoistLoop(loop)`

Moves `loop` one level upward in the tree, making it the parent of its current parent. Equivalent to loop interchange.

**Inverse:** `SinkLoop`

**Preconditions:**
- `loop`'s current parent is a `LoopLevel` (not `ProgramNode`).
- `loop` is the only child of its parent (makes the operation unambiguous).

---

### `SinkLoop(loop)`

Moves `loop` one level downward into its single `LoopLevel` child. Equivalent to loop interchange.

**Inverse:** `HoistLoop`

**Preconditions:**
- `loop` has exactly one `LoopLevel` child (otherwise the target is ambiguous).
- No DDG dependency from `loop`'s body (outside the child) to a `ComputeNode` inside the child would be violated by the new nesting order.

---

## Summary Table

| Concept | Representation |
|---|---|
| Full program | `ProgramNode` |
| One GPU kernel launch | Direct child `LoopLevel` of `ProgramNode` |
| One tiled loop | `LoopLevel(dim, bound, parallel)` |
| One primitive operation | `ComputeNode(op, output_dims, carried_dims)` |
| Data flow between ops | DDG edge `(producer_id, consumer_id, tensor)` |
| Kernel boundary | Implicit: each top-level `LoopLevel` = one kernel |
| Parallel loop | `LoopLevel.parallel=True` → `ct.bid` / grid dim |
| Serial loop | `LoopLevel.parallel=False` → for-loop in kernel body |
| Accumulation loop | `LoopLevel.parallel=False` + `dim ∈ ComputeNode.carried_dims` |
| Batch / head parallelism | CUDA grid level; not in AST |
| Primary fusion mechanism | `LowerCompute` / `RaiseCompute` |
| Loop tiling | `SplitLoop` / `MergeLoop` |
| Parallelism control | `Parallelize` / `Serialize` |
| Correctness guarantee | Inductive: atomic grammar is correct; every valid rewrite preserves DDG semantics |

---

## Future Work: Cross-Iteration Accumulation and Online Softmax

The current grammar represents cross-iteration accumulation via `carried_dims` on `ComputeNode`. The renderer handles the initialisation (`acc = 0.0`) and accumulation (`acc +=`) pattern mechanically. However, **online softmax** (as used in FlashAttention) requires a mathematically more complex form of cross-iteration state.

### Why online softmax is not just scheduling

FlashAttention resolves the problem of tiled softmax not through scheduling, but through a mathematically different algorithm. It applies a rescaling correction at the end of each `SEQ_KV` tile iteration — adjusting the accumulated partial output `O` by `exp(m_prev - m_new)` to compensate for the updated running maximum. This correction requires `ComputeNode`s that do not exist in the original DDG (`UpdateMax`, `Rescale`, etc.) and changes the semantics of existing nodes.

The DDG topology itself changes. FlashAttention is not a reordering of the original computation — it is a mathematically equivalent but algorithmically distinct computation that requires a separate correctness proof and a new atomic grammar.

### Recommendation

Treat FlashAttention as an unreachable upper bound to compare against rather than a reachable optimum from the current atomic grammar. The grammar still expresses a rich optimization space — arbitrary fusion via `LowerCompute`, tiling of parallel dimensions via `SplitLoop`, loop reordering, and parallelism control. The gap between the atomic grammar (8 separate kernels) and a fully fused non-online attention kernel is substantial and worth exploring.

Supporting online softmax would require introducing a new atomic grammar for the online algorithm with a different DDG topology, and a new type of loop-carried state that the rewrite rules understand — a research contribution beyond the scope of the current implementation.
