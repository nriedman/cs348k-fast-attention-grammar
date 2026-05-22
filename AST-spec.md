# Kernel Grammar AST Specification

## Overview

This document specifies the Abstract Syntax Tree (AST) used to represent GPU kernel programs in the grammar-based optimization framework. The framework is inspired by "Design for Descent" (Kodnongbua et al., 2025), which applies gradient-descent-style optimization to shape grammars. Here, the "shape" is a GPU kernel, the "render" function compiles and benchmarks the kernel, and the "objective" is kernel runtime on a fixed benchmark.

The AST is **kernel-agnostic**: the node types and rewrite rules place no assumptions on what computation is being performed. The kernel being optimized is determined entirely by the atomic grammar supplied as the initial state. The first application of this framework targets an attention kernel (see The Atomic Grammar below), but the framework is designed to generalize.

The AST has two components:
- **Structure**: a tree of nodes representing the kernel's loop nest and fusion topology
- **Parameters**: scalar values attached to nodes, optimized within a fixed structure (analogous to continuous parameters in the paper)

Rewrite rules change the structure; parameter updates change parameter values within a fixed structure. These two phases alternate during optimization.

### Correctness by Construction

Correctness is guaranteed inductively, not by global inspection:

1. The atomic grammar is correct by definition — it is the naive, obviously-correct implementation of the target computation.
2. Every rewrite rule is **semantics-preserving by design** — it reorganizes the schedule (loop order, fusion, tiling) without changing what is computed, provided its data-dependency preconditions are satisfied.
3. Therefore every state reachable from the atomic grammar by valid rewrites is also correct.

No part of the system needs to check whether the set of `Compute` nodes is "complete" or "correct." Correctness is a property of the rewrite rules, not of the AST validator. The only validity constraints that matter at runtime are those that prevent a rewrite from violating a data dependency.

---

## Node Types

### 1. `ProgramNode` (root)

The root of the AST. Contains one or more `KernelScope` children, each compiled into a separate CUDA kernel launch, executed in order. When there is only one `KernelScope`, the entire computation is fused into a single kernel.

**Children:** one or more `KernelScope` nodes, in execution order.

**Parameters:** none.

---

### 2. `KernelScope`

Represents a single GPU kernel launch. Its subtree defines the loop nest and the computations performed within that kernel. The renderer maps this node to a CuTile kernel function.

**Children:** one or more `LoopLevel` nodes (possibly nested), with `Compute` nodes at the leaves.

**Parameters:** none. Grid and block dimensions are derived by the renderer from the enclosed `LoopLevel` structure.

**Validity:**
- Must contain at least one `Compute` node.
- All data dependencies among enclosed `Compute` nodes must be satisfiable within the scope — a `Compute` node may not consume the output of a node in a different `KernelScope` unless that output has been materialized to global memory by the earlier scope.
- The shared memory footprint implied by the fusion structure must not exceed the hardware limit (48KB default; 96KB with dynamic shared memory on Ampere+). The renderer computes this statically; validity checking uses the same estimate.

---

### 3. `LoopLevel`

Represents a single tiled loop over one logical dimension of the computation. Nested `LoopLevel` nodes represent nested loops.

**Children:** one or more `LoopLevel` or `Compute` nodes.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `dim` | string | The logical dimension this loop iterates over (e.g. `SEQ_Q`, `SEQ_KV`, `HEAD_DIM`). The set of valid dimensions is defined by the atomic grammar, not by the framework. |
| `tile_size` | int | Number of elements processed per iteration along `dim`. Must be a power of 2 and evenly divide the total size of `dim`. |

**Tags:**

| Tag | Values | Description |
|---|---|---|
| `loop_type` | `parallel` \| `reduction` | Whether iterations of this loop are independent (`parallel`) or must be serialized because enclosed `Compute` nodes reduce over this dimension (`reduction`). Set at node creation and kept consistent during rewrites. The renderer uses this tag to select the correct CuTile primitives: parallel loops map to block/warp tiling APIs; reduction loops map to sequential iteration. |

**Validity:**
- `tile_size` must be a power of 2.
- `tile_size` must evenly divide the total size of `dim`.
- A `reduction` loop may not appear as an ancestor of a `parallel` loop over a dimension it reduces over.

---

### 4. `Compute`

A leaf node representing a single primitive operation. `Compute` nodes are always leaves — they have no `LoopLevel` or `Compute` children. Their inputs and outputs are defined by the data dependency graph, not by the tree structure.

**Children:** none.

**Parameters:** none at present. Future work may add parameters such as accumulation precision or output dtype.

**Tags:**

| Tag | Type | Description |
|---|---|---|
| `op` | string | Identifies the primitive operation (e.g. `MatMul`, `RowMax`, `Exp`). Defined by the atomic grammar. The framework does not interpret or validate `op` values — the renderer is responsible for mapping each `op` to the correct CuTile primitive. |

---

## Data Dependency Graph

The data dependency graph (DDG) is a **DAG** that runs alongside the tree structure. It encodes data flow between `Compute` nodes. The tree structure encodes loop nesting and fusion boundaries; the DDG encodes what data each operation needs and produces. These are separate concerns.

Each edge in the DDG has the form:
```
(producer: Compute, consumer: Compute, tensor: str)
```
The output tensor of `producer` is consumed by `consumer`. The renderer guarantees `producer` executes before `consumer` within an iteration. If both nodes are in the same `KernelScope`, the intermediate tensor may live in shared memory or registers depending on the loop structure. If they are in different `KernelScope`s, the tensor must be materialized to global memory by the first scope and loaded by the second.

The DDG is always a DAG — it contains no cycles. Cross-iteration accumulation (loop-carried state) is explicitly out of scope for this version of the grammar; see Future Work.

### Dependency preservation under rewrites

Every rewrite rule must preserve the DDG in the following sense: after the rewrite, every edge `(producer, consumer, tensor)` must still be satisfiable — either both nodes remain in the same `KernelScope`, or the tensor is materialized to global memory between scopes. This is the one correctness invariant that rewrite preconditions are responsible for enforcing. The DDG itself does not change under any rewrite; only the tree structure changes.

---

## The Atomic Grammar (Initial State)

The optimizer starts from the *atomic grammar*: the simplest correct implementation of the target computation, with no fusion and no non-trivial tiling. All intermediate tensors pass through global memory. This state is correct by construction and serves as the fixed starting point from which all rewrites proceed.

**The atomic grammar is supplied externally** — it is not defined by the framework. For the attention kernel, the atomic grammar groups the computation into three unfused kernel scopes. Batch and head dimensions are handled at the CUDA grid level (one block per head per batch element) and do not appear as `LoopLevel` nodes.

```
ProgramNode
├── KernelScope                             # kernel 1: QK matmul + scale
│   ├── LoopLevel(SEQ_Q,  tile_size=64, parallel)
│   │   └── LoopLevel(SEQ_KV, tile_size=64, parallel)
│   │       ├── Compute(MatMul_QK)          # S = Q @ K^T
│   │       └── Compute(Scale)              # S = S / sqrt(d)
│
├── KernelScope                             # kernel 2: softmax
│   └── LoopLevel(SEQ_Q,  tile_size=64, parallel)
│       ├── Compute(RowMax)                 # m = max(S, dim=-1)
│       ├── Compute(Subtract)               # S = S - m
│       ├── Compute(Exp)                    # P = exp(S)
│       ├── Compute(RowSum)                 # l = sum(P, dim=-1)
│       └── Compute(Divide)                 # A = P / l
│
└── KernelScope                             # kernel 3: PV matmul
    └── LoopLevel(SEQ_Q,  tile_size=64, parallel)
        └── LoopLevel(SEQ_KV, tile_size=64, parallel)
            └── Compute(MatMul_PV)          # O = A @ V
```

The DDG for the attention atomic grammar:

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

Each rewrite rule transforms the tree structure while leaving the DDG unchanged. Rules come in inverse pairs (satisfying the Reversibility property from the paper). Preconditions are stated purely in terms of data dependency preservation — no other validity constraints are imposed.

---

### `SplitLoop(loop: LoopLevel, outer_tile: int, inner_tile: int)`
Replaces `loop` with two nested `LoopLevel` nodes over the same dimension, where the outer iterates in steps of `outer_tile * inner_tile` and the inner iterates in steps of `inner_tile`.

**Inverse:** `MergeLoop`

**Preconditions:**
- `loop.loop_type == parallel`. Splitting a `reduction` loop is forbidden because enclosed `Compute` nodes that reduce over this dimension require access to the full extent of the dimension within a single iteration. Splitting the loop would cause each iteration to see only a tile, producing an incorrect partial reduction. See Future Work for a finer-grained treatment of this constraint.
- `outer_tile * inner_tile == loop.tile_size`
- Both `outer_tile` and `inner_tile` are powers of 2.

**Parameter initialization:** both child loops inherit `loop.dim` and `loop.loop_type`. `outer.tile_size = outer_tile`, `inner.tile_size = inner_tile`.

---

### `MergeLoop(outer: LoopLevel, inner: LoopLevel)`
Collapses two directly nested `LoopLevel` nodes over the same dimension into one with `tile_size = outer.tile_size * inner.tile_size`.

**Inverse:** `SplitLoop`

**Preconditions:**
- `inner` is the only child of `outer`.
- `outer.dim == inner.dim`.
- `outer.loop_type == inner.loop_type`.
- The merged `tile_size` evenly divides the total size of `dim`.

---

### `FuseKernels(scopeA: KernelScope, scopeB: KernelScope)`
Merges two adjacent `KernelScope` siblings into one, eliminating the global memory materialization of tensors produced by `scopeA` and consumed by `scopeB`.

**Inverse:** `UnfuseKernels`

**Preconditions:**
- `scopeA` immediately precedes `scopeB` under `ProgramNode`.
- No tensor produced by `scopeA` and consumed by `scopeB` is also consumed by any other `KernelScope` — otherwise removing the global materialization would break that consumer.
- All data dependencies among the `Compute` nodes of the fused scope are satisfiable within a single scope (i.e. the DDG has no cycle that would require a scope boundary).
- The combined shared memory footprint does not exceed the hardware limit.

**Parameter initialization:** the loop structures of both scopes are preserved inside the fused scope. The renderer inserts synchronization barriers as needed.

---

### `UnfuseKernels(scope: KernelScope, splitPoint: Compute)`
Splits `scope` into two at `splitPoint`: all `Compute` nodes that `splitPoint` transitively depends on remain in the first scope; `splitPoint` and its transitive dependents move to the second scope.

**Inverse:** `FuseKernels`

**Preconditions:**
- The partition induced by `splitPoint` is valid: all inputs to the second scope are produced by the first scope (no dependency is left unsatisfied across the new boundary).
- Both resulting scopes contain at least one `Compute` node.

---

### `ReorderLoops(loopA: LoopLevel, loopB: LoopLevel)`
Swaps two sibling `LoopLevel` nodes, changing the order of the corresponding loops.

**Inverse:** `ReorderLoops` (self-inverse)

**Preconditions:**
- `loopA` and `loopB` are siblings under the same parent node.
- No `Compute` node inside `loopA` has a DDG dependency (in either direction) on a `Compute` node inside `loopB`.

---

### `HoistLoop(loop: LoopLevel)`
Moves `loop` one level upward in the tree, making it the parent of its current parent.

**Inverse:** `SinkLoop`

**Preconditions:**
- `loop`'s current parent is a `LoopLevel` (not a `KernelScope`).
- No DDG dependency requires any `Compute` node in the parent loop (but outside `loop`) to execute before a `Compute` node inside `loop`, in a way that would be violated by the new loop order.

---

### `SinkLoop(loop: LoopLevel)`
Moves `loop` one level downward in the tree, making it a child of one of its current children.

**Inverse:** `HoistLoop`

**Preconditions:**
- `loop` has exactly one `LoopLevel` child (the target child is otherwise ambiguous).
- No DDG dependency requires `loop` to execute before its child in a way that would be violated by the new nesting order.

---

## Summary Table

| Concept | Representation |
|---|---|
| Full program | `ProgramNode` |
| One GPU kernel launch | `KernelScope` |
| One tiled loop | `LoopLevel(dim, tile_size, loop_type)` |
| One primitive operation | `Compute(op)` |
| Data flow between ops | Edge in DDG `(producer, consumer, tensor)` |
| Batch / head parallelism | CUDA grid level; not in AST |
| Shared memory allocation | Derived by renderer from fusion structure |
| Splitting reduction loops | Forbidden; `SplitLoop` requires `loop_type == parallel` |
| Cross-iteration accumulation | Out of scope; see Future Work |
| Correctness guarantee | Inductive: atomic grammar is correct; every valid rewrite preserves DDG semantics |

---

## Future Work: Cross-Iteration Accumulation

The current grammar cannot represent kernels that accumulate state across loop iterations — for example, FlashAttention's online softmax, which maintains a running max `m` and running sum `l` across tiles of the `SEQ_KV` dimension, avoiding full materialization of the attention score matrix.

Supporting this would unlock significantly better memory-bandwidth utilization and is the primary optimization that separates FlashAttention from a naively fused kernel.

### What it requires

Each `Compute` node would need to declare, in addition to its standard DDG inputs, a set of *carried inputs*: tensors that are initialized before a reduction loop begins and updated at the end of each iteration. For `RowMax`, the carried input is `m` (initialized to `-inf`); for `RowSum`, it is `l` (initialized to `0.0`); for `MatMul_PV`, it is `O` (initialized to `0.0`).

Critically, carried inputs are semantically distinct from standard DDG inputs. A standard input is fresh data fetched or computed *this iteration* — for `RowMax`, this is the current tile of `S_scaled`. A carried input is register-resident state from the *previous iteration* — for `RowMax`, this is the running max accumulated so far. The operation `RowMax` computes `m_new = max(m_prev, rowmax(S_scaled_tile))`, combining both.

### Why it is not in the DDG

A carried input creates a cycle: `RowMax` both produces and consumes `m`, making the dependency graph non-acyclic if represented as a standard edge. The DDG must remain a DAG for topological ordering (and therefore correct code generation) to be well-defined. Carried inputs must therefore be declared separately from the DDG, as part of the op registry, and wired by the renderer independently of the standard dependency analysis.

### Why it complicates the rewrite rules

The rewrite rules as currently defined only enforce DDG preservation. This is sufficient for correctness when there are no carried inputs. With carried inputs, an additional invariant is required: for each `Compute` node `N` with a carried input over dimension `d`, every rewrite must preserve the property that `N` is enclosed by a loop over `d` that covers the full extent of `d` without interruption by a `KernelScope` boundary. Without this, a rewrite such as `UnfuseKernels` could split a scope in a way that `N` only sees a subset of the tiles — producing an incorrect partial accumulation that the DDG preservation check would not catch.

Enforcing this invariant requires each rewrite rule to be aware of which `Compute` nodes have carried inputs and over which dimensions, adding non-trivial precondition logic to `UnfuseKernels`, `HoistLoop`, and `SinkLoop` in particular.

### Carry-over and tiling reduction loops

The current grammar's blanket prohibition on splitting `reduction` loops is conservative but necessary. The underlying problem is that a `Compute` node which reduces over a dimension — such as `RowMax` over `SEQ_KV` — must see the full extent of that dimension to produce a correct result. Splitting the loop gives it only a tile per iteration, which is incorrect.

The natural instinct is to fix this with carry-over: `RowMax` accumulates a running max across tiles, carrying it forward each iteration. This works for `RowMax` in isolation. But it does not work for the nodes that consume `RowMax`'s output within the same iteration. `Subtract` needs the *global* row max to correctly shift the scores before `Exp` — but with a tiled loop, `RowMax` has only seen one tile when `Subtract` runs. The partial max is wrong, making `Subtract`, `Exp`, `RowSum`, `Divide`, and `MatMul_PV` all produce incorrect intermediate values within each iteration.

FlashAttention resolves this not through scheduling, but through a mathematically different algorithm. It applies a rescaling correction at the end of each iteration — adjusting the accumulated partial output `O` by `exp(m_prev - m_new)` to compensate for the updated running max. This correction requires new `Compute` nodes that do not exist in the original DDG, and changes the semantics of several existing nodes. The DDG topology itself changes. FlashAttention is not a reordering of the original computation — it is a mathematically equivalent but algorithmically distinct computation that requires a separate correctness proof.

This means carry-over cannot be added as a pure scheduling extension to the existing grammar. Supporting FlashAttention-style tiled softmax would require introducing a new atomic grammar for the online algorithm, with a different set of `Compute` nodes and a different DDG, and establishing separately that this new atomic grammar computes the same function as the original. It is a research contribution in its own right, well beyond the scope of a first implementation.

### Recommendation

Enforce the `loop_type == parallel` precondition on `SplitLoop`, omit carry-over entirely, and treat FlashAttention as an unreachable upper bound to compare against rather than a reachable optimum. The grammar still expresses a rich optimization space — arbitrary fusion, tiling of parallel dimensions, and loop reordering — and the gap between the atomic grammar and a fully fused non-online kernel is already substantial and worth exploring.