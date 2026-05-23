## Building Fast Fused Attention Kernels with Grammars Designed for Descent

Building on the work in [*Design for Descent: What Makes a Shape Grammar Easy to Optimize?*](https://dl.acm.org/doi/pdf/10.1145/3757377.3764004), I will design a grammar representation of ML kernel code that will “render” (cross-compile) to CuTile. I will deploy the modified descent algorithm for optimizing grammar rewrite rules to implement a performant attention kernel. The loss will be the runtime as measured by deploying a benchmark task on a single GPU. By comparing the partial kernel’s runtime against an extremely simple one as the base case and a state-of-the-art fused attention kernel for the GPU architecture as the number of grammars considered increases, I will evaluate the efficacy of my grammar's design.

The key questions I will be considering are:

1. How can I represent an attention kernel as a grammar? What primitives and rewrite rules can I define that allow me to programmatically define correct attention kernels of varying performance?
2. Building on (1), how can I extend the grammar representation to make it easy to optimize? Are the principles described in the [original paper](https://dl.acm.org/doi/pdf/10.1145/3757377.3764004) applicable to ML kernels, or are there different considerations that apply?

## Methodology

To answer my research questions, I will proceed in three phases:

1. Design a grammar representation of correct attention a kernel.
2. Implement a system that "renders" (cross-compiles) a grammar representation to a `cuTile` program.
3. Build a training pipeline that mutates a given grammar as described in the [original paper](https://dl.acm.org/doi/pdf/10.1145/3757377.3764004), then evaluates it by cross compiling to `cuTile` and running an attention task (loss is the runtime).
4. Evaluate a given rendered kernel by timing how long it takes to compute the output over a set of random inputs.

Once the systems for training a grammar and evaluating an output are in place, I can experiment with different grammars and design philosophies. I can perform ablation studies to understand the effect that design decisions have on how quickly a given grammar can converge to state-of-the-art performance (if at all).

## Grammar Design

### How to Design a cuTile Kernel

The first question I need to answer is: How can I represent an attention kernel as a grammar? What primitive and rewrite rules can I define that allow me to programatically define correct attention kernels of varying performance?

I decided to take a principled approach to this question by first understanding what the programming model is for writing kernels in `cuTile`. As a review, [cuTile](https://docs.nvidia.com/cuda/cutile-python/) is a DSL made by NVIDIA to provide kernel engineers with a "tile code" abstraction. Instead of designing a kernel by manually assigning work to and synchronizing threads in a warp block, the body of a `cuTile` kernel is composed of basic operations over **tiles**. A tile is an immutable tensor that only exists in kernel code, and instead of mutating the memory itself, the programmer specifies when to load the tile from global memory, when to store it back, and what operations to perform on it. The compiler then optimizes the low-level control of threads, synchrony, and async memory access patterns.

Under the Tile abstraction, designing a kernel becomes an exercise in assigning what work gets done to what Tile and when. To understand the types of decisions being made, I wrote a simple two-stage pipeline in `dev/playground.py`. This demonstrates two basic operations which together inform a surprisingly deep undersanding of how to make design decisions in a `cuTile` kernel.

### Loops over Tiles

When a kernel launches, it follows CUDA by assigning work to a 3D grid of blocks, which execute in parallel. Following the tile-first programming model, instead of thinking about threads in warps in grids, a `cuTile` kernel is best conceptualized as a deeply nested for-loop in which every iteration at every level is executed at the same time. The loops iterate over tiles in each dimension of the output, and the body of the kernel computes a single output tile. Thus, a `cuTile` kernel is a massive parallel for-loop over tiles.

> [!NOTE]
> `cuTile` is an expressive DSL. I'm sure there are other ways to think about how work is assigned to the body of a kernel block. Here I share one perspective which I happen to find convincing.

By exploring the two stages of the pipeline, we can better understand how loops over tiles pervade the `cuTile` kernel design process.

The first operation (stage 2 in the pipeline, my apologies) is a simple element-wise addition of two matrices. The kernel launches a grid of blocks, each of which is assigned a tile of the output to compute. The data dependency of the compute operation over the tile reveals what input tiles must be loaded to shared memory. Because the operation is element-wise, a single tile of the output depends on a single, corresponding tile in each input -- these are fetched from global memory before the compute operation. At the end of the kernel, the output tile is stored to global memory. Each block in the grid computes a disjoint tile, together producing the entire output.

To summarize, we trivially identified a correct kernel by first mapping output to input tiles through the compute operation, then issuing loads and stores as required before and after the computation. In pseudocode:

```
Load(input_a_, bid.x, bid.y, TILE_X, TILE_Y)            // load required tiles from input
Load(input_b_, bid.x, bid.y, TILE_X, TILE_Y)

output_ = Compute(input_a, input_b_)

Store(output, bid.x, bid.y, TILE_X, TILE_Y, output_)    // store the result cooperatively
```

This is a basic, generalizable pattern.

The second operation (stage 1 in the pipeline, my apologies) illustrates what happens when the data dependency is non-trivial. In this case, we are performing a matrix multiplication, so a single tile in the output depends on entire rows and columns in the first and second input, respectively. For the trivial example, we load a large tile from each input that spans the entire range of dependent indices. However, for large inputs, this will quickly fill the shared memory on device (an L4 GPU has 48 KB of shared memory by default -- not much!).

This brings us to our first idea on how to design a better kernel: split the `K` dimension into `K / TILE_K` tiles of size `TILE_K`, and compute partial matrix multiplications of the inputs of the more-manageable size of `(TILE_X, TILE_K)` and `(TILE_K, TILE_Y)`. By delegating fine-tune control of threads and warp blocks to the compiler, the programmer is free to make decisions in terms of which dimension to break into tiles, and how big those tiles should be.

Applying this change, we are adding a sequential loop in the body of the grid-parallel kernel:

```
for tk in K / TILE_K:

    Load(input_a_, bid.x + tk, bid.y,      TILE_N, TILE_K)    // load the k'th tile along the row
    Load(input_b_, bid.x,      bid.y + tk, TILE_K, TILE_M)    // load the k'th tile along the column

    output_ = Compute(input_a, input_b_)

    Store(output, bid.x, bid.y, TILE_N, TILE_M, output_)
```

This is all well and good, but there's something missing. Because the compute operation is reducing along a dimension in the inputs that is not present in the output, the same region in the output is being written in each iteration. This is incorrect -- instead of overwriting the partial result, the reduction should be accumulated in a `(TILE_N, TILE_M)`-shaped tensor. Then, after the full output tile is computed, the result can be written back to global memory. Thus, the full kernel looks like:

```
acc_ = cp.zeros((TILE_N, TILE_M))

for tk in K / TILE_K:

    Load(input_a_, bid.x + tk, bid.y,      TILE_N, TILE_K)    // load the k'th tile along the row
    Load(input_b_, bid.x,      bid.y + tk, TILE_K, TILE_M)    // load the k'th tile along the column

    _ = Compute(input_a, input_b_, acc_)                      // compute with ct.mma to accumulate into acc_

Store(output, bid.x, bid.y, TILE_N, TILE_M, acc_)             // store complete tile from acc_
```

This illustrates a key insight: just as the `@ct.kernel` is analogous to a parallel for-loop, computation in the body of a kernel can be further sub-divided into tiles and iterated over sequentially (in reality, the JIT compiler will apply heavy optimizations to these "sequential" for-loops, but the mental model is the same). When this sub-tiling is applied, if the iterations all write to the same region in the output (which happens when the sub-tiling dimension does not appear in the output), then the computation is a reduction and we need to accumulate partial results. Otherwise, we're free to sub-tile at any time to reduce shared memory pressure.

### What About Loads and Stores?

I did something slightly deceitful in the previous example. When transitioning to using an accumulator, I raised the terminal `Store` outside the bounds of the loop and into a higher scope. Why was that allowed? It was, but don't take my word for it.

A load or a store can be hoisted up out of a loop or sunk into a loop. By changing the level at which the memory operation is initiated, the characteristics of memory traffic between global and shared memory change. Making decisions about when to materialize a tile is an extremely important way to optimize performance: sometimes it's better to load in one big buffer, then compute small sections of it in the body of the sequential loop. Other times, it's better to load many smaller tiles. These tradeoffs are captured in the basic question: **At what scope should I initiate the load or store?**

The level of the load/store determines its memory footprint in shared memory. The memory footprint is a function of data dependency: in a sequential loop like the matrix multiplication example above, hoisting the load would require loading in the entire row once again, since the load offset in the loop depends on the iteration of the loop. The store, on the other hand, could be hoisted without changing its footprint because the offset in the destination did not depend on the iteration index.

The memory footprint and data dependency, respectively, enforce upper and lower bounds on where a memory operation can be placed. The higher the scope, the larger the potential memory footprint of a load. Shared memory is finite, and so a load could fail if it attempts to loard more data than is available. The lower the loop placement, the narrower the scope that the memory operation is available in. If a Load is sunk lower than a compute node that depends on it, the compute node has no initialized memory to read from.

Overall, the case study in `dev/playground.py` illustrates key concepts about a) what the conceptual components of a `cuTile` grammar are; and b) how design decisions can be framed as mechanically adjusting the structure and parameters of those components. Let's formalize this.

### The Resulting Grammar

| **Node** | **Description** | **Parameters** | **Constraints** |
| --- | --- | --- | --- |
| **Program** | The root of the AST. | None | Must have no ancestor. |
| **ParallelLoop** | A single `ct.kernel` that launches a grid of parallel blocks. | tile_sizes (the size of tile in each dimension of the _output_) | tile_sizes must all be powers of two and divide into the corresponding dimension. When there are more than three tile dimensions, the earlier ones are compressed along the first grid dimension until a 3D grid can be defined. The grid must be wellformed (dims < 65,535). A **ParallelLoop** must be a direct child of **Program** and must have at least one child. |
| **SequentialLoop** | A sub-tiling that iterates sequentially over one dimension inside a **ParallelLoop**. | dim (the dimension to tile over), tile_size (the size of the tile) | tile_size must be a power of two and divide evenly into the sub-tiled dimension. **SequentialLoop** can only appear as an ancestor of a **ParallelLoop**. |
| **Load** | Transfer a tile from global memory to shared memory. | tensor, offset, tile_shape | There must be room in shared memory for the new tensor. Loads are leaf nodes. The **Load** must exist at or above the scope of the highest node that reads from it. |
| **Store** | Transfer a tile from shared memory to global memory. | dst_tensor, offset, tile_shape, src_tensor | Stores are leaf nodes. Cannot be evaluated before a **Load** for each input has been evaluated in the current scope. |
| **Compute** | Perform an operation on the given inputs, and optionally aggregate in a given accumulator. | op (the operation to perform), accumulate (Bool, whether or not to accumulate) | Computes are leaf nodes. |

The nodes of the grammar are composed to form a tree. A the top is the **Program**. This includes host-side initialization and orchestration boilderplate. The children of the **Program** are **ParallelLoop**s, sorted and evaluated in typographical order by the data dependency graph. Each **ParallelLoop** is a separate kernel launch, and the tile dimensions and sizes are parameters that the optimization stage will modify.

**ParallelLoop**s cannot be childless. Inside the body of the kernel are sub-tiling **SequentialLoop**s, **Load**s, **Store**s, and **Compute** nodes.

We define an _atomic grammar_ as the naive stating point. Every compute stage is reduced to a single, simple operation. Each of these is wrapped in a **ParallelLoop**, triggering a separate kernel launch. If necessary, a **SequentialLoop** with an accumulator is applied if the naive **Load** requires more memory than is available in shared memory.

Now that we have an atomic grammar, we need to talk about how to rewrite them.

> [!Note]
>Through investigating the ins and outs of how this grammar will behave, it became apparent that data dependency is important to track. For example, the memory footprint of a `Load` needs to be derived from what indices in the inputs a given output tile depends on. To derive this dependency, we can pull a page from [Halide's](https://people.csail.mit.edu/jrk/halide-pldi13.pdf) handbook and define a simple declarative language that specifies a) the stages of a pipeline (which are each represented as a **ParallelLoop** in the naive atomic grammar); and b) the exact data dependency between tiles of the outputs and inputs. After declaring the stages and dependencies, the atomic grammar can be trivially generated, seeding the auto-tuning search.

### Rewrite Rules

Based on the case study explored above, we can write down a few general rewrite rules:

| **Rule** | **Input(s)** | **Effect** | **Constraint(s)** |
| --- | --- | --- | --- |
| **Hoist** | _memop_: **Load** or **Store**, _loop_: **SequentialLoop** | Lift _memop_ out of _loop_'s scope. The tile shape is expanded to have capacity for the entire memory footprint defined by the output tiles computed over all iterations of _loop_, if applicable. A **Load** is placed before _loop_ in the new scope, and a **Store** is placed afterwards. | Sum of all buffers implied by _memop_ must fit in shared memory. _loop_ must be an immediate parent of _memop_ in the grammar AST. The body of _loop_ cannot be empty after this operation. |
| **Sink** | _memop_: **Load** or **Store**, _loop_: **SequentialLoop** | The inverse of **Hoist**. Lowers _memop_ into the body of _loop_, constricting _memop_'s shape as applicable by the narrowed data dependency. _memop_ is placed at either the beginning or the end of the loop body (whichever is closer to where it was before the sink). | A load must be at or above the loop level of the highest compute node that reads from it. _loop_ must be in the same loop scope as _memop_ (they share a parent in the AST), and they must be directly adjacent. |
| **Subtile** | _dim_: **Int**, _comp_: **Computation** | Break up _comp_ into tiles along _dim_ by wrapping the **Computation** in a **SequentialLoop**. The shape of the inputs to _comp_ are changed to reflect the new tiling if the input has dimension _dim_. Initially, the tile size of the new dimension is set to the size of _dim_ so that the loop has only one iteration. If the output region doesn't change across loop iterations, set the _accumulate_ flag in **Compute** and insert accumulator boilderplate around the new loop. | Can only be applied to a **Compute** node (the body of the new loop is just that operation). _dim_ must be a dimension that exists in the scope of the enclosing loop. There must be space in shared memory for any necessary accumulator. (Optional) _dim_ must appear in at least one input or ouptut shape. |
| **Unwrap** | _loop_: **SequentialLoop** | Remoes a **SequentialLoop**, promoting its contents back to the enclosing scope. | The change must be trivial, so _loop_ must only contain a single **Compute** node in its body, and _loop_'s tile size must be equal to its dimension size s.t. there is only one iteration of the loop. |
| **InterchangeLoop** | _inner_: **SequentialLoop**, _outer_: **SequentialLoop** | Swaps the nested _inner_ and _outer_ loops so that _inner_ replaces _outer_, and vice versa. | The two **SequentialLoops** must only contain a single **Compute** node in the body of _inner_. |
| **Reorder** | _first_: any **Node**, _second_: any **Node** | Swap the exectution order of two adjacent **Node**s in the same scope. | There must be no dependency between the nodes, or any sub-nodes in the case of loops. Shared memory must be able to accommodate the change. |
| **Merge** | _a_: **SequentialLoop** or **ParallelLoop**, _b_: **SequantialLoop** or **ParallelLoop** | Combine two adjacent loops of the same kind into one loop. The body of the resulting loop will be the body of _a_ concatenated with the body of _b_ in relative execution order. | The loops must be adjacent. The loops must have the same tile dimensions and sizes. For parallel loops, the output of _a_ must be consumed by _b_ **and no other Node**. _b_ must be the sole consumer of the product of _a_. |
| **EliminateRoundTrip** | _send_: **Store**, _receive_: **Load** | When _receive_ immediately follows _send_ over the same tile, remove both nodes. | _send_ must immediately precede _receive_. They must be memory operations over the same tensor, at the same offset, with the same tile shape. |
 
These rewrite rules are designed to guarantee correctness by construction -- no rewrite rule changes _what_ is computed, only _what tile_ work is assigned to and _when_.

Together, these rules capture 80% of the design decisions that go in to writing a fast kernel. I designed them to be expressive enough to give an optimizer ample room to explore, but narrow and orthogonal enough to not dilute the search space with redundant actions. They are well on their way to obeying the principles laid out in [*Design for Descent: What Makes a Shape Grammar Easy to Optimize?*](https://dl.acm.org/doi/pdf/10.1145/3757377.3764004), and form a principled foundation on which to build the rest of the project.

> [!Note]
> From my work so far, it is clear to see that kernels derived from my grammars will not reach the level of _FlashAttention2_. The authors of _FlastAttention2_ leveraged a wonderful algebraic property of exponentials to compute Softmax online, removing the need to write the scores to global memory. Unfortunately, this technique requires a) changing what the math / algorithm does; and b) supporting inter-iteration dependencies (e.g. the computation at _k + 1_ is a function of the computation at _k_). This is a whole can of worms that (while extremely interesting) fall well within the Future Work section for the purposes of this project.

### Open Questions

A few questions remain. For one, I would like to design an inverse to the **Merge** rewrite rule. **Merge** is loop fusion -- it would be nice to support loop fission with a **Split** rule. This would both allow me to fully test **Reversability** as a design principle and also give the optimizer more tools to work with

For another, I still need to build the "renderer" (compiler) that takes a grammar as defined above an outputs a cuTile kernel. Then, I need to build the optimization pipeline that takes an atomic grammar and applies Stochastic Rewrite Descent to improve performance. Now that the grammar is well defined, the next steps are clear and should be straight forward.

Finally, while it's clear that attention is more complicated than the toy pipeline in `dev/playground.py`, it's not by much. By first stepping through a simple case study, I was able to methodically identify design principles and programming models that informed a rigorous, polished grammar. Attention is the same thing, just with 8 stages instead of 2.

## Evaluation

To evaluate the product of a grammar, I wrote a benchmark script in `benchmark.py`. The benchmark records how long it takes to compute the forward pass of attention over a set of random inputs. The dimensions can be set to arbitrary values, but in the evaluation we will use the same settings as described in [Section 4.1 of FlashAttention2](https://arxiv.org/pdf/2307.08691). I will start by evauating the naive base case that constitutes the atomic initial state of the grammar (no fusion, no nesting loops, no tiling, reading/writing from global memory), as well as measuring the state of the art for the GPU I have access to (in this case, [FlashAttention2](https://arxiv.org/pdf/2307.08691), which was developed for the A100 GPU).

The evaluation pipeline is ready for deployment on a Google Cloud Compute GPU-equipped VM. As an initial experiment, I evaluated a naive baseline and FlashAttention2 for sequenc length 2048. They achieved:

- **Naive:** `4.25 TFLOP/s`
- **FlashAttention2:** `50.62 TFLOP/s`

The naive and SOTA performance will act as floor and ceiling, respectively, for what a grammar can achieve. By evaluating the grammar at intervals throughout the training cycle, I can track the gains in performance over time. Comparing different training trajectories as they (hopefully) trace from the naive to the SOTA will allow me to measure the effect that grammar design decisions have on how quickly a performant kernel can be found.
