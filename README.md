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

## Evaluation

To evaluate the product of a grammar, I wrote a benchmark script in `benchmark.py`. The benchmark records how long it takes to compute the forward pass of attention over a set of random inputs. The dimensions can be set to arbitrary values, but in the evaluation we will use the same settings as described in [Section 4.1 of FlashAttention2](https://arxiv.org/pdf/2307.08691). I will start by evauating the naive base case that constitutes the atomic initial state of the grammar (no fusion, no nesting loops, no tiling, reading/writing from global memory), as well as measuring the state of the art for the GPU I have access to (in this case, [FlashAttention2](https://arxiv.org/pdf/2307.08691), which was developed for the A100 GPU).

The evaluation pipeline is ready for deployment on a Google Cloud Compute GPU-equipped VM. As an initial experiment, I evaluated a naive baseline and FlashAttention2 for sequenc length 2048. They achieved:

- **Naive:** `4.25 TFLOP/s`
- **FlashAttention2:** `50.62 TFLOP/s`

The naive and SOTA performance will act as floor and ceiling, respectively, for what a grammar can achieve. By evaluating the grammar at intervals throughout the training cycle, I can track the gains in performance over time. Comparing different training trajectories as they (hopefully) trace from the naive to the SOTA will allow me to measure the effect that grammar design decisions have on how quickly a performant kernel can be found.
