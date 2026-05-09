## Building Fast Fused Attention Kernels with Grammars Designed for Descent

Building on the work in [*Design for Descent: What Makes a Shape Grammar Easy to Optimize?*](https://dl.acm.org/doi/pdf/10.1145/3757377.3764004), I will design a grammar representation of ML kernel code that will “render” (cross-compile) to CuTile. I will deploy the modified descent algorithm for optimizing grammar rewrite rules to implement a performant attention kernel. The loss will be the runtime as measured by deploying a benchmark task on a single GPU. By comparing the partial kernel’s runtime against an extremely simple one as the base case and a state-of-the-art fused attention kernel for the GPU architecture as the number of grammars considered increases, I will evaluate the efficacy of my grammar's design.

The key questions I will be considering are:

1. How can I represent an attention kernel as a grammar? What primitives and rewrite rules can I define that allow me to programmatically define correct attention kernels of varying performance?
2. Building on (1), how can I extend the grammar representation to make it easy to optimize? Are the principles described in the [original paper](https://dl.acm.org/doi/pdf/10.1145/3757377.3764004) applicable to ML kernels, or are there different considerations that apply?

## Methodology

To answer my research questions, I will proceed in three phases:

1. Design a grammar representation of correct attention a kernel.
2. Implement a system that translates a grammar representation to a `cuTile` program.
3. Build a training pipeline that mutates a given grammar as described in the [original paper](https://dl.acm.org/doi/pdf/10.1145/3757377.3764004), then evaluates it by cross compiling to `cuTile` and running an attention task (loss is the runtime).
4. Evaluate

