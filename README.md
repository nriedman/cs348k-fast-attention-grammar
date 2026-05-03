## Building Fast Fused Attention Kernels with Grammars Designed for Descent

Building on the work in Design for Descent: What Makes a Shape Grammar Easy to Optimize?, I will design a grammar representation of ML kernel code that will “render” (cross-compile) to CuTile. I will deploy the modified descent algorithm for optimizing grammar rewrite rules to implement a fused attention kernel. The loss will be the runtime as measured by deploying a benchmark task on a single GPU. By comparing the partial kernel’s runtime against an extremely simple as the base case and a state-of-the-art fused attention kernel for the GPU architecture as the number of grammars considered increases, I will evaluate the efficacy of my grammar.

