"""
benchmark.py — Attention Kernel Benchmark
==========================================
Measures the runtime of a single attention kernel on a standardised task.

Usage
-----
    python benchmark.py \
        --kernel path/to/kernel.py \
        --kernel-fn attention \
        [--batch 8] [--heads 16] [--seq 2048] [--dim 64] \
        [--warmup 50] [--iters 200] \
        [--dtype fp16|bf16|fp32] \
        [--label my_kernel] \
        [--output results/my_kernel.json]

Kernel interface
----------------
The file passed to --kernel must expose a callable with the signature:
    fn(q, k, v) -> out
where q, k, v, and out are all (B, H, S, D) float CUDA tensors.
"""

import argparse
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Callable

import torch

DTYPE_MAP = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


def load_kernel(path: str, fn_name: str) -> Callable:
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Kernel file not found: {p}")
    spec = importlib.util.spec_from_file_location("_kernel", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    if not hasattr(mod, fn_name):
        raise AttributeError(f"'{p}' has no attribute '{fn_name}'")
    return getattr(mod, fn_name)


def benchmark(fn: Callable, q, k, v, warmup: int, iters: int) -> dict:
    # Warmup
    for _ in range(warmup):
        fn(q, k, v)
    torch.cuda.synchronize()

    # GPU-event timing
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    start_ev.record()
    for _ in range(iters):
        fn(q, k, v)
    end_ev.record()
    torch.cuda.synchronize()
    mean_gpu_ms = start_ev.elapsed_time(end_ev) / iters

    # Per-iter wall-clock distribution
    latencies = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn(q, k, v)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1e3)

    return {
        "mean_gpu_ms":    mean_gpu_ms,
        "mean_wall_ms":   statistics.mean(latencies),
        "median_wall_ms": statistics.median(latencies),
        "stdev_wall_ms":  statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
        "min_wall_ms":    min(latencies),
        "max_wall_ms":    max(latencies),
        "iters":          iters,
    }


def parse_args():
    p = argparse.ArgumentParser(description="Attention kernel benchmark")
    p.add_argument("--kernel",    required=True, help="Path to kernel Python file")
    p.add_argument("--kernel-fn", default="attention", help="Function name in kernel file")
    p.add_argument("--label",     default=None, help="Human-readable name for this run")
    p.add_argument("--batch",     type=int, default=8)
    p.add_argument("--heads",     type=int, default=16)
    p.add_argument("--seq",       type=int, default=2048)
    p.add_argument("--dim",       type=int, default=64)
    p.add_argument("--warmup",    type=int, default=50)
    p.add_argument("--iters",     type=int, default=200)
    p.add_argument("--dtype",     choices=["fp16", "bf16", "fp32"], default="fp16")
    p.add_argument("--output",    default=None, help="Path to write JSON results")
    return p.parse_args()


def main():
    args = parse_args()
    dtype = DTYPE_MAP[args.dtype]

    if not torch.cuda.is_available():
        print("[ERROR] CUDA not available.", file=sys.stderr)
        sys.exit(1)

    device   = torch.cuda.current_device()
    dev_name = torch.cuda.get_device_name(device)
    label    = args.label or Path(args.kernel).stem

    print(f"[bench] Device : {dev_name}")
    print(f"[bench] Kernel : {label}  ({args.kernel}::{args.kernel_fn})")
    print(f"[bench] Config : B={args.batch} H={args.heads} S={args.seq} D={args.dim} dtype={args.dtype}")

    try:
        fn = load_kernel(args.kernel, args.kernel_fn)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    shape = (args.batch, args.heads, args.seq, args.dim)
    q = torch.randn(shape, dtype=dtype, device="cuda")
    k = torch.randn(shape, dtype=dtype, device="cuda")
    v = torch.randn(shape, dtype=dtype, device="cuda")

    stats = benchmark(fn, q, k, v, warmup=args.warmup, iters=args.iters)

    # FLOP count: 4·B·H·S²·D  (QKᵀ + AV, both matmuls)
    flops = 4.0 * args.batch * args.heads * args.seq ** 2 * args.dim
    stats["tflops"] = flops / (stats["mean_gpu_ms"] * 1e-3) / 1e12

    print(f"[bench] mean GPU : {stats['mean_gpu_ms']:.3f} ms  |  {stats['tflops']:.2f} TFLOP/s")

    result = {
        "label":  label,
        "device": dev_name,
        "config": {
            "batch": args.batch, "heads": args.heads,
            "seq":   args.seq,   "dim":   args.dim, "dtype": args.dtype,
        },
        "stats": stats,
    }

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[bench] Results → {out}")


if __name__ == "__main__":
    main()