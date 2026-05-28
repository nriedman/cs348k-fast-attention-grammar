"""
bench.py — Generic cuTile kernel benchmark
===========================================
Times a generated kernel module's host callable. Unlike the attention-specific
harness, input shapes and the FLOP count come from the module's KERNEL_META, so
it works for any Program our emitter produces.

Usage
-----
    python bench.py --kernel generated_kernel.py [--kernel-fn fn] \
        [--warmup 50] [--iters 200] [--dtype fp16|fp32] \
        [--label my_kernel] [--output results/my_kernel.json]

Kernel module interface
-----------------------
The module must expose:
  - a callable `fn(*inputs) -> output`  (cupy arrays)
  - KERNEL_META = {"inputs": [[name, [dims...]], ...],
                   "output": [name, [dims...]],
                   "flops": int, "flops_exact": bool}
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Callable

import cupy as cp

DTYPE_MAP = {"fp16": cp.float16, "fp32": cp.float32}


def load_module(path: str):
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Kernel file not found: {p}")
    spec = importlib.util.spec_from_file_location("_kernel", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def make_inputs(meta: dict, dtype) -> list:
    return [cp.random.randn(*dims, dtype=dtype) for _name, dims in meta["inputs"]]


def benchmark(fn: Callable, inputs: list, warmup: int, iters: int) -> dict:
    try:
        for _ in range(warmup):
            fn(*inputs)
        cp.cuda.runtime.deviceSynchronize()

        # GPU-event timing over the whole loop, divided by iters
        start_ev, end_ev = cp.cuda.Event(), cp.cuda.Event()
        start_ev.record()
        for _ in range(iters):
            fn(*inputs)
        end_ev.record()
        end_ev.synchronize()

        return {
            "mean_gpu_ms": cp.cuda.get_elapsed_time(start_ev, end_ev) / iters,
            "iters": iters,
            "failed": False,
        }
    except Exception as e:
        return {
            "mean_gpu_ms": float("inf"),
            "iters": iters,
            "failed": True,
            "error": str(e),
        }


def parse_args():
    p = argparse.ArgumentParser(description="Generic cuTile kernel benchmark")
    p.add_argument("--kernel",    required=True, help="Path to generated kernel module")
    p.add_argument("--kernel-fn", default="fn", help="Host callable name")
    p.add_argument("--label",     default=None)
    p.add_argument("--warmup",    type=int, default=50)
    p.add_argument("--iters",     type=int, default=200)
    p.add_argument("--dtype",     choices=list(DTYPE_MAP), default="fp32")
    p.add_argument("--output",    default=None, help="Path to write JSON results")
    return p.parse_args()


def main():
    args = parse_args()
    dtype = DTYPE_MAP[args.dtype]

    if cp.cuda.runtime.getDeviceCount() == 0:
        print("[ERROR] No CUDA device available.", file=sys.stderr)
        sys.exit(1)

    dev_name = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())["name"].decode()
    label = args.label or Path(args.kernel).stem

    try:
        mod = load_module(args.kernel)
        fn = getattr(mod, args.kernel_fn)
        meta = mod.KERNEL_META
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    print(f"[bench] Device : {dev_name}")
    print(f"[bench] Kernel : {label}  ({args.kernel}::{args.kernel_fn})")
    print(f"[bench] Inputs : {meta['inputs']}  dtype={args.dtype}")

    inputs = make_inputs(meta, dtype)
    stats = benchmark(fn, inputs, warmup=args.warmup, iters=args.iters)

    if stats["failed"]:
        print(f"[bench] FAILED : {stats['error']}")
        perf = "  |  FAILED"
        stats["tflops"] = None
    else:
        flops = meta.get("flops")
        if flops and meta.get("flops_exact", False):
            stats["tflops"] = flops / (stats["mean_gpu_ms"] * 1e-3) / 1e12
            perf = f"  |  {stats['tflops']:.2f} TFLOP/s"
        else:
            stats["tflops"] = None
            perf = "  |  TFLOP/s n/a"
    print(f"[bench] mean GPU : {stats['mean_gpu_ms']:.3f} ms{perf}")

    result = {
        "label":  label,
        "device": dev_name,
        "config": {"dtype": args.dtype, "inputs": meta["inputs"], "output": meta["output"]},
        "stats":  stats,
    }
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[bench] Results -> {out}")


if __name__ == "__main__":
    main()