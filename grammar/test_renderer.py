"""
test_renderer.py — correctness tests for the cuTile kernel renderer
===================================================================
Builds programs that exercise every AST node and every placement/nesting case,
emits a runnable module via `emit_module`, executes it, and checks the result
against a PyTorch (or numpy) reference.

Two execution backends, auto-selected:
  * GPU  — real `cuda.tile` + `cupy`, the generated kernel runs natively.
  * CPU  — a numpy backend that models cuTile's tile semantics (load = slice
           copy, store = slice write, mma = acc + a@b, alloc = uninitialised).
           `ct.alloc` is filled with NaN so any incompletely-written buffer
           surfaces as a test failure rather than passing silently.

Usage
-----
    python test_renderer.py                 # auto: GPU if available, else CPU
    python test_renderer.py --device cpu
    python test_renderer.py --device gpu
    python test_renderer.py --ref torch     # force the reference backend
    python test_renderer.py -k matmul       # only tests whose name matches
    python test_renderer.py -v              # show the generated source on failure
"""

from __future__ import annotations
import argparse
import sys
import types
from dataclasses import dataclass
from typing import Callable

import numpy as np

from AST import (
    Program, ParallelLoop, Load, Store, Compute, ReductionLoop, SpatialLoop,
    emit_module, CuTileRenderer, RenderCache, program_flops, structural_key,
)


# ==========================================================================
# Execution backend: real cuTile/cupy on GPU, or a numpy mock on CPU.
# ==========================================================================
def _install_cpu_mock() -> None:
    """Register numpy-backed `cuda.tile` and `cupy` in sys.modules so a
    generated module's `import cuda.tile as ct; import cupy as cp` picks them up.
    The mock models cuTile tile semantics exactly enough to validate indexing,
    slicing, accumulation, and grid decode."""
    ctx = {"bid": (0, 0, 0)}

    def _slices(idx, shape):
        return tuple(slice(i * s, i * s + s) for i, s in zip(idx, shape))

    ct = types.ModuleType("cuda.tile")
    ct.kernel = lambda fn: fn
    ct.bid = lambda n: ctx["bid"][n]
    ct.load = lambda t, idx, shape: t[_slices(idx, shape)].copy()

    def _store(t, idx, val):
        t[_slices(idx, tuple(val.shape))] = val
    ct.store = _store

    ct.zeros = lambda shape, dtype=np.float32: np.zeros(shape, dtype=dtype)
    # NaN-filled: an incompletely-scattered buffer will corrupt the result.
    ct.alloc = lambda shape, dtype=np.float32: np.full(shape, np.nan, dtype=dtype)
    ct.matmul = lambda a, b: a @ b
    ct.mma = lambda a, b, acc: acc + a @ b
    ct.add = lambda a, b: a + b
    ct.maximum = lambda x, y: np.maximum(x, y)
    ct.exp = lambda x: np.exp(x)
    ct.cdiv = lambda a, b: -(-a // b)

    def _launch(stream, grid, kern, args):
        gx, gy, gz = (list(grid) + [1, 1, 1])[:3]
        for i in range(gx):
            for j in range(gy):
                for k in range(gz):
                    ctx["bid"] = (i, j, k)
                    kern(*args)
    ct.launch = _launch

    class _Const:
        def __class_getitem__(cls, item):
            return cls
    ct.Constant = _Const

    cp = types.ModuleType("cupy")
    cp.zeros = lambda shape, dtype=np.float32: np.zeros(shape, dtype=dtype)
    cp.float16, cp.float32 = np.float16, np.float32
    cp.asarray = lambda a: np.asarray(a)
    cp.asnumpy = lambda a: np.asarray(a)
    cp.random = types.SimpleNamespace(
        randn=lambda *shape, dtype=np.float32: np.random.randn(*shape).astype(dtype))
    cp.cuda = types.ModuleType("cupy.cuda")
    cp.cuda.get_current_stream = lambda: None
    cp.cuda.runtime = types.SimpleNamespace(deviceSynchronize=lambda: None)

    cuda_pkg = types.ModuleType("cuda")
    cuda_pkg.tile = ct
    sys.modules["cuda"] = cuda_pkg
    sys.modules["cuda.tile"] = ct
    sys.modules["cupy"] = cp


def setup_backend(device: str):
    """Return (cp_module, is_gpu). `device` is 'auto' | 'cpu' | 'gpu'."""
    def _try_gpu():
        try:
            import cupy as cp            # noqa: F401
            import cuda.tile             # noqa: F401
            return cp
        except Exception:
            return None

    if device == "gpu":
        cp = _try_gpu()
        if cp is None:
            print("[error] --device gpu but cuda.tile/cupy unavailable.", file=sys.stderr)
            sys.exit(2)
        return cp, True
    if device == "auto":
        cp = _try_gpu()
        if cp is not None:
            return cp, True
    _install_cpu_mock()
    import cupy as cp                    # the mock
    return cp, False


# ==========================================================================
# Reference math: torch if available (honouring "like PyTorch"), else numpy.
# Reference functions are written once against this small adapter.
# ==========================================================================
class _NumpyRef:
    name = "numpy"
    matmul = staticmethod(lambda a, b: a @ b)
    add = staticmethod(lambda a, b: a + b)
    mul = staticmethod(lambda a, b: a * b)
    relu = staticmethod(lambda a: np.maximum(a, 0.0))
    exp = staticmethod(lambda a: np.exp(a))
    asarray = staticmethod(lambda a: np.asarray(a))
    tonumpy = staticmethod(lambda a: np.asarray(a))


class _TorchRef:
    name = "torch"

    def __init__(self, torch):
        self.t = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"

    def matmul(self, a, b): return self.t.matmul(a, b)
    def add(self, a, b): return a + b
    def mul(self, a, b): return a * b
    def relu(self, a): return self.t.relu(a)
    def exp(self, a): return self.t.exp(a)
    def asarray(self, a): return self.t.as_tensor(np.asarray(a), device=self.dev)
    def tonumpy(self, a): return a.detach().to("cpu").numpy()


def setup_reference(ref: str):
    def _try_torch():
        try:
            import torch
            return _TorchRef(torch)
        except Exception:
            return None
    if ref == "torch":
        r = _try_torch()
        if r is None:
            print("[error] --ref torch but torch unavailable.", file=sys.stderr)
            sys.exit(2)
        return r
    if ref == "auto":
        r = _try_torch()
        if r is not None:
            return r
    return _NumpyRef()


# ==========================================================================
# Test harness
# ==========================================================================
@dataclass
class Case:
    name: str
    build: Callable[[], Program]              # () -> Program
    ref: Callable[[object, dict], object]     # (RefMath, inputs) -> array
    rtol: float = 1e-3
    atol: float = 1e-3


def _exec_module(src: str) -> dict:
    ns: dict = {"__name__": "_generated"}     # not "__main__": skip the demo block
    exec(compile(src, "<generated>", "exec"), ns)
    return ns


def run_case(case: Case, cp, is_gpu: bool, refmath, verbose: bool):
    prog = case.build()
    src = emit_module(prog)
    try:
        ns = _exec_module(src)
    except Exception as e:
        return False, f"module failed to import/compile: {e}", src
    fn, meta = ns["fn"], ns["KERNEL_META"]

    rng = np.random.default_rng(0)
    inputs_np = {name: rng.standard_normal(tuple(shape)).astype(np.float32)
                 for name, shape in meta["inputs"]}

    # reference (torch/numpy)
    ins_ref = {k: refmath.asarray(v) for k, v in inputs_np.items()}
    expected = np.asarray(refmath.tonumpy(case.ref(refmath, ins_ref)), dtype=np.float32)

    # kernel: inputs in fn arg order, on the execution backend
    to_dev = (lambda a: cp.asarray(a)) if is_gpu else (lambda a: a)
    args = [to_dev(inputs_np[name]) for name, _ in meta["inputs"]]
    try:
        out = fn(*args)
    except Exception as e:
        return False, f"kernel raised at runtime: {e}", src
    got = np.asarray(cp.asnumpy(out) if is_gpu else out, dtype=np.float32)

    if got.shape != expected.shape:
        return False, f"shape {got.shape} != reference {expected.shape}", src
    if not np.all(np.isfinite(got)):
        return False, "output contains NaN/Inf (incomplete write or bad index)", src
    ok = np.allclose(got, expected, rtol=case.rtol, atol=case.atol)
    abs_err = float(np.max(np.abs(got - expected)))
    denom = np.maximum(np.abs(expected), 1e-12)
    rel_err = float(np.max(np.abs(got - expected) / denom))
    msg = f"max|abs|={abs_err:.2e} max|rel|={rel_err:.2e}"
    return ok, msg, src


# ==========================================================================
# Program builders — one per node / placement / nesting case.
# ==========================================================================
def p_elementwise_add():
    a, b = Load("a", ["n", "m"]), Load("b", ["n", "m"])
    s = Compute("add", [a, b])
    return Program({"a": (256, 256), "b": (256, 256), "o": (256, 256)},
                   [ParallelLoop("o", (64, 64), ("n", "m"),
                                 [a, b, s, Store("o", s, ["n", "m"])])])


def p_relu():
    a = Load("a", ["n", "m"]); r = Compute("relu", [a])
    return Program({"a": (256, 256), "o": (256, 256)},
                   [ParallelLoop("o", (64, 64), ("n", "m"),
                                 [a, r, Store("o", r, ["n", "m"])])])


def p_mul():
    a, b = Load("a", ["n", "m"]), Load("b", ["n", "m"])
    s = Compute("mul", [a, b])
    return Program({"a": (256, 256), "b": (256, 256), "o": (256, 256)},
                   [ParallelLoop("o", (64, 64), ("n", "m"),
                                 [a, b, s, Store("o", s, ["n", "m"])])])


def p_exp():
    a = Load("a", ["n", "m"]); e = Compute("exp", [a])
    return Program({"a": (256, 256), "o": (256, 256)},
                   [ParallelLoop("o", (64, 64), ("n", "m"),
                                 [a, e, Store("o", e, ["n", "m"])])])


def p_relu_of_sum():
    a, b = Load("a", ["n", "m"]), Load("b", ["n", "m"])
    s = Compute("add", [a, b]); r = Compute("relu", [s])
    return Program({"a": (256, 256), "b": (256, 256), "o": (256, 256)},
                   [ParallelLoop("o", (64, 64), ("n", "m"),
                                 [a, b, s, r, Store("o", r, ["n", "m"])])])


def p_matmul_full_k():
    x, y = Load("x", ["n", "k"]), Load("y", ["k", "m"])
    mm = Compute("matmul", [x, y])
    return Program({"x": (256, 128), "y": (128, 256), "c": (256, 256)},
                   [ParallelLoop("c", (64, 64), ("n", "m"),
                                 [x, y, mm, Store("c", mm, ["n", "m"])])])


def p_matmul_reduction():
    x, y = Load("x", ["n", "k"]), Load("y", ["k", "m"])
    mm = Compute("matmul", [x, y])
    red = ReductionLoop("k", 32, [x, y, mm], mm)
    return Program({"x": (256, 128), "y": (128, 256), "c": (256, 256)},
                   [ParallelLoop("c", (64, 64), ("n", "m"),
                                 [red, Store("c", red, ["n", "m"])])])


def p_spatial_relu():            # SpatialLoop, store INSIDE
    a = Load("a", ["n", "m"]); r = Compute("relu", [a])
    sp = SpatialLoop("n", 32, [a, r, Store("o", r, ["n", "m"])])
    return Program({"a": (256, 256), "o": (256, 256)},
                   [ParallelLoop("o", (64, 64), ("n", "m"), [sp])])


def p_load_outside():            # Load OUTSIDE the SpatialLoop -> sliced read
    a = Load("a", ["n", "m"]); r = Compute("relu", [a])
    sp = SpatialLoop("n", 32, [r, Store("o", r, ["n", "m"])])
    return Program({"a": (256, 256), "o": (256, 256)},
                   [ParallelLoop("o", (64, 64), ("n", "m"), [a, sp])])


def p_store_outside():           # Store OUTSIDE -> ct.alloc buffer + scatter
    a = Load("a", ["n", "m"]); r = Compute("relu", [a])
    sp = SpatialLoop("n", 32, [a, r])
    return Program({"a": (256, 256), "o": (256, 256)},
                   [ParallelLoop("o", (64, 64), ("n", "m"),
                                 [sp, Store("o", r, ["n", "m"])])])


def p_reduction_in_spatial():    # ReductionLoop nested in SpatialLoop
    x, y = Load("x", ["n", "k"]), Load("y", ["k", "m"])
    mm = Compute("matmul", [x, y])
    red = ReductionLoop("k", 32, [x, y, mm], mm)
    sp = SpatialLoop("n", 32, [red, Store("c", red, ["n", "m"])])
    return Program({"x": (256, 128), "y": (128, 256), "c": (256, 256)},
                   [ParallelLoop("c", (64, 64), ("n", "m"), [sp])])


def p_spatial_in_reduction():    # SpatialLoop nested in ReductionLoop (acc expand)
    x, y = Load("x", ["n", "k"]), Load("y", ["k", "m"])
    mm = Compute("matmul", [x, y])
    sp = SpatialLoop("n", 32, [x, y, mm])
    red = ReductionLoop("k", 32, [sp], mm)
    return Program({"x": (256, 128), "y": (128, 256), "c": (256, 256)},
                   [ParallelLoop("c", (64, 64), ("n", "m"),
                                 [red, Store("c", red, ["n", "m"])])])


def p_fused_matmul_add():        # two stages: o = (x @ y) + b
    x, y = Load("x", ["n", "k"]), Load("y", ["k", "m"])
    mm = Compute("matmul", [x, y])
    red = ReductionLoop("k", 32, [x, y, mm], mm)
    s1 = ParallelLoop("a", (64, 64), ("n", "m"), [red, Store("a", red, ["n", "m"])])
    a2, b2 = Load("a", ["n", "m"]), Load("b", ["n", "m"])
    add = Compute("add", [a2, b2])
    s2 = ParallelLoop("o", (32, 32), ("n", "m"), [a2, b2, add, Store("o", add, ["n", "m"])])
    return Program({"x": (256, 128), "y": (128, 256), "b": (256, 256),
                    "a": (256, 256), "o": (256, 256)}, [s1, s2])


def p_batched_4d_relu():         # 4-D output -> grid collapse/decode
    x = Load("x", ["b0", "b1", "n", "m"]); r = Compute("relu", [x])
    return Program({"x": (2, 4, 128, 128), "y": (2, 4, 128, 128)},
                   [ParallelLoop("y", (1, 1, 64, 64), ("b0", "b1", "n", "m"),
                                 [x, r, Store("y", r, ["b0", "b1", "n", "m"])])])


CASES = [
    Case("elementwise_add",     p_elementwise_add,     lambda M, i: M.add(i["a"], i["b"])),
    Case("relu",                p_relu,                lambda M, i: M.relu(i["a"])),
    Case("mul",                 p_mul,                 lambda M, i: M.mul(i["a"], i["b"])),
    Case("exp",                 p_exp,                 lambda M, i: M.exp(i["a"])),
    Case("relu_of_sum",         p_relu_of_sum,         lambda M, i: M.relu(M.add(i["a"], i["b"]))),
    Case("matmul_full_k",       p_matmul_full_k,       lambda M, i: M.matmul(i["x"], i["y"])),
    Case("matmul_reduction",    p_matmul_reduction,    lambda M, i: M.matmul(i["x"], i["y"])),
    Case("spatial_relu",        p_spatial_relu,        lambda M, i: M.relu(i["a"])),
    Case("load_outside",        p_load_outside,        lambda M, i: M.relu(i["a"])),
    Case("store_outside",       p_store_outside,       lambda M, i: M.relu(i["a"])),
    Case("reduction_in_spatial", p_reduction_in_spatial, lambda M, i: M.matmul(i["x"], i["y"])),
    Case("spatial_in_reduction", p_spatial_in_reduction, lambda M, i: M.matmul(i["x"], i["y"])),
    Case("fused_matmul_add",    p_fused_matmul_add,    lambda M, i: M.add(M.matmul(i["x"], i["y"]), i["b"])),
    Case("batched_4d_relu",     p_batched_4d_relu,     lambda M, i: M.relu(i["x"])),
]


# ==========================================================================
# Renderer property checks (non-numeric): cache dedup, flops, inline render.
# ==========================================================================
def property_checks():
    results = []

    # 1. structural cache dedups equivalent programs (skips re-emission).
    cache = RenderCache()
    _, h1 = cache.module(p_matmul_reduction())
    _, h2 = cache.module(p_matmul_reduction())   # independently rebuilt, identical
    results.append(("cache_dedup", (not h1) and h2,
                    f"first hit={h1} second hit={h2}"))

    # 2. matmul FLOPs are exact and equal 2*M*N*K.
    flops, exact = program_flops(p_matmul_reduction())
    results.append(("flops_matmul", exact and flops == 2 * 256 * 256 * 128,
                    f"flops={flops} exact={exact}"))

    # 3. inline renderer produces source for every case without raising.
    ok_inline, bad = True, None
    for c in CASES:
        try:
            CuTileRenderer().render(c.build())
        except Exception as e:                   # pragma: no cover
            ok_inline, bad = False, f"{c.name}: {e}"
            break
    results.append(("inline_render_smoke", ok_inline, bad or "all cases rendered"))
    return results


# ==========================================================================
# Runner
# ==========================================================================
def main():
    ap = argparse.ArgumentParser(description="cuTile renderer correctness tests")
    ap.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto")
    ap.add_argument("--ref", choices=["auto", "numpy", "torch"], default="auto")
    ap.add_argument("-k", "--filter", default=None, help="substring of test name to run")
    ap.add_argument("-v", "--verbose", action="store_true", help="print source on failure")
    args = ap.parse_args()

    cp, is_gpu = setup_backend(args.device)
    refmath = setup_reference(args.ref)
    print(f"[test] execution backend : {'GPU (cuda.tile+cupy)' if is_gpu else 'CPU (numpy mock)'}")
    print(f"[test] reference backend : {refmath.name}")
    print("-" * 64)

    cases = [c for c in CASES if not args.filter or args.filter in c.name]
    passed = failed = 0
    for c in cases:
        ok, msg, src = run_case(c, cp, is_gpu, refmath, args.verbose)
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {c.name:<22} {msg}")
        if ok:
            passed += 1
        else:
            failed += 1
            if args.verbose:
                print("\n".join("        " + ln for ln in src.splitlines()))

    if not args.filter:
        print("-" * 64)
        for name, ok, msg in property_checks():
            mark = "PASS" if ok else "FAIL"
            print(f"  [{mark}] {name:<22} {msg}")
            passed += 1 if ok else 0
            failed += 0 if ok else 1

    print("-" * 64)
    print(f"[test] {passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()