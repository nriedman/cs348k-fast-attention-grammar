"""Dev-only: mock cuda.tile + cupy with numpy to execute a generated module
on CPU and verify host plumbing + kernel logic. NOT shipped as the runtime."""
import sys, types, importlib.util
import numpy as np

# ---- mock cupy ----
cp = types.ModuleType("cupy")
cp.zeros = lambda shape, dtype=np.float32: np.zeros(shape, dtype=dtype)
cp.float32, cp.float16 = np.float32, np.float16
cp.random = types.SimpleNamespace(
    randn=lambda *shape, dtype=np.float32: np.random.randn(*shape).astype(dtype))
cp.asnumpy = lambda x: np.asarray(x)
cp.cuda = types.ModuleType("cupy.cuda")
cp.cuda.get_current_stream = lambda: None
cp.cuda.runtime = types.SimpleNamespace(deviceSynchronize=lambda: None)
sys.modules["cupy"] = cp

# ---- mock cuda.tile ----
cuda_pkg = types.ModuleType("cuda")
ct = types.ModuleType("cuda.tile")
_ctx = {"bid": (0, 0, 0)}
def _sl(idx, shape): return tuple(slice(i*s, i*s+s) for i, s in zip(idx, shape))
ct.kernel = lambda fn: fn
ct.bid = lambda n: _ctx["bid"][n]
ct.load = lambda t, idx, shape: t[_sl(idx, shape)]
def _store(t, idx, val): t[_sl(idx, tuple(val.shape))] = val
ct.store = _store
ct.matmul = lambda a, b: a @ b
ct.add = lambda a, b: a + b
ct.mul = lambda a, b: a * b
ct.maximum = lambda x, y: np.maximum(x, y)
ct.exp = lambda x: np.exp(x)
ct.cdiv = lambda a, b: -(-a // b)
def _launch(stream, grid, kern, args):
    gx, gy, gz = (list(grid) + [1, 1, 1])[:3]
    for i in range(gx):
        for j in range(gy):
            for k in range(gz):
                _ctx["bid"] = (i, j, k)
                kern(*args)
ct.launch = _launch
class _Const:
    def __class_getitem__(cls, item): return cls
ct.Constant = _Const
cuda_pkg.tile = ct
sys.modules["cuda"], sys.modules["cuda.tile"] = cuda_pkg, ct

# ---- run the generated module ----
spec = importlib.util.spec_from_file_location("fused_kernel", "fused_kernel.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

np.random.seed(0)
x = np.random.randn(1024, 512).astype(np.float32)
y = np.random.randn(512, 1024).astype(np.float32)
b = np.random.randn(1024, 1024).astype(np.float32)
o = mod.fn(b, x, y)                      # fn signature is (b, x, y)
ref = x @ y + b
print("fn(b,x,y) -> o shape:", o.shape)
print("matches x@y + b :", np.allclose(o, ref, rtol=1e-3, atol=1e-3))
print("META flops       :", mod.KERNEL_META["flops"], "(exact:", mod.KERNEL_META["flops_exact"], ")")