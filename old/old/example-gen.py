from kernel_ast import (Program, ParallelLoop, Load, Store, Compute,
                        SpatialLoop, emit_module)

# o = relu(a), subtiling the n output dim by SUB_TS_n = 32 (block tile TS_n = 64).
N, M = 1024, 1024
a_ = Load("a", ["n", "m"])
r  = Compute("relu", [a_])
st = Store("o", r, ["n", "m"])
sp = SpatialLoop(axis="n", tile=32, body=[a_, r, st])

prog = Program(
    tensors={"a": (N, M), "o": (N, M)},
    body=[ParallelLoop(out="o", tile_shape=(64, 64), index_vars=("n", "m"), body=[sp])],
)
src = emit_module(prog)
open("example.py", "w").write(src)
print(src)