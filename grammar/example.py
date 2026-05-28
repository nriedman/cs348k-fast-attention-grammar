from AST import Program, ParallelLoop, Load, Store, Compute, emit_module

# Fused two-stage kernel, mirroring playground.py:
#   stage 1: a = x @ y      (x:(N,K), y:(K,M) -> a:(N,M))
#   stage 2: o = a + b      (a,b:(N,M) -> o:(N,M))
N, M, K = 1024, 1024, 512

# stage 1
x_ = Load("x", ["n", "k"])      # n tiled, k whole
y_ = Load("y", ["k", "m"])      # k whole, m tiled
mm = Compute("matmul", [x_, y_])
s1 = ParallelLoop(out="a", tile_shape=(64, 64), index_vars=("n", "m"),
                  body=[x_, y_, mm, Store("a", mm, ["n", "m"])])

# stage 2
a_ = Load("a", ["n", "m"])
b_ = Load("b", ["n", "m"])
ad = Compute("add", [a_, b_])
s2 = ParallelLoop(out="o", tile_shape=(32, 32), index_vars=("n", "m"),
                  body=[a_, b_, ad, Store("o", ad, ["n", "m"])])

prog = Program(
    tensors={"x": (N, K), "y": (K, M), "b": (N, M), "a": (N, M), "o": (N, M)},
    body=[s1, s2],
)

src = emit_module(prog)
open("kernels/example.py", "w").write(src)
print(src)