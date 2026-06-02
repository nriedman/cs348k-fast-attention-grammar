from grammar.kernel_ast import (Program, ParallelLoop, Load, Store, Compute,
                        SpatialLoop, emit_module)

def build_mlp_ln(B: int, Dm: int, H: int, eps: float = 1e-5,
                 tile: int | None = None, subtile_k: int | None = None) -> Program:
    """MLP + residual + LayerNorm as separate stages. `tile` sets a uniform
    starting output-tile size (None => atomic: full-extent tiles, so the search
    must discover tiling). `subtile_k` (if set) pre-wraps each matmul in a
    reduction loop tiling its contraction by that amount -- this makes the
    matmuls emit as ct.mma K-loops from the start, avoiding the pathologically
    slow cuTile compile of a full-K single-tile ct.matmul. The search can still
    retune the K-tile and apply every other rewrite. Axis names: b (batch),
    dm (feature), hid (hidden) -- distinct from tensor names to avoid clashes."""
    def t2(d0, d1):
        return (32, 32) if tile is None else (min(tile, d0), min(tile, d1))

    from grammar.rewrites import subtile_reduction
    import grammar.rewrites as R

    # h = X @ W1  -> [B, H], reduces dm
    X = Load("X", ["b", "dm"]); W1 = Load("W1", ["dm", "hid"])
    h = Compute("matmul", [X, W1])
    s_h = ParallelLoop("h", t2(B, H), ("b", "hid"), [X, W1, h, Store("h", h, ["b", "hid"])])

    # a = relu(h) -> [B, H]
    hL = Load("h", ["b", "hid"]); a = Compute("relu", [hL])
    s_a = ParallelLoop("a", t2(B, H), ("b", "hid"), [hL, a, Store("a", a, ["b", "hid"])])

    # y = a @ W2  -> [B, Dm], reduces hid
    aL = Load("a", ["b", "hid"]); W2 = Load("W2", ["hid", "dm"])
    y = Compute("matmul", [aL, W2])
    s_y = ParallelLoop("y", t2(B, Dm), ("b", "dm"), [aL, W2, y, Store("y", y, ["b", "dm"])])

    # r = y + X  -> [B, Dm]  (residual)
    yL = Load("y", ["b", "dm"]); XL = Load("X", ["b", "dm"]); r = Compute("add", [yL, XL])
    s_r = ParallelLoop("r", t2(B, Dm), ("b", "dm"), [yL, XL, r, Store("r", r, ["b", "dm"])])

    # out = LayerNorm(r) over dm:
    #   s1 = sum(r); s2 = sum(r*r); mean = s1/Dm; var = s2/Dm - mean^2;
    #   out = (r - mean) / sqrt(var + eps)
    rL = Load("r", ["b", "dm"]); s1 = Compute("rowsum", [rL], axis="dm")
    r2a, r2b = Load("r", ["b", "dm"]), Load("r", ["b", "dm"])
    rr = Compute("mul", [r2a, r2b]); s2 = Compute("rowsum", [rr], axis="dm")
    mean = Compute("mulc", [s1], const=1.0 / Dm)
    meansq = Compute("mulc", [s2], const=1.0 / Dm)
    mean2 = Compute("mul", [mean, mean])
    var = Compute("sub", [meansq, mean2])
    vareps = Compute("addc", [var], const=eps)
    std = Compute("sqrt", [vareps])
    r3 = Load("r", ["b", "dm"])
    centered = Compute("sub", [r3, mean])
    out = Compute("div", [centered, std])
    s_ln = ParallelLoop("out", (B, Dm), ("b", "dm"),
                        [rL, s1, r2a, r2b, rr, s2, mean, meansq, mean2, var,
                         vareps, std, r3, centered, out, Store("out", out, ["b", "dm"])])

    tens = {"X": (B, Dm), "W1": (Dm, H), "W2": (H, Dm),
            "h": (B, H), "a": (B, H), "y": (B, Dm), "r": (B, Dm), "out": (B, Dm)}
    prog = Program(tens, [s_h, s_a, s_y, s_r, s_ln])

    if subtile_k is not None:
        # wrap each matmul's contraction in a reduction loop so it emits as a
        # ct.mma K-loop (not a full-K single ct.matmul), THEN sink the operand
        # loads into the loop. Sinking is essential: without it the loads stay
        # hoisted and materialize a full-contraction-width tile (e.g. 64x1024),
        # which exhausts memory during cuTile/ptxas compilation. Sunk, each
        # iteration loads only a TS x TS slice.
        prog = subtile_reduction(prog, h, "dm", tile=min(subtile_k, Dm))
        def bare_matmuls(p):
            return [s for loop in p.body for s in loop.body
                    if isinstance(s, Compute) and s.op == "matmul"]
        for mm in bare_matmuls(prog):
            prog = subtile_reduction(prog, mm, "hid", tile=min(subtile_k, H))
        # sink every hoisted load into its sibling reduction loop
        changed = True
        while changed:
            changed = False
            for loop in prog.body:
                redloops = [s for s in loop.body if isinstance(s, ReductionLoop)]
                for s in list(loop.body):
                    if isinstance(s, Load):
                        for rl in redloops:
                            if R.can_sink(prog, s, rl)[0]:
                                prog = R.sink(prog, s, rl); changed = True; break
                        if changed:
                            break
                if changed:
                    break
    return prog

# o = relu(a), subtiling the n output dim by SUB_TS_n = 32 (block tile TS_n = 64).
# N, M = 1024, 1024
# a_ = Load("a", ["n", "m"])
# r  = Compute("relu", [a_])
# st = Store("o", r, ["n", "m"])
# sp = SpatialLoop(axis="n", tile=32, body=[a_, r, st])

# prog = Program(
#     tensors={"a": (N, M), "o": (N, M)},
#     body=[ParallelLoop(out="o", tile_shape=(64, 64), index_vars=("n", "m"), body=[sp])],
# )

prog = build_mlp_ln(256, 512, 1024)
src = emit_module(prog)
open("example.py", "w").write(src)
print(src)