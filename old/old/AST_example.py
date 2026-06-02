from grammar.old.AST import *

def main():
    TILE_X = 32
    TILE_Y = 64

    X = Dim.get("X", TILE_X * 4)
    Y = Dim.get("Y", TILE_Y * 8)

    A = Tensor.get("A", (X, Y))
    B = Tensor.get("B", (X, Y))

    O = Tensor.get("O", (X, Y))

    g = Program(
        children=[
            ParallelLoop(
                tiling=Tiling.from_dict({X: TILE_X, Y: TILE_Y}),
                children=[
                    Load(A),
                    Compute("ElementwiseAdd", A_, B_, O_),
                    Store(O, O_)
                ]
            )
        ]
    )

if __name__ == "__main__":
    main()