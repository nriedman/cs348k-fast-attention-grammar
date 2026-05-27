from __future__ import annotations
from dataclasses import dataclass, field

from enum import Enum, auto

# MARK: Support

class MemorySpace(Enum):
    GLOBAL = auto()
    SHARED = auto()

@dataclass(frozen=True)
class Dim:
    """
        An identifiable symbol representing a dimension.
    
        Use Dim.get() to construct and compare with `is` or `==`
        (e.g. `loop.dim is load.dim` or `loop.dim == load.dim`).
    """

    name: str
    size: int | None = None

    # Static, global namespace
    _registry: ClassVar[dict[str, Dim]] = {}

    @classmethod
    def get(cls, name: str, size: int | None = None) -> Dim:
        if name not in cls._registry:
            cls._registry[name] = cls(name, size)
        return cls._registry[name]

    def __repr__(self) -> str:
        return f"Dim({self.name!r})" if self.size is None else f"Dim({self.name!r}, {self.size})"

@dataclass(frozen=True)
class Tiling:
    """
        Maps each output dimension to its tiling size.
        Enforces dimension ordering and tiling constraints.

        Use `==` to compare.
    """

    sizes: tuple[tuple[Dim, int], ...]

    @classmethod
    def from_dict(cls, d: dict[Dim, int]) -> Tiling:
        return cls(sizes=tuple(d.items()))
    
    def __getitem__(self, dim: Dim) -> int:
        for d, s in self.sizes:
            if d is dim:
                return s
        raise KeyError(dim)

    def dims(self) -> tuple[Dim, ...]:
        return tuple(d for d, _ in self.sizes)
    
    def __repr__(self) -> str:
        inner = ", ".join(f"{d.name}: {s}" for d, s in self.sizes)
        return f"Tiling({{{inner}}})"

@dataclass(frozen=True)
class Tensor:
    """
        An identifiable symbol representing a tensor.

        Use Tensor.get() to construct and `is` or `==` to compare.
    """
    name: str
    dims: tuple[Dim, ...]  # ordered — defines the axis layout

    _registry: ClassVar[dict[str, Tensor]] = {}

    @classmethod
    def get(cls, name: str, dims: tuple[Dim, ...]) -> Tensor:
        if name not in cls._registry:
            cls._registry[name] = cls(name, dims)
        return cls._registry[name]

    def rank(self) -> int:
        return len(self.dims)

    def __repr__(self) -> str:
        dims = ", ".join(d.name for d in self.dims)
        return f"Tensor({self.name!r}, ({dims}))"

@dataclass(frozen=True)
class TileRef:
    """A reference to a specific tile of a tensor.
    
    `offset` and `tiling` are both keyed by the tensor's dims, so
    axis identity is always unambiguous.
    """
    tensor: Tensor
    offset: dict[Dim, int]      # which tile (in tile-index space, not element space)
    tiling: Tiling              # tile size in each dimension
    memory_space: MemorySpace

    def __post_init__(self):
        if set(self.offset.keys()) != set(self.tensor.dims):
            raise ValueError("offset must cover all tensor dimensions")
        if set(self.tiling.dims()) != set(self.tensor.dims):
            raise ValueError("tiling must cover all tensor dimensions")

# MARK: Base Types

@dataclass
class GrammarNode:
    """Abstract base for all nodes in the grammar."""
    pass

@dataclass
class BranchNode(GrammarNode):
    """Base for nodes that have children."""
    children: list[GrammarNode] = field(default_factory=list, kw_only=True)

@dataclass
class LeafNode(GrammarNode):
    """Base for terminal nodes."""
    pass

# MARK: Branch Nodes

@dataclass
class Program(BranchNode):
    """Root node of the program."""
    pass

@dataclass
class ParallelLoop(BranchNode):
    """A single `ct.kernel` that launches a grid of paralllel blocks."""

    # The size of tile in each dimension of the **output**.
    #
    # Constraints:
    #   - Each element must be a power of two and divide evenly into its
    #     corresponding dimension.
    #   - The grid resulting from <tile_sizes> must be well formed:
    #        TODO: Find what these bounds are
    tiling: Tiling

@dataclass
class SequentialLoop(BranchNode):
    """
        A sub-tiling that iterates sequentially over one dimension
        inside a `ParallelLoop`.
    """

    tiling: Tiling

    def __post_init__(self):
        if len(self.tiling.sizes) != 1:
            raise ValueError(
                f"SequentialLoop tiles over exactly one dimension, got {len(self.tiling.sizes)}"
            )
    
    @property
    def dim(self) -> Dim:
        return self.tiling.sizes[0][0]

    @property
    def tile_size(self) -> int:
        return self.tiling.sizes[0][1]

# MARK: Leaf Nodes

class Load(LeafNode):
    """Move a tile from global to shared memory."""
    src: TileRef

class Store(LeafNode):
    """Move a tile from shared to global memory."""
    dst: TileRef        # global memory
    src: TileRef        # shared memory

    def __post_init__(self):
        if self.src.tiling != self.dst.tiling:
            raise ValueError(
                f"Store src and dst tile sizes must match: "
                f"{self.src.tiling} != {self.dst.tiling}"
            )


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
                Compute("ElementwiseAdd", A_, B_, O_)
                Store(O, O_)
            ]
        )
    ]
)