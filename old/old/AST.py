from __future__ import annotations
from typing import ClassVar
from dataclasses import dataclass, field
from enum import Enum, auto

# MARK: Support

class DType(Enum):
    F16  = 2
    BF16 = 2
    F32  = 4
    I32  = 4
    F64  = 8

    @property
    def bytes(self) -> int:
        return self.value

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
    dtype: DType

    _registry: ClassVar[dict[str, Tensor]] = {}

    @classmethod
    def get(cls, name: str, dims: tuple[Dim, ...], dtype: DType = DType.F16) -> Tensor:
        if name not in cls._registry:
            cls._registry[name] = cls(name, dims, dtype)
        return cls._registry[name]

    def rank(self) -> int:
        return len(self.dims)

    def __repr__(self) -> str:
        dims = ", ".join(d.name for d in self.dims)
        return f"Tensor({self.name!r}, ({dims}))"

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

# MARK: Program

@dataclass
class Program(BranchNode):
    """Root node of the program."""
    pass

# MARK: Parallel Loop

@dataclass
class ParallelLoop(BranchNode):
    """A single `ct.kernel` that launches a grid of paralllel blocks."""

    tiling: Tiling

# MARK: Single Dim Loops

@dataclass
class SingleDimLoop(BranchNode):
    tiling: Tiling

    def __post_init__(self):
        if len(self.tiling.sizes) != 1:
            raise ValueError(
                f"{type(self).__name__} tiles over exactly one dimension, "
                f"got {len(self.tiling.sizes)}"
            )
    
    @property
    def dim(self) -> Dim:
        return self.tiling.sizes[0][0]

    @property
    def tile_size(self) -> int:
        return self.tiling.sizes[0][1]

@dataclass
class SpatialLoop(SingleDimLoop):
    """
        Tiles over a single spatial dimension.
        
        Each iteration produces an independent output tile — no accumulation
        across iterations.
    """
    pass

@dataclass
class ReductionLoop(SingleDimLoop):
    """
        Tiles over a single reduction dimension.

        Iterations accumulate into the enclosed Compute node's accumulator.
    """
    pass

# MARK: Load

@dataclass
class Load(LeafNode):
    """
        Move a tile from global to shared memory.
        
        Tile shape and offset are inferred from the enclosing ParallelLoop's
        tiling and the block's bid at compile time.
    """
    src: Tensor
    out: Tensor = field(init=False)

    def __post_init__(self):
        self.out = Tensor.get(f"{self.src.name}_", self.src.dims)       # TODO: dims of output should be the tile size

# MARK: Store

@dataclass
class Store(LeafNode):
    """
        Move a tile from shared to global memory.
        
        Tile shape and offset are inferred from the enclosing ParallelLoop's
        tiling and the block's bid at compile time.
    """
    dst: Tensor
    src: Tensor

# MARK: Compute

@dataclass(frozen=True)
class ComputeOp:
    """Describes a compute operation and its reduction semantics."""
    name: str
    reducible: bool
    reduction_op: ReductionOp | None = None

    def __post_init__(self):
        if self.reducible and self.reduction_op is None:
            raise ValueError(f"Reducible op {self.name!r} must specify a reduction_op")
        if not self.reducible and self.reduction_op is not None:
            raise ValueError(f"Non-reducible op {self.name!r} must not specify a reduction_op")

    def __repr__(self) -> str:
        return self.name

class ReductionOp(Enum):
    MMA = ("ct.mma", "ct.zeros")
    ADD = ("ct.add", "ct.zeros")
    MAX = ("ct.max", "ct.neg_inf")
    MIN = ("ct.min", "ct.pos_inf")

    def __init__(self, instruction: str, identity: str):
        self.instruction = instruction
        self.identity = identity

class Op:
    MATMUL          = ComputeOp("Matmul",         reducible=True,  reduction_op=ReductionOp.MMA)
    ELEMENTWISE_ADD = ComputeOp("ElementwiseAdd", reducible=False)
    ELEMENTWISE_MUL = ComputeOp("ElementwiseMul", reducible=False)
    RELU            = ComputeOp("ReLU",           reducible=False)
    SUM             = ComputeOp("Sum",            reducible=True,  reduction_op=ReductionOp.ADD)
    MAX             = ComputeOp("Max",            reducible=True,  reduction_op=ReductionOp.MAX)
    MIN             = ComputeOp("Min",            reducible=True,  reduction_op=ReductionOp.MIN)

@dataclass
class Compute(LeafNode):
    """Apply an operation to a set of input tiles, producing a single output tile."""
    op: ComputeOp
    inputs: tuple[Tile, ...]
    out: Tile

    @property
    def reducible(self) -> bool:
        return self.op.reducible

    @property
    def reduction_op(self) -> ReductionOp | None:
        return self.op.reduction_op