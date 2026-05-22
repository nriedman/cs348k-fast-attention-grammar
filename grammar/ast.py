"""
Core AST node types and the Grammar container.

Design notes:
- All nodes are frozen dataclasses with __slots__ for fast attribute access and
  value-based hashing. Immutability enables structural sharing across rewrites.
- The entire ProgramNode executes as a single GPU kernel launch. Sibling
  LoopLevel children of ProgramNode run sequentially within that kernel, with
  an implicit barrier between each sibling to enforce data-dependency ordering.
- LoopLevel uses `bound` (number of iterations; power of 2) and `parallel: bool`
  instead of the old `tile_size` and `loop_type`.
- ComputeNode carries `output_dims` and `carried_dims` declared by the grammar
  author. These are the only op-semantic knowledge the framework requires:
    output_dims:  frozenset of dimension names that appear in the op's output
                  (determines write address per loop iteration).
    carried_dims: frozenset of dimension names over which the op accumulates
                  state across loop iterations (requires a serial loop; the
                  renderer emits an initialised accumulator before the loop
                  and += inside it).
- The DDG is stored alongside the tree in Grammar. Rewrite rules mutate the
  tree but never the DDG.
- ComputeNode carries a node_id (monotonically increasing int) for DDG identity.
- Grammar.node_index is a dict mapping node_id -> ComputeNode, built once on
  construction and reused for O(1) DDG lookups.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

ASTNode = Union["ProgramNode", "LoopLevel", "ComputeNode"]
LoopChild = Union["LoopLevel", "ComputeNode"]

# ---------------------------------------------------------------------------
# Node-ID counter (module-level, intentionally simple)
# ---------------------------------------------------------------------------

_node_counter: int = 0


def _next_id() -> int:
    global _node_counter
    _node_counter += 1
    return _node_counter


def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


# ---------------------------------------------------------------------------
# Leaf node
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ComputeNode:
    """
    Leaf node: a single primitive operation.
    Children: none. Data flow is encoded in the DDG, not the tree.

    Tags:
        op:           identifies the primitive (e.g. 'MatMul_QK', 'RowMax').
        node_id:      unique identity for DDG edge references.
        output_dims:  dimensions that appear in this op's output tensor.
                      A parallel loop over dim D writes to a different output
                      slice each iteration; D must be in output_dims.
        carried_dims: dimensions over which this op accumulates across loop
                      iterations. A loop over dim D where D ∈ carried_dims
                      must be serial (parallel=False). The renderer emits an
                      accumulator initialised before the loop and += inside it.
    """

    op: str
    node_id: int
    output_dims: frozenset  # frozenset[str]
    carried_dims: frozenset  # frozenset[str]

    @staticmethod
    def make(
        op: str,
        output_dims: tuple | set | frozenset = (),
        carried_dims: tuple | set | frozenset = (),
    ) -> "ComputeNode":
        """Create a new ComputeNode with a fresh unique node_id."""
        return ComputeNode(
            op=op,
            node_id=_next_id(),
            output_dims=frozenset(output_dims),
            carried_dims=frozenset(carried_dims),
        )


# ---------------------------------------------------------------------------
# Interior nodes
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LoopLevel:
    """
    A single tiled loop over one logical dimension.

    Parameters:
        dim:      logical dimension name (e.g. 'SEQ_Q', 'SEQ_KV', 'HEAD_DIM').
        bound:    number of iterations; must be a power of 2.
                  SplitLoop produces two nested LoopLevels where
                  outer.bound * inner.bound == original.bound.
        parallel: True  → parallelised across GPU threads/blocks (maps to
                          ct.bid / grid dim in the renderer).
                  False → serial loop in the kernel body (for-loop).
                  Changed by the Parallelize / Serialize rewrite rules.
        children: tuple of LoopLevel or ComputeNode, in execution order.
    """

    dim: str
    bound: int
    parallel: bool
    children: tuple  # tuple[LoopChild, ...]

    def __post_init__(self) -> None:
        if not _is_pow2(self.bound):
            raise ValueError(f"bound must be a power of 2, got {self.bound}")
        if not self.children:
            raise ValueError("LoopLevel must have at least one child")


@dataclass(frozen=True, slots=True)
class ProgramNode:
    """
    Root of the AST. The entire ProgramNode maps to a single GPU kernel launch.

    Direct child LoopLevel nodes execute sequentially within that kernel. An
    implicit synchronization barrier is inserted between each sibling to ensure
    that intermediate tensors written by one stage are visible to the next.
    The renderer derives grid/block dimensions from the LoopLevel structure;
    they are not stored in the AST.

    Children: one or more LoopLevel nodes.
    """

    children: tuple  # tuple[LoopLevel, ...]

    def __post_init__(self) -> None:
        if not self.children:
            raise ValueError("ProgramNode must have at least one child")


# ---------------------------------------------------------------------------
# DDG edge
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DDGEdge:
    """
    One edge in the data-dependency graph.

        producer_id: node_id of the producing ComputeNode.
        consumer_id: node_id of the consuming ComputeNode.
        tensor:      name of the tensor flowing along this edge.

    The DDG is always a DAG. Rewrite rules never modify it.
    If producer and consumer are in the same kernel (share a root-to-leaf
    path in the tree), the tensor may live in shared memory or registers.
    Otherwise it must be materialised to global memory.
    """

    producer_id: int
    consumer_id: int
    tensor: str


# ---------------------------------------------------------------------------
# Grammar container
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class Grammar:
    """
    A (program tree, DDG) pair — the complete state of one grammar instance.

    Attributes:
        program:    root ProgramNode.
        ddg:        tuple of DDGEdge; never mutated by rewrite rules.
        node_index: dict mapping node_id -> ComputeNode; built once at init.
    """

    program: ProgramNode
    ddg: tuple  # tuple[DDGEdge, ...]
    node_index: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._build_index(self.program)

    def _build_index(self, node: ASTNode) -> None:
        if isinstance(node, ComputeNode):
            self.node_index[node.node_id] = node
        else:
            for child in node.children:
                self._build_index(child)

    def producers_of(self, consumer_id: int) -> list[DDGEdge]:
        """All DDG edges whose consumer is consumer_id."""
        return [e for e in self.ddg if e.consumer_id == consumer_id]

    def consumers_of(self, producer_id: int) -> list[DDGEdge]:
        """All DDG edges whose producer is producer_id."""
        return [e for e in self.ddg if e.producer_id == producer_id]

    def transitive_producers(self, node_id: int, within: set[int] | None = None) -> set[int]:
        """
        Return the set of node_ids that node_id transitively depends on.
        If `within` is provided, only follow edges whose producer_id is in that set.
        """
        result: set[int] = set()
        queue = [node_id]
        while queue:
            nid = queue.pop()
            for e in self.ddg:
                if e.consumer_id == nid:
                    pid = e.producer_id
                    if pid not in result:
                        if within is None or pid in within:
                            result.add(pid)
                            queue.append(pid)
        return result
