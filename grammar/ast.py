"""
Core AST node types and the Grammar container.

Design notes:
- All nodes are frozen dataclasses with __slots__ for fast attribute access and
  value-based hashing. Immutability enables structural sharing across rewrites.
- Parameters (dim, tile_size) live directly on LoopLevel nodes — no separate
  param dict — because the parameter space IS the node structure here.
- The DDG is stored alongside the tree in Grammar. Rewrite rules mutate the
  tree but never the DDG.
- ComputeNode carries a node_id (monotonically increasing int) for DDG identity.
  Two ComputeNodes with the same op but different node_ids are different nodes.
- Grammar.node_index is a dict mapping node_id -> ComputeNode, built once on
  construction and reused for O(1) DDG lookups.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

ASTNode = Union["ProgramNode", "KernelScope", "LoopLevel", "ComputeNode"]
LoopChild = Union["LoopLevel", "ComputeNode"]
KernelChild = Union["LoopLevel", "ComputeNode"]

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
        op:      identifies the primitive (e.g. 'MatMul_QK', 'RowMax').
        node_id: unique identity for DDG edge references. Not part of
                 structural equality / tree hash — see utils.tree_hash.
    """

    op: str
    node_id: int

    @staticmethod
    def make(op: str) -> "ComputeNode":
        """Create a new ComputeNode with a fresh unique node_id."""
        return ComputeNode(op=op, node_id=_next_id())


# ---------------------------------------------------------------------------
# Interior nodes
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LoopLevel:
    """
    A single tiled loop over one logical dimension.

    Parameters:
        dim:       logical dimension name (e.g. 'SEQ_Q', 'SEQ_KV', 'HEAD_DIM').
        tile_size: elements processed per iteration; must be a power of 2.
        loop_type: 'parallel' | 'reduction'.
        children:  tuple of LoopLevel or ComputeNode, in execution order.
    """

    dim: str
    tile_size: int
    loop_type: str  # 'parallel' | 'reduction'
    children: tuple  # tuple[LoopChild, ...]

    def __post_init__(self) -> None:
        if self.loop_type not in ("parallel", "reduction"):
            raise ValueError(
                f"loop_type must be 'parallel' or 'reduction', got {self.loop_type!r}"
            )
        if not _is_pow2(self.tile_size):
            raise ValueError(f"tile_size must be a power of 2, got {self.tile_size}")
        if not self.children:
            raise ValueError("LoopLevel must have at least one child")


@dataclass(frozen=True, slots=True)
class KernelScope:
    """
    One GPU kernel launch. Grid/block dims are derived by the renderer from the
    enclosed LoopLevel structure; they are not stored in the AST.

    Children: one or more LoopLevel (or Compute) nodes.
    """

    children: tuple  # tuple[KernelChild, ...]

    def __post_init__(self) -> None:
        if not self.children:
            raise ValueError("KernelScope must have at least one child")


@dataclass(frozen=True, slots=True)
class ProgramNode:
    """
    Root of the AST. Holds one or more KernelScope nodes in execution order.
    """

    children: tuple  # tuple[KernelScope, ...]

    def __post_init__(self) -> None:
        if not self.children:
            raise ValueError("ProgramNode must have at least one KernelScope")


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

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def _build_index(self, node: ASTNode) -> None:
        if isinstance(node, ComputeNode):
            self.node_index[node.node_id] = node
        else:
            for child in node.children:
                self._build_index(child)

    # ------------------------------------------------------------------
    # DDG query helpers (O(|ddg|) — ddg is small in practice)
    # ------------------------------------------------------------------

    def producers_of(self, consumer_id: int) -> list[DDGEdge]:
        """All DDG edges whose consumer is consumer_id."""
        return [e for e in self.ddg if e.consumer_id == consumer_id]

    def consumers_of(self, producer_id: int) -> list[DDGEdge]:
        """All DDG edges whose producer is producer_id."""
        return [e for e in self.ddg if e.producer_id == producer_id]

    def transitive_producers(self, node_id: int, within: set[int] | None = None) -> set[int]:
        """
        Return the set of node_ids that node_id transitively depends on
        (i.e. all ancestor producers in the DDG).

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
