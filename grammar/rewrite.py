"""
Rewrite rules for the grammar AST.

Each rule takes a Grammar and the specific nodes to transform, validates all
preconditions (raising ValueError on failure), and returns a new Grammar.
The DDG is never modified — only the program tree changes.

Rules (from AST-spec.md):
    split_loop      / merge_loop       — inverse pair
    fuse_kernels    / unfuse_kernels   — inverse pair
    reorder_loops                      — self-inverse
    hoist_loop      / sink_loop        — inverse pair

Design: rules are plain functions, not classes. The search algorithm calls
them directly and wraps them in try/except to detect inapplicable rewrites.
"""

from __future__ import annotations

from .ast import (
    ASTNode,
    ComputeNode,
    DDGEdge,
    Grammar,
    KernelScope,
    LoopLevel,
    ProgramNode,
)
from .utils import (
    all_compute_nodes,
    find_parent,
    replace_node,
)


def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


# ---------------------------------------------------------------------------
# SplitLoop / MergeLoop
# ---------------------------------------------------------------------------


def split_loop(
    grammar: Grammar,
    loop: LoopLevel,
    outer_tile: int,
    inner_tile: int,
) -> Grammar:
    """
    Replace `loop` with two nested LoopLevels over the same dimension.
    The outer iterates with tile_size=outer_tile; the inner with tile_size=inner_tile.
    loop.children become the children of the inner loop.

    Preconditions (spec § SplitLoop):
        loop.loop_type == 'parallel'
        outer_tile * inner_tile == loop.tile_size
        both outer_tile and inner_tile are powers of 2
    """
    if loop.loop_type != "parallel":
        raise ValueError(
            f"SplitLoop requires loop_type == 'parallel', got {loop.loop_type!r}"
        )
    if outer_tile * inner_tile != loop.tile_size:
        raise ValueError(
            f"outer_tile * inner_tile ({outer_tile} * {inner_tile} = "
            f"{outer_tile * inner_tile}) must equal loop.tile_size ({loop.tile_size})"
        )
    if not _is_pow2(outer_tile):
        raise ValueError(f"outer_tile must be a power of 2, got {outer_tile}")
    if not _is_pow2(inner_tile):
        raise ValueError(f"inner_tile must be a power of 2, got {inner_tile}")

    inner = LoopLevel(
        dim=loop.dim,
        tile_size=inner_tile,
        loop_type="parallel",
        children=loop.children,
    )
    outer = LoopLevel(
        dim=loop.dim,
        tile_size=outer_tile,
        loop_type="parallel",
        children=(inner,),
    )
    return replace_node(grammar, loop, outer)


def merge_loop(
    grammar: Grammar,
    outer: LoopLevel,
    inner: LoopLevel,
) -> Grammar:
    """
    Collapse two directly nested LoopLevels over the same dimension into one.
    merged.tile_size = outer.tile_size * inner.tile_size.

    Preconditions (spec § MergeLoop):
        inner is the only child of outer
        outer.dim == inner.dim
        outer.loop_type == inner.loop_type

    Note: the spec also requires the merged tile_size to evenly divide the
    total size of `dim`. That divisibility check requires external knowledge
    of dimension sizes and is left to the caller / search algorithm.
    """
    if len(outer.children) != 1 or outer.children[0] is not inner:
        raise ValueError("inner must be the only child of outer for MergeLoop")
    if outer.dim != inner.dim:
        raise ValueError(
            f"outer.dim ({outer.dim!r}) must equal inner.dim ({inner.dim!r})"
        )
    if outer.loop_type != inner.loop_type:
        raise ValueError(
            f"outer.loop_type ({outer.loop_type!r}) must equal "
            f"inner.loop_type ({inner.loop_type!r})"
        )

    merged = LoopLevel(
        dim=outer.dim,
        tile_size=outer.tile_size * inner.tile_size,
        loop_type=outer.loop_type,
        children=inner.children,
    )
    return replace_node(grammar, outer, merged)


# ---------------------------------------------------------------------------
# FuseKernels / UnfuseKernels
# ---------------------------------------------------------------------------


def fuse_kernels(
    grammar: Grammar,
    scope_a: KernelScope,
    scope_b: KernelScope,
) -> Grammar:
    """
    Merge two adjacent KernelScope siblings into one, eliminating the global
    memory round-trip for tensors produced by scope_a and consumed by scope_b.

    Preconditions (spec § FuseKernels):
        scope_a immediately precedes scope_b under ProgramNode
        no A→B tensor is also consumed by a third scope
        (shared memory limit check is deferred to the renderer)
    """
    program = grammar.program
    kids = program.children

    idx_a = next((i for i, c in enumerate(kids) if c is scope_a), None)
    idx_b = next((i for i, c in enumerate(kids) if c is scope_b), None)

    if idx_a is None:
        raise ValueError("scope_a not found in ProgramNode.children")
    if idx_b is None:
        raise ValueError("scope_b not found in ProgramNode.children")
    if idx_b != idx_a + 1:
        raise ValueError(
            f"scope_a (idx={idx_a}) must immediately precede scope_b (idx={idx_b})"
        )

    a_ids = {n.node_id for n in all_compute_nodes(scope_a)}
    b_ids = {n.node_id for n in all_compute_nodes(scope_b)}

    # ids of compute nodes in all other scopes
    other_ids: set[int] = set()
    for i, s in enumerate(kids):
        if i != idx_a and i != idx_b:
            for n in all_compute_nodes(s):
                other_ids.add(n.node_id)

    # Tensors that flow from A to B
    a_to_b_tensors = {
        e.tensor
        for e in grammar.ddg
        if e.producer_id in a_ids and e.consumer_id in b_ids
    }
    # If any of those tensors also flow to a third scope, fusion is illegal
    for e in grammar.ddg:
        if e.producer_id in a_ids and e.consumer_id in other_ids:
            if e.tensor in a_to_b_tensors:
                raise ValueError(
                    f"Cannot fuse: tensor '{e.tensor}' is produced by scope_a, "
                    f"consumed by scope_b AND by another scope. Removing the global "
                    f"materialization would break the third consumer."
                )

    fused = KernelScope(children=scope_a.children + scope_b.children)
    new_kids = kids[:idx_a] + (fused,) + kids[idx_b + 1 :]
    return Grammar(ProgramNode(new_kids), grammar.ddg)


def unfuse_kernels(
    grammar: Grammar,
    scope: KernelScope,
    split_point: ComputeNode,
) -> Grammar:
    """
    Split `scope` into two at `split_point`.

    All ComputeNodes that `split_point` transitively depends on (within the
    scope) stay in the first scope. `split_point` and all ComputeNodes that
    transitively depend on it move to the second scope.

    The loop-level tree structure is partitioned accordingly: a LoopLevel that
    contains nodes destined for both scopes is duplicated in both, and empty
    branches are pruned.

    Preconditions (spec § UnfuseKernels):
        split_point is inside scope
        both resulting scopes contain at least one ComputeNode
        all inputs to the second scope are produced by the first scope
        (the transitive-dep partition guarantees this by construction)
    """
    kids = grammar.program.children
    idx = next((i for i, c in enumerate(kids) if c is scope), None)
    if idx is None:
        raise ValueError("scope not found in ProgramNode.children")

    scope_nodes = all_compute_nodes(scope)
    scope_ids = {n.node_id for n in scope_nodes}

    if split_point.node_id not in scope_ids:
        raise ValueError("split_point is not inside scope")

    # Compute nodes that split_point transitively depends on → scope 1
    scope1_ids = grammar.transitive_producers(split_point.node_id, within=scope_ids)
    # split_point + everything that depends on it → scope 2
    scope2_ids = scope_ids - scope1_ids

    if not scope1_ids:
        raise ValueError("UnfuseKernels: first scope would be empty")
    if not scope2_ids:
        raise ValueError("UnfuseKernels: second scope would be empty")

    # Partition the tree structure
    s1_children, s2_children = _partition_children(scope.children, scope1_ids, scope2_ids)

    if not s1_children:
        raise ValueError("UnfuseKernels: first scope tree is empty after partition")
    if not s2_children:
        raise ValueError("UnfuseKernels: second scope tree is empty after partition")

    scope1 = KernelScope(tuple(s1_children))
    scope2 = KernelScope(tuple(s2_children))
    new_kids = kids[:idx] + (scope1, scope2) + kids[idx + 1 :]
    return Grammar(ProgramNode(new_kids), grammar.ddg)


def _partition_subtree(
    node: ASTNode,
    scope1_ids: set[int],
    scope2_ids: set[int],
) -> tuple[ASTNode | None, ASTNode | None]:
    """
    Recursively partition a subtree.
    Returns (subtree_for_scope1, subtree_for_scope2); either may be None if
    no compute nodes for that scope exist in this subtree.
    """
    if isinstance(node, ComputeNode):
        if node.node_id in scope1_ids:
            return node, None
        if node.node_id in scope2_ids:
            return None, node
        return None, None  # not in either scope (shouldn't happen)

    s1_children: list[ASTNode] = []
    s2_children: list[ASTNode] = []
    for child in node.children:
        s1, s2 = _partition_subtree(child, scope1_ids, scope2_ids)
        if s1 is not None:
            s1_children.append(s1)
        if s2 is not None:
            s2_children.append(s2)

    if isinstance(node, LoopLevel):
        s1_node = (
            LoopLevel(node.dim, node.tile_size, node.loop_type, tuple(s1_children))
            if s1_children
            else None
        )
        s2_node = (
            LoopLevel(node.dim, node.tile_size, node.loop_type, tuple(s2_children))
            if s2_children
            else None
        )
    else:
        raise TypeError(f"_partition_subtree: unexpected node type {type(node)}")

    return s1_node, s2_node


def _partition_children(
    children: tuple,
    scope1_ids: set[int],
    scope2_ids: set[int],
) -> tuple[list[ASTNode], list[ASTNode]]:
    """Partition a sequence of children into two lists (scope1, scope2)."""
    s1: list[ASTNode] = []
    s2: list[ASTNode] = []
    for child in children:
        n1, n2 = _partition_subtree(child, scope1_ids, scope2_ids)
        if n1 is not None:
            s1.append(n1)
        if n2 is not None:
            s2.append(n2)
    return s1, s2


# ---------------------------------------------------------------------------
# ReorderLoops
# ---------------------------------------------------------------------------


def reorder_loops(
    grammar: Grammar,
    loop_a: LoopLevel,
    loop_b: LoopLevel,
) -> Grammar:
    """
    Swap two sibling LoopLevel nodes, changing loop execution order.
    Self-inverse.

    Preconditions (spec § ReorderLoops):
        loop_a and loop_b are siblings under the same parent node
        no DDG dependency (in either direction) between compute nodes
        inside loop_a and compute nodes inside loop_b
    """
    parent_a = find_parent(grammar.program, loop_a)
    parent_b = find_parent(grammar.program, loop_b)

    if parent_a is None or parent_b is None:
        raise ValueError("Both loops must exist in the grammar tree")
    if parent_a is not parent_b:
        raise ValueError("loop_a and loop_b must be siblings under the same parent")

    parent = parent_a

    # DDG dependency check
    a_ids = {n.node_id for n in all_compute_nodes(loop_a)}
    b_ids = {n.node_id for n in all_compute_nodes(loop_b)}

    for e in grammar.ddg:
        if (e.producer_id in a_ids and e.consumer_id in b_ids) or (
            e.producer_id in b_ids and e.consumer_id in a_ids
        ):
            raise ValueError(
                f"ReorderLoops: DDG dependency between loop_a and loop_b "
                f"via tensor '{e.tensor}' — reordering would violate data flow"
            )

    # Swap
    new_children = tuple(
        loop_b if c is loop_a else (loop_a if c is loop_b else c)
        for c in parent.children
    )
    new_parent = _rebuild_with_children(parent, new_children)
    return replace_node(grammar, parent, new_parent)


# ---------------------------------------------------------------------------
# HoistLoop / SinkLoop
# ---------------------------------------------------------------------------


def hoist_loop(grammar: Grammar, loop: LoopLevel) -> Grammar:
    """
    Move `loop` one level upward, making it the parent of its current parent.
    Equivalent to loop interchange: if P is the parent, the result is
        loop(children=(P(children=loop.children),))
    replacing P in the original tree.

    Preconditions (spec § HoistLoop):
        loop's parent is a LoopLevel (not a KernelScope or ProgramNode)
        loop is the only child of its parent (makes the operation unambiguous)
        no DDG dependency violated by the new loop order
    """
    parent = find_parent(grammar.program, loop)
    if not isinstance(parent, LoopLevel):
        raise ValueError(
            "HoistLoop: loop's parent must be a LoopLevel (not KernelScope/ProgramNode)"
        )
    if len(parent.children) != 1 or parent.children[0] is not loop:
        raise ValueError(
            "HoistLoop: loop must be the only child of its parent "
            "(otherwise the hoist target is ambiguous)"
        )

    # DDG check: no compute node outside `loop` (but inside `parent`) needs
    # to execute before a node inside `loop`. Since loop is the only child,
    # there are no such nodes — the check is trivially satisfied.

    # Interchange: loop becomes outer, parent becomes inner
    new_inner = LoopLevel(parent.dim, parent.tile_size, parent.loop_type, loop.children)
    new_outer = LoopLevel(loop.dim, loop.tile_size, loop.loop_type, (new_inner,))
    return replace_node(grammar, parent, new_outer)


def sink_loop(grammar: Grammar, loop: LoopLevel) -> Grammar:
    """
    Move `loop` one level downward into its single LoopLevel child.
    Equivalent to loop interchange: if `child` is the one LoopLevel child,
    the result is child(children=(loop(children=child.children), *others))
    replacing `loop` in the original tree.

    Preconditions (spec § SinkLoop):
        loop has exactly one LoopLevel child (otherwise target is ambiguous)
        no DDG dependency violated by the new loop order
    """
    loop_children = [c for c in loop.children if isinstance(c, LoopLevel)]
    if len(loop_children) != 1:
        raise ValueError(
            f"SinkLoop: loop must have exactly one LoopLevel child, "
            f"found {len(loop_children)}"
        )

    child = loop_children[0]
    other_children = tuple(c for c in loop.children if c is not child)

    # DDG check: no compute node in `loop` (outside `child`) depends on
    # a compute node inside `child` in a way that requires loop to execute first.
    outer_ids = {n.node_id for n in all_compute_nodes(loop) if n not in all_compute_nodes(child)}
    inner_ids = {n.node_id for n in all_compute_nodes(child)}

    for e in grammar.ddg:
        if e.producer_id in outer_ids and e.consumer_id in inner_ids:
            raise ValueError(
                f"SinkLoop: DDG dependency from outer loop body to inner loop "
                f"via tensor '{e.tensor}' — sinking would reorder dependent ops"
            )

    # Interchange: child becomes outer, loop becomes inner
    new_inner = LoopLevel(loop.dim, loop.tile_size, loop.loop_type, child.children)
    new_outer = LoopLevel(
        child.dim, child.tile_size, child.loop_type, (new_inner,) + other_children
    )
    return replace_node(grammar, loop, new_outer)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _rebuild_with_children(node: ASTNode, new_children: tuple) -> ASTNode:
    """Reconstruct a non-leaf node with a new children tuple."""
    if isinstance(node, LoopLevel):
        return LoopLevel(node.dim, node.tile_size, node.loop_type, new_children)
    if isinstance(node, KernelScope):
        return KernelScope(new_children)
    if isinstance(node, ProgramNode):
        return ProgramNode(new_children)
    raise TypeError(f"_rebuild_with_children: unexpected type {type(node)}")
