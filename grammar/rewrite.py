"""
Rewrite rules for the grammar AST.

Each rule takes a Grammar and the specific nodes to transform, validates all
preconditions (raising ValueError on failure), and returns a new Grammar.
The DDG is never modified — only the program tree changes.

Rules (from AST-spec.md):
    split_loop      / merge_loop       — inverse pair
    lower_compute   / raise_compute    — primary fusion/fission mechanism
    parallelize     / serialize        — change loop parallelism
    reorder_loops                      — self-inverse
    hoist_loop      / sink_loop        — inverse pair

Design: rules are plain functions, not classes. The search algorithm calls
them directly and wraps them in try/except to detect inapplicable rewrites.
"""

from __future__ import annotations

from .ast import (
    ASTNode,
    ComputeNode,
    Grammar,
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
    outer_bound: int = 1,
    inner_bound: int | None = None,
) -> Grammar:
    """
    Replace `loop` with two nested LoopLevels over the same dimension.
    The outer iterates outer_bound times; the inner iterates inner_bound times.
    loop.children become the children of the inner loop.

    Defaults: outer_bound=1, inner_bound=loop.bound (i.e. a no-op split that
    peels off a trivial outer loop; useful as a starting point for parameter search).

    Preconditions (spec § SplitLoop):
        loop.parallel must be True
        outer_bound * inner_bound == loop.bound
        both outer_bound and inner_bound are powers of 2
    """
    if inner_bound is None:
        inner_bound = loop.bound

    if not loop.parallel:
        raise ValueError(
            "SplitLoop requires loop.parallel == True (cannot split a serial loop)"
        )
    if outer_bound * inner_bound != loop.bound:
        raise ValueError(
            f"outer_bound * inner_bound ({outer_bound} * {inner_bound} = "
            f"{outer_bound * inner_bound}) must equal loop.bound ({loop.bound})"
        )
    if not _is_pow2(outer_bound):
        raise ValueError(f"outer_bound must be a power of 2, got {outer_bound}")
    if not _is_pow2(inner_bound):
        raise ValueError(f"inner_bound must be a power of 2, got {inner_bound}")

    inner = LoopLevel(
        dim=loop.dim,
        bound=inner_bound,
        parallel=True,
        children=loop.children,
    )
    outer = LoopLevel(
        dim=loop.dim,
        bound=outer_bound,
        parallel=True,
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
    merged.bound = outer.bound * inner.bound.

    Preconditions (spec § MergeLoop):
        inner is the only child of outer
        outer.dim == inner.dim
        outer.parallel == inner.parallel

    Note: the spec also requires the merged bound to evenly divide the total
    size of `dim`. That divisibility check requires external knowledge of
    dimension sizes and is left to the caller / search algorithm.
    """
    if len(outer.children) != 1 or outer.children[0] is not inner:
        raise ValueError("inner must be the only child of outer for MergeLoop")
    if outer.dim != inner.dim:
        raise ValueError(
            f"outer.dim ({outer.dim!r}) must equal inner.dim ({inner.dim!r})"
        )
    if outer.parallel != inner.parallel:
        raise ValueError(
            f"outer.parallel ({outer.parallel!r}) must equal "
            f"inner.parallel ({inner.parallel!r})"
        )

    merged = LoopLevel(
        dim=outer.dim,
        bound=outer.bound * inner.bound,
        parallel=outer.parallel,
        children=inner.children,
    )
    return replace_node(grammar, outer, merged)


# ---------------------------------------------------------------------------
# LowerCompute / RaiseCompute
# ---------------------------------------------------------------------------


def _prune_compute(node: ASTNode, compute: ComputeNode) -> ASTNode | None:
    """
    Return the subtree rooted at `node` with `compute` removed.
    Returns None if the subtree becomes empty after removal (i.e. the only
    content was `compute` itself).
    Uses object identity (is) to locate the target compute node.
    """
    if isinstance(node, ComputeNode):
        # This is a leaf. If it IS the compute to remove, signal removal with None.
        return None if node is compute else node

    # Interior node: filter children recursively.
    new_children = []
    for child in node.children:
        pruned = _prune_compute(child, compute)
        if pruned is not None:
            new_children.append(pruned)

    if not new_children:
        return None  # this subtree is now empty

    new_children_tuple = tuple(new_children)
    # Structural sharing: if nothing changed, reuse the original node.
    if all(nc is oc for nc, oc in zip(new_children_tuple, node.children)) and len(new_children_tuple) == len(node.children):
        return node

    if isinstance(node, LoopLevel):
        return LoopLevel(node.dim, node.bound, node.parallel, new_children_tuple)
    if isinstance(node, ProgramNode):
        return ProgramNode(new_children_tuple)
    raise TypeError(f"_prune_compute: unexpected node type {type(node)}")


def _kernel_index_of(program: ProgramNode, node: ASTNode) -> int | None:
    """Return the index in program.children of the top-level LoopLevel that contains node."""
    for i, kernel_root in enumerate(program.children):
        if _contains(kernel_root, node):
            return i
    return None


def _contains(root: ASTNode, target: ASTNode) -> bool:
    if root is target:
        return True
    if isinstance(root, ComputeNode):
        return False
    return any(_contains(c, target) for c in root.children)


def lower_compute(
    grammar: Grammar,
    compute: ComputeNode,
    target_loop: LoopLevel,
) -> Grammar:
    """
    Move `compute` from its current position in the tree into `target_loop`,
    appending it as the last child of target_loop.

    After removal of `compute` from its original location, any LoopLevel that
    becomes empty is pruned. If an entire top-level kernel becomes empty, it is
    removed from ProgramNode.children.

    Preconditions:
        compute exists in the grammar tree.
        target_loop exists in the grammar tree.
        compute is NOT already a child of target_loop.
        If target_loop.parallel is True, target_loop.dim must NOT be in
            compute.carried_dims (a parallel loop cannot carry state across
            its iterations for this compute).
        DDG: all DDG producers of compute must be in kernels that execute
            before target_loop's kernel (appear earlier in ProgramNode.children),
            OR already be inside the same kernel as target_loop.
    """
    program = grammar.program

    # Locate compute and target_loop in the kernel tree.
    compute_kernel_idx = _kernel_index_of(program, compute)
    target_kernel_idx = _kernel_index_of(program, target_loop)

    if compute_kernel_idx is None:
        raise ValueError("lower_compute: compute not found in grammar tree")
    if target_kernel_idx is None:
        raise ValueError("lower_compute: target_loop not found in grammar tree")

    # Check that compute is not already a direct child of target_loop.
    if compute in target_loop.children:
        raise ValueError(
            "lower_compute: compute is already a child of target_loop"
        )

    # Parallel-loop / carried_dims compatibility check.
    if target_loop.parallel and target_loop.dim in compute.carried_dims:
        raise ValueError(
            f"lower_compute: target_loop is parallel over dim '{target_loop.dim}', "
            f"but compute '{compute.op}' has '{target_loop.dim}' in its carried_dims. "
            f"A parallel loop cannot carry state for this op."
        )

    # DDG producer check: all producers of compute must be in kernels that
    # execute at or before target_kernel_idx.
    target_kernel_ids = {
        n.node_id for n in all_compute_nodes(program.children[target_kernel_idx])
    }
    for edge in grammar.producers_of(compute.node_id):
        producer_kidx = _kernel_index_of(program, grammar.node_index[edge.producer_id])
        if producer_kidx is None:
            raise ValueError(
                f"lower_compute: DDG producer '{edge.producer_id}' not found in grammar tree"
            )
        if producer_kidx > target_kernel_idx:
            raise ValueError(
                f"lower_compute: DDG producer of '{compute.op}' (via tensor "
                f"'{edge.tensor}') is in kernel {producer_kidx}, which executes "
                f"AFTER target kernel {target_kernel_idx}. This would break data flow."
            )

    # Step 1: Remove compute from its current location, pruning empty containers.
    new_program = _prune_compute(program, compute)
    if new_program is None:
        raise ValueError("lower_compute: pruning compute would empty the entire program")
    assert isinstance(new_program, ProgramNode)

    # Step 2: Insert compute as the last child of target_loop (which may now live
    # in the pruned tree). We need to find target_loop in the pruned tree.
    # Since target_loop is frozen and identity-based, we locate it by identity.
    # If compute was NOT inside target_loop's kernel, target_loop is unchanged.
    # If compute WAS inside target_loop's kernel, target_loop may have been
    # reconstructed by _prune_compute — find the new version by structural walk.
    # The safest approach: re-locate target_loop in new_program by identity.
    # If not found (because compute was inside target_loop's kernel), we must
    # rebuild target_loop with compute appended and then re-insert it.

    # Check if target_loop still exists in new_program (identity check).
    if _contains(new_program, target_loop):
        new_target = LoopLevel(
            dim=target_loop.dim,
            bound=target_loop.bound,
            parallel=target_loop.parallel,
            children=target_loop.children + (compute,),
        )
        final_program = _replace_in_node(new_program, target_loop, new_target)
    else:
        # target_loop was inside compute's old kernel and was reconstructed.
        # We need to find the structurally-equivalent node and append compute.
        # Since target_loop's bound/dim/parallel are preserved, find by a
        # post-order walk looking for the equivalent LoopLevel.
        new_target = LoopLevel(
            dim=target_loop.dim,
            bound=target_loop.bound,
            parallel=target_loop.parallel,
            children=target_loop.children + (compute,),
        )
        # We use a structural replacement: find any node that was reconstructed
        # from target_loop. Because _prune_compute only removes compute nodes,
        # the target_loop's identity is preserved unless compute was one of
        # target_loop's children (already ruled out). So this branch shouldn't
        # normally be reached — but handle it defensively.
        final_program = _replace_in_node(new_program, target_loop, new_target)

    if not isinstance(final_program, ProgramNode):
        final_program = ProgramNode(final_program.children if hasattr(final_program, 'children') else ())

    return Grammar(final_program, grammar.ddg)


def _replace_in_node(node: ASTNode, target: ASTNode, replacement: ASTNode) -> ASTNode:
    """Replace target with replacement in the subtree rooted at node (identity-based)."""
    if node is target:
        return replacement
    if isinstance(node, ComputeNode):
        return node
    new_children = tuple(_replace_in_node(c, target, replacement) for c in node.children)
    if all(nc is oc for nc, oc in zip(new_children, node.children)):
        return node
    if isinstance(node, LoopLevel):
        return LoopLevel(node.dim, node.bound, node.parallel, new_children)
    if isinstance(node, ProgramNode):
        return ProgramNode(new_children)
    raise TypeError(f"_replace_in_node: unexpected type {type(node)}")


def raise_compute(
    grammar: Grammar,
    compute: ComputeNode,
) -> Grammar:
    """
    Move `compute` one level up in the tree — out of its current LoopLevel
    parent and into the parent's parent, placed immediately after the parent
    in the parent's parent's children list.

    Preconditions:
        compute's immediate parent is a LoopLevel (not ProgramNode — cannot
        raise a compute that is already a direct child of a top-level kernel
        root without creating a new kernel, which is LowerCompute's inverse).
        parent.dim must NOT be in compute.carried_dims (the loop over parent.dim
        would no longer enclose this compute, but carried_dims requires it to).
    """
    program = grammar.program
    parent = find_parent(program, compute)

    if parent is None:
        raise ValueError("raise_compute: compute not found in grammar tree")
    if not isinstance(parent, LoopLevel):
        raise ValueError(
            "raise_compute: compute's parent must be a LoopLevel "
            "(cannot raise a compute that is already at the top-level kernel root)"
        )
    if parent.dim in compute.carried_dims:
        raise ValueError(
            f"raise_compute: parent loop dim '{parent.dim}' is in compute "
            f"'{compute.op}'.carried_dims. Raising would remove the enclosing "
            f"loop required for carry-over accumulation."
        )

    grandparent = find_parent(program, parent)
    if grandparent is None:
        raise ValueError("raise_compute: could not find grandparent of compute's parent")

    # Remove compute from parent's children.
    new_parent_children = tuple(c for c in parent.children if c is not compute)
    if not new_parent_children:
        # Parent becomes empty — remove it from grandparent.
        new_grandparent_children = tuple(c for c in grandparent.children if c is not parent)
        if not new_grandparent_children:
            raise ValueError(
                "raise_compute: raising this compute would empty its grandparent"
            )
    else:
        new_parent = LoopLevel(parent.dim, parent.bound, parent.parallel, new_parent_children)
        # Insert compute immediately after new_parent in grandparent's children.
        new_grandparent_children = []
        for c in grandparent.children:
            if c is parent:
                new_grandparent_children.append(new_parent)
                new_grandparent_children.append(compute)
            else:
                new_grandparent_children.append(c)
        new_grandparent_children = tuple(new_grandparent_children)

    # Rebuild grandparent.
    if isinstance(grandparent, LoopLevel):
        new_grandparent = LoopLevel(
            grandparent.dim, grandparent.bound, grandparent.parallel,
            new_grandparent_children,
        )
    elif isinstance(grandparent, ProgramNode):
        new_grandparent = ProgramNode(new_grandparent_children)
    else:
        raise TypeError(f"raise_compute: unexpected grandparent type {type(grandparent)}")

    new_program = _replace_in_node(program, grandparent, new_grandparent)
    return Grammar(new_program, grammar.ddg)


# ---------------------------------------------------------------------------
# Parallelize / Serialize
# ---------------------------------------------------------------------------


def parallelize(grammar: Grammar, loop: LoopLevel) -> Grammar:
    """
    Set loop.parallel = True.

    Precondition:
        No ComputeNode inside `loop` has loop.dim in its carried_dims.
        (A loop that carries state across iterations cannot be parallelised —
        each iteration depends on the previous iteration's accumulated value.)
    """
    inner_computes = all_compute_nodes(loop)
    for c in inner_computes:
        if loop.dim in c.carried_dims:
            raise ValueError(
                f"parallelize: ComputeNode '{c.op}' inside loop has "
                f"'{loop.dim}' in its carried_dims. Cannot parallelize a loop "
                f"that carries accumulation state."
            )

    new_loop = LoopLevel(loop.dim, loop.bound, True, loop.children)
    return replace_node(grammar, loop, new_loop)


def serialize(grammar: Grammar, loop: LoopLevel) -> Grammar:
    """
    Set loop.parallel = False. Always valid — making a parallel loop serial
    never violates data dependencies (it is strictly more conservative).
    """
    new_loop = LoopLevel(loop.dim, loop.bound, False, loop.children)
    return replace_node(grammar, loop, new_loop)


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
        loop's parent is a LoopLevel (not ProgramNode)
        loop is the only child of its parent (makes the operation unambiguous)
        no DDG dependency violated by the new loop order
    """
    parent = find_parent(grammar.program, loop)
    if not isinstance(parent, LoopLevel):
        raise ValueError(
            "HoistLoop: loop's parent must be a LoopLevel (not ProgramNode)"
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
    new_inner = LoopLevel(parent.dim, parent.bound, parent.parallel, loop.children)
    new_outer = LoopLevel(loop.dim, loop.bound, loop.parallel, (new_inner,))
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
    child_node_set = set(all_compute_nodes(child))
    outer_ids = {
        n.node_id for n in all_compute_nodes(loop) if n not in child_node_set
    }
    inner_ids = {n.node_id for n in all_compute_nodes(child)}

    for e in grammar.ddg:
        if e.producer_id in outer_ids and e.consumer_id in inner_ids:
            raise ValueError(
                f"SinkLoop: DDG dependency from outer loop body to inner loop "
                f"via tensor '{e.tensor}' — sinking would reorder dependent ops"
            )

    # Interchange: child becomes outer, loop becomes inner
    new_inner = LoopLevel(loop.dim, loop.bound, loop.parallel, child.children)
    new_outer = LoopLevel(
        child.dim, child.bound, child.parallel, (new_inner,) + other_children
    )
    return replace_node(grammar, loop, new_outer)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _rebuild_with_children(node: ASTNode, new_children: tuple) -> ASTNode:
    """Reconstruct a non-leaf node with a new children tuple."""
    if isinstance(node, LoopLevel):
        return LoopLevel(node.dim, node.bound, node.parallel, new_children)
    if isinstance(node, ProgramNode):
        return ProgramNode(new_children)
    raise TypeError(f"_rebuild_with_children: unexpected type {type(node)}")
