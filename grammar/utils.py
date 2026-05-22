"""
Tree traversal utilities, structural hashing, and in-place node replacement.

All functions operate on immutable nodes and return new objects; none mutate
the input Grammar. Structural sharing is used wherever possible: if a subtree
is unchanged by a replacement, the same object is returned.
"""

from __future__ import annotations

import hashlib
from typing import Iterator

from .ast import (
    ASTNode,
    ComputeNode,
    Grammar,
    LoopLevel,
    ProgramNode,
)

# ---------------------------------------------------------------------------
# Traversal
# ---------------------------------------------------------------------------


def all_compute_nodes(node: ASTNode) -> list[ComputeNode]:
    """Return every ComputeNode in the subtree rooted at node (pre-order)."""
    if isinstance(node, ComputeNode):
        return [node]
    result: list[ComputeNode] = []
    for child in node.children:
        result.extend(all_compute_nodes(child))
    return result


def all_nodes_of_type(node: ASTNode, node_type: type) -> list[ASTNode]:
    """Return every node of the given type in the subtree (pre-order)."""
    result: list[ASTNode] = []
    if isinstance(node, node_type):
        result.append(node)
    if not isinstance(node, ComputeNode):
        for child in node.children:
            result.extend(all_nodes_of_type(child, node_type))
    return result


def find_parent(root: ASTNode, target: ASTNode) -> ASTNode | None:
    """
    Return the immediate parent of `target` in the tree rooted at `root`,
    or None if target is the root or not found.
    Uses identity comparison (is), not value equality.
    """
    if isinstance(root, ComputeNode):
        return None
    for child in root.children:
        if child is target:
            return root
        result = find_parent(child, target)
        if result is not None:
            return result
    return None


def iter_nodes(root: ASTNode) -> Iterator[ASTNode]:
    """Yield every node in the subtree (pre-order)."""
    yield root
    if not isinstance(root, ComputeNode):
        for child in root.children:
            yield from iter_nodes(child)


# ---------------------------------------------------------------------------
# Structural hashing (ignores node_id)
# ---------------------------------------------------------------------------


def _struct_str(node: ASTNode) -> str:
    """Build a canonical string for the subtree, ignoring node_id."""
    if isinstance(node, ComputeNode):
        out = ",".join(sorted(node.output_dims))
        car = ",".join(sorted(node.carried_dims))
        return f"C({node.op},out=[{out}],car=[{car}])"
    if isinstance(node, LoopLevel):
        kids = ",".join(_struct_str(c) for c in node.children)
        par = "par" if node.parallel else "ser"
        return f"L({node.dim},{node.bound},{par},[{kids}])"
    if isinstance(node, ProgramNode):
        kids = ",".join(_struct_str(c) for c in node.children)
        return f"P([{kids}])"
    raise TypeError(f"Unknown node type: {type(node)}")


def tree_hash(grammar: Grammar) -> str:
    """
    SHA-256 of the program tree's canonical string representation.
    node_ids are excluded — this hash is suitable as a compilation cache key.
    Two grammars with identical structure and parameters hash identically.
    """
    raw = _struct_str(grammar.program)
    return hashlib.sha256(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Structural replacement (with sharing)
# ---------------------------------------------------------------------------


def _replace_in_node(node: ASTNode, target: ASTNode, replacement: ASTNode) -> ASTNode:
    """
    Return the subtree rooted at `node` with `target` replaced by `replacement`.
    Uses object identity (is) to locate the target.
    Returns the same object if nothing changed (structural sharing).
    """
    if node is target:
        return replacement
    if isinstance(node, ComputeNode):
        return node  # leaf, not the target
    new_children = tuple(_replace_in_node(c, target, replacement) for c in node.children)
    # Structural sharing: if no child changed by identity, reuse this node.
    if all(nc is oc for nc, oc in zip(new_children, node.children)):
        return node
    if isinstance(node, LoopLevel):
        return LoopLevel(node.dim, node.bound, node.parallel, new_children)
    if isinstance(node, ProgramNode):
        return ProgramNode(new_children)
    raise TypeError(f"Unknown node type: {type(node)}")


def replace_node(grammar: Grammar, target: ASTNode, replacement: ASTNode) -> Grammar:
    """
    Return a new Grammar with `target` replaced by `replacement` in the tree.
    The DDG is shared unchanged.
    """
    new_program = _replace_in_node(grammar.program, target, replacement)
    if new_program is grammar.program:
        return grammar  # nothing changed
    return Grammar(new_program, grammar.ddg)


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------


def pretty_print(grammar: Grammar) -> str:
    """Return a human-readable multi-line string of the grammar tree."""
    lines: list[str] = []
    _fmt_node(grammar.program, "", True, lines)
    lines.append("")
    lines.append("DDG edges:")
    for e in grammar.ddg:
        prod = grammar.node_index.get(e.producer_id)
        cons = grammar.node_index.get(e.consumer_id)
        pname = prod.op if prod else f"id={e.producer_id}"
        cname = cons.op if cons else f"id={e.consumer_id}"
        lines.append(f"  {pname} --[{e.tensor}]--> {cname}")
    return "\n".join(lines)


def _fmt_node(node: ASTNode, prefix: str, last: bool, lines: list[str]) -> None:
    connector = "└── " if last else "├── "
    if isinstance(node, ProgramNode):
        lines.append("ProgramNode")
        for i, child in enumerate(node.children):
            _fmt_node(child, "", i == len(node.children) - 1, lines)
        return
    if isinstance(node, LoopLevel):
        par_str = "parallel" if node.parallel else "serial"
        lines.append(
            f"{prefix}{connector}LoopLevel(dim={node.dim}, "
            f"bound={node.bound}, {par_str})"
        )
        child_prefix = prefix + ("    " if last else "│   ")
        for i, child in enumerate(node.children):
            _fmt_node(child, child_prefix, i == len(node.children) - 1, lines)
        return
    if isinstance(node, ComputeNode):
        out = "{" + ", ".join(sorted(node.output_dims)) + "}"
        car = "{" + ", ".join(sorted(node.carried_dims)) + "}"
        lines.append(
            f"{prefix}{connector}Compute({node.op}, "
            f"output_dims={out}, carried_dims={car})"
        )
        return
    raise TypeError(f"Unknown node type: {type(node)}")
