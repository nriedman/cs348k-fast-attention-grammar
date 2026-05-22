"""
grammar — parametric shape grammar representation for GPU kernel optimization.

Public API:

    Node types:
        ComputeNode, LoopLevel, ProgramNode

    DDG / Grammar:
        DDGEdge, Grammar

    Atomic grammar:
        attention_atomic_grammar()

    Tree utilities:
        tree_hash, pretty_print
        all_compute_nodes, all_nodes_of_type, find_parent, iter_nodes

    Rewrite rules:
        split_loop, merge_loop
        lower_compute, raise_compute
        parallelize, serialize
        reorder_loops
        hoist_loop, sink_loop
"""

from .ast import (
    ComputeNode,
    DDGEdge,
    Grammar,
    LoopLevel,
    ProgramNode,
)
from .atomic import attention_atomic_grammar
from .rewrite import (
    hoist_loop,
    lower_compute,
    merge_loop,
    parallelize,
    raise_compute,
    reorder_loops,
    serialize,
    sink_loop,
    split_loop,
)
from .utils import (
    all_compute_nodes,
    all_nodes_of_type,
    find_parent,
    iter_nodes,
    pretty_print,
    tree_hash,
)

__all__ = [
    # Node types
    "ComputeNode",
    "DDGEdge",
    "Grammar",
    "LoopLevel",
    "ProgramNode",
    # Atomic grammar
    "attention_atomic_grammar",
    # Rewrite rules
    "split_loop",
    "merge_loop",
    "lower_compute",
    "raise_compute",
    "parallelize",
    "serialize",
    "reorder_loops",
    "hoist_loop",
    "sink_loop",
    # Utilities
    "tree_hash",
    "pretty_print",
    "all_compute_nodes",
    "all_nodes_of_type",
    "find_parent",
    "iter_nodes",
]
