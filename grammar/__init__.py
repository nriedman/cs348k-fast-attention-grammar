"""
grammar — parametric shape grammar representation for GPU kernel optimization.

Public API:

    Node types:
        ComputeNode, LoopLevel, KernelScope, ProgramNode

    DDG / Grammar:
        DDGEdge, Grammar

    Atomic grammar:
        attention_atomic_grammar()

    Tree utilities:
        tree_hash, pretty_print
        all_compute_nodes, all_nodes_of_type, find_parent, scope_of

    Rewrite rules:
        split_loop, merge_loop
        fuse_kernels, unfuse_kernels
        reorder_loops
        hoist_loop, sink_loop
"""

from .ast import (
    ComputeNode,
    DDGEdge,
    Grammar,
    KernelScope,
    LoopLevel,
    ProgramNode,
)
from .atomic import attention_atomic_grammar
from .rewrite import (
    fuse_kernels,
    hoist_loop,
    merge_loop,
    reorder_loops,
    sink_loop,
    split_loop,
    unfuse_kernels,
)
from .utils import (
    all_compute_nodes,
    all_nodes_of_type,
    find_parent,
    pretty_print,
    scope_of,
    tree_hash,
)

__all__ = [
    # Node types
    "ComputeNode",
    "DDGEdge",
    "Grammar",
    "KernelScope",
    "LoopLevel",
    "ProgramNode",
    # Atomic grammar
    "attention_atomic_grammar",
    # Rewrite rules
    "split_loop",
    "merge_loop",
    "fuse_kernels",
    "unfuse_kernels",
    "reorder_loops",
    "hoist_loop",
    "sink_loop",
    # Utilities
    "tree_hash",
    "pretty_print",
    "all_compute_nodes",
    "all_nodes_of_type",
    "find_parent",
    "scope_of",
]
