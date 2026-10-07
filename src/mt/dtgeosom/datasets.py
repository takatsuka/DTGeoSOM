# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Masahiro Takatsuka. See the NOTICE file for attribution terms.
"""
The synthetic data of Bui and Takatsuka (WSOM 2007, section 4.1): a binary tree as a distance matrix.

    from mt.dtgeosom.datasets import binary_tree, tree_path

    x, names = binary_tree(depth=3)     # (15, 15): x[i, j] = hops between tree nodes i+1 and j+1
    tree_path(8, 13)                    # [8, 4, 2, 1, 3, 6, 13]

Nodes are numbered as in a heap: the root is 1 and the children of node k are 2k and 2k + 1.
Each node is one sample, described by its graph distances to all the nodes, so samples close in
the tree are close in data space and a path on a well-trained SOM between two nodes should pass
the tree nodes in between.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy import ndarray


def _ancestors(k: int) -> list[int]:
    chain = [k]
    while k > 1:
        k //= 2
        chain.append(k)
    return chain


def tree_path(a: int, b: int) -> list[int]:
    """The nodes of the binary tree from node a to node b (inclusive), numbered from 1."""
    up_a, up_b = _ancestors(int(a)), _ancestors(int(b))
    common = next(k for k in up_a if k in up_b)
    return up_a[:up_a.index(common) + 1] + list(reversed(up_b[:up_b.index(common)]))


def binary_tree(depth: int = 3) -> tuple[ndarray, list[str]]:
    """
    (x, names): the (n, n) graph-distance matrix of a full binary tree with `depth` levels below the
    root (n = 2^(depth+1) - 1 nodes; 15 for depth 3), and the node names '1' ... 'n'.
    """
    n = 2 ** (depth + 1) - 1
    x = np.array([[len(tree_path(i, j)) - 1 for j in range(1, n + 1)] for i in range(1, n + 1)], dtype=float)
    return x, [str(i) for i in range(1, n + 1)]


def path_agrees_with_tree(labels: Sequence, a: int, b: int, groups: Sequence[set] = ()) -> bool:
    """
    True when the tree nodes met along a SOM path (`labels`, in order) are exactly the nodes of the
    tree path from a to b, in the same order.  Nodes that share a neuron (e.g. siblings mapped to the
    same neuron, as nodes 8 and 9 in the paper) may be listed in `groups`; they are then counted as one.
    """
    merge = {}
    for g in groups:
        g = {int(v) for v in g}
        for v in g:
            merge[v] = min(g)

    def canon(seq):
        out = []
        for v in seq:
            c = merge.get(int(v), int(v))
            if not out or out[-1] != c:
                out.append(c)
        return out

    return canon(labels) == canon(tree_path(a, b))
