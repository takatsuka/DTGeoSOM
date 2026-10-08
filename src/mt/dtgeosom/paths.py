# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Masahiro Takatsuka. See the NOTICE file for attribution terms.
"""
Reading a path off a distance map (Bui and Takatsuka, WSOM 2007, Algorithm 1), and the path object.

Once the distance transform has given every neuron its distance to the goal, a path from the start
neuron is found by walking downhill on the distance map until the goal (distance 0) is reached.

    rule='exact'     (default) move to the neighbour m for which  distance[m] + cost(m -> here)  is
                     smallest -- the neighbour the wave actually came from -- so the walk is a
                     cheapest path for the step cost used by the transform.
    rule='steepest'  Algorithm 1 as published: move to the neighbour with the lowest distance.  With
                     the paper's 'node' step cost both rules pick the same neurons (up to ties).
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy import ndarray

from mt.dtgeosom.lattice import Lattice
from mt.dtgeosom.transform import StepCost, Transform, step_cost_function


class NoPathError(RuntimeError):
    """The goal cannot be reached from the start."""


def descend(lattice: Lattice, transform: Transform, start: int, *, rule: str = 'exact') -> ndarray:
    """
    The neurons of a path from `start` to the transform's goal, start first.

    :param lattice: the Lattice the transform was computed on
    :param transform: a Transform (distance_transform(...))
    :param start: the start neuron
    :param rule: 'exact' (follow the wave back) or 'steepest' (Algorithm 1)
    """
    if rule not in ('exact', 'steepest'):
        raise ValueError("rule must be 'exact' or 'steepest'")
    dist, goal = transform.distance, transform.goal
    start = int(start)
    if not np.isfinite(dist[start]):
        raise NoPathError(f'neuron {goal} cannot be reached from neuron {start}')
    if rule == 'exact' and transform.method == 'paper':
        rule = 'steepest'                                  # the paper's map has no single step cost
    cost = step_cost_function(lattice, transform.step, transform.diagonal_factor) if rule == 'exact' else None

    path, seen, c = [start], {start}, start
    while c != goal:
        nb = lattice.neighbours[c]
        reachable = np.isfinite(dist[nb])
        if not reachable.any():
            raise NoPathError(f'stuck at neuron {c}')
        if rule == 'steepest':
            score = np.where(reachable, dist[nb], np.inf)
        else:
            score = np.array([dist[m] + cost(int(m), c, bool(d)) if ok else np.inf
                              for m, d, ok in zip(nb, lattice.diagonal[c], reachable, strict=True)])
        # among equally good neighbours prefer the one lower on the map, then the one geometrically nearer the
        # goal (a straighter path when many are equally short, e.g. counting hops), then one not yet visited
        near = (np.linalg.norm(lattice.positions[nb] - lattice.positions[goal], axis=1)
                if lattice.positions is not None else np.zeros(len(nb)))
        order = np.lexsort((np.isin(nb, list(seen)), np.round(near, 9), dist[nb], np.round(score, 12)))
        nxt = int(nb[order[0]])
        if rule == 'steepest' and dist[nxt] >= dist[c] and dist[c] > 0:
            raise NoPathError(f'no neighbour of neuron {c} is lower on the distance map')
        if nxt in seen:
            raise NoPathError(f'the descent returned to neuron {nxt}')
        path.append(nxt)
        seen.add(nxt)
        c = nxt
    return np.array(path, dtype=int)


def path_cost(lattice: Lattice, nodes: Sequence[int], step: StepCost = 'node', diagonal_factor: float = 1.0) -> float:
    """Cost of walking `nodes` (start ... goal) with the given step cost (the wave runs goal -> start)."""
    f = step_cost_function(lattice, step, diagonal_factor)
    total = 0.0
    for a, b in zip(nodes[:-1], nodes[1:], strict=True):     # walker a -> b, so the wave went b -> a
        total += f(int(b), int(a), lattice.is_diagonal(int(b), int(a)))
    return float(total)


@dataclass
class SOMPath:
    """
    A path between two neurons of a SOM.

    nodes        neuron indices from start to goal
    cost         distance of the start on the final distance map
    kind         'shortest', 'flattest', 'hops' or 'edge' (see mt.dtgeosom.pathfinder.KINDS)
    transform    the (last) Transform: its .distance is the distance map
    threshold    the U-height threshold of the floodplain that holds the path (flattest paths)
    thresholds   every threshold tried, in order (flattest paths)
    """

    nodes: ndarray
    cost: float
    kind: str
    transform: Transform
    threshold: float | None = None
    thresholds: list[float] = field(default_factory=list)

    @property
    def start(self) -> int:
        return int(self.nodes[0])

    @property
    def goal(self) -> int:
        return int(self.nodes[-1])

    @property
    def distance(self) -> ndarray:
        """The distance map the path was read from."""
        return self.transform.distance

    def __len__(self) -> int:
        return len(self.nodes)

    @property
    def hops(self) -> int:
        """Number of steps between neighbouring neurons."""
        return len(self.nodes) - 1

    def states(self, som) -> ndarray:
        """(len, dim) weight vectors along the path: the intermediate states from start to goal."""
        return np.asarray(som.weights)[self.nodes]

    def max_height(self, u_height: ndarray) -> float:
        """The highest U-height the path climbs over."""
        return float(np.max(np.asarray(u_height)[self.nodes]))

    def visited_labels(self, node_labels: Sequence) -> list:
        """
        The labels met along the path, in order and without repeats.

        :param node_labels: one entry per neuron: None, a label, or a list/dict of labels
                            (e.g. som.bmu_labels(...) or som.node_labels(...))
        """
        seen: list = []
        for i in self.nodes:
            entry = node_labels[int(i)]
            if entry is None:
                continue
            labels = sorted(entry) if isinstance(entry, dict | set | list | tuple) else [entry]
            for lab in labels:
                if lab not in seen:
                    seen.append(lab)
        return seen

    def __repr__(self) -> str:
        extra = f', threshold={self.threshold:.4g}' if self.threshold is not None else ''
        return f'SOMPath({self.kind}, {self.start} -> {self.goal}, {len(self)} neurons, cost={self.cost:.4g}{extra})'
