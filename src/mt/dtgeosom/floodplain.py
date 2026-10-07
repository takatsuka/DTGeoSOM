# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Masahiro Takatsuka. See the NOTICE file for attribution terms.
"""
Floodplain analysis: the flattest, shortest path (Bui and Takatsuka, WSOM 2007, Algorithm 3).

A plain shortest path may cross a "mountain range" of the U-matrix -- a cluster border where
no data lie -- because the SOM's lattice joins the two sides.  The flattest path stays on the
*floodplain*: the neurons whose U-height is at most a threshold.

    1. threshold = max(u[start], u[goal])
    2. run the distance transform towards the goal, ignoring every neuron above the threshold
    3. if the start was not reached, raise the threshold to the lowest U-height among the ignored
       neurons the wave bumped into, and go back to 2
    4. descend the distance map from the start (Algorithm 1)

The final threshold is the smallest one at which start and goal are connected -- the *minimax*
(bottleneck) height between them -- so ``threshold='minimax'`` finds it directly with a union-find
sweep over the neurons sorted by U-height and runs the transform once.  Both give the same path.
"""
from __future__ import annotations

import numpy as np
from numpy import ndarray

from mt.dtgeosom.lattice import Lattice
from mt.dtgeosom.paths import NoPathError, SOMPath, descend
from mt.dtgeosom.transform import StepCost, distance_transform, step_cost_table


def floodplain(lattice: Lattice, threshold: float, *always: int) -> ndarray:
    """(n,) bool: neurons whose U-height is at most `threshold` (plus the neurons `always`)."""
    mask = lattice.u_height <= threshold
    for i in always:
        mask[int(i)] = True
    return mask


def minimax_threshold(lattice: Lattice, start: int, goal: int) -> float:
    """
    The smallest threshold t >= max(u[start], u[goal]) for which start and goal are joined by
    neurons of U-height <= t: the height of the lowest pass between them.
    """
    u = lattice.u_height
    start, goal = int(start), int(goal)
    parent = np.arange(lattice.n)

    def find(i):
        root = i
        while parent[root] != root:
            root = parent[root]
        while parent[i] != root:
            parent[i], i = root, parent[i]
        return root

    floor = max(u[start], u[goal])
    order = np.argsort(u, kind='stable')
    added = np.zeros(lattice.n, dtype=bool)
    level = floor
    for i in order:
        if u[i] > floor and find(start) == find(goal):
            break
        added[i] = True
        level = max(level, u[i])
        for m in lattice.neighbours[i]:
            if added[m]:
                ra, rb = find(int(i)), find(int(m))
                if ra != rb:
                    parent[ra] = rb
    if find(start) != find(goal):
        raise NoPathError(f'neurons {start} and {goal} are not connected')
    return float(level)


def flattest_path(lattice: Lattice, start: int, goal: int, *, step: StepCost = 'node',
                  diagonal_factor: float = 1.0, method: str = 'wavefront', threshold: str = 'iterative',
                  rule: str = 'exact') -> SOMPath:
    """
    The flattest, shortest path from `start` to `goal` (Algorithm 3).

    :param threshold: 'iterative' (the paper: raise the threshold until the start is reached) or
                      'minimax' (compute the final threshold first, then run the transform once)
    :param step, diagonal_factor, method: as for mt.dtgeosom.transform.distance_transform
    :param rule: how to descend the distance map: 'exact' or 'steepest' (see mt.dtgeosom.paths)
    """
    start, goal = int(start), int(goal)
    u = lattice.u_height
    costs = None if method == 'paper' else step_cost_table(lattice, step, diagonal_factor)

    if threshold == 'minimax':
        levels = [minimax_threshold(lattice, start, goal)]
    elif threshold == 'iterative':
        levels = [float(max(u[start], u[goal]))]
    else:
        raise ValueError("threshold must be 'iterative' or 'minimax'")

    while True:
        t = levels[-1]
        dt = distance_transform(lattice, goal, step=step, diagonal_factor=diagonal_factor,
                                allowed=floodplain(lattice, t, start, goal), method=method, costs=costs)
        if np.isfinite(dt.distance[start]):
            break
        if not np.isfinite(dt.next_threshold):
            raise NoPathError(f'neurons {start} and {goal} are not connected')
        levels.append(float(dt.next_threshold))

    nodes = descend(lattice, dt, start, rule=rule)
    return SOMPath(nodes, float(dt.distance[start]), 'flattest', dt, threshold=levels[-1], thresholds=levels)


def shortest_path(lattice: Lattice, start: int, goal: int, *, step: StepCost = 'node', diagonal_factor: float = 1.0,
                  method: str = 'wavefront', rule: str = 'exact') -> SOMPath:
    """The shortest path from `start` to `goal` over the whole lattice (Algorithms 2 and 1)."""
    dt = distance_transform(lattice, int(goal), step=step, diagonal_factor=diagonal_factor, method=method)
    nodes = descend(lattice, dt, int(start), rule=rule)
    return SOMPath(nodes, float(dt.distance[int(start)]), 'shortest', dt)
