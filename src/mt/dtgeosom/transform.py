# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Masahiro Takatsuka. See the NOTICE file for attribution terms.
"""
The distance transform on a SOM lattice (Bui and Takatsuka, WSOM 2007, Algorithm 2).

A *distance wave* starts at the goal neuron (distance 0) and is propagated to the neighbours of
every neuron it reaches, through a first-in-first-out queue.  When a neuron is reached, or reached
again more cheaply, it gets the smaller distance and goes back into the queue, so the wave keeps
spreading until nothing can be improved.  The result is a *distance map*: for every neuron, the
cost of the cheapest walk from it to the goal over the lattice.  A path is then read off the map
by descending it from the start neuron (Algorithm 1, see mt.dtgeosom.paths).

    from mt.dtgeosom.lattice import Lattice
    from mt.dtgeosom.transform import distance_transform

    lattice = Lattice.from_som(som)
    dt = distance_transform(lattice, goal=17)          # dt.distance: (n,) cost to reach neuron 17
    dt.distance[42]                                    # inf where the goal cannot be reached

Cost of one step (``step``)
    The paper propagates the U-height (the *average difference* between a neuron's weight vector
    and its neighbours') of the neuron the wave moves into:  n.distance = c.distance + n.u_height.
    The cost of a path is then the sum of the U-heights of its neurons (all but the goal), so
    paths avoid the "mountain ranges" of the U-matrix.

    'node'   (default, the paper)  cost(c -> n) = u[n]
    'mean'                         cost(c -> n) = (u[c] + u[n]) / 2      (the same in both directions)
    'edge'                         cost(c -> n) = |w_c - w_n|           (distance between the two weight
                                                                         vectors: shortest path in data space)
    a callable f(c, n) -> cost     any other non-negative cost

    ``diagonal_factor`` multiplies the cost of a step to a diagonal neighbour of the 2D index array
    (default 1: on the sphere every neighbour is one ring away; the paper used sqrt(2)).

Propagation (``method``)
    'wavefront' (default)  the paper's FIFO-queue distance wave, with every improved neuron re-queued
                           (a label-correcting search; a neuron already waiting is not queued twice)
    'dijkstra'             Dijkstra's algorithm with a binary heap -- the same distance map, for checks
    'paper'                Algorithm 2 transcribed line by line: a newly reached neuron n gets
                           c.distance + n.u_height, and a neuron already reached is improved with
                           c.distance + sqrt(2) c.u_height across a diagonal of the 2D index array and
                           c.distance + n.u_height otherwise.  (``step`` and ``diagonal_factor`` are
                           ignored.)  The first two rules disagree, so the map depends on the order
                           in which neurons are visited -- kept for reproducing the published results.

Restricting the wave (``allowed``)
    Neurons outside the boolean mask ``allowed`` are ignored: the wave never enters them.  The
    floodplain analysis (Algorithm 3, mt.dtgeosom.floodplain) uses this to keep paths below a
    U-height threshold; the transform reports the lowest U-height among the ignored neurons it
    bumped into (``next_threshold``), which is where the threshold goes next.
"""
from __future__ import annotations

import heapq
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy import ndarray

from mt.dtgeosom.lattice import SQRT2, Lattice

STEPS = ('node', 'mean', 'edge')
METHODS = ('wavefront', 'dijkstra', 'paper')

StepCost = str | Callable[[int, int], float]


@dataclass
class Transform:
    """
    The result of a distance transform.

    distance        (n,) cost of the cheapest walk from each neuron to the goal (inf: unreachable)
    goal            the goal neuron
    allowed         (n,) bool, the neurons the wave was allowed to enter
    next_threshold  lowest U-height of an ignored neuron next to the reached region (inf if none)
    pushes          number of times a neuron was put into the queue (the work done)
    method, step    how it was computed
    """

    distance: ndarray
    goal: int
    allowed: ndarray
    next_threshold: float
    pushes: int
    method: str
    step: StepCost
    diagonal_factor: float

    @property
    def reached(self) -> ndarray:
        """(n,) bool: neurons from which the goal can be reached."""
        return np.isfinite(self.distance)

    def as_paper(self) -> ndarray:
        """The distance map in the paper's notation: -1 for neurons the wave never reached."""
        return np.where(np.isfinite(self.distance), self.distance, -1.0)


# ---------------------------------------------------------------------- step costs
def step_cost_function(lattice: Lattice, step: StepCost = 'node',
                       diagonal_factor: float = 1.0) -> Callable[[int, int, bool], float]:
    """f(c, n, diagonal) -> cost of the wave moving from neuron c into its neighbour n."""
    u = lattice.u_height
    factor = float(diagonal_factor)
    if callable(step):
        def base(c, n):
            return float(step(c, n))
    elif step == 'node':
        def base(c, n):
            return u[n]
    elif step == 'mean':
        def base(c, n):
            return 0.5 * (u[c] + u[n])
    elif step == 'edge':
        if lattice.weights is None:
            raise ValueError("step='edge' needs the weight vectors (build the Lattice with from_som)")
        w = lattice.weights

        def base(c, n):
            return float(np.linalg.norm(w[c] - w[n]))
    else:
        raise ValueError(f'step must be one of {STEPS} or a callable, not {step!r}')

    if factor == 1.0:
        return lambda c, n, diagonal: base(c, n)
    return lambda c, n, diagonal: base(c, n) * (factor if diagonal else 1.0)


def step_cost_table(lattice: Lattice, step: StepCost = 'node', diagonal_factor: float = 1.0) -> list[ndarray]:
    """cost[c][k]: cost of the wave moving from neuron c into lattice.neighbours[c][k]."""
    f = step_cost_function(lattice, step, diagonal_factor)
    table = []
    for c, (nb, diag) in enumerate(zip(lattice.neighbours, lattice.diagonal, strict=True)):
        table.append(np.array([f(c, int(n), bool(d)) for n, d in zip(nb, diag, strict=True)], dtype=float))
    if any((t < 0).any() for t in table):
        raise ValueError('step costs must be non-negative')
    return table


# ------------------------------------------------------------------ the transform
def distance_transform(lattice: Lattice, goal: int, *, step: StepCost = 'node', diagonal_factor: float = 1.0,
                       allowed: ndarray | None = None, method: str = 'wavefront',
                       costs: list[ndarray] | None = None) -> Transform:
    """
    Distance transform of `lattice` towards `goal`; see the module docstring.

    :param lattice: a Lattice (Lattice.from_som(som))
    :param goal: index of the goal neuron
    :param step: cost of one step: 'node' (the paper), 'mean', 'edge' or a callable f(c, n)
    :param diagonal_factor: multiplies the cost of steps across diagonals of the 2D index array
    :param allowed: optional (n,) bool mask of the neurons the wave may enter (the goal always may)
    :param method: 'wavefront' (the paper's queue), 'dijkstra', or 'paper' (Algorithm 2 verbatim)
    :param costs: optional precomputed step_cost_table(lattice, step, diagonal_factor)
    """
    n = lattice.n
    goal = int(goal)
    if not 0 <= goal < n:
        raise IndexError(f'goal {goal} is not a neuron (0..{n - 1})')
    if method not in METHODS:
        raise ValueError(f'method must be one of {METHODS}, not {method!r}')
    ok = np.ones(n, dtype=bool) if allowed is None else np.array(allowed, dtype=bool, copy=True)
    if ok.shape != (n,):
        raise ValueError(f'allowed must have shape ({n},)')
    ok[goal] = True

    if method == 'paper':
        distance, nxt, pushes = _paper(lattice, goal, ok)
    else:
        if costs is None:
            costs = step_cost_table(lattice, step, diagonal_factor)
        run = _wavefront if method == 'wavefront' else _dijkstra
        distance, nxt, pushes = run(lattice, costs, goal, ok)
    return Transform(distance, goal, ok, nxt, pushes, method, step, float(diagonal_factor))


def _wavefront(lattice: Lattice, costs: list[ndarray], goal: int, ok: ndarray):
    """The paper's FIFO distance wave (label-correcting): exact for non-negative costs."""
    u, nbs = lattice.u_height, lattice.neighbours
    dist = np.full(lattice.n, np.inf)
    dist[goal] = 0.0
    queued = np.zeros(lattice.n, dtype=bool)
    q = deque([goal])
    queued[goal] = True
    pushes, nxt = 1, np.inf
    while q:
        c = q.popleft()
        queued[c] = False
        dc = dist[c]
        for n, cost in zip(nbs[c], costs[c], strict=True):
            if not ok[n]:
                nxt = min(nxt, u[n])                     # an ignored neuron: remember its height
                continue
            nd = dc + cost
            if nd < dist[n]:
                dist[n] = nd
                if not queued[n]:
                    q.append(n)
                    queued[n] = True
                    pushes += 1
    return dist, float(nxt), pushes


def _dijkstra(lattice: Lattice, costs: list[ndarray], goal: int, ok: ndarray):
    u, nbs = lattice.u_height, lattice.neighbours
    dist = np.full(lattice.n, np.inf)
    dist[goal] = 0.0
    heap = [(0.0, goal)]
    done = np.zeros(lattice.n, dtype=bool)
    pushes, nxt = 1, np.inf
    while heap:
        dc, c = heapq.heappop(heap)
        if done[c]:
            continue
        done[c] = True
        for n, cost in zip(nbs[c], costs[c], strict=True):
            if not ok[n]:
                nxt = min(nxt, u[n])
                continue
            nd = dc + cost
            if nd < dist[n]:
                dist[n] = nd
                heapq.heappush(heap, (nd, int(n)))
                pushes += 1
    return dist, float(nxt), pushes


def _paper(lattice: Lattice, goal: int, ok: ndarray):
    """Algorithms 2 / 3 of Bui and Takatsuka (2007), line by line (-1 = not reached yet)."""
    u, nbs, diag = lattice.u_height, lattice.neighbours, lattice.diagonal
    dist = np.full(lattice.n, -1.0)
    dist[goal] = 0.0
    q = deque([goal])
    pushes, nxt = 1, np.inf
    while q:
        c = q.popleft()
        d = u[c] * SQRT2
        for n, is_diag in zip(nbs[c], diag[c], strict=True):
            if not ok[n]:                                # Algorithm 3: "ignore n"
                nxt = min(nxt, u[n])
                continue
            if dist[n] == -1:
                dist[n] = dist[c] + u[n]
                q.append(n)
                pushes += 1
            elif is_diag and dist[n] > dist[c] + d:
                dist[n] = dist[c] + d
                q.append(n)
                pushes += 1
            elif dist[n] > dist[c] + u[n]:
                dist[n] = dist[c] + u[n]
                q.append(n)
                pushes += 1
                # (the paper also copies n.distance to n's duplicate neurons on the seams of the
                #  2D index array; GeoSOM's neurons are unique, so there is nothing to copy)
    return np.where(dist < 0, np.inf, dist), float(nxt), pushes
