# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Masahiro Takatsuka. See the NOTICE file for attribution terms.
"""
PathFinder: paths between neurons of a trained GeoSOM -- shortest, flattest, fewest hops, or through data space.

    from mt.geosom.GeoSOM import GeoSOM
    from mt.dtgeosom import PathFinder

    som = GeoSOM(8).train(data)
    finder = PathFinder(som)
    p = finder.shortest_path(data[3], data[120])     # samples (mapped to their BMUs) or neuron indices
    q = finder.flattest_path(data[3], data[120])     # stays on the floodplain of the U-matrix
    p.nodes, p.cost, p.states(som)                   # neurons, cost, weight vectors along the path
    finder.path(a, b, 'hops')                        # any of KINDS (below)

Kinds of path (KINDS)
    'shortest'  distance transform with the PathFinder's step cost (default: the U-height of every neuron
                entered, as in Bui and Takatsuka 2007) -- avoids high U-heights, but may still cross a ridge
    'flattest'  floodplain analysis (Algorithm 3): the shortest path that stays below the lowest U-height
                threshold at which start and goal are connected -- goes round the ridges
    'hops'      the fewest steps on the geodesic grid, ignoring the data (the shortest route on the lattice
                itself; among equally short ones, the geometrically straightest)
    'edge'      distance transform with |w_c - w_n|, the distance between neighbouring weight vectors, as the
                step cost -- the shortest walk through data space along the map's neurons

Works with any trained mt.geosom map that has `neighbours`, `u_matrix()` and `weights` (GeoSOM,
PlaneSOM, LineSOM).  The lattice and its U-matrix are taken from the SOM when the PathFinder is
made; call refresh() after training the SOM further.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy import ndarray

from mt.dtgeosom import floodplain as fp
from mt.dtgeosom.lattice import Lattice
from mt.dtgeosom.paths import SOMPath, descend
from mt.dtgeosom.transform import StepCost, Transform, distance_transform, step_cost_table

KINDS = ('shortest', 'flattest', 'hops', 'edge')


class PathFinder:
    """
    Distance-transform path finding on a trained SOM.

    :param som: a trained mt.geosom map (GeoSOM, PlaneSOM, ...)
    :param step: cost of one step: 'node' (default, the paper: the U-height of the neuron entered),
                 'mean', 'edge' (distance between the weight vectors), or a callable f(c, n)
    :param diagonal_factor: cost factor for diagonal neighbours of the 2D index array (paper: sqrt(2))
    :param method: 'wavefront' (default, the paper's queue), 'dijkstra', or 'paper' (Algorithm 2 verbatim)
    :param rule: how paths are read off the distance map: 'exact' (default) or 'steepest' (Algorithm 1)
    :param u_height: optional per-neuron heights instead of som.u_matrix()
    """

    def __init__(self, som, *, step: StepCost = 'node', diagonal_factor: float = 1.0, method: str = 'wavefront',
                 rule: str = 'exact', u_height: ndarray | None = None):
        self.som = som
        self.step, self.diagonal_factor, self.method, self.rule = step, float(diagonal_factor), method, rule
        self._u_height = u_height
        self.refresh()

    @classmethod
    def paper(cls, som) -> PathFinder:
        """The settings of Bui and Takatsuka (2007): Algorithm 2 verbatim and steepest descent."""
        return cls(som, method='paper', rule='steepest', diagonal_factor=float(np.sqrt(2.0)))

    def refresh(self):
        """Re-reads the lattice and U-matrix from the SOM (after it has been trained further)."""
        self.lattice = Lattice.from_som(self.som, self._u_height)
        self._tables: dict = {}
        self._costs = None if self.method == 'paper' else self._table(self.step)
        return self

    def _table(self, step):
        key = step if isinstance(step, str) else id(step)
        if key not in self._tables:
            self._tables[key] = step_cost_table(self.lattice, step, self.diagonal_factor)
        return self._tables[key]

    # ------------------------------------------------------------------ helpers
    @property
    def u_height(self) -> ndarray:
        return self.lattice.u_height

    def neuron(self, x) -> int:
        """A neuron index: `x` itself if it is an integer, else the best matching unit of the sample `x`."""
        if isinstance(x, int | np.integer):
            if not 0 <= int(x) < self.lattice.n:
                raise IndexError(f'neuron {x} out of range (0..{self.lattice.n - 1})')
            return int(x)
        v = np.asarray(x, dtype=float)
        if v.ndim != 1:
            raise ValueError('give a neuron index or one sample (a 1D vector)')
        return int(self.som.bmu(v[None, :])[0])

    def _kw(self):
        return dict(step=self.step, diagonal_factor=self.diagonal_factor, method=self.method)

    # ------------------------------------------------------------------ queries
    def distance_map(self, goal, allowed: ndarray | None = None) -> Transform:
        """The distance transform towards `goal` (a neuron or a sample); .distance is the map."""
        return distance_transform(self.lattice, self.neuron(goal), allowed=allowed, costs=self._costs, **self._kw())

    def shortest_path(self, start, goal) -> SOMPath:
        """Shortest path over the whole map (Algorithms 2 and 1).  start, goal: neurons or samples."""
        return fp.shortest_path(self.lattice, self.neuron(start), self.neuron(goal), rule=self.rule, **self._kw())

    def flattest_path(self, start, goal, threshold: str = 'iterative') -> SOMPath:
        """Flattest, shortest path on the floodplain of the U-matrix (Algorithm 3)."""
        return fp.flattest_path(self.lattice, self.neuron(start), self.neuron(goal), threshold=threshold,
                                rule=self.rule, **self._kw())

    def hop_path(self, start, goal) -> SOMPath:
        """The path with the fewest steps on the geodesic grid (the data are ignored)."""
        return self._fixed_step_path(start, goal, 'hops')

    def edge_path(self, start, goal) -> SOMPath:
        """The shortest walk through data space: step cost = distance between neighbouring weight vectors."""
        return self._fixed_step_path(start, goal, 'edge')

    def path(self, start, goal, kind: str = 'shortest') -> SOMPath:
        """A path of one of KINDS: 'shortest', 'flattest', 'hops' or 'edge'."""
        find = {'shortest': self.shortest_path, 'flattest': self.flattest_path,
                'hops': self.hop_path, 'edge': self.edge_path}.get(kind)
        if find is None:
            raise ValueError(f'kind must be one of {KINDS}, not {kind!r}')
        return find(start, goal)

    def all_paths(self, start, goal, kinds=KINDS) -> dict[str, SOMPath]:
        """{kind: path} for several kinds between the same two neurons (or samples)."""
        a, b = self.neuron(start), self.neuron(goal)
        return {kind: self.path(a, b, kind) for kind in kinds}

    def _fixed_step_path(self, start, goal, step: str) -> SOMPath:
        a, b = self.neuron(start), self.neuron(goal)
        method = 'wavefront' if self.method == 'paper' else self.method
        dt = distance_transform(self.lattice, b, step=step, diagonal_factor=1.0, method=method,
                                costs=step_cost_table(self.lattice, step) if self.diagonal_factor != 1.0
                                else self._table(step))
        nodes = descend(self.lattice, dt, a, rule='exact')
        return SOMPath(nodes, float(dt.distance[a]), step, dt)

    def floodplain(self, threshold: float) -> ndarray:
        """(n,) bool: neurons with U-height <= threshold."""
        return fp.floodplain(self.lattice, threshold)

    def paths(self, pairs: Sequence[tuple], kind: str = 'shortest') -> list[SOMPath]:
        """Paths for several (start, goal) pairs."""
        return [self.path(a, b, kind) for a, b in pairs]


def shortest_path(som, start, goal, **kwargs) -> SOMPath:
    """PathFinder(som, **kwargs).shortest_path(start, goal)."""
    return PathFinder(som, **kwargs).shortest_path(start, goal)


def flattest_path(som, start, goal, threshold: str = 'iterative', **kwargs) -> SOMPath:
    """PathFinder(som, **kwargs).flattest_path(start, goal)."""
    return PathFinder(som, **kwargs).flattest_path(start, goal, threshold)
