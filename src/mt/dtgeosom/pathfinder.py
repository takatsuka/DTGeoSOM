# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Masahiro Takatsuka. See the NOTICE file for attribution terms.
"""
PathFinder: shortest and flattest paths between neurons of a trained GeoSOM.

    from mt.geosom.GeoSOM import GeoSOM
    from mt.dtgeosom import PathFinder

    som = GeoSOM(8).train(data)
    finder = PathFinder(som)
    p = finder.shortest_path(data[3], data[120])     # samples (mapped to their BMUs) or neuron indices
    q = finder.flattest_path(data[3], data[120])     # stays on the floodplain of the U-matrix
    p.nodes, p.cost, p.states(som)                   # neurons, cost, weight vectors along the path

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
from mt.dtgeosom.paths import SOMPath
from mt.dtgeosom.transform import StepCost, Transform, distance_transform, step_cost_table


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
        self._costs = None if self.method == 'paper' else step_cost_table(self.lattice, self.step,
                                                                           self.diagonal_factor)
        return self

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

    def floodplain(self, threshold: float) -> ndarray:
        """(n,) bool: neurons with U-height <= threshold."""
        return fp.floodplain(self.lattice, threshold)

    def paths(self, pairs: Sequence[tuple], kind: str = 'shortest') -> list[SOMPath]:
        """Paths for several (start, goal) pairs."""
        find = self.shortest_path if kind == 'shortest' else self.flattest_path
        return [find(a, b) for a, b in pairs]


def shortest_path(som, start, goal, **kwargs) -> SOMPath:
    """PathFinder(som, **kwargs).shortest_path(start, goal)."""
    return PathFinder(som, **kwargs).shortest_path(start, goal)


def flattest_path(som, start, goal, threshold: str = 'iterative', **kwargs) -> SOMPath:
    """PathFinder(som, **kwargs).flattest_path(start, goal)."""
    return PathFinder(som, **kwargs).flattest_path(start, goal, threshold)
