# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Masahiro Takatsuka. See the NOTICE file for attribution terms.
"""
Lattice: the neighbour graph of a trained SOM, in the form the distance transform needs.

    from mt.dtgeosom.lattice import Lattice

    lattice = Lattice.from_som(som)        # a trained mt.geosom GeoSOM (or PlaneSOM)
    lattice.neighbours[i]                  # neuron indices next to neuron i
    lattice.u_height                       # (n,) U-matrix value of every neuron
    lattice.diagonal[i]                    # True where that neighbour is a *diagonal* one in the
                                           # Wu-Takatsuka 2D index array (see below)

Diagonal neighbours
    GeoSOM stores its geodesic dome in the indexed 2D data structure of Wu and Takatsuka (2006):
    the icosahedron is cut open and its vertices are kept in a 2D array.  In that array a neuron
    has six neighbours: four *direct* ones (offsets (+-1, 0) and (0, +-1)) and two *diagonal*
    ones (offsets (+1, +1) and (-1, -1)).  Bui and Takatsuka (WSOM 2007) propagate distances
    across diagonal neighbours with a factor sqrt(2) (the `d` of their Algorithms 2 and 3).  The
    flag is recovered here from the stored (x, y) of every vertex, so that rule can be reproduced.
    On the sphere every neighbour is one ring away, so the default cost models treat all six alike.

Duplicate neurons
    The 2D array stores vertices on the seams of the unfolded icosahedron more than once.  The
    paper copies every distance update to those duplicates; GeoSOM numbers the *unique* neurons,
    so here every neuron exists once and no copying is needed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy import ndarray

SQRT2 = float(np.sqrt(2.0))


@dataclass
class Lattice:
    """
    The neighbour graph of a SOM.

    :param neighbours: list of int arrays, the neighbours of every neuron
    :param u_height: (n,) U-matrix height of every neuron (mean distance to its neighbours' weights)
    :param weights: optional (n, dim) weight vectors (for the 'edge' cost model)
    :param diagonal: optional list of bool arrays parallel to `neighbours` (diagonal in the 2D index array)
    :param positions: optional (n, k) coordinates of the neurons on the lattice (unit vectors on a sphere); used
                      to break ties between equally short paths in favour of the geometrically straighter one
    """

    neighbours: list[ndarray]
    u_height: ndarray
    weights: ndarray | None = None
    diagonal: list[ndarray] = field(default=None)  # type: ignore[assignment]
    positions: ndarray | None = None

    def __post_init__(self):
        self.neighbours = [np.asarray(nb, dtype=int) for nb in self.neighbours]
        self.u_height = np.asarray(self.u_height, dtype=float)
        if len(self.u_height) != len(self.neighbours):
            raise ValueError(f'{len(self.u_height)} U-heights for {len(self.neighbours)} neurons')
        if self.diagonal is None:
            self.diagonal = [np.zeros(len(nb), dtype=bool) for nb in self.neighbours]
        else:
            self.diagonal = [np.asarray(d, dtype=bool) for d in self.diagonal]
        if self.weights is not None:
            self.weights = np.asarray(self.weights, dtype=float)
        if self.positions is not None:
            self.positions = np.asarray(self.positions, dtype=float)

    # ------------------------------------------------------------------ builders
    @classmethod
    def from_som(cls, som, u_height: ndarray | None = None) -> Lattice:
        """
        The lattice of a trained mt.geosom map (GeoSOM, PlaneSOM or LineSOM).

        :param u_height: optional per-neuron heights to use instead of som.u_matrix()
        """
        if getattr(som, 'weights', None) is None:
            raise ValueError('the SOM has no weights yet: initialise and train it first')
        heights = som.u_matrix() if u_height is None else np.asarray(u_height, dtype=float)
        neighbours = [np.asarray(nb, dtype=int) for nb in som.neighbours]
        diagonal = _diagonal_flags(som, neighbours)
        positions = next((getattr(som, a) for a in ('points', 'positions', 'init_coords')
                          if getattr(som, a, None) is not None), None)
        return cls(neighbours, heights, weights=np.asarray(som.weights, dtype=float), diagonal=diagonal,
                   positions=positions)

    @classmethod
    def from_edges(cls, n: int, edges, u_height, weights=None) -> Lattice:
        """A lattice from an (E, 2) array of neighbouring pairs (any graph)."""
        nbs: list[list[int]] = [[] for _ in range(n)]
        for a, b in np.asarray(edges, dtype=int):
            if a != b:
                nbs[a].append(b)
                nbs[b].append(a)
        return cls([np.unique(nb) for nb in nbs], u_height, weights)

    # ------------------------------------------------------------------ queries
    @property
    def n(self) -> int:
        return len(self.neighbours)

    def is_diagonal(self, a: int, b: int) -> bool:
        """True when b is a diagonal neighbour of a in the 2D index array."""
        hit = np.nonzero(self.neighbours[a] == b)[0]
        return bool(len(hit) and self.diagonal[a][hit[0]])

    def edges(self) -> ndarray:
        """(E, 2) neighbouring pairs, a < b."""
        pairs = [(a, b) for a, nb in enumerate(self.neighbours) for b in nb if a < b]
        return np.array(pairs, dtype=int).reshape(-1, 2)


def _diagonal_flags(som, neighbours: list[ndarray]) -> list[ndarray]:
    """For a GeoSOM: which neighbours are diagonal (offset (+1, +1) or (-1, -1)) in the dome's 2D array."""
    flags = [np.zeros(len(nb), dtype=bool) for nb in neighbours]
    dome = getattr(som, 'dome', None)
    if dome is None:                                           # PlaneSOM / LineSOM: no index array
        return flags
    stored: dict[int, list[tuple[int, int]]] = {}
    for v in dome.get_all_vertices():
        stored.setdefault(int(som.index_map[v.id]), []).append((int(v.x), int(v.y)))
    for a, nb in enumerate(neighbours):
        for k, b in enumerate(nb):
            offsets = {(bx - ax, by - ay) for ax, ay in stored.get(a, ()) for bx, by in stored.get(int(b), ())}
            direct = any(abs(dx) + abs(dy) == 1 for dx, dy in offsets)
            diag = any(abs(dx) == 1 and abs(dy) == 1 for dx, dy in offsets)
            flags[a][k] = diag and not direct
    return flags
