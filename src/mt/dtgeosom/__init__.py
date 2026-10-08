# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Masahiro Takatsuka. See the NOTICE file for attribution terms.
"""
mt.dtgeosom -- path finding on a spherical SOM with the distance transform and floodplain analysis.

Implements M. Bui and M. Takatsuka, "Path finding on a spherical SOM using the distance transform
and floodplain analysis", Proc. 6th International Workshop on Self-Organizing Maps (WSOM 2007),
Bielefeld, on top of GeoSOM (mt.geosom).

    from mt.geosom.GeoSOM import GeoSOM
    from mt.dtgeosom import PathFinder

    som = GeoSOM(8).train(data)
    finder = PathFinder(som)
    finder.shortest_path(data[0], data[1])     # Algorithm 2 (distance transform) + Algorithm 1 (descent)
    finder.flattest_path(data[0], data[1])     # Algorithm 3 (floodplain analysis)

Modules
    mt.dtgeosom.lattice      Lattice: the SOM's neighbour graph, U-heights, diagonal flags of the 2D index array
    mt.dtgeosom.transform    distance_transform: the distance wave (wavefront / dijkstra / paper), step costs
    mt.dtgeosom.paths        descend (Algorithm 1), SOMPath, path_cost
    mt.dtgeosom.floodplain   flattest_path (Algorithm 3), shortest_path, minimax_threshold, floodplain
    mt.dtgeosom.pathfinder   PathFinder: all of it for a trained GeoSOM / PlaneSOM
    mt.dtgeosom.datasets     the binary-tree data of the paper's section 4.1
    mt.dtgeosom.gui          PathViewer: paths on GeoSOM's rotatable map projection; PathPicker: click two
                             neurons to see the path between them (needs matplotlib)
"""

from mt.dtgeosom.floodplain import flattest_path as lattice_flattest_path
from mt.dtgeosom.floodplain import minimax_threshold
from mt.dtgeosom.floodplain import shortest_path as lattice_shortest_path
from mt.dtgeosom.lattice import Lattice
from mt.dtgeosom.pathfinder import KINDS, PathFinder, flattest_path, shortest_path
from mt.dtgeosom.paths import NoPathError, SOMPath, descend, path_cost
from mt.dtgeosom.transform import Transform, distance_transform

__version__ = '1.0.0'

__author__ = 'Masahiro Takatsuka, Michael Bui'
__maintainer__ = 'Masahiro Takatsuka <masa@takatsuka.org>'

__all__ = [
    'KINDS', 'Lattice', 'PathFinder', 'SOMPath', 'Transform', 'NoPathError',
    'distance_transform', 'descend', 'path_cost', 'minimax_threshold',
    'shortest_path', 'flattest_path', 'lattice_shortest_path', 'lattice_flattest_path',
    '__version__',
]
