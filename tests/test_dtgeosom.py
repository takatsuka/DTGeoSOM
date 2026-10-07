# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Masahiro Takatsuka. See the NOTICE file for attribution terms.
import itertools

import numpy as np
import pytest
from mt.geosom.GeoSOM import GeoSOM
from mt.geosom.PlaneSOM import PlaneSOM

from mt.dtgeosom import (
    Lattice,
    NoPathError,
    PathFinder,
    descend,
    distance_transform,
    lattice_flattest_path,
    minimax_threshold,
    path_cost,
)
from mt.dtgeosom.datasets import binary_tree, path_agrees_with_tree, tree_path
from mt.dtgeosom.floodplain import floodplain


# ---------------------------------------------------------------------------- fixtures
@pytest.fixture(scope='module')
def tree_som():
    x, names = binary_tree(3)
    som = GeoSOM(2, seed=0)
    som.initialise(dataset=x)
    som.train(x, epochs=150, sigma=3.0, mode='online', learning_rate=(0.8, 0.02))
    return som, x, [int(n) for n in names]


@pytest.fixture(scope='module')
def big_som():
    rng = np.random.default_rng(1)
    centres = rng.normal(size=(4, 5)) * 4
    x = np.vstack([c + rng.normal(size=(60, 5)) for c in centres])
    return GeoSOM(6, seed=0).train(x, epochs=15), x


def line_lattice(heights):
    n = len(heights)
    return Lattice.from_edges(n, [(i, i + 1) for i in range(n - 1)], heights)


def brute_force(lattice, goal, step='node'):
    """Bellman-Ford on the lattice, the reference distance map."""
    from mt.dtgeosom.transform import step_cost_function
    f = step_cost_function(lattice, step)
    d = np.full(lattice.n, np.inf)
    d[goal] = 0
    for _ in range(lattice.n):
        for c in range(lattice.n):
            for n, dg in zip(lattice.neighbours[c], lattice.diagonal[c], strict=True):
                d[n] = min(d[n], d[c] + f(c, int(n), bool(dg)))
    return d


# ---------------------------------------------------------------------------- datasets
def test_tree_paths():
    assert tree_path(8, 13) == [8, 4, 2, 1, 3, 6, 13]
    assert tree_path(1, 15) == [1, 3, 7, 15]
    assert tree_path(5, 5) == [5]
    x, names = binary_tree(3)
    assert x.shape == (15, 15) and len(names) == 15
    assert x[7, 12] == 6 and np.allclose(x, x.T) and (np.diag(x) == 0).all()
    assert path_agrees_with_tree([8, 9, 4, 2], 9, 2, groups=[{8, 9}])
    assert not path_agrees_with_tree([8, 13], 8, 13)


# ---------------------------------------------------------------------------- lattice
def test_lattice_from_geosom(tree_som):
    som, _, _ = tree_som
    lat = Lattice.from_som(som)
    assert lat.n == 42
    assert np.allclose(lat.u_height, som.u_matrix())
    n_diag = sum(int(d.sum()) for d in lat.diagonal)
    assert n_diag == 2 * 40                       # a third of the 120 dome edges are diagonals of the 2D array
    for a in range(lat.n):
        for b, d in zip(lat.neighbours[a], lat.diagonal[a], strict=True):
            assert lat.is_diagonal(int(b), a) == bool(d)    # symmetric


def test_untrained_som_rejected():
    som = GeoSOM(2)
    with pytest.raises(ValueError):
        Lattice.from_som(som)


# ---------------------------------------------------------------------------- transform
@pytest.mark.parametrize('step', ['node', 'mean', 'edge'])
def test_wavefront_equals_dijkstra_and_bellman_ford(big_som, step):
    som, _ = big_som
    lat = Lattice.from_som(som)
    for goal in (0, 57, 300):
        w = distance_transform(lat, goal, step=step, method='wavefront').distance
        d = distance_transform(lat, goal, step=step, method='dijkstra').distance
        assert np.allclose(w, d)
    small = Lattice.from_som(GeoSOM(3, seed=0).train(som.weights[:50], epochs=3))
    assert np.allclose(distance_transform(small, 5, step=step).distance, brute_force(small, 5, step))


def test_paper_method_runs_and_reaches_everything(tree_som):
    som, _, _ = tree_som
    lat = Lattice.from_som(som)
    dt = distance_transform(lat, 3, method='paper')
    assert np.isfinite(dt.distance).all() and dt.distance[3] == 0
    assert (dt.as_paper() >= 0).all()


def test_node_cost_is_sum_of_heights_on_a_line():
    lat = line_lattice([1.0, 2.0, 3.0, 4.0])
    dt = distance_transform(lat, 0)
    assert np.allclose(dt.distance, [0, 2, 5, 9])
    assert list(descend(lat, dt, 3)) == [3, 2, 1, 0]
    assert path_cost(lat, [3, 2, 1, 0]) == pytest.approx(9.0)


def test_allowed_mask_and_next_threshold():
    lat = line_lattice([0.1, 5.0, 0.2, 0.3])
    dt = distance_transform(lat, 0, allowed=lat.u_height <= 1.0)
    assert np.isinf(dt.distance[1:]).all() and dt.next_threshold == 5.0
    with pytest.raises(NoPathError):
        descend(lat, dt, 3)


def test_bad_arguments(tree_som):
    lat = Lattice.from_som(tree_som[0])
    with pytest.raises(IndexError):
        distance_transform(lat, 99)
    with pytest.raises(ValueError):
        distance_transform(lat, 0, method='astar')
    with pytest.raises(ValueError):
        distance_transform(lat, 0, step='banana')


# ---------------------------------------------------------------------------- paths
@pytest.mark.parametrize('step', ['node', 'mean', 'edge'])
def test_descent_gives_optimal_connected_paths(big_som, step):
    som, _ = big_som
    finder = PathFinder(som, step=step)
    rng = np.random.default_rng(0)
    for a, b in rng.integers(0, som.n_nodes, size=(15, 2)):
        p = finder.shortest_path(int(a), int(b))
        assert p.start == a and p.goal == b
        assert len(set(p.nodes.tolist())) == len(p)                 # no repeats
        for u, v in zip(p.nodes[:-1], p.nodes[1:], strict=True):
            assert v in som.neighbours[u]                          # moves between neighbours only
        assert path_cost(finder.lattice, p.nodes, step) == pytest.approx(p.cost, rel=1e-9, abs=1e-12)


def test_steepest_rule_matches_exact_cost_for_node_step(big_som):
    som, _ = big_som
    exact, steep = PathFinder(som), PathFinder(som, rule='steepest')
    for a, b in [(0, 100), (5, 300), (77, 3)]:
        steep_cost = path_cost(exact.lattice, steep.shortest_path(a, b).nodes)
        assert steep_cost == pytest.approx(exact.shortest_path(a, b).cost)


def test_samples_are_mapped_to_bmus(big_som):
    som, x = big_som
    finder = PathFinder(som)
    p = finder.shortest_path(x[0], x[-1])
    bmu = som.bmu(x[[0, -1]])
    assert p.start == bmu[0] and p.goal == bmu[1]
    assert p.states(som).shape == (len(p), x.shape[1])


def test_same_start_and_goal(big_som):
    p = PathFinder(big_som[0]).flattest_path(9, 9)
    assert list(p.nodes) == [9] and p.cost == 0


# ---------------------------------------------------------------------------- floodplain
def test_flattest_path_goes_round_the_mountain():
    #   0 - 1 - 2         a ring of 6 neurons; neuron 1 is a mountain, the way round is longer
    #   |       |         but flat
    #   5 - 4 - 3
    heights = np.array([0.1, 9.0, 0.1, 0.2, 0.2, 0.2])
    ring = [(i, (i + 1) % 6) for i in range(6)]
    lat = Lattice.from_edges(6, ring, heights)
    short = lattice_flattest_path(lat, 0, 2, threshold='minimax')
    assert list(short.nodes) == [0, 5, 4, 3, 2] and short.threshold == pytest.approx(0.2)
    assert short.thresholds == [pytest.approx(0.2)]
    it = lattice_flattest_path(lat, 0, 2)
    assert list(it.nodes) == [0, 5, 4, 3, 2] and it.thresholds == [pytest.approx(0.1), pytest.approx(0.2)]


def test_iterative_threshold_equals_minimax(big_som):
    som, _ = big_som
    finder = PathFinder(som)
    rng = np.random.default_rng(3)
    for a, b in rng.integers(0, som.n_nodes, size=(20, 2)):
        it = finder.flattest_path(int(a), int(b))
        mm = finder.flattest_path(int(a), int(b), threshold='minimax')
        assert it.threshold == pytest.approx(mm.threshold)
        assert it.threshold == pytest.approx(minimax_threshold(finder.lattice, int(a), int(b)))
        assert it.cost == pytest.approx(mm.cost)
        assert it.thresholds == sorted(it.thresholds)
        # the path stays on the floodplain, and no lower floodplain connects start and goal
        assert finder.u_height[it.nodes].max() <= it.threshold + 1e-12
        lower = it.threshold - 1e-9
        if lower >= max(finder.u_height[a], finder.u_height[b]):
            dt = distance_transform(finder.lattice, int(b), allowed=floodplain(finder.lattice, lower, a, b))
            assert not np.isfinite(dt.distance[a])


def test_flattest_is_never_higher_than_shortest(big_som):
    finder = PathFinder(big_som[0])
    u = finder.u_height
    for a, b in itertools.combinations([0, 40, 200, 333, 361], 2):
        s, f = finder.shortest_path(a, b), finder.flattest_path(a, b)
        assert f.max_height(u) <= s.max_height(u) + 1e-12
        assert f.cost >= s.cost - 1e-9


# ---------------------------------------------------------------------------- the paper's experiment
def test_binary_tree_flattest_paths_follow_the_tree(tree_som):
    """Section 4.1: flattest paths agree with the tree more often than plain shortest paths."""
    som, x, names = tree_som
    bmu = som.bmu(x)
    labels = som.node_labels(x, names)
    groups = [set(d) for d in labels if len(d) > 1]
    finder = PathFinder(som)
    good = {'shortest': 0, 'flattest': 0}
    for a, b in itertools.combinations(range(1, 16), 2):
        if bmu[a - 1] == bmu[b - 1]:
            continue
        for kind, find in (('shortest', finder.shortest_path), ('flattest', finder.flattest_path)):
            p = find(int(bmu[a - 1]), int(bmu[b - 1]))
            good[kind] += path_agrees_with_tree(p.visited_labels(labels), a, b, groups)
    assert good['flattest'] > good['shortest']


# ---------------------------------------------------------------------------- other lattices
def test_plane_som():
    rng = np.random.default_rng(0)
    x = rng.random((200, 3))
    som = PlaneSOM(8, 10, seed=0).train(x, epochs=10)
    finder = PathFinder(som)
    p = finder.flattest_path(0, som.n_nodes - 1)
    assert p.start == 0 and p.goal == som.n_nodes - 1
    assert sum(int(d.sum()) for d in finder.lattice.diagonal) == 0


# ---------------------------------------------------------------------------- gui
def test_path_viewer(tmp_path, tree_som):
    pytest.importorskip('matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    from mt.dtgeosom.gui import PathViewer, great_circle, plot_paths

    som, x, names = tree_som
    finder = PathFinder(som)
    s, f = finder.shortest_path(0, 30), finder.flattest_path(0, 30)
    v = plot_paths(som, [s, f], x, names, floodplain=f.threshold, path=str(tmp_path / 'p.png'),
                   projection='Wagner III')
    assert isinstance(v, PathViewer) and len(v.paths) == 2 and (tmp_path / 'p.png').exists()
    v.rotate(30, 10)
    v.clear_paths()
    assert v.paths == []
    arc = great_circle(som.points[[0, 1]])
    assert np.allclose(np.linalg.norm(arc, axis=1), 1)
