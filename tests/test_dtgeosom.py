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


def test_path_picker(tree_som):
    pytest.importorskip('matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    from types import SimpleNamespace

    from mt.dtgeosom.gui import PathPicker

    som, x, names = tree_som
    assert PathPicker(PathFinder(som), x, names).kinds == ['flattest']      # the default: the flattest path
    picker = PathPicker(PathFinder(som), x, names, kinds=['shortest'])
    assert picker.selecting == 'start'                      # no path yet: the first clicks choose the end points
    picker.select_node(3)                                   # a click on neuron 3: the start
    assert picker._pending_start == 3 and picker.selecting == 'goal' and picker.paths == []
    picker.select_node(30)                                  # the goal: the shortest path appears
    assert picker.goal == 30 and [p.kind for p in picker.paths] == ['shortest']
    assert picker.paths[0].start == 3 and picker.paths[0].goal == 30

    key = SimpleNamespace(inaxes=None)

    def press(k):
        key.key = k
        picker._on_path_key(key)

    for k in ('2', '3', '4'):                               # flattest, hops, edge
        press(k)
    assert [p.kind for p in picker.paths] == ['shortest', 'flattest', 'hops', 'edge']
    press('1')                                              # hide shortest
    assert [p.kind for p in picker.paths] == ['flattest', 'hops', 'edge']
    press('m')                                              # distance maps: flattest, hops, edge, then off
    assert picker.distance_kind == 'flattest'
    press('m')
    press('m')
    assert picker.distance_kind == 'edge'
    press('m')
    assert picker.distance_kind is None
    press('x')                                              # swap
    assert picker.start == 30 and picker.goal == 3 and picker.paths[0].start == 30
    press('a')
    assert len(picker.paths) == 4
    picker.pick(7)                                          # not selecting: a click only inspects
    assert picker.inspected == 7 and picker.start == 30 and len(picker.paths) == 4
    press('enter')                                          # Enter: choose new end points
    picker.pick(7)
    picker.pick(12)
    assert (picker.start, picker.goal, picker.selecting) == (7, 12, None) and len(picker.paths) == 4
    press('c')                                              # clear: no paths, selecting again
    assert picker.start is None and picker.paths == [] and picker.selecting == 'start'
    picker.pick(5)
    assert picker.selecting == 'goal' and picker.start is None
    with pytest.raises(ValueError):
        PathPicker(PathFinder(som), kinds=['bananas'])


# ---------------------------------------------------------------------------- kinds of path
def bfs_hops(lattice, a):
    from collections import deque
    d = np.full(lattice.n, -1)
    d[a] = 0
    q = deque([a])
    while q:
        c = q.popleft()
        for n in lattice.neighbours[c]:
            if d[n] < 0:
                d[n] = d[c] + 1
                q.append(n)
    return d


def test_all_kinds(big_som):
    from mt.dtgeosom import KINDS
    som, x = big_som
    finder = PathFinder(som)
    rng = np.random.default_rng(5)
    for a, b in rng.integers(0, som.n_nodes, size=(10, 2)):
        a, b = int(a), int(b)
        paths = finder.all_paths(a, b)
        assert list(paths) == list(KINDS)
        hops = bfs_hops(finder.lattice, b)[a]
        for kind, p in paths.items():
            assert p.kind == kind and p.start == a and p.goal == b
            assert p.hops >= hops
        assert paths['hops'].hops == hops and paths['hops'].cost == hops      # the fewest steps
        edge = distance_transform(finder.lattice, b, step='edge', method='dijkstra').distance[a]
        assert paths['edge'].cost == pytest.approx(edge)                       # the shortest walk in data space
        assert path_cost(finder.lattice, paths['edge'].nodes, 'edge') == pytest.approx(edge)
        assert paths['shortest'].cost <= path_cost(finder.lattice, paths['hops'].nodes) + 1e-9
    assert finder.path(0, 5, 'hops').hops == bfs_hops(finder.lattice, 5)[0]
    assert finder.path(0, 5).kind == 'flattest'                            # the default kind
    assert [p.kind for p in finder.paths([(0, 5), (7, 9)])] == ['flattest', 'flattest']
    with pytest.raises(ValueError):
        finder.path(0, 5, 'scenic')


def test_hop_paths_are_straight_on_the_sphere():
    """Among equally short hop paths the descent keeps to the great circle (no needless zigzag)."""
    som = GeoSOM(8, seed=0).train(np.random.default_rng(0).random((100, 3)), epochs=2)
    finder = PathFinder(som)
    for a, b in [(5, 400), (100, 600), (17, 333)]:
        p = finder.hop_path(a, b)
        pts = som.points[p.nodes]
        normal = np.cross(pts[0], pts[-1])
        normal /= np.linalg.norm(normal)
        assert np.abs(pts @ normal).max() < 2.5 * som.ring_length            # stays near the great circle


def test_path_check_boxes_and_legend(tree_som):
    pytest.importorskip('matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    from types import SimpleNamespace

    from mt.dtgeosom.gui import PathPicker, plot_paths

    som, x, names = tree_som
    picker = PathPicker(PathFinder(som), x, names, kinds=['shortest', 'flattest'])
    picker.set_endpoints(3, 30)
    boxes = picker._kind_checks
    assert boxes.get_status() == [True, True, False, False, True]          # shortest flattest hops edge floodplain
    assert picker._flood_artist is not None
    boxes.set_active(2)                                                     # tick hops
    assert [p.kind for p in picker.paths] == ['shortest', 'flattest', 'hops']
    boxes.set_active(0)                                                     # untick shortest
    assert [p.kind for p in picker.paths] == ['flattest', 'hops']
    boxes.set_active(4)                                                     # hide the floodplain crosses
    assert not picker.show_flood and picker._flood_artist is None
    picker._on_path_key(SimpleNamespace(inaxes=None, key='4'))             # keys update the boxes
    assert boxes.get_status() == [False, True, True, True, False]
    picker.next_distance_map()
    assert 'flattest' in picker._distance_button.label.get_text()
    legend_entry = next(a for a, i in picker._legend_map.items() if picker.paths[i].kind == 'hops')
    picker._on_legend_pick(SimpleNamespace(artist=legend_entry))           # clicking the legend hides it too
    assert 'hops' not in picker.kinds and boxes.get_status()[2] is False

    finder = PathFinder(som)
    viewer = plot_paths(som, list(finder.all_paths(3, 30).values()), x, names)
    assert viewer.path_visible('hops')
    viewer.set_path_visible('hops', False)
    assert not viewer.path_visible('hops') and viewer.path_visible(0)
    entry = next(a for a, i in viewer._legend_map.items() if i == 0)
    viewer._on_legend_pick(SimpleNamespace(artist=entry))
    assert not viewer.path_visible(0)
    viewer.toggle_path(viewer.paths[0])
    assert viewer.path_visible(0) and len(viewer.paths) == 4                # hidden paths are kept, not removed


def test_inspect_neurons_on_the_path(tree_som):
    pytest.importorskip('matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    from types import SimpleNamespace

    from mt.dtgeosom.gui import PathPicker

    som, x, names = tree_som
    picker = PathPicker(PathFinder(som), x, names, kinds=['shortest', 'hops'])
    picker.select_node(3)
    picker.select_node(30)
    path = picker.result['shortest']
    middle = int(path.nodes[len(path) // 2])
    picker.select_node(middle)                                  # a click on the path: inspect, keep the paths
    assert (picker.start, picker.goal, picker.inspected, picker.selected) == (3, 30, middle, middle)
    assert picker.paths and picker.positions(middle)['shortest'] == len(path) // 2
    assert f'neuron {middle}' in picker._status.get_text()

    key = SimpleNamespace(inaxes=None, key='n')
    picker._on_path_key(key)                                    # n: the next neuron along the path
    assert picker.inspected == int(path.nodes[len(path) // 2 + 1]) and picker.goal == 30
    key.key = 'b'
    picker._on_path_key(key)
    picker._on_path_key(key)
    assert picker.inspected == int(path.nodes[len(path) // 2 - 1])
    for _ in range(len(path) + 2):                              # stops at the start
        picker.step(-1)
    assert picker.inspected == 3 and picker.start == 3

    off = next(i for i in range(som.n_nodes) if not picker.on_path(i))
    picker.select_node(off)                                     # a click off the paths: inspect, paths stay
    assert picker.inspected == off and picker.goal == 30 and picker.paths
    assert 'not on a path' in picker._status.get_text()

    picker._on_select_button()                                  # the button: select new end points
    assert picker.selecting == 'start' and 'START' in picker._select_button.label.get_text()
    picker._on_select_button()                                  # pressed again: cancel, the paths stay
    assert picker.selecting is None and picker.goal == 30 and picker.paths
    picker._on_select_button()
    picker.select_node(9)
    assert picker.selecting == 'goal' and picker.goal == 30     # the old paths stay until the goal is clicked
    key.key = 'escape'
    picker._on_path_key(key)                                    # Esc cancels the half-made selection
    assert picker.selecting is None and (picker.start, picker.goal) == (3, 30)
    picker._on_select_button()
    picker.select_node(9)
    picker.select_node(20)
    assert (picker.start, picker.goal) == (9, 20) and picker.result['shortest'].start == 9
    assert picker._select_button.label.get_text() == 'Select start & goal'
