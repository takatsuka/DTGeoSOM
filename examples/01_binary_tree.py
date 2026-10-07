# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Masahiro Takatsuka. See the NOTICE file for attribution terms.
"""
01 -- The synthetic experiment of Bui and Takatsuka (WSOM 2007, section 4.1): paths through a binary tree.

A binary tree of depth 3 (15 nodes, figure 6 of the paper) becomes 15 samples with 15 attributes:
sample i holds the graph distances from tree node i to every node.  A two-frequency GeoSOM
(42 neurons) is trained on them with the paper's settings (online, learning rate 0.8, initial
radius 3, 150 epochs).  Then, for every pair of tree nodes,

  * the shortest path (distance transform, Algorithm 2, + descent, Algorithm 1) and
  * the flattest path (floodplain analysis, Algorithm 3)

are found between their neurons, and a path counts as *correct* when the tree nodes it passes are
exactly those of the tree path, in order (as for figures 7-9).  The script prints both success
rates, the pair 8 -> 13 that the paper discusses, and saves a Wagner III map with both paths.

Run:
    python examples/01_binary_tree.py                       # writes examples/output/01_binary_tree.png
    python examples/01_binary_tree.py --show                # interactive, rotatable map
    python examples/01_binary_tree.py --paper               # Algorithm 2 verbatim (sqrt(2) diagonals)
    python examples/01_binary_tree.py --seed 4 --pair 9 14
"""
import argparse
import itertools
import os

from mt.geosom.GeoSOM import GeoSOM

from mt.dtgeosom import PathFinder
from mt.dtgeosom.datasets import binary_tree, path_agrees_with_tree, tree_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--frequency', type=int, default=2, help='GeodesicDome frequency (2: 42 neurons)')
    ap.add_argument('--epochs', type=int, default=150)
    ap.add_argument('--pair', type=int, nargs=2, default=(8, 13), help='tree nodes to draw a path between')
    ap.add_argument('--paper', action='store_true', help="the paper's Algorithm 2 verbatim")
    ap.add_argument('--show', action='store_true', help='open the interactive map')
    ap.add_argument('--save', default=os.path.join(os.path.dirname(__file__), 'output', '01_binary_tree.png'))
    args = ap.parse_args()

    x, names = binary_tree(3)
    nodes = [int(n) for n in names]
    som = GeoSOM(args.frequency, seed=args.seed)
    som.initialise(dataset=x)
    som.train(x, epochs=args.epochs, sigma=3.0, mode='online', learning_rate=(0.8, 0.02))
    print(f'{som!r}  QE {som.quantisation_error(x):.3f}  TE {som.topographic_error(x):.3f}')

    bmu = som.bmu(x)
    labels = som.node_labels(x, nodes)
    shared = [set(d) for d in labels if len(d) > 1]
    if shared:
        print('tree nodes sharing a neuron:', ', '.join('/'.join(map(str, sorted(g))) for g in shared))

    finder = PathFinder.paper(som) if args.paper else PathFinder(som)
    good = {'shortest': 0, 'flattest': 0}
    pairs = [(a, b) for a, b in itertools.combinations(nodes, 2) if bmu[a - 1] != bmu[b - 1]]
    for a, b in pairs:
        for kind in good:
            find = finder.shortest_path if kind == 'shortest' else finder.flattest_path
            p = find(int(bmu[a - 1]), int(bmu[b - 1]))
            good[kind] += path_agrees_with_tree(p.visited_labels(labels), a, b, shared)
    print(f'\npaths that follow the tree ({len(pairs)} pairs of nodes on different neurons):')
    for kind, n in good.items():
        print(f'  {kind:9s} {n:4d}  ({100 * n / len(pairs):.0f}%)')

    a, b = args.pair
    print(f'\ntree path {a} -> {b}: {"-".join(map(str, tree_path(a, b)))}')
    shown = []
    for kind in ('shortest', 'flattest'):
        find = finder.shortest_path if kind == 'shortest' else finder.flattest_path
        p = find(int(bmu[a - 1]), int(bmu[b - 1]))
        met = p.visited_labels(labels)
        ok = path_agrees_with_tree(met, a, b, shared)
        extra = f', U-height threshold {p.threshold:.3f} after {len(p.thresholds)} round(s)' if p.threshold else ''
        print(f'  {kind:9s} {"-".join(map(str, met)):24s} {"correct" if ok else "INCORRECT":9s} '
              f'{len(p)} neurons, cost {p.cost:.3f}, highest U {p.max_height(finder.u_height):.3f}{extra}')
        shown.append(p)

    if args.show or args.save:
        if not args.show:
            import matplotlib
            matplotlib.use('Agg')
        from mt.dtgeosom.gui import plot_paths
        legend = [f'{p.kind}: {"-".join(map(str, p.visited_labels(labels)))}' for p in shown]
        viewer = plot_paths(som, shown, x, names, names=legend, floodplain=shown[1].threshold,
                            projection='Wagner III', node_labels='all', label_size=10, edges=True,
                            title=f'Binary tree on GeoSOM({args.frequency}): paths from tree node {a} to {b} '
                                  f'(x: neurons above the flattest path\'s floodplain)')
        if args.save:
            os.makedirs(os.path.dirname(args.save) or '.', exist_ok=True)
            viewer.save(args.save, dpi=130)
            print(f'\nsaved {args.save}')
        if args.show:
            viewer.show()


if __name__ == '__main__':
    main()
