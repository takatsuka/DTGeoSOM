# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Masahiro Takatsuka. See the NOTICE file for attribution terms.
"""
02 -- Paths between two samples of real data: what has to change to get from one state to another?

A GeoSOM is trained on one of the sample datasets shipped with geosom (default: wine, 13
attributes).  Two samples are picked -- by default the two of different classes whose neurons are
furthest apart on the sphere -- and four kinds of path are found between their best matching units:

  * shortest  -- the paper's distance transform, step cost = U-height of the neuron entered
  * flattest  -- floodplain analysis: the shortest path that never climbs above the lowest
                 U-height threshold at which the two neurons are connected
  * hops      -- the fewest steps on the geodesic grid, ignoring the data
  * edge      -- the distance transform with the distance between neighbouring weight vectors
                 as the step cost (the shortest walk in data space through the map's neurons)

For each path the script prints the classes it passes, the highest U-height it climbs and the
attributes that change most from start to goal, and writes a map with all four paths.  The weight
vectors along a path (`path.states(som)`) are the intermediate states between the two samples.

Run:
    python examples/02_paths_between_samples.py                     # wine, saves examples/output/02_*.png
    python examples/02_paths_between_samples.py --data penguins --show
    python examples/02_paths_between_samples.py --start 0 --goal 150 --frequency 6
"""
import argparse
import os

import numpy as np
from mt.geosom import datasets
from mt.geosom.GeoSOM import GeoSOM

from mt.dtgeosom import PathFinder


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', default='wine', help='iris, penguins, wine, clusters, or a CSV file')
    ap.add_argument('--frequency', type=int, default=8)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--start', type=int, help='index of the start sample')
    ap.add_argument('--goal', type=int, help='index of the goal sample')
    ap.add_argument('--show', action='store_true')
    ap.add_argument('--save', default=os.path.join(os.path.dirname(__file__), 'output'))
    args = ap.parse_args()

    ds = datasets.load(args.data)
    x = datasets.standardise(ds.data)
    labels = ds.labels if ds.labels is not None else np.array(['?'] * len(x))
    names = list(ds.feature_names) if ds.feature_names is not None else [f'a{i}' for i in range(x.shape[1])]
    som = GeoSOM(args.frequency, seed=0).train(x, epochs=args.epochs)
    print(f'{som!r}  QE {som.quantisation_error(x):.3f}  TE {som.topographic_error(x):.3f}')

    bmu = som.bmu(x)
    if args.start is None or args.goal is None:            # the two classes furthest apart on the map
        best = (-1.0, 0, 0)
        for i in range(len(x)):
            other = np.nonzero(labels != labels[i])[0]
            if len(other) == 0:
                continue
            d = som.points[bmu[other]] @ som.points[bmu[i]]
            j = int(other[np.argmin(d)])
            best = max(best, (float(np.arccos(np.clip(d.min(), -1, 1))), i, j))
        _, args.start, args.goal = best
    a, b = args.start, args.goal
    print(f'from sample {a} ({labels[a]}) to sample {b} ({labels[b]}): neurons {bmu[a]} -> {bmu[b]}\n')

    node_labels = som.node_labels(x, labels)
    finder = PathFinder(som)
    paths = finder.all_paths(x[a], x[b])                  # shortest, flattest, hops, edge
    for kind, p in paths.items():
        empty = sum(1 for i in p.nodes if not node_labels[i])
        passed = ' > '.join(map(str, p.visited_labels(node_labels)))
        print(f'{kind:9s} {p.hops:3d} hops ({empty} neurons without data), highest U-height '
              f'{p.max_height(finder.u_height):.3f}, classes passed: {passed}')

    states = paths['flattest'].states(som)
    change = states[-1] - states[0]
    order = np.argsort(-np.abs(change))[:5]
    print('\nlargest changes from start to goal (standardised units):')
    for k in order:
        print(f'  {names[k]:28s} {change[k]:+.2f}')

    if args.show or args.save:
        if not args.show:
            import matplotlib
            matplotlib.use('Agg')
        from mt.dtgeosom.gui import plot_paths
        viewer = plot_paths(som, list(paths.values()), x, labels, names=list(paths),
                            floodplain=paths['flattest'].threshold, projection='Wagner III',
                            title=f'{args.data}: paths from sample {a} ({labels[a]}) to {b} ({labels[b]})')
        if args.save:
            os.makedirs(args.save, exist_ok=True)
            out = os.path.join(args.save, f'02_paths_{os.path.splitext(os.path.basename(args.data))[0]}.png')
            viewer.save(out, dpi=130)
            print(f'\nsaved {out}')
        if args.show:
            viewer.show()


if __name__ == '__main__':
    main()
