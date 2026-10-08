# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Masahiro Takatsuka. See the NOTICE file for attribution terms.
"""
03 -- Pick two points on the map with the mouse and see the paths between them.

A GeoSOM is trained (default: the UCI wine data shipped with geosom) and shown on a rotatable map
(mt.dtgeosom.gui.PathPicker).  Then

    'Select start & goal' button (top left, or Enter): the next click is the start neuron, the one after
                     it the goal -- the paths are found and drawn.  The button (or Esc) cancels.  At the
                     beginning, and after c, the picker is already selecting: just click two neurons.
    click            otherwise, inspects a neuron: its attributes appear on the right, and the line under
                     the map says which step of which path it is (n / b: the next / previous neuron along
                     the path)
    1                shortest -- distance transform, step cost = U-height (the paper)        cyan
    2  (or f)        flattest -- floodplain analysis (the paper); x = above its floodplain  magenta
    3                hops     -- fewest steps on the geodesic grid, ignoring the data          yellow
    4                edge     -- shortest walk through data space (weight-vector distances)    lime
                     each key shows / hides that path;  a  shows all four
    'paths' box      on the left: the same as check boxes (one per path, in its colour), a box for the
                     floodplain crosses, and a button that cycles the distance maps; clicking an entry of
                     the legend on the map also hides that path
    m                colour the map by a distance map towards the goal (each press: the next path's map,
                     then back to the U-matrix)
    x                swap start and goal          c   clear
    drag             rotate the sphere (plus all of SOMViewer's keys: arrows, p projection, l labels, ...)

The panel under the map lists, for each path, its number of hops, its cost, the highest U-height
it climbs, and the classes of the samples it passes.  Click a neuron's attribute panel on the right
to inspect it as usual.

Run:
    python examples/03_interactive_paths.py                       # wine
    python examples/03_interactive_paths.py --paths all           # all four kinds from the start
    python examples/03_interactive_paths.py --data penguins --paths shortest,flattest
    python examples/03_interactive_paths.py --data tree           # the paper's binary tree (GeoSOM(2))
    python examples/03_interactive_paths.py --start 10 --goal 500 --paths all --save examples/output/03_picker.png
"""
import argparse

import numpy as np
from mt.geosom import datasets
from mt.geosom.GeoSOM import GeoSOM

from mt.dtgeosom import KINDS, PathFinder
from mt.dtgeosom.datasets import binary_tree


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', default='wine', help="iris, penguins, wine, clusters, a CSV file, or 'tree'")
    ap.add_argument('--frequency', type=int, help='GeodesicDome frequency (default 8; 2 for the tree)')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--paths', default='shortest',
                    help=f"kinds of path to show at first, comma-separated, from {', '.join(KINDS)}; or 'all'")
    ap.add_argument('--flattest', action='store_true', help='same as adding flattest to --paths')
    ap.add_argument('--step', default='node', choices=['node', 'mean', 'edge', 'hops'],
                    help="step cost of the 'shortest' and 'flattest' paths (default node = U-height, the paper)")
    ap.add_argument('--projection', default='Wagner III')
    ap.add_argument('--start', type=int, help='start neuron (optional: pre-select a path)')
    ap.add_argument('--goal', type=int, help='goal neuron')
    ap.add_argument('--save', help='write the window to this image file and exit (no window)')
    args = ap.parse_args()

    if args.save:
        import matplotlib
        matplotlib.use('Agg')
    from mt.dtgeosom.gui import PathPicker

    if args.data == 'tree':
        x, labels = binary_tree(3)
        som = GeoSOM(args.frequency or 2, seed=0)
        som.initialise(dataset=x)
        som.train(x, epochs=150, sigma=3.0, mode='online', learning_rate=(0.8, 0.02))
        extra = dict(sample_names=labels, node_labels='all', label_size=10, edges=True)
    else:
        ds = datasets.load(args.data)
        x = datasets.standardise(ds.data)
        labels = ds.labels if ds.labels is not None else np.array(['?'] * len(x))
        som = GeoSOM(args.frequency or 8, seed=0).train(x, epochs=args.epochs)
        extra = {}
    print(f'{som!r}  QE {som.quantisation_error(x):.3f}  TE {som.topographic_error(x):.3f}')

    kinds = list(KINDS) if args.paths == 'all' else [k.strip() for k in args.paths.split(',') if k.strip()]
    picker = PathPicker(PathFinder(som, step=args.step), x, labels, kinds=kinds, flattest=args.flattest,
                        projection=args.projection, **extra)
    if args.start is not None and args.goal is not None:
        picker.set_endpoints(args.start, args.goal)
        picker.centre_on()
        for kind, p in picker.result.items():
            print(f'{kind:8s} {p.hops:3d} hops, cost {p.cost:8.3f}: {" ".join(map(str, p.nodes))}')
    if args.save:
        picker.save(args.save, dpi=130)
        print(f'saved {args.save}')
    else:
        picker.show()


if __name__ == '__main__':
    main()
