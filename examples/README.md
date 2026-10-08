# Examples

| Script | What |
|---|---|
| `01_binary_tree.py` | The paper's section 4.1: a 15-node binary tree on a GeoSOM(2); shortest vs flattest paths for every pair of tree nodes, and the 8 → 13 path of figures 8 and 9 |
| `02_paths_between_samples.py` | Shortest, flattest, fewest-hops and weight-distance paths between two samples of iris / penguins / wine / clusters, with the attributes that change most along the way |

| `03_interactive_paths.py` | Interactive: press *Select start & goal* (or Enter) and click a start and a goal neuron on the rotatable map, and the paths are drawn; any other click shows that neuron's attributes and its step along the paths (`n` / `b` walk along the path); the *paths* check boxes (or keys `1`–`4`) turn the shortest, flattest, fewest-hops and data-space (`edge`) paths on and off, `a` shows all, `m` cycles through their distance maps, `x` swaps, `c` clears (`--data tree` for the paper's binary tree) |

01 and 02 save a PNG to `output/` by default; add `--show` for the interactive, rotatable map (needs a GUI matplotlib
backend; in Jupyter use `%matplotlib widget`).
