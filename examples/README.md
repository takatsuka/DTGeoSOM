# Examples

| Script | What |
|---|---|
| `01_binary_tree.py` | The paper's section 4.1: a 15-node binary tree on a GeoSOM(2); shortest vs flattest paths for every pair of tree nodes, and the 8 → 13 path of figures 8 and 9 |
| `02_paths_between_samples.py` | Shortest, flattest and weight-distance paths between two samples of iris / penguins / wine / clusters, with the attributes that change most along the way |

Both save a PNG to `output/` by default; add `--show` for the interactive, rotatable map (needs a GUI matplotlib
backend; in Jupyter use `%matplotlib widget`).
