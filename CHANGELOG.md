# Changelog

## 0.1.0 — 2026-10-08

First version: the method of Bui and Takatsuka (WSOM 2007) on GeoSOM.

- `distance_transform`: the distance wave of Algorithm 2 (`'wavefront'`), a Dijkstra check, and the printed
  algorithm verbatim (`'paper'`); step costs `'node'` (U-height, as in the paper), `'mean'`, `'edge'` or a callable;
  `diagonal_factor` for diagonals of the Wu–Takatsuka 2D index array.
- `descend`: Algorithm 1 (`'steepest'`) and an exact back-tracking rule.
- `flattest_path`: floodplain analysis, Algorithm 3 (`'iterative'`), and a direct `'minimax'` threshold.
- `PathFinder` for trained GeoSOM / PlaneSOM / LineSOM maps; samples or neuron indices as endpoints.
- `gui.PathViewer` / `plot_paths`: paths and the floodplain on GeoSOM's rotatable map projection.
- `datasets.binary_tree`: the synthetic data of section 4.1; examples reproducing figures 8 and 9.
- GitHub Actions: tests (lint; pytest on Linux, macOS and Windows with Python 3.10–3.14; the examples), and
  automatic releases: pushing a new version to `main` publishes it to PyPI and GitHub (`RELEASING.md`).
