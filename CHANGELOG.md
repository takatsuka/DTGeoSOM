# Changelog

## Unreleased

- `flattest` is now the default kind of path: `PathFinder.path(a, b)` and `paths(pairs)` use it when no kind is
  given, the picker (`PathPicker`, example 03) opens showing the flattest path, and example 03's `--paths` defaults
  to `flattest`.
- The picker draws a lone path at a normal line width (it used the thinnest of the nested widths).
- `docs/path_methods.md`: a guide to the four kinds of path, with their algorithms and a worked example.

## 0.1.0 — 2026-10-08

First version: the method of Bui and Takatsuka (WSOM 2007) on GeoSOM.

- `distance_transform`: the distance wave of Algorithm 2 (`'wavefront'`), a Dijkstra check, and the printed
  algorithm verbatim (`'paper'`); step costs `'node'` (U-height, as in the paper), `'mean'`, `'edge'` or a callable;
  `diagonal_factor` for diagonals of the Wu–Takatsuka 2D index array.
- `descend`: Algorithm 1 (`'steepest'`) and an exact back-tracking rule.
- `flattest_path`: floodplain analysis, Algorithm 3 (`'iterative'`), and a direct `'minimax'` threshold.
- `PathFinder` for trained GeoSOM / PlaneSOM / LineSOM maps; samples or neuron indices as endpoints.
- `gui.PathViewer` / `plot_paths`: paths and the floodplain on GeoSOM's rotatable map projection.
- Four kinds of path (`KINDS`, `PathFinder.path` / `all_paths`): `'shortest'` (U-height distance transform),
  `'flattest'` (floodplain), `'hops'` (fewest steps on the geodesic grid; step cost `'hops'`) and `'edge'`
  (shortest walk through data space).  Equally short paths are resolved towards the geometrically straightest.
- `gui.PathPicker` and `examples/03_interactive_paths.py`: click a start and a goal neuron; a check box per kind of
  path (or keys 1-4) turns it on or off, another the floodplain crosses; a button (or `m`) cycles the distance maps.
- `PathPicker`: a *Select start & goal* button (or Enter; Esc cancels) chooses the end points; other clicks
  inspect a neuron -- its attributes and its step along each path; `n` / `b` (or `step(±1)`) walk along the path.
- `PathViewer.set_path_visible` / `toggle_path`; clicking a legend entry shows or hides that path.
- `datasets.binary_tree`: the synthetic data of section 4.1; examples reproducing figures 8 and 9.
- GitHub Actions: tests (lint; pytest on Linux, macOS and Windows with Python 3.10–3.14; the examples), and
  publishing to PyPI when a GitHub release is published (`RELEASING.md`).
