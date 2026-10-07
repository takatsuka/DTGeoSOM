# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Masahiro Takatsuka. See the NOTICE file for attribution terms.
"""
Paths drawn on GeoSOM's rotatable map projection (needs matplotlib: pip install "geosom[interactive]").

    from mt.dtgeosom import PathFinder
    from mt.dtgeosom.gui import PathViewer

    finder = PathFinder(som)
    viewer = PathViewer(som, data, labels=labels, projection='Wagner III')
    viewer.add_path(finder.shortest_path(a, b), color='white', label='shortest')
    viewer.add_path(finder.flattest_path(a, b), color='magenta', label='flattest')
    viewer.show_floodplain(viewer.paths[-1].threshold)          # grey out the neurons above it
    viewer.show()                                               # or viewer.save('paths.png')

PathViewer is mt.geosom.gui.SOMViewer (U-matrix colours, rotation, neuron inspector, projections
-- the paper used Wagner III) with paths on top.  Every step between neighbouring neurons is drawn
along the great circle joining them, and broken where it crosses the edge of the map, so the paths
stay right while the sphere is rotated.
"""
from __future__ import annotations

import numpy as np
from mt.geosom.gui import SOMViewer
from numpy import ndarray

from mt.dtgeosom.paths import SOMPath

_SAMPLES = 12                               # points per great-circle step


def great_circle(points: ndarray, samples: int = _SAMPLES) -> ndarray:
    """Unit vectors along the great circles joining consecutive rows of `points` ((k, 3) -> (m, 3))."""
    p = np.asarray(points, dtype=float)
    if len(p) < 2:
        return p
    out = [p[:1]]
    t = np.linspace(0.0, 1.0, samples + 1)[1:, None]
    for a, b in zip(p[:-1], p[1:], strict=True):
        omega = np.arccos(np.clip(a @ b, -1.0, 1.0))
        if omega < 1e-9:
            seg = np.repeat(b[None, :], len(t), axis=0)
        else:
            seg = (np.sin((1 - t) * omega) * a + np.sin(t * omega) * b) / np.sin(omega)
        out.append(seg)
    return np.vstack(out)


def _break_at_seam(xy: ndarray, jump: float) -> ndarray:
    """Inserts NaN rows where consecutive projected points jump across the map (the map's edge)."""
    if len(xy) < 2:
        return xy
    gaps = np.nonzero(np.linalg.norm(np.diff(xy, axis=0), axis=1) > jump)[0]
    return np.insert(xy, gaps + 1, np.nan, axis=0) if len(gaps) else xy


class PathViewer(SOMViewer):
    """
    SOMViewer with paths.  All SOMViewer arguments are accepted (som, data, labels, feature_names,
    layer, projection, view, title, figsize, ...).
    """

    def __init__(self, som, data=None, labels=None, **kwargs):
        self.paths: list[SOMPath] = []
        self._path_artists: list[tuple] = []           # (path, line, start marker, goal marker, xyz)
        self._flood_artist = None
        self._flood_nodes: ndarray | None = None
        super().__init__(som, data, labels, **kwargs)
        self.on_rotate(lambda _v: self._update_paths())
        self.on_projection(lambda _v: self._update_paths())

    # ------------------------------------------------------------------ public
    def add_path(self, path: SOMPath, color: str = 'white', label: str | None = None, width: float = 2.5):
        """Draws `path` (a SOMPath) with markers at its start (circle) and goal (star)."""
        xyz = great_circle(self.som.points[path.nodes])
        line, = self.ax.plot([], [], '-', color=color, lw=width, zorder=6, solid_capstyle='round',
                             label=label or f'{path.kind} {path.start}->{path.goal}')
        halo, = self.ax.plot([], [], '-', color='black', lw=width + 1.6, zorder=5, alpha=0.55)
        start, = self.ax.plot([], [], 'o', ms=9, mfc=color, mec='black', mew=1.2, zorder=7)
        goal, = self.ax.plot([], [], '*', ms=15, mfc=color, mec='black', mew=1.0, zorder=7)
        for a in (line, halo, start, goal):
            a.set_clip_path(self._outline)
        self.paths.append(path)
        self._path_artists.append((path, line, halo, start, goal, xyz))
        self._update_paths()
        self.ax.legend(handles=[a[1] for a in self._path_artists], loc='lower left', fontsize=8, framealpha=0.85)
        self.fig.canvas.draw_idle()
        return self

    def centre_on(self, *paths: SOMPath):
        """Rotates the map so that the neurons of `paths` (default: all paths shown) are in the middle."""
        paths = paths or tuple(self.paths)
        if not paths:
            return self
        pts = self.som.points[np.concatenate([p.nodes for p in paths])]
        c = pts.mean(axis=0)
        if np.linalg.norm(c) < 1e-9:                     # the path goes all the way round: use its middle
            c = pts[len(pts) // 2]
        c = c / np.linalg.norm(c)
        self.set_view(np.degrees(np.arcsin(np.clip(c[2], -1, 1))), np.degrees(np.arctan2(c[1], c[0])))
        self._update_paths()
        return self

    def clear_paths(self):
        """Removes every path."""
        for _p, *artists, _xyz in self._path_artists:
            for a in artists:
                a.remove()
        self._path_artists.clear()
        self.paths.clear()
        legend = self.ax.get_legend()
        if legend is not None:
            legend.remove()
        self.fig.canvas.draw_idle()
        return self

    def show_floodplain(self, threshold: float | None, u_height: ndarray | None = None):
        """Marks the neurons above `threshold` (off the floodplain) with grey crosses; None hides them."""
        if self._flood_artist is not None:
            self._flood_artist.remove()
            self._flood_artist, self._flood_nodes = None, None
        if threshold is not None:
            u = self.som.u_matrix() if u_height is None else np.asarray(u_height)
            self._flood_nodes = np.nonzero(u > threshold)[0]
            xy = self._project(self.som.points[self._flood_nodes])
            self._flood_artist = self.ax.scatter(xy[:, 0], xy[:, 1], marker='x', s=14, c='0.25',
                                                 linewidths=0.8, zorder=4.5)
            self._flood_artist.set_clip_path(self._outline)
        self.fig.canvas.draw_idle()
        return self

    # ----------------------------------------------------------------- drawing
    def _update_paths(self):
        if not getattr(self, '_path_artists', None) and self._flood_artist is None:
            return
        outline = self.map.outline()
        jump = 0.25 * float(np.ptp(outline[:, 0]))
        for path, line, halo, start, goal, xyz in self._path_artists:
            xy = _break_at_seam(self._project(xyz), jump)
            line.set_data(xy[:, 0], xy[:, 1])
            halo.set_data(xy[:, 0], xy[:, 1])
            ends = self._project(self.som.points[[path.start, path.goal]])
            start.set_data(ends[:1, 0], ends[:1, 1])
            goal.set_data(ends[1:, 0], ends[1:, 1])
        if self._flood_artist is not None:
            self._flood_artist.set_offsets(self._project(self.som.points[self._flood_nodes]))

    def _overlay_artists(self):
        extra = [a for _p, *artists, _xyz in getattr(self, '_path_artists', []) for a in artists]
        if getattr(self, '_flood_artist', None) is not None:
            extra.append(self._flood_artist)
        return super()._overlay_artists() + extra


def plot_paths(som, paths, data=None, labels=None, *, colors=('cyan', 'magenta', 'yellow', 'white'),
               names: list[str] | None = None, floodplain: float | None = None, centre: bool = True,
               path: str | None = None, **kwargs) -> PathViewer:
    """
    A PathViewer showing `paths` (SOMPaths) with legend entries `names`, centred on them (centre=True),
    with the neurons above the U-height `floodplain` crossed out, and saved to `path` when given.
    kwargs go to PathViewer (e.g. projection='Wagner III', view=(lat, lon), node_labels='all').
    """
    viewer = PathViewer(som, data, labels, **kwargs)
    for i, p in enumerate(paths):
        viewer.add_path(p, color=colors[i % len(colors)], label=names[i] if names else None)
    if centre and 'view' not in kwargs:
        viewer.centre_on()
    if floodplain is not None:
        viewer.show_floodplain(floodplain)
    if path is not None:
        viewer.save(path, dpi=150)
    return viewer
