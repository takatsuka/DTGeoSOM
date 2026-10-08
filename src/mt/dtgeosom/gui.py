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
from mt.geosom.gui._figure import size_inches
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
        self.fig.canvas.mpl_connect('pick_event', self._on_legend_pick)

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
        self._draw_legend()
        self.fig.canvas.draw_idle()
        return self

    def set_path_visible(self, which, on: bool = True):
        """
        Shows or hides a path without removing it.  `which`: its index in `paths`, the SOMPath itself, or its
        kind ('shortest', 'flattest', ...; every path of that kind).  Clicking its legend entry does the same.
        """
        for i, (path, *artists, _xyz) in enumerate(self._path_artists):
            if which is path or which == i or (isinstance(which, str) and path.kind == which):
                for a in artists:
                    a.set_visible(bool(on))
        self._draw_legend()
        self.fig.canvas.draw_idle()
        return self

    def path_visible(self, which) -> bool:
        """True if the path (index, SOMPath or kind) is shown."""
        for i, (path, line, *_rest) in enumerate(self._path_artists):
            if which is path or which == i or (isinstance(which, str) and path.kind == which):
                return line.get_visible()
        return False

    def toggle_path(self, which):
        """Shows the path if hidden, hides it if shown."""
        return self.set_path_visible(which, not self.path_visible(which))

    def _draw_legend(self):
        if not self._path_artists:
            return
        lines = [a[1] for a in self._path_artists]
        legend = self.ax.legend(handles=lines, loc='lower left', fontsize=8, framealpha=0.85,
                                title='click to show / hide', title_fontsize=7)
        self._legend_map = {}
        for entry, text, line in zip(legend.get_lines(), legend.get_texts(), lines, strict=True):
            on = line.get_visible()
            entry.set_alpha(1.0 if on else 0.2)
            text.set_alpha(1.0 if on else 0.4)
            for artist in (entry, text):
                artist.set_picker(6)
                self._legend_map[artist] = lines.index(line)

    def _on_legend_pick(self, event):
        i = getattr(self, '_legend_map', {}).get(event.artist)
        if i is not None and i < len(self._path_artists):
            self._legend_clicked(i)

    def _legend_clicked(self, i: int):
        self.toggle_path(i)

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
        self._legend_map = {}
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


def plot_paths(som, paths, data=None, labels=None, *, colors=('cyan', 'magenta', 'yellow', 'lime'),
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


def _free_keys(*keys):
    """Takes `keys` out of matplotlib's own shortcuts (e.g. f = full screen, c = back) so the picker gets them."""
    import matplotlib as mpl
    for name in [k for k in mpl.rcParams if k.startswith('keymap.')]:
        mpl.rcParams[name] = [k for k in mpl.rcParams[name] if k not in keys]


class PathPicker(PathViewer):
    """
    Pick two neurons with the mouse and see the paths between them.

        from mt.dtgeosom import PathFinder
        from mt.dtgeosom.gui import PathPicker

        PathPicker(PathFinder(som), data, labels, kinds=('shortest', 'flattest', 'hops', 'edge')).show()

    The kinds of path (mt.dtgeosom.pathfinder.KINDS), each in its own colour:
        1  shortest   distance transform, step cost = U-height (the paper's shortest path)       cyan
        2  flattest   floodplain analysis (the paper's flattest, shortest path) + its floodplain  magenta
        3  hops       fewest steps on the geodesic grid, ignoring the data                        yellow
        4  edge       shortest walk through data space (distance between neighbouring weights)    lime

    Mouse and keyboard (plus everything SOMViewer offers: drag to rotate, layers, projections, ...)
        'Select start & goal' button (or Enter): the next click is the start neuron, the one after it the
                     goal -- the paths are found and drawn.  Press the button again (or Esc) to cancel.  With
                     no path yet (at the beginning, or after c) the picker is already selecting.
        click        otherwise, inspects the neuron: its attributes appear on the right, and the line under
                     the map says which step of which path it is (or that it is not on a path)
        n / b        inspect the next / previous neuron along the path
        'paths' box  (left) a check box per kind of path, one for the floodplain crosses, and a
                     button that cycles the distance maps; clicking a legend entry hides that path too
        1 2 3 4      show / hide that kind of path        a   show all four kinds
        f            same as 2 (flattest)
        m            colour the map by a distance map towards the goal: each press shows the next shown
                     kind's map, then goes back to the layer
        x            swap start and goal                   c   clear

    :param finder: a PathFinder for the trained GeoSOM (its step cost is used by 'shortest' and 'flattest')
    :param data, labels: optional samples and their labels (classes passed along the path are reported)
    :param kinds: the kinds of path to show at first (default: 'shortest' only)
    :param flattest: shorthand for adding 'flattest' to `kinds`
    :param kwargs: anything PathViewer / SOMViewer accepts (projection, view, node_labels, title, ...)
    """

    COLOURS = {'shortest': 'cyan', 'flattest': 'magenta', 'hops': 'yellow', 'edge': 'lime'}
    KEYS = {'1': 'shortest', '2': 'flattest', '3': 'hops', '4': 'edge', 'f': 'flattest'}
    HELP = ('Select start & goal (button, Enter) then click two neurons  ·  other clicks inspect a neuron, '
            'n / b: next / previous on the path\n'
            'paths: boxes on the left or keys 1-4, a all  ·  m distance map  ·  x swap  ·  c clear')

    def __init__(self, finder, data=None, labels=None, *, kinds=('shortest',), flattest: bool = False, **kwargs):
        from mt.dtgeosom.pathfinder import KINDS
        self.finder = finder
        self.start: int | None = None
        self.goal: int | None = None
        kinds = list(kinds) + (['flattest'] if flattest and 'flattest' not in kinds else [])
        unknown = [k for k in kinds if k not in KINDS]
        if unknown:
            raise ValueError(f'unknown kinds {unknown}; choose from {KINDS}')
        self.kinds: list[str] = [k for k in KINDS if k in kinds]
        self.distance_kind: str | None = None          # whose distance map colours the map (None: the layer)
        self.inspected: int | None = None              # the neuron on a path whose attributes are shown
        self._step_kind: str | None = None             # the path n / b step along
        self._summary: list[str] = []
        self.selecting: str | None = 'start'           # 'start' / 'goal' while choosing the end points, else None
        self._pending_start: int | None = None
        self.result: dict[str, SOMPath] = {}
        kwargs.setdefault('title', 'Click a start neuron, then a goal neuron')
        super().__init__(finder.som, data, labels, **kwargs)
        self._node_labels = (finder.som.node_labels(np.asarray(data, dtype=float), list(labels))
                             if data is not None and labels is not None else None)
        box = self.ax.get_position()
        # grows upwards from just above ProjectionViewer's status line, into the space under the map's outline
        self._status = self.fig.text(box.x0 + box.width / 2, 0.025 + 0.22 / size_inches(self.fig)[1], self.HELP,
                                     ha='center', va='bottom', fontsize=8.5, family='monospace', linespacing=1.15)
        self.show_flood = True                           # the crosses on neurons above the floodplain
        self._build_path_controls()
        self.on_select(self._on_neuron)
        self.fig.canvas.mpl_connect('key_press_event', self._on_path_key)
        _free_keys('f', 'm', 'c', 'x', 'a', 'n', 'b', 'enter', 'escape', '1', '2', '3', '4')

    # --------------------------------------------------------------- controls
    FLOOD_LABEL = 'floodplain ✕'

    def _build_path_controls(self):
        """The 'paths' box: a check box per kind of path and for the floodplain, and the distance-map button."""
        from matplotlib.widgets import Button, CheckButtons

        from mt.dtgeosom.pathfinder import KINDS
        fig = self.fig
        self._path_syncing = False
        # The box goes in the left-hand column, between the class legend (top) and the projection chooser.  The
        # projection chooser is made more compact to make room; Reset view and the layer list stay where they are.
        # Sizes are in inches, so the layout holds at any window height.
        height = size_inches(fig)[1]
        rows = len(KINDS) + 1
        legend_bottom = self._class_legend_bottom()
        n_proj = len(self._radio.labels) if self._radio is not None else 0
        # rows in inches: (projection row, path row, gap, button, title) -- roomy, then tighter for short windows
        for proj_row, row, gap, button_h, title_h in ((0.24, 0.2, 0.07, 0.27, 0.22), (0.19, 0.16, 0.04, 0.24, 0.2),
                                                      (0.16, 0.14, 0.03, 0.22, 0.18)):
            proj_h = min(0.045 * n_proj + 0.01, n_proj * proj_row / height + 0.01) if n_proj else 0.0
            base = 0.51 + proj_h + (0.01 if n_proj else 0.0)
            boxes_h = rows * row / height
            select_h = button_h + 0.04
            if base + (boxes_h * height + gap + button_h + title_h + gap + select_h) / height <= legend_bottom:
                break
        if self._radio is not None:
            self._radio.ax.set_position([0.01, 0.51, 0.13, proj_h])
        x0, width = 0.012, 0.128
        bax = fig.add_axes([x0 + 0.005, base, width - 0.01, button_h / height])
        cax = fig.add_axes([x0, base + (button_h + gap) / height, width, boxes_h], frameon=False)
        cax.set_title('paths', fontsize=9, loc='left')
        sax = fig.add_axes([x0, base + (button_h + gap + title_h + gap) / height + boxes_h, width, select_h / height])
        self._select_button = Button(sax, '', color='#cfe8ff', hovercolor='#a8d4ff')
        self._select_button.label.set_fontsize(8.5)
        self._select_button.label.set_fontweight('bold')
        self._select_button.on_clicked(self._on_select_button)
        labels = [f'{k} ({n})' for n, k in enumerate(KINDS, start=1)] + [self.FLOOD_LABEL]
        status = [k in self.kinds for k in KINDS] + [self.show_flood]
        colours = [self.COLOURS[k] for k in KINDS] + ['0.35']
        try:                                                   # coloured boxes need matplotlib >= 3.7
            self._kind_checks = CheckButtons(cax, labels, status,
                                             check_props={'facecolor': 'black'},
                                             frame_props={'edgecolor': 'black', 'facecolor': colours})
        except TypeError:
            self._kind_checks = CheckButtons(cax, labels, status)
        for text in self._kind_checks.labels:
            text.set_fontsize(8)
        self._kind_checks.on_clicked(self._on_kind_check)
        self._distance_button = Button(bax, '')
        self._distance_button.label.set_fontsize(7)
        self._distance_button.on_clicked(lambda _event: self.next_distance_map())
        self._sync_path_controls()

    def _class_legend_bottom(self) -> float:
        """Figure y of the bottom of SOMViewer's class legend (top left), or a little below the top if none."""
        legend = self.fig.legends[0] if self.fig.legends else None
        if legend is None:
            return 1.0 - 0.3 / size_inches(self.fig)[1]
        try:
            box = legend.get_window_extent(self.fig.canvas.get_renderer())
            return float(self.fig.transFigure.inverted().transform((0, box.y0))[1]) - 0.01
        except Exception:                                    # noqa: BLE001  (no renderer: assume a full legend)
            return 1.0 - 1.35 / size_inches(self.fig)[1]

    def _sync_path_controls(self):
        """Makes the check boxes and the button match the current state (without firing callbacks)."""
        from mt.dtgeosom.pathfinder import KINDS
        if getattr(self, '_kind_checks', None) is None:
            return
        self._path_syncing = True
        try:
            wanted = [k in self.kinds for k in KINDS] + [self.show_flood]
            for i, (have, want) in enumerate(zip(self._kind_checks.get_status(), wanted, strict=True)):
                if have != want:
                    self._kind_checks.set_active(i)
            self._distance_button.label.set_text(f'distance map (m): {self.distance_kind or "off"}')
            selecting = getattr(self, 'selecting', None)
            if selecting is None:
                text, colour = 'Select start & goal', '#cfe8ff'
            elif self.result:
                text, colour = f'click the {selecting.upper()}  ·  cancel', '#ffe08a'
            else:
                text, colour = f'click the {selecting.upper()} neuron', '#ffe08a'
            self._select_button.label.set_text(text)
            self._select_button.color = colour
            self._select_button.ax.set_facecolor(colour)
        finally:
            self._path_syncing = False

    def _on_kind_check(self, label):
        if self._path_syncing:
            return
        if label == self.FLOOD_LABEL:
            self.set_show_floodplain(not self.show_flood)
        else:
            self.toggle_kind(label.split(' ')[0])

    def _legend_clicked(self, i: int):
        self.toggle_kind(self.paths[i].kind)

    # ------------------------------------------------------------------ public
    def set_show_floodplain(self, on: bool = True):
        """Shows or hides the crosses on the neurons above the flattest path's floodplain."""
        self.show_flood = bool(on)
        flat = self.result.get('flattest')
        self.show_floodplain(flat.threshold if (flat is not None and self.show_flood) else None,
                             self.finder.u_height)
        self._sync_path_controls()
        return self

    @property
    def flattest(self) -> bool:
        return 'flattest' in self.kinds

    @property
    def distance_shown(self) -> bool:
        return self.distance_kind is not None

    def pick(self, neuron: int):
        """Does what a click on `neuron` does (for scripts and tests)."""
        self._take(int(neuron))
        return self

    def set_endpoints(self, start: int, goal: int):
        """Sets both neurons and finds the paths."""
        self.start, self.goal = int(start), int(goal)
        self.selecting, self._pending_start = None, None
        self._solve()
        return self

    def begin_selection(self):
        """Starts choosing new end points: the next click is the start, the one after it the goal (button / Enter)."""
        self.selecting, self._pending_start = 'start', None
        self._sync_path_controls()
        self._set_status('\n'.join(self._summary + ['click the START neuron, then the GOAL neuron   ·   '
                                                     'Esc or the button cancels', self.HELP]))
        return self

    def cancel_selection(self):
        """Stops choosing end points and keeps the paths already shown (button / Esc).  Without paths it stays on."""
        if not self.result:
            return self.begin_selection()
        self.selecting, self._pending_start = None, None
        self._sync_path_controls()
        self._set_status('\n'.join(self._summary + [self.HELP]))
        return self

    def _on_select_button(self, _event=None):
        if self.selecting is not None and self.result:
            self.cancel_selection()
        else:
            self.begin_selection()

    def set_kinds(self, kinds):
        """Shows exactly these kinds of path (any of 'shortest', 'flattest', 'hops', 'edge')."""
        from mt.dtgeosom.pathfinder import KINDS
        self.kinds = [k for k in KINDS if k in set(kinds)]
        if self.distance_kind not in self.kinds:
            self.distance_kind = None
        if self.start is not None and self.goal is not None:
            self._solve()
        else:
            self._refresh_layer()
        self._sync_path_controls()
        return self

    def toggle_kind(self, kind: str):
        """Shows `kind` if hidden, hides it if shown."""
        return self.set_kinds([k for k in self.kinds if k != kind] if kind in self.kinds else self.kinds + [kind])

    def set_flattest(self, on: bool):
        return self.set_kinds([k for k in self.kinds if k != 'flattest'] + (['flattest'] if on else []))

    def show_distance_map(self, kind: str | bool | None = True):
        """
        Colours the map by the distance to the goal of path `kind` (True: the first kind shown; False / None:
        back to the current layer).  Neurons a flattest path may not enter are drawn at the top of the scale.
        """
        if kind is True:
            kind = next((k for k in self.kinds if k in self.result), None)
        self.distance_kind = kind if kind in self.result else None
        self._refresh_layer()
        self._sync_path_controls()
        return self

    def next_distance_map(self):
        """Cycles the map colours: the distance map of each kind shown, then the layer again."""
        shown = [k for k in self.kinds if k in self.result]
        if not shown:
            return self.show_distance_map(None)
        i = shown.index(self.distance_kind) + 1 if self.distance_kind in shown else 0
        return self.show_distance_map(shown[i] if i < len(shown) else None)

    def clear(self):
        """Forgets start, goal and the paths."""
        self.start = self.goal = None
        self.result = {}
        self.distance_kind = None
        self.inspected = None
        self._summary = []
        self.selecting, self._pending_start = 'start', None
        self.clear_paths()
        self.show_floodplain(None)
        self._refresh_layer()
        self._sync_path_controls()
        self._set_status(self.HELP)
        return self

    # ----------------------------------------------------------------- internals
    def _refresh_layer(self):
        if self.distance_kind is None:
            self.set_layer(self.layer)
        else:
            d = np.array(self.result[self.distance_kind].distance, dtype=float)
            finite = np.isfinite(d)
            d[~finite] = d[finite].max() if finite.any() else 0.0
            norm = self._show_layer_colours(d, 'viridis')
            self._update_colourbar(norm, 'viridis')
            self._cax.set_ylabel(f'distance to the goal ({self.distance_kind})', fontsize=8)
        self.fig.canvas.draw_idle()

    def select_node(self, node):
        """As SOMViewer.select_node; a click on the neuron already selected counts too (e.g. as the start)."""
        same = node is not None and self.selected is not None and int(node) == self.selected
        super().select_node(node)
        if same:
            self._take(int(node))

    def _on_neuron(self, viewer):
        if viewer.selected is not None:
            self._take(int(viewer.selected))

    def _take(self, neuron: int):
        if self.selecting == 'start':
            self._pending_start = neuron
            self.selecting = 'goal'
            self._sync_path_controls()
            self._set_status('\n'.join(self._summary + [f'start: neuron {neuron}{self._labels_at(neuron)}   ·   '
                                                         'now click the GOAL neuron', self.HELP]))
        elif self.selecting == 'goal':
            self.start, self.goal = self._pending_start, neuron
            self._pending_start, self.selecting = None, None
            self._solve()
        else:                                                   # not selecting: inspect the neuron
            self._inspect(neuron)

    def _solve(self):
        self.clear_paths()
        self.show_floodplain(None)
        f = self.finder
        self.result = {kind: f.path(self.start, self.goal, kind) for kind in self.kinds}
        lines = []
        widths = dict(zip(self.result, (5.5, 4.0, 2.7, 1.5)[-len(self.result):] if self.result else (), strict=True))
        for kind, p in self.result.items():
            passed = ''
            if self._node_labels is not None:
                met = p.visited_labels(self._node_labels)
                passed = '  passes: ' + ' > '.join(map(str, met[:6])) + (' …' if len(met) > 6 else '')
            lines.append(f'{kind:8s} {p.hops:3d} hops  cost {p.cost:8.3f}  '
                         f'highest U {p.max_height(f.u_height):.3f}{passed}')
            # nested strokes, widest first, so that stretches shared by several paths show every colour
            self.add_path(p, color=self.COLOURS[kind], label=f'{kind} {p.start}→{p.goal}', width=widths[kind])
        if 'flattest' in self.result and self.show_flood:
            self.show_floodplain(self.result['flattest'].threshold, f.u_height)
        if self.distance_kind not in self.result:
            self.distance_kind = None
        self._refresh_layer()
        self._sync_path_controls()
        if not lines:
            lines = ['(no kind of path selected: press 1, 2, 3 or 4)']
        self._summary = lines
        self.inspected = None
        self._set_status('\n'.join(lines + [self.HELP]))

    # ------------------------------------------------------ inspecting the path
    def on_path(self, neuron: int) -> bool:
        """True if `neuron` lies on one of the paths shown."""
        return any(int(neuron) in p.nodes for p in self.result.values())

    def positions(self, neuron: int) -> dict[str, int]:
        """{kind: step} for every shown path through `neuron` (step 0 = the start)."""
        return {kind: int(np.nonzero(p.nodes == int(neuron))[0][0])
                for kind, p in self.result.items() if int(neuron) in p.nodes}

    def step(self, delta: int = 1):
        """
        Inspects the next (delta=1) or previous (-1) neuron along the path being inspected (the first shown
        path through the inspected neuron; from the start if none is inspected yet).  Keys n / b.
        """
        if not self.result:
            return self
        if self.inspected is None or not self.on_path(self.inspected):
            kind = self._step_kind if self._step_kind in self.result else next(iter(self.result))
            i = -1 if delta > 0 else len(self.result[kind].nodes)
        else:
            here = self.positions(self.inspected)
            kind = self._step_kind if self._step_kind in here else next(iter(here))
            i = here[kind]
        nodes = self.result[kind].nodes
        j = min(max(i + delta, 0), len(nodes) - 1)
        self._step_kind = kind
        self.select_node(int(nodes[j]))                      # -> _on_neuron -> _inspect (shows the attributes)
        return self

    def _inspect(self, neuron: int):
        """The attribute panel shows `neuron` (SOMViewer does that); the status line says where it is on the paths."""
        self.inspected = int(neuron)
        here = self.positions(neuron)
        if self._step_kind not in here:
            self._step_kind = next(iter(here), None)
        where = (',  '.join(f'{kind} step {i}/{self.result[kind].hops}' for kind, i in here.items())
                 or 'not on a path')
        u = self.finder.u_height[neuron]
        line = f'▶ neuron {neuron}{self._labels_at(neuron)}:  {where}   (U-height {u:.3f})'
        self._sync_path_controls()
        self._set_status('\n'.join(getattr(self, '_summary', []) + [line, self.HELP]))

    def _labels_at(self, neuron: int) -> str:
        if self._node_labels is None or not self._node_labels[neuron]:
            return ''
        return ' (' + ', '.join(map(str, sorted(self._node_labels[neuron]))) + ')'

    def _set_status(self, text: str):
        self._status.set_text(text)
        self.fig.canvas.draw_idle()

    def _on_path_key(self, event):
        if event.inaxes is not None and event.inaxes is not self.ax:
            return
        key = event.key or ''
        if key in self.KEYS:
            self.toggle_kind(self.KEYS[key])
        elif key == 'a':
            from mt.dtgeosom.pathfinder import KINDS
            self.set_kinds(KINDS)
        elif key == 'm':
            self.next_distance_map()
        elif key == 'enter':
            self.begin_selection()
        elif key == 'escape':
            self.cancel_selection()
        elif key in ('n', 'b'):
            self.step(1 if key == 'n' else -1)
        elif key == 'c':
            self.clear()
        elif key == 'x' and self.start is not None and self.goal is not None:
            self.set_endpoints(self.goal, self.start)
