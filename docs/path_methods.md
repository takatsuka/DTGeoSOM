# Shortest-path methods in DTGeoSOM

DTGeoSOM finds four kinds of path between two neurons of a trained SOM. They answer different questions.
**`flattest` is the default**: it is what `finder.path(a, b)` returns when no kind is given, and the only path the
interactive picker shows when it opens.

| Kind | Question it answers | Cost of one step c → n | From the paper? |
|---|---|---|---|
| [`shortest`](#2-shortest--the-distance-transform-on-the-u-matrix) | Which route avoids high U-heights as much as possible overall? | `u[n]`, the U-height of the neuron entered | Yes: Algorithms 2 and 1 |
| [`flattest`](#3-flattest--floodplain-analysis) | Which route never climbs higher than it has to? | `u[n]`, but only over neurons with `u ≤ t` | Yes: Algorithm 3 |
| [`hops`](#4-hops--fewest-steps-on-the-geodesic-grid) | Which route takes the fewest steps on the map? | `1` | No |
| [`edge`](#5-edge--shortest-walk-through-data-space) | Which route is shortest in the data space, moving only between neighbouring neurons? | `‖w_c − w_n‖` | No |

All four are computed by one piece of machinery — a **distance transform** over the lattice followed by a **descent**
from the start — and differ only in the step cost and, for `flattest`, in which neurons the wave may enter.
Section 1 describes that machinery; sections 2–5 describe each kind; section 6 works through an example; section 7
helps choose between them.

The method is from M. Bui and M. Takatsuka, *Path finding on a spherical SOM using the distance transform and
floodplain analysis*, WSOM 2007. [algorithm.md](algorithm.md) compares the implementation with the printed pseudocode
line by line.

```python
from mt.dtgeosom import PathFinder

finder = PathFinder(som)                 # a trained GeoSOM (PlaneSOM and LineSOM work too)
finder.shortest_path(a, b)               # a, b: neuron indices or samples (mapped to their BMUs)
finder.flattest_path(a, b)
finder.hop_path(a, b)
finder.edge_path(a, b)
finder.path(a, b)                        # the default kind: flattest
finder.all_paths(a, b)                   # {'shortest': ..., 'flattest': ..., 'hops': ..., 'edge': ...}
```

---

## 1. The common machinery

### 1.1 The lattice

`Lattice.from_som(som)` collects what every method needs:

| Field | Meaning |
|---|---|
| `neighbours[i]` | the neurons next to neuron `i` on the map (6 on a geodesic dome, 5 at the 12 icosahedron corners) |
| `u_height[i]` | the U-height of neuron `i`: the mean distance from `w_i` to its neighbours' weight vectors (`som.u_matrix()`) |
| `weights[i]` | the weight vector `w_i` (used by `edge`) |
| `positions[i]` | where the neuron sits on the map (a unit vector on the sphere); used only to break ties |
| `diagonal[i][k]` | whether the k-th neighbour is a *diagonal* one in GeoSOM's 2D index array (used only by the paper's √2 rule) |

High U-heights mark cluster borders — the "mountain ranges" of the U-matrix landscape, where the map stretches across
a gap with little or no data. Low U-heights are the valleys and plains where the data lie.

GeoSOM numbers each neuron once, so the duplicate seam vertices of the 2D index array, which the paper had to update
separately, need no special treatment.

### 1.2 The distance transform

The distance transform (`distance_transform(lattice, goal, step=…, allowed=…, method=…)`, in `transform.py`) gives
every neuron its **distance to the goal**: the cost of the cheapest walk from it to the goal over the lattice. It
works like a wave spreading out from the goal, as in robot motion planning (Jarvis 1985):

```
distance[goal] = 0;  distance[every other neuron] = ∞
Q = [goal]                                         # a first-in-first-out queue
while Q is not empty:
    c = Q.pop_front()
    for each neighbour n of c:
        if n is not allowed:                       # outside the floodplain (section 3)
            remember u[n] as a candidate next threshold
            continue
        d = distance[c] + cost(c → n)
        if d < distance[n]:                        # reached for the first time, or more cheaply
            distance[n] = d
            if n is not already in Q: Q.push_back(n)
```

- **Why it is exact.** A neuron goes back into the queue whenever its distance improves, so the wave keeps running
  until no distance can be lowered. For non-negative step costs this is a label-correcting shortest-path search
  (the Bellman–Ford–Moore queue method), and it ends with the exact cheapest-walk distance for every neuron.
- **Cost.** Typically close to linear in the number of neurons. On a 642-neuron dome (1 920 edges) the wave makes
  about 640–720 queue insertions; the worst case is O(V·E).
- **Unreachable neurons** keep the distance ∞ (the paper writes −1; `Transform.as_paper()` converts).

Three ways to run the wave (`method=`):

| `method` | What runs | Use |
|---|---|---|
| `'wavefront'` (default) | the queue above — the paper's distance transform | normal use |
| `'dijkstra'` | Dijkstra's algorithm with a binary heap, O(E log V) | gives the same distance map; the tests use it as a check |
| `'paper'` | Algorithm 2 exactly as printed (section 2.4) | reproducing the published results |

The result is a `Transform`: `.distance` (the distance map), `.goal`, `.allowed`, `.next_threshold` (the lowest
U-height among the ignored neurons the wave touched) and `.pushes` (queue insertions — the work done).

### 1.3 Reading the path off the distance map

Once every neuron knows its distance to the goal, the path is found by walking downhill from the start until the goal
(distance 0) is reached — the paper's Algorithm 1 (`descend(lattice, transform, start, rule=…)`, in `paths.py`):

```
path = [start];  c = start
while c ≠ goal:
    choose the next neuron m among the neighbours of c   (rule below)
    path.append(m);  c = m
```

| `rule` | Next neuron | Notes |
|---|---|---|
| `'exact'` (default) | the neighbour `m` with the smallest `distance[m] + cost(m → c)` — the neuron the wave actually came from | always a cheapest path for the step cost used |
| `'steepest'` | the neighbour with the smallest `distance[m]` — Algorithm 1 as printed | the same path as `'exact'` for the `shortest` and `flattest` costs, because there `cost(m → c) = u[c]` is the same for every `m` |

**Ties.** Several neighbours can be equally good, especially when counting hops. Ties are broken in this order: the
neighbour lower on the distance map, then the one **geometrically nearest the goal** on the map (so equally short
paths come out as straight as possible), then one not yet visited. A walk that would revisit a neuron, or a start the
wave never reached, raises `NoPathError`.

### 1.4 What a path's cost means

The wave runs from the goal outwards, so `cost(c → n)` is paid when the *wave* enters `n`. Seen from the walker going
from the start to the goal, each step from neuron `a` to its neighbour `b` costs `cost(b → a)`. For `shortest`, that
makes the cost of a path **the sum of the U-heights of all its neurons except the goal**. `SOMPath.cost` is the
start's value on the distance map; `path_cost(lattice, nodes, step)` recomputes it for any list of neurons.

---

## 2. `shortest` — the distance transform on the U-matrix

**The paper's path finding (Algorithms 2 and 1).** `finder.shortest_path(a, b)`.

### 2.1 Idea

Treat the U-matrix as a landscape: the U-height of a neuron is how hard it is to pass through it. The shortest path
is the route whose neurons have the smallest total U-height. It prefers valleys (similar neighbouring neurons,
where data are dense), so the weight vectors along it change gradually — the intermediate states between the start
and the goal.

### 2.2 Algorithm

```
1.  distance = distance_transform(lattice, goal, step='node')     # cost(c → n) = u[n]
2.  path     = descend(lattice, distance, start)
```

### 2.3 Properties

- It minimises the **sum** of U-heights, not the **largest** one. A short hop across a narrow ridge can cost less
  in total than a long detour across low ground, so the shortest path may still cross a mountain range — a stretch
  of map with no data behind it. The paper's figures 8 and 10 show this; `flattest` (section 3) fixes it.
- The cost depends on the direction only through the endpoints: going a → b counts `u[a]` but not `u[b]`, and b → a
  the reverse.
- Variants: `PathFinder(som, step='mean')` uses `(u[c] + u[n]) / 2`, which is symmetric; `diagonal_factor=√2`
  makes steps across diagonals of the 2D index array dearer, as in the paper. On the sphere all six neighbours are
  one ring apart, so the default treats them alike.

### 2.4 The printed version (`PathFinder.paper(som)`)

Algorithm 2 as printed differs from 2.2 in one detail: a neuron reached for the **first** time gets
`distance[c] + u[n]`, but a neuron reached **again** across a diagonal of the 2D index array is lowered to
`distance[c] + √2·u[c]` — a different cost for the same kind of step. The result therefore depends on the order in
which neurons are visited and is not the shortest-path distance for any single cost. `PathFinder.paper(som)`
reproduces it (with steepest descent) so the published results can be checked; use the default otherwise.

---

## 3. `flattest` — floodplain analysis

**The paper's flattest, shortest path (Algorithm 3), and DTGeoSOM's default.** `finder.flattest_path(a, b,
threshold='iterative')`, or `finder.path(a, b)`.

### 3.1 Idea

Flood the landscape. Every neuron whose U-height is at most a threshold `t` is on the *floodplain*; the rest are
dry hills the path may not enter. Start with the water as low as possible — just high enough to cover the start and
the goal — and raise it only until the two are connected. Then take the `shortest` path on that floodplain. The
result climbs no higher than it must: it goes round mountain ranges whenever a way round exists, staying in the part
of the map that represents data.

### 3.2 Algorithm (iterative, as in the paper)

```
t = max(u[start], u[goal])
loop:
    allowed  = { neurons n : u[n] ≤ t } ∪ {start, goal}
    distance = distance_transform(lattice, goal, step='node', allowed=allowed)
    if distance[start] < ∞: break                        # connected on this floodplain
    t = distance.next_threshold                          # lowest U-height of an ignored neuron the wave touched
    if t = ∞: raise NoPathError                           # the map is disconnected (never on a sphere)
path = descend(lattice, distance, start)
```

`SOMPath.threshold` is the final `t`; `SOMPath.thresholds` lists every value tried.

### 3.3 Why the final threshold is the lowest possible

Each round raises `t` to the lowest U-height on the boundary of the region the wave reached. Any route from the start
to the goal has to leave that region, so it must pass through a neuron at least that high. The loop therefore never
raises the water higher than needed, and it stops at the **minimax** (bottleneck) height: the smallest `t` for which
some route from start to goal stays at or below `t`.

### 3.4 Algorithm (direct: `threshold='minimax'`)

The iterative loop may take many rounds — on the 642-neuron wine map the path from neuron 10 to neuron 500 took 54,
each a full distance transform. The minimax height can instead be found directly, and the transform then run once:

```
floor = max(u[start], u[goal])
union-find over the neurons;  add neurons in increasing order of U-height,
    joining each to its already-added neighbours
stop as soon as start and goal are in the same set and every neuron with u ≤ floor has been added
t = the U-height of the last neuron added (at least floor)
```

This costs O(E log V) for the sort and nearly linear time for the union-find. Both ways give the same threshold and
the same path; the tests check this on random pairs.

### 3.5 Properties

- The highest U-height on the path is the minimax height — never more than on any other route.
- Within the floodplain it is a `shortest` path, so among the flattest routes it prefers low total U-height.
- It is usually longer than the other kinds (more hops), since detours round ridges are the point.
- The floodplain can be shown on the map: the crosses marked by `PathViewer.show_floodplain(threshold)`, or the
  *floodplain ✕* box in the picker.

---

## 4. `hops` — fewest steps on the geodesic grid

**A reference path, not from the paper.** `finder.hop_path(a, b)`.

### 4.1 Idea

Ignore the data completely and count steps: the route with the fewest moves between neighbouring neurons. On the
sphere this follows the great circle between the two neurons as closely as the grid allows. It shows what the
lattice itself suggests, so comparing it with the other paths shows how much the data bend them.

### 4.2 Algorithm

```
1.  distance = distance_transform(lattice, goal, step='hops')     # cost(c → n) = 1
2.  path     = descend(lattice, distance, start)                  # ties broken towards the goal on the map
```

With unit costs the distance map is the hop count to the goal, the same as a breadth-first search.

### 4.3 Properties

- `SOMPath.cost == SOMPath.hops`, the length of the shortest route on the grid.
- Many routes are usually equally short. The geometric tie-break (section 1.3) picks the one that keeps closest to
  the straight line on the map instead of zigzagging; the tests check that it stays within a few rings of the great
  circle.
- It crosses ridges and empty regions without hesitation — by design.

---

## 5. `edge` — shortest walk through data space

**A reference path, not from the paper.** `finder.edge_path(a, b)`.

### 5.1 Idea

Measure each step by how much the weight vector actually changes: `cost(c → n) = ‖w_c − w_n‖`, the Euclidean
distance between neighbouring neurons' weight vectors. The path is then the shortest walk through the data space,
moving only along the map's neighbour links — a discrete geodesic on the map's surface in data space.

### 5.2 Algorithm

```
1.  distance = distance_transform(lattice, goal, step='edge')     # cost(c → n) = ‖w_c − w_n‖
2.  path     = descend(lattice, distance, start, rule='exact')
```

### 5.3 Properties

- Symmetric: the cost of a → b equals that of b → a.
- `SOMPath.cost` is the length of the walk in data space. It is never shorter than the straight-line distance
  `‖w_start − w_goal‖`, and comparing the two shows how far the map bends between the endpoints.
- It charges each step for the change it actually makes, where `shortest` charges for the neighbourhood around
  each neuron (the U-height is an average over all its neighbours). The two often agree; they differ near borders,
  where one step can be small even though a neuron's other neighbours are far away.
- Like `shortest`, it minimises a sum, so it can still cross a short ridge if that is cheaper overall.

---

## 6. A worked example

A ring of eight neurons, with the start at neuron 0 and the goal at neuron 2. The direct way passes through neuron 1,
a narrow ridge; the way round passes neurons 7, 6, 5, 4, 3 on low ground. Each neuron has a one-dimensional weight.

| Neuron | 0 (start) | 1 (ridge) | 2 (goal) | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| U-height `u` | 0.10 | 0.60 | 0.10 | 0.15 | 0.15 | 0.15 | 0.15 | 0.15 |
| weight `w` | 0.00 | 2.00 | 0.40 | 0.05 | 0.10 | 0.15 | 0.20 | 0.25 |

| Kind | Path | Cost | Why |
|---|---|---|---|
| `shortest` | 0 → 1 → 2 | 0.70 = u₀ + u₁ | the ridge (0.60) is cheaper than five low neurons plus the start (0.85) |
| `flattest` | 0 → 7 → 6 → 5 → 4 → 3 → 2 | 0.85 | t starts at 0.10 (start and goal), rises once to 0.15, and the way round connects; the ridge stays dry |
| `hops` | 0 → 1 → 2 | 2 hops | two steps beat six |
| `edge` | 0 → 7 → 6 → 5 → 4 → 3 → 2 | 0.80 | the weights jump 0 → 2 → 0.4 across the ridge (3.6) but change gently the long way |

`shortest` and `flattest` disagree exactly as in the paper's figures 8 and 9: the shortest path takes the ridge, the
flattest goes round it.

On real data — the UCI wine data on a 642-neuron GeoSOM, from neuron 10 to neuron 500
(`examples/03_interactive_paths.py --start 10 --goal 500 --paths all`):

| Kind | Hops | Cost | Highest U-height on the path |
|---|---|---|---|
| `shortest` | 13 | 9.823 | 1.100 |
| `flattest` | 19 | 14.820 | 0.974 (54 threshold rounds) |
| `hops` | 13 | 13 | 1.100 |
| `edge` | 13 | 10.528 | 1.100 |

Three paths take the direct route over a ridge of height 1.10; the flattest path detours by six hops to stay below
0.974, passing through the cultivar-1 region instead of crossing the border.

---

## 7. Choosing a method

| You want… | Use |
|---|---|
| a sensible path when unsure — the default | `flattest` |
| the paper's path through the U-matrix landscape | `shortest` |
| a path that stays where the data are and never crosses an empty border it can avoid | `flattest` |
| the intermediate states between two samples, as smooth as possible | `flattest`, then compare with `shortest` |
| the geometric route on the map, ignoring the data | `hops` |
| the shortest change in attribute space, moving between neighbouring neurons | `edge` |
| to reproduce the published figures exactly | `PathFinder.paper(som)` with `shortest` / `flattest` |

Where all four agree the map is flat between the two neurons. Where `flattest` departs from the others it is going
round a ridge; where `edge` departs from `shortest`, single steps and neighbourhood averages disagree near a border.

---

## 8. Quick reference

| | `shortest` | `flattest` | `hops` | `edge` |
|---|---|---|---|---|
| Call | `shortest_path(a, b)` | `flattest_path(a, b, threshold)` | `hop_path(a, b)` | `edge_path(a, b)` |
| `path(a, b, kind)` | `'shortest'` | `'flattest'` (the default) | `'hops'` | `'edge'` |
| Step cost | `u[n]` (or `PathFinder(step=…)`) | the same | `1` | `‖w_c − w_n‖` |
| Neurons allowed | all | `u ≤ t` | all | all |
| Transforms run | 1 | 1 per round (`'iterative'`) or 1 (`'minimax'`) | 1 | 1 |
| Picker key / colour | `1` / cyan | `2` or `f` / magenta | `3` / yellow | `4` / lime |

Every call returns a `SOMPath`: `nodes`, `start`, `goal`, `kind`, `cost`, `hops`, `distance` (the distance map),
`threshold` / `thresholds` (flattest), `states(som)` (the weight vectors along the path), `max_height(u)` and
`visited_labels(node_labels)`. `PathFinder(method='dijkstra')` and `PathFinder(rule='steepest')` change how every
kind is computed (sections 1.2 and 1.3).
