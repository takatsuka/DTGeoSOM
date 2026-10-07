# The algorithms, and how they map onto GeoSOM

Bui and Takatsuka (WSOM 2007) describe three algorithms for path finding on the Geodesic SOM. This page states
each one as implemented, and notes every place where the code departs from the printed pseudocode.

## Setting

A trained GeoSOM has `n` neurons on a geodesic dome. Each neuron `i` has a weight vector `w_i` and a **U-height**
`u_i` — the mean distance between `w_i` and the weight vectors of its neighbours (`som.u_matrix()`; the paper's
"avg diff"). High U-heights are cluster borders: the "mountain ranges" of the U-matrix landscape.

`Lattice.from_som(som)` collects what the algorithms need: the neighbours of every neuron, `u`, the weights, and
for each neighbour whether it is a *diagonal* one in the Wu–Takatsuka 2D index array (below).

## Algorithm 2 — the distance transform

```
distance[goal] = 0, every other neuron unreached;  Q = [goal]
while Q is not empty:
    c = Q.pop_front()
    for every neighbour n of c that is allowed:
        if distance[c] + cost(c → n) < distance[n]:
            distance[n] = distance[c] + cost(c → n);  Q.push_back(n)   (unless n is already waiting)
```

`distance_transform(lattice, goal, method='wavefront')`. Because every improved neuron is queued again, the wave
converges to the exact cheapest-walk distance for any non-negative step cost (a label-correcting search);
`method='dijkstra'` computes the same map with a heap and is used by the tests as a check.

**Step cost.** In the paper a neuron entered by the wave adds its own U-height: `cost(c → n) = u_n` (`step='node'`,
the default). The cost of a path is then the sum of the U-heights of its neurons except the goal, so paths prefer
valleys. Alternatives: `'mean'` = (u_c + u_n)/2 (symmetric), `'edge'` = |w_c − w_n| (the length of the walk in data
space), or any callable `f(c, n)`.

**Unreached neurons.** The paper marks them with −1; here they are `inf` (`Transform.as_paper()` gives −1).

### Diagonal neighbours

GeoSOM keeps its dome in the indexed 2D data structure of Wu and Takatsuka (2006): the icosahedron is unfolded and its
vertices stored in a 2D array, where each neuron has four *direct* neighbours (offsets (±1, 0), (0, ±1)) and two
*diagonal* ones (offsets (+1, +1), (−1, −1)). The paper propagates across diagonals with `d = √2 · u_c`. On the sphere
all six neighbours are one ring apart — the √2 is a property of the array, not of the map — so by default all
neighbours are treated alike. `diagonal_factor=np.sqrt(2)` multiplies the cost of diagonal steps, and
`Lattice.diagonal` exposes the flags (exactly a third of the dome's edges are diagonals).

### Duplicate neurons

The 2D array stores vertices on the seams of the unfolded icosahedron more than once, and the paper copies every
distance update to those duplicates. GeoSOM numbers the unique neurons (`som.index_map` maps stored vertices to them),
so every neuron exists once and there is nothing to copy.

### The printed version

`method='paper'` transcribes Algorithm 2 line by line: a newly reached neuron gets `c.distance + n.u`; an already
reached one is lowered to `c.distance + √2 · c.u` across a diagonal, and to `c.distance + n.u` otherwise. The two
rules use different costs for the same step, so the result depends on the visiting order and is not a shortest-path
distance for any single cost; it is kept for reproducing the published results. `PathFinder.paper(som)` selects it
together with steepest descent.

## Algorithm 1 — reading the path off the distance map

Starting at the start neuron, step to a neighbour lower on the distance map until the goal (distance 0) is reached.
`descend(..., rule='steepest')` takes the lowest neighbour, as printed. `rule='exact'` (default) takes the neighbour
`m` minimising `distance[m] + cost(m → current)` — the neighbour the wave came from — so the walk is a cheapest path
for whatever step cost was used. With `step='node'` the two coincide (up to ties), since `cost(m → current) = u_current`
is the same for every `m`. Ties are broken towards the neighbour nearer the goal; a walk that would revisit a neuron
raises `NoPathError`.

## Algorithm 3 — floodplain analysis: the flattest, shortest path

```
t = max(u_start, u_goal)
repeat:
    distance transform towards the goal, ignoring neurons with u > t
    if the start was reached: stop
    t = lowest U-height among the ignored neurons the wave ran into
descend from the start
```

`flattest_path(..., threshold='iterative')`. All neurons with `u ≤ t` form the *floodplain*; start and goal are
always on it. The transform records the lowest U-height of the ignored neurons it touches (`Transform.next_threshold`),
which is the paper's "next max u-height".

**The final threshold is the minimax height.** Each raise goes to the lowest neuron on the boundary of the goal's
region, and any walk from the start must cross that boundary, so the threshold never overshoots: the loop ends at the
smallest `t ≥ max(u_start, u_goal)` at which start and goal are connected — the height of the lowest pass between them.
`threshold='minimax'` (or `minimax_threshold`) finds that value directly with a union-find sweep over the neurons in
order of U-height and then runs the transform once. The tests check that both give the same threshold and path cost.

## Results reproduced

`examples/01_binary_tree.py` builds the paper's 15-node binary tree (15 × 15 graph-distance matrix), trains a GeoSOM(2)
(42 neurons) online with learning rate 0.8, initial radius 3, 150 epochs, and checks every pair of tree nodes. With
seed 0: 46 % of the shortest paths and 82 % of the flattest paths follow the tree; for 8 → 13 the shortest path is
8-9-12-13 across the mountain range (figure 8) and the flattest path 8-4-2-1-3-6-13 (figure 9), found after three
threshold rounds. The NSW public-library benchmarking data of section 4.2 are not included.
