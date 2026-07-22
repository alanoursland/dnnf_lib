# External-compiler comparison: D4

*(d4 v1 (crillab), built from source; same DIMACS inputs; our compiler
= boolean path, smooth=True. Counts cross-checked.)*

| instance | ours nodes / time | d4 nodes / time | model counts |
|---|---|---|---|
| random 3-CNF n=40 r=2.0 | 17,624 / 0.44 s | 11,255 / 0.12 s | **match** (26,035,345) |
| grid 14x14 | 2,811 / 0.32 s | 648 / 0.05 s | **match** (1.0e19) |
| pigeonhole 7→6 (UNSAT) | 1 / 0.08 s | 2 / 0.01 s | **match** (0) |

Findings: (1) **independent validation** — an industrial compiler
agrees with ours on every count; (2) d4 is 4–6x faster and produces
1.6–4.3x smaller circuits, largest gap on spatial structure (grid) —
consistent with our own dtree experiments: decomposition quality is
the differentiator, and our FM-lite partitioner does not close it;
(3) the honest conclusion for the paper: the pure-Python compiler is a
correct reference implementation, and production use at scale should
route through external compilers (the `.nnf`/driver interop path)
while all downstream queries remain ours.

Retires limitation 10. Caveat: d4's arc-format output is not yet
parsed into our evaluators (its self-reported stats and counts are
used here); the c2d format path is the one with full round-trip.
