# 6. Real telemetry, and an external baseline

**Real data.** Section 5's telemetry was sampled with the modeled
error rates, so misspecification was structurally absent. We therefore
evaluated on real instrumented hardware: the UCI hydraulic-rig
condition-monitoring dataset [HelwigEtAl2015] — 2,205 physical load
cycles with labeled component conditions. Diagnosing the cooler (3
conditions) and pump leakage (3 levels) from quantile-bucketed
cycle-mean sensor features, with constraints (mode → observed bucket
support) and priors taken from even cycles and evaluation on the 1,102
odd cycles, evidence entering as ε-smoothed soft observations:

| target | top-1 accuracy | mean posterior on truth |
|---|---|---|
| cooler | 0.868 | 0.872 |
| pump leakage | 0.720 | 0.762 |

The result that matters is **calibration**. Binned by reported
confidence: above 0.8, observed accuracy is **1.000** for both targets
(917 and 598 cycles); in the 0.4–0.6 band, observed accuracy is 0.26
(cooler) and 0.16 (pump) — overconfident by 2–3×. Exact inference on a
misspecified model is exactly this: where the crude bucket-support
constraints suffice, the posteriors are perfect probabilities; where
classes overlap, the model's confusion is reported *as if it were the
world's*. The deployment guidance follows directly — trust the
confident calls, treat mid-range posteriors as "model doesn't know,"
and let the miscalibrated bins point at where the sensor model needs
refinement. We regard measuring this, rather than hypothesizing it,
as a contribution of the study.

**External baseline.** We built the D4 compiler [LagniezMarquis2017]
from source and ran it on shared DIMACS instances against our
reference compiler:

| instance | ours (nodes / s) | D4 (nodes / s) | model counts |
|---|---|---|---|
| random 3-CNF, n=40 | 17,624 / 0.44 | 11,255 / 0.12 | match |
| grid 14×14 | 2,811 / 0.32 | 648 / 0.05 | match |
| pigeonhole 7→6 | 1 / 0.08 | 2 / 0.01 | match (UNSAT) |

Counts agree on every instance — independent validation of the
compiler by an industrial tool, which we value above the performance
comparison. That comparison is nonetheless honest: D4 is 4–6× faster
with 1.6–4.3× smaller circuits, the widest gap on spatial structure,
consistent with our own finding that decomposition quality (D4's
hypergraph partitioning versus our lightweight heuristics) is the
differentiator. The conclusion we draw is architectural, not
competitive: the pure-Python compiler is a correct, readable reference
implementation; production-scale compilation should route through
external compilers via the interchange path the library already
provides, while every downstream capability of Sections 3–5 —
diagnosis, tracking, learning — operates unchanged on the imported
circuits.
