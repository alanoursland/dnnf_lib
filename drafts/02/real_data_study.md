# Real-telemetry study: hydraulic test rig (UCI dataset 447)

*(Numbers from `drafts/02/real_data_study.py`; data: 2,205 real load
cycles from an instrumented hydraulic rig with labeled component
conditions — Helwig et al.; NOT sampled from our model.)*

**Setup.** Diagnose the cooler (3 conditions) and pump leakage (3
levels) from cycle-mean sensor features (cooling efficiency/power;
volumetric flow/efficiency), quantized to 8 quantile buckets. The
compiled model's constraints (mode → observed bucket support) and
priors come from the even cycles; evaluation on the 1,102 odd cycles.
Evidence enters as epsilon-smoothed soft observations (ch.6 machinery).

**Results.**

| target | top-1 accuracy | mean posterior on truth |
|---|---|---|
| cooler | 0.868 | 0.872 |
| pump leakage | 0.720 | 0.762 |

**The calibration story — the point of the exercise.** Binned by
reported confidence:

- cooler: conf > 0.8 → **1.000 observed accuracy** (917 cycles);
  conf 0.4–0.6 → 0.258 observed (120); conf 0.2–0.4 → 0.123 (65).
- pump: conf > 0.8 → **1.000** (598); 0.6–0.8 → 0.908 (153);
  0.4–0.6 → **0.160** (351).

Where the model is confident, it is *perfectly* calibrated on real
data. Where it hedges, it is **overconfident** — mid-range posteriors
overstate accuracy by 2–3x. That is misspecification made visible:
bucket-support constraints + i.i.d. smoothing are too crude exactly in
the class-overlap regions, and exact inference faithfully reports the
model's confusion as if it were the world's. The practical reading for
deployment: trust the confident calls, and treat mid-range posteriors
as "the model doesn't know" rather than as probabilities — or refine
the sensor model where the bins expose it (finer features per
confusion pair, learned conditional noise).

This retires half of limitation 4 (real data, real misspecification,
calibration measured) and sharpens the other half: rate learning and
tracking on real *sequences* remain untested.
