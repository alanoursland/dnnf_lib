# Case study: a residential solar+battery system, end to end

*(Draft section for the DX paper. Every number is produced by
`drafts/02/case_study.py` — one script, seeded, CPU-only.)*

**The system.** Four components with 3/2/3/2 modes (PV array, charge
controller, battery, inverter), two internal flow states, and four
noisy sensors (2–5% error rates) — a realistic small residential
installation, and representative of the model *shape* the architecture
targets: local structure, discrete modes, cheap unreliable sensing.

## Offline: compile once

The model compiles to a **152-node** circuit (302 edges, 48
finite-domain variables including sensor-noise auxiliaries) in **4 ms**;
serialized, the deployable artifact is **5.2 KiB** and loads in 0.2 ms
from a header that states all resource bounds up front
(single-allocation loading, after the flight-software practice of the
architecture's ancestors). Everything below runs against this one
artifact.

## Online 1: snapshot diagnosis under ambiguity

Evidence: sunny, PV current present, **no charging**, house powered.
Ranked exact posteriors (query time 1.9 ms):

| rank | explanation | posterior |
|---|---|---|
| 1 | all nominal (a sensor glitched) | 0.603 |
| 2 | controller stuck off | 0.299 |
| 3 | battery weak | 0.032 |
| 4 | PV degraded | 0.025 |

Two properties worth noting are structural, not tuned: the system
*prefers a single sensor glitch* to a component fault because the
noise rates and fault priors say so — and it is honestly torn (0.60 /
0.30) rather than falsely decisive.

## Online 2: active sensing

`value_of_information` on the same evidence ranks the unread signals:
charge-current truth (0.599 nats expected entropy reduction) and the
SOC trend (0.458) far above PV internals (0.043). Reading the top
recommendation — `soc_rising=False` — resolves the ambiguity:
controller-stuck jumps to **0.740** and the sensor-glitch story
collapses. This is the diagnose → check → observe loop as a query
type.

## Online 3: a month of telemetry, one injected fault

Battery degradation transitions (`ok→weak` 0.4%/day, absorbing
`dead`), thirty days of sensor readings sampled from ground truth
**with the modeled error rates**, fault injected on day 13:

| day | P(ok) | P(weak) |
|---|---|---|
| 12 | 1.000 | 0.000 |
| 13 (fault) | 0.926 | 0.074 |
| 14 | 0.382 | **0.618** |
| 15 | 0.031 | 0.969 |
| 30 | 0.000 | 0.995 |

Detection (P(weak) > 0.5) on **day 14** — one day of corroboration
after onset, exactly what 5% sensor noise should cost. No thresholds
were tuned; the delay *is* the posterior arithmetic.

## Offline again: learning priors from the fleet

2,000 snapshot observation records (sensor readings only — mode labels
never observed) sampled from a fleet whose true battery prior is
(0.90, 0.08, 0.02). EM from three different starting priors:

| start (weak) | fitted (ok/weak/dead) | iterations |
|---|---|---|
| 0.33 | 0.879 / 0.101 / 0.020 | 16 |
| 0.10 | 0.879 / 0.101 / 0.020 | 6 |
| 0.60 | 0.879 / 0.101 / 0.020 | 18 |

Multi-start agreement to 3×10⁻⁴ (identifiable); recovered rates within
sampling error of truth; log-likelihood non-decreasing in every run.
The learned priors immediately parameterize the diagnosis and tracking
queries above — the loop closes on one artifact.

## Summary claim for the paper

One 5 KiB compiled circuit served exact ranked diagnosis (2 ms),
information-optimal sensor selection, a month of fault tracking with
1-day detection latency, and fleet-scale prior learning — with zero
per-query modeling work after the single 4 ms compile.
