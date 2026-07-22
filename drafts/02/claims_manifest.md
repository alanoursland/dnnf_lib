# Claims manifest — Paper 2: Compiled Diagnosis, Made Differentiable

One row per empirical claim. Verify final LaTeX against this table.
Regenerate: `python drafts/02/case_study.py` (seed 2026),
`python drafts/02/real_data_study.py` (deterministic given the dataset
at /tmp/hydraulic), `python bench/scale.py`, d4 per
`drafts/02/d4_comparison.md`. Values reproduced 2026-07-22.

## Case study (§5) — `case_study.py`

| # | claim | exact value |
|---|---|---|
| C1 | circuit size | 152 nodes, 302 edges, 48 FD variables |
| C2 | compile time | 4 ms (timing varies run-to-run; a later cold run measured 63 ms — quote "milliseconds" or re-measure at submission) |
| C3 | serialized artifact / load | 5.2 KiB / 0.2 ms, header states all bounds |
| C4 | diagnosis query time | 1.9 ms |
| C5 | no-charging scenario ranking | all-nominal 0.6032 · ctrl stuck 0.2986 · battery weak 0.0321 · pv degraded 0.0254 |
| C6 | VOI ranking (nats) | chg 0.5985 · soc_rising 0.4577 · pv_i 0.0429 |
| C7 | after observing soc_rising=False | ctrl stuck 0.7402 (top); battery weak 0.0795 |
| C8 | tracked month, P(weak) by day | d12: 0.000 · d13 (onset): 0.074 · d14: 0.618 · d15: 0.969 · d16: 0.998 · d30: 0.995 |
| C9 | detection latency (P>0.5) | day 14, one day after onset |
| C10 | fleet learning setup | N=2,000 sensor-only records; true battery prior (0.90, 0.08, 0.02); starts weak=0.10/0.33/0.60 |
| C11 | fitted priors (all three starts) | (0.879, 0.101, 0.020); 6/16/18 iterations |
| C12 | multi-start spread on weak rate | 2.93e-4 |
| C13 | log-likelihood traces | non-decreasing, all runs; e.g. −1.5216 → −0.7050 from worst start |

## Real data (§6) — `real_data_study.py`, UCI dataset 447

| # | claim | exact value |
|---|---|---|
| C14 | dataset | 2,205 real hydraulic-rig load cycles, labeled; train=even, test=1,102 odd cycles |
| C15 | cooler top-1 / mean posterior on truth | 0.868 / 0.872 |
| C16 | pump-leak top-1 / mean posterior on truth | 0.720 / 0.762 |
| C17 | cooler calibration by confidence bin | [0.2–0.4]: 0.123 (n=65) · [0.4–0.6]: 0.258 (n=120) · [0.8–1.0]: 1.000 (n=917) |
| C18 | pump calibration by confidence bin | [0.4–0.6]: 0.160 (n=351) · [0.6–0.8]: 0.908 (n=153) · [0.8–1.0]: 1.000 (n=598) |
| C19 | the headline phrasing | confident bins perfectly calibrated; mid-range overconfident 2–3× (0.4–0.6 bins: stated ~0.5, observed 0.16–0.26) |

## External baseline (§6) — `d4_comparison.md`

| # | claim | exact value |
|---|---|---|
| C20 | random 3-CNF n=40 r=2.0 | ours 17,624 nodes / 0.44 s · d4 11,255 / 0.12 s · counts match (26,035,345) |
| C21 | grid 14×14 | ours 2,811 / 0.32 s · d4 648 / 0.05 s · counts match (10,008,547,669,287,945,680) |
| C22 | pigeonhole 7→6 | ours 1 node / 0.08 s · d4 2 / 0.01 s · both UNSAT (count 0) |
| C23 | summary ratios | d4 4–6× faster; circuits 1.6–4.3× smaller; counts match on 3/3 |

## Queries & scale (§3) — `bench/scale.py`, `docs/SCALE.md`

| # | claim | exact value |
|---|---|---|
| C24 | modes-first on adders (constrained vs free nodes) | 1-bit: 368 vs 147 · 2-bit: 3,589 vs 317 · ≥4-bit: exceeded budget (reported "intractable by 4 bits") |
| C25 | modes-first on process lines | identical node counts at every tested size (5–120 stages) — "free" |
| C26 | scale, free order | 32-bit adder (160 fault modes, 1,313 FD vars): 5,777 nodes, 0.77 s · 200-stage line (400 mode vars): 9,201 nodes, 31 s |
| C27 | MPE≠MAP ranking divergence exists | constructed case in `tests/test_map.py::test_map_and_mpe_rankings_differ` (also tutorial ch5: 0.359 vs 0.621) |
| C28 | FD vs one-hot circuit size | ~40% smaller on diagnosis chains (596 → 351 nodes at 25 components) |

## Tracking & learning (§4) — test suite

| # | claim | exact value |
|---|---|---|
| C29 | beam tracking exact when covering | matches hand-coded HMM forward filter, rel 1e-9, 5 obs sequences | `tests/test_tracking.py` |
| C30 | correlated dynamics exact | matches exact joint forward filter, rel 1e-9 | `tests/test_torch_learn.py::test_transition_fn_correlated_failure` |
| C31 | EM/SGD shared fixed point | ~1e-4 agreement with each other and analytic MLE | `tests/test_torch_learn.py`; paper-1 manifest C9–C13 |
| C32 | neural front-end without observable labels | detector learns correct direction from Gaussian raw signals, supervision = hard alarm channel + structure only | `tests/test_observation.py` |
| C33 | GPU marginals | one backward pass = all posterior marginals, verified vs enumeration | `tests/test_torch.py::test_marginals_match_brute_force` |

**Notes for the rewrite.** (a) C2: re-time on the submission machine;
all other case-study numbers are deterministic under seed 2026.
(b) C21's count in the draft is written "1.0e19" — full value here if
you prefer exact. (c) C24's "intractable" = exceeded a 40 s/ratio
budget guard, not a proven blowup — keep the hedge. (d) The C19
phrasing "2–3×" derives from 0.5/0.26 ≈ 1.9 and 0.5/0.16 ≈ 3.1; "2–3×"
is the honest rounding. (e) Dataset citation: Helwig, Pignanelli &
Schütze, IEEE I2MTC 2015; UCI ML Repository #447.
