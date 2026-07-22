# Claims manifest — Paper 1: Exact Responsibilities

One row per empirical claim. Verify your final LaTeX against this
table, not against the draft prose. All values reproduced 2026-07-22;
regenerate via `python -m pytest tests/test_implicit_em.py -v` and the
snippet in `reports/implicit_em_lab_report.md`. E1 is deterministic;
E2 uses seed 5; float values are exact unless marked (~).

| # | claim | exact value | source |
|---|---|---|---|
| C1 | testbed circuit (Exp. 1) | 44 nodes, 7 OR (LSE) nodes, 15 FD variables | lab report §Exp.1; `test_implicit_em.py::test_1` |
| C2 | max \|gradient − responsibility\| over all leaves, at depth | 2.22e-16 (float64 machine epsilon) | same |
| C3 | per-variable gradient sums (simplex) | 1.0 within 1e-9 | same |
| C4 | observed variable's responsibility on its observed value | 1.0 within 1e-9 | same |
| C5 | illustrative recovered posterior P(v1 \| alarm) | ok 0.4737 / bad 0.5263 | lab report §Exp.1 |
| C6 | illustrative recovered posterior P(v2 \| alarm) | ok 0.6205 / weak 0.1163 / bad 0.2632 | same |
| C7 | Exp. 2 setup | true P(bad)=0.3; fp=fn=0.1; N=2,500 alarm-only obs; both starts at 0.5 | `test_implicit_em.py::test_2` |
| C8 | observed alarm frequency (seed 5) | 0.3532 | lab report §Exp.2 |
| C9 | analytic MLE (0.3532−0.1)/0.8 | 0.316500 | same |
| C10 | EM estimate (tol 1e-13) | 0.316500 (17 iterations) | same |
| C11 | SGD estimate (Adam, 500 epochs, lr 0.1) | 0.316500 | same |
| C12 | average responsibility r̄ at EM convergence | 0.316500 (= prior; stationarity confirmed, abs 1e-5) | same |
| C13 | final average log-likelihood, EM and SGD | −0.649405 both (identical) | same |
| C14 | four-way coincidence precision | six decimal places | derived from C9–C12 |
| C15 | gauge direction: uniform likelihood scale by c | log-evidence shifts by exactly log c (c=10 verified, abs 1e-9) | `test_implicit_em.py::test_3a` |
| C16 | gauge direction: effect on posteriors | none — every posterior invariant, abs 1e-9 | same |
| C17 | no clamped observation: collapsed vs truthful detector | collapsed (input-ignoring) achieves strictly higher likelihood | `test_implicit_em.py::test_3b` |
| C18 | one hard evidence channel added | ordering reverses; truthful wins | same |
| C19 | degeneracies encountered before the theorem was known to implementers | development history: gauge blowup and posterior collapse hit as bugs during `torch_learn` development, fixed by log-softmax + grounding, later recognized as the theorem's conditions | session history; `neximode/torch_learn.py` docstrings; `tests/test_observation.py` |
| C20 | broader EM/SGD agreement (supporting, different benchmark) | agree with each other and analytic MLE to ~1e-4 | `tests/test_torch_learn.py::test_sgd_recovers_prior_and_matches_em` |
| C21 | corollary claim | Darwiche's differential semantics = ∂logWMC/∂log w = P(value \| e) verified against brute-force enumeration on randomized circuits | `tests/test_torch.py::test_marginals_match_brute_force` |

**Notes for the rewrite.** (a) C2's "machine epsilon" phrasing is
literal: 2.220446049250313e-16 = 2⁻⁵². (b) C13's likelihoods are
*average per observation*. (c) C19 is a historical claim, not a
numerical one — keep its wording within what the session record
supports ("during development, before the paper was read"). (d) If any
library change shifts these values, rerun the two commands at top and
diff this file.
