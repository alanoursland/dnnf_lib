# 4. Experiments

All experiments are deterministic or seeded, run in double precision on
CPU, and live in the library's continuous-integration suite
(`tests/test_implicit_em.py`); the numbers below are reported in
`reports/implicit_em_lab_report.md`. Reproduction:

```bash
pip install -e .[torch,dev] && python -m pytest tests/test_implicit_em.py -v
```

## 4.1 Gradients are responsibilities, at depth

**Setup.** A two-level diagnosis model: modes $v_1 \in \{ok, bad\}$ and
$v_2 \in \{ok, weak, bad\}$ with priors; intermediate flows
$f_1 = (v_1{=}ok)$, $f_2 = f_1 \wedge (v_2 \neq bad)$; a noisy alarm
sensor on $\neg f_2$ (false-positive and false-negative rate 0.05,
introducing hidden glitch variables). Compiled: 44 nodes with 7 OR
(LSE) nodes over 15 finite-domain variables — nested mixtures, not a
single LSE. Evidence: alarm on.

**Method.** Autograd computes
$\partial \log \mathrm{WMC} / \partial \log w$ for all 27 leaves in one
backward pass. Ground truth for every variable's posterior is computed
independently, without differentiation, as WMC ratios (themselves
validated elsewhere in the suite against exhaustive enumeration).

**Result.** Maximum absolute difference between gradient and exact
responsibility over all leaves: **2.2e-16** — double-precision machine
epsilon. Per-variable gradients lie on the simplex (sums 1 within
1e-9); the observed variable carries responsibility 1 on its observed
value. The identity survives composition exactly.

*Note.* For a variable inside the evidence, a naive
"override-and-renormalize" posterior computes a likelihood ratio, not a
conditional; the gradient correctly reports responsibility 1. The
distinction surfaced while constructing the test and is retained in its
documentation.

## 4.2 EM and gradient ascent share one fixed point

**Setup.** One mode $v \in \{ok, bad\}$, true $P(bad) = 0.3$; a noisy
alarm (rates 0.1). $N = 2{,}500$ observations of the alarm *only*,
sampled from the true model. Both learners start from the wrong prior
$P(bad) = 0.5$. The setting is identifiable:
$P(\text{alarm}) = 0.8\,p + 0.1$, so the maximum likelihood estimate is
analytic. Observed alarm frequency: 0.3532.

**Result.**

| quantity | value |
|---|---|
| analytic MLE $(0.3532 - 0.1)/0.8$ | 0.316500 |
| exact EM (17 iterations) | 0.316500 |
| Adam on softmax logits (500 epochs) | 0.316500 |
| average responsibility $\bar r$ at EM convergence | 0.316500 |
| final avg log-likelihood, EM / SGD | $-0.649405$ / $-0.649405$ |

Four independently computed quantities coincide to six decimals, and
the stationarity condition $\bar r = \pi$ is confirmed directly. The
log-likelihood trace of EM is non-decreasing (guaranteed) and SGD
reaches the same value.

## 4.3 Deleting the conditions produces the predicted degeneracies

Both degeneracies below were first encountered as *bugs* during
development of the library's neural observation-model trainer, before
the companion paper was read by the developers, and were repaired by
measures later recognized as the theorem's structural conditions. We
reproduce them deliberately.

**(a) No normalization: unbounded gauge direction.** Scaling a soft
evidence vector uniformly by $c$ shifts $\log P(\text{evidence})$ by
exactly $\log c$ (verified for $c{=}10$ to 1e-9) while leaving *every*
posterior unchanged (verified to 1e-9 across all variables). The
objective is unbounded along a semantically inert direction, and an
optimizer given unnormalized outputs rides it: the gradient along the
gauge direction is not a responsibility. Per-block log-softmax removes
the direction identically — the gradient-descent analogue of the
M-step's simplex projection. (Contrast genuine likelihood
singularities, e.g. Gaussian variance collapse [Bishop2006], where the
runaway direction changes the model; here it is pure gauge, so
normalization fixes it exactly rather than regularizing it.)

**(b) No clamping: posterior collapse.** With no hard evidence
anywhere, a constant detector that always reports the a-priori likely
value achieves strictly higher likelihood than a confident truthful
detector: the objective is satisfiable without consulting the input,
the analogue of posterior collapse in latent-variable models
[BowmanEtAl2016]. Adding a single hard-observed channel that the
constraint structure links to the soft channel reverses the ordering —
truth then outperforms collapse, and a detector trained in this
grounded setting learns the correct input-output direction with no
labels for the soft observable (`tests/test_observation.py`). This is
the constrained regime's clamping term ("the label declares
responsibility 1" [Oursland2025]), implemented literally as
annihilation of competing values.
