# Lab report: Implicit EM in compiled circuits

**Date**: 2026-07-21 · **System**: dnnf_lib (branch
`claude/dnnf-logic-library-3nb814`) · **Test module**:
`tests/test_implicit_em.py` (all experiments run in CI) · **Reference**:
Oursland, *Gradient Descent as Implicit EM*
(github.com/alanoursland/gradient_descent_as_implicit_em)

## Objective

Verify the reference paper's central identity — for log-sum-exp
objectives over distances, ∂L/∂d_i = −r_i (the gradient with respect to
each distance is the negative posterior responsibility; Fisher's
identity) — in a model class where it can be checked against ground
truth: smooth deterministic DNNF circuits, whose log-space evaluation is
a DAG of LSE nodes over neg-log-weight distances. Additionally, verify
the EM/SGD fixed-point equivalence the identity implies, and confirm
that violating the paper's structural conditions (normalization;
clamped observation) produces the predicted degeneracies.

## Apparatus

- Compiled finite-domain d-DNNF circuits (`dnnf.fd`); exact reference
  posteriors by weighted-model-count ratios (`CompiledSystem.posteriors`),
  themselves validated elsewhere against exhaustive enumeration.
- Differentiable evaluator (`dnnf.torch_backend.TorchCircuit`,
  log-prob semiring, float64): gradients via one backward pass.
- Learners: exact EM (`fit_priors`) and Adam SGD with softmax
  reparameterization (`fit_priors_torch`).

## Experiment 1 — Gradients are responsibilities, at depth

**Setup.** A two-level diagnosis model (modes `v1`∈{ok,bad},
`v2`∈{ok,weak,bad}; intermediate flows `f1`,`f2`; noisy `alarm` sensor
with fp=fn=0.05, adding hidden glitch variables). Compiled: 44 nodes,
7 OR (LSE) nodes, 15 FD variables — genuinely nested mixtures, not a
single LSE. Evidence: `alarm=True`.

**Method.** Compute ∂logWMC/∂(leaf log-weight) for every (variable,
value) leaf by autograd; independently compute every P(var=value |
alarm) by WMC ratios; compare.

**Results.**

- Max |gradient − responsibility| over all leaves: **2.22e-16**
  (double-precision machine epsilon — exact to representation).
- Per-variable gradients sum to 1 (simplex) to 1e-9; the observed
  variable carries responsibility 1 on its observed value.
- Illustrative posteriors recovered by the backward pass:
  P(v1|alarm) = (ok 0.4737, bad 0.5263);
  P(v2|alarm) = (ok 0.6205, weak 0.1163, bad 0.2632).

**Note.** For a variable *inside* the evidence, the naive
"override-and-ratio" computation yields a likelihood ratio, not a
conditional; the gradient correctly reports responsibility 1. This
distinction surfaced during test construction and is now documented.

## Experiment 2 — EM and SGD share one fixed point

**Setup.** Single mode `v`∈{ok,bad}, true P(bad)=0.3, noisy alarm
(fp=fn=0.1). 2500 observations of the alarm **only**, sampled from the
true model (`sample_state`). Both learners start from the wrong prior
P(bad)=0.5. Identifiability: P(alarm)=0.8·p+0.1 is invertible in p.

**Results** (observed alarm frequency 0.3532 ⇒ analytic MLE
p̂ = (0.3532−0.1)/0.8 = 0.316500):

| quantity | value |
|---|---|
| analytic MLE | 0.316500 |
| EM estimate (17 iterations to tol 1e-13) | 0.316500 |
| SGD estimate (Adam, 500 epochs, lr 0.1) | 0.316500 |
| average posterior responsibility r̄ at EM convergence | 0.316500 |
| final avg log-lik, EM / SGD | −0.649405 / −0.649405 |

All four quantities coincide to six decimals; the stationarity
condition ∇θ log P = r̄ − π = 0 is confirmed directly (prior equals
average responsibility). EM's M-step (simplex projection) and SGD's
softmax reparameterization are the same fixed-point equation iterated
hard versus softly.

## Experiment 3 — Predicted degeneracies under condition violation

Both failure modes below were first encountered *during development of
the observation-model trainer, before the reference paper was read*,
and only later recognized as its structural conditions — they are
observations, not constructions.

**3a. No normalization → unbounded gauge direction.** Scaling a soft
(virtual) evidence vector uniformly by c shifts log-evidence by exactly
log c (verified: +log 10 to 1e-9) while leaving **every posterior
unchanged** (verified to 1e-9 across all variables). The objective is
unbounded along a semantically inert direction; the gradient along it
is not a responsibility. Per-block log-softmax removes the direction
exactly — the SGD analogue of the M-step's simplex projection.

**3b. No clamping → posterior collapse.** With no hard evidence
anywhere, a constant detector always reporting the a-priori likely
value achieves strictly higher likelihood than a confident truthful
detector (the objective is satisfiable without using the input). Adding
one hard-observed channel (`alarm`) that the structure links to the
soft channel reverses the ordering: truth outperforms collapse. This is
the paper's "the label declares responsibility 1" clamping term,
implemented literally as −inf annihilation of competing values.

## Conclusions

1. The reference identity holds exactly (machine precision) in a
   nontrivially composed LSE DAG; composition is handled by the chain
   rule composing local softmax responsibilities — equivalently,
   Darwiche's differential semantics of d-DNNF is Fisher's identity
   applied recursively, a bridge between the implicit-EM thesis and the
   knowledge-compilation literature.
2. The EM/SGD equivalence implied by the identity is quantitative, not
   qualitative: same fixed point, same likelihood, six-decimal
   agreement with the analytic MLE from partially observed data.
3. The structural conditions are load-bearing in practice: each
   violation produced its predicted degeneracy in an independent
   development setting, and restoring the exact named condition cured
   it. As design guidance: normalize at every choice point; ground
   every soft channel with at least one clamped observation.

## Reproduction

```bash
pip install -e .[torch,dev]
python -m pytest tests/test_implicit_em.py -v
```

Numbers in this report were produced with seed 5 (experiment 2) on
CPU float64; experiment 1 is deterministic.
