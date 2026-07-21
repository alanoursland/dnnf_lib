# Proposal 1: Standalone paper — "Exact Responsibilities: Compiled
# Circuits as a Testbed for Gradient Descent as Implicit EM"

**Form**: a new, self-contained paper whose role is to *support*
*Gradient Descent as Implicit EM* (Oursland) with an exactly solvable
instantiation — cited as its empirical/structural companion, not merged
into it. **Venue**: arXiv first; natural workshop targets are TPM
(Tractable Probabilistic Modeling, UAI) or an ICML/NeurIPS workshop on
theory of learning; the knowledge-compilation framing also fits KR/KoCoa
audiences.

## Working title options

- *Exact Responsibilities: Verifying Implicit EM in Compiled
  Probabilistic Circuits*
- *Where Gradients Are Provably Responsibilities: d-DNNF Circuits as an
  Implicit-EM Laboratory*

## Thesis (one paragraph)

The parent paper argues that LSE-over-distances objectives make
gradient descent perform implicit EM: dL/dd_i = −r_i, with three
structural conditions (exponentiated distances, normalization,
gradient updates) and three regimes (unsupervised, conditional,
constrained). In neural networks that structure is interpretive. This
paper exhibits a model class where it is **explicit and exactly
solvable** — smooth deterministic DNNF circuits, whose log-space
evaluation is a DAG of LSE nodes over neg-log-weight distances — and
verifies every element of the thesis against ground truth computable by
enumeration: gradients equal exact posterior responsibilities at
arbitrary composition depth; EM and SGD provably share the stationarity
condition r̄ = π and empirically land on the same MLE; and deliberately
violating each structural condition produces precisely the degeneracy
the theorem predicts (unbounded gauge direction without normalization;
posterior collapse without clamping). Because the degeneracies were
first encountered *independently during system development* and only
later recognized as the theorem's conditions, they function as
out-of-sample confirmations rather than constructed illustrations.

## Why a standalone paper works (and what it adds beyond the parent)

1. **Depth.** The parent's identity is usually stated for one LSE. The
   circuit case exercises *composition*: local softmax responsibilities
   chain through an arbitrary DAG into exact global posteriors — and
   this is exactly Darwiche's differential semantics of d-DNNF (2003),
   giving the implicit-EM thesis a previously unremarked anchor in the
   knowledge-compilation literature. That bridge (Fisher's identity ↔
   circuit differential semantics) is itself a contribution.
2. **Ground truth.** Every responsibility is checkable by exhaustive
   enumeration; every fixed point by analytic MLE. No appeal to
   interpretation of learned representations is needed.
3. **The regimes appear as APIs, not analogies.** Unsupervised =
   free-competition prior learning; constrained = hard-evidence
   annihilation (the "label declares responsibility 1" operation,
   implemented literally as a −inf mask); conditional = internal
   (non-root) LSE nodes with responsibility-weighted gradients.
4. **Failure modes as evidence.** Both predicted degeneracies were hit
   in practice before the theorem was known to the implementers, then
   cured by restoring exactly the named conditions (log-softmax = the
   SGD analogue of the M-step's simplex projection; grounding evidence
   = the clamping term). A theorem whose preconditions are
   rediscovered by breaking things is doing explanatory work.

## Claims → existing evidence (all machine-checked, in dnnf_lib)

| Claim | Evidence |
|---|---|
| ∂logWMC/∂log w = exact posterior responsibility, at depth; simplex-normalized per variable; observed vars carry responsibility 1 | `tests/test_implicit_em.py::test_1`; `tests/test_torch.py::test_marginals_match_brute_force` (randomized, vs. enumeration) |
| EM (M-step projection) and SGD (softmax reparam) share the fixed point r̄ = π; both hit the analytic MLE (~1e-4) | `tests/test_implicit_em.py::test_2`; `tests/test_torch_learn.py` |
| No normalization → unbounded pure-gauge direction, posteriors invariant | `tests/test_implicit_em.py::test_3a` |
| No clamping → input-ignoring detector optimal; one hard channel reverses it | `tests/test_implicit_em.py::test_3b`; `tests/test_observation.py` |
| Neural front-end learns from structure + anchor with no observable labels | `tests/test_observation.py` |

## Outline (target 6–8 pages)

1. Introduction: the parent thesis; the gap between interpretive and
   exact instantiations; contributions.
2. Background: d-DNNF, semiring evaluation, neg-log weights as
   distances; the differential semantics.
3. The correspondence: OR = LSE mixture node, AND = independent factor;
   Theorem 1 composed through a DAG = circuit backward pass;
   responsibility flow = E-step (relation to "circuit flows" in the
   probabilistic-circuits literature).
4. Exact verification: the three experiments with numbers.
5. The three regimes as system operations (table + mask semantics).
6. Predicted degeneracies, observed: gauge explosion and posterior
   collapse — with the development-history framing.
7. Discussion: what transfers back to neural networks (the conditions
   as design checklist: normalize per-choice-point, ground every soft
   channel); limits (circuits are the linear/exactly-marginalizable
   case; deep nets add representation learning the testbed lacks).
8. Reproducibility: the library, the test module, one command.

## What remains to do

- Optional new figure/experiment: identity error vs. circuit depth and
  vs. batch of random weighted circuits (expected: machine precision,
  flat) — half a day with existing machinery.
- Writing. All experiments exist and are in CI.
