# Proposal 2: Compiled model-based diagnosis, made differentiable

**Form**: full paper. **Venue**: DX (International Workshop on
Principles of Diagnosis) — the historical home of this lineage; a
journal version (e.g., AIJ or JAIR) becomes plausible with one real
hardware case study added.

## Thesis

The DNNF-compiled diagnosis architecture of the early-2000s NASA/JPL
systems — compile the system model offline into a small circuit, answer
observation queries online in time linear in circuit size, read leaf
weights as negative log probabilities — is revisited with capabilities
that did not exist then. The neg-log reading, made rigorous as semiring
evaluation, extends all the way to gradient-based learning: the same
compiled artifact that ranks diagnoses also *simulates* the system,
*tracks* it through time, *learns its failure priors from telemetry*,
and *trains neural sensor front-ends without observable labels*. The
architecture's twenty-year-old offline/online split turns out to be
exactly the CPU/GPU split of modern differentiable computing.

## Contributions and existing evidence

1. **A native finite-domain decision-DNNF compiler and query engine.**
   Leaves are atomic assignments (`valve=stuck_closed`), decisions
   branch d-ways; ~40% smaller circuits than boolean one-hot lowering
   on mode-rich models. All queries are semiring sweeps; ordered
   enumeration and sampling reuse one lazy k-best machinery.
   *Evidence*: `neximode/fd.py`; brute-force cross-validation in
   `tests/test_fd.py`; size comparison in `bench/run.py` history.
2. **Exact marginal MAP over modes via constrained compilation, with an
   empirical structure-sensitivity result.** Modes-first orders make
   summed-posterior diagnosis ranking exact. Measured: the constraint
   is *free* on chain-structured monitoring models (identical circuit
   sizes at every tested size) and *exponentially catastrophic* on
   gate-level networks (adders: intractable by 4 bits) — a crisp
   empirical face for the NP^PP-hardness of marginal MAP, with
   practical guidance attached.
   *Evidence*: `docs/SCALE.md` finding 3; `bench/scale.py`;
   `tests/test_map.py` (brute-force validation incl. a case where MAP
   and MPE rankings provably differ).
3. **Scale characterization.** Circuit size scales linearly on
   structured models (32-bit adder with 160 stuck-at fault modes: 5.8k
   nodes; 200-stage process line, 400 mode variables: 9.2k nodes);
   diagnosis quality holds at every size (single-fault pinpointing;
   sensor-exoneration reasoning). Compile-time constants, not
   representation, are the wall.
   *Evidence*: `docs/SCALE.md`, `bench/scale.py`.
4. **Temporal mode tracking.** Beam filtering over joint mode
   assignments; exact HMM filtering when the beam covers the space;
   command-conditioned and correlated (previous-joint-state-dependent)
   dynamics, verified against exact forward filters.
   *Evidence*: `neximode/tracking.py`, `tests/test_tracking.py`,
   `tests/test_learning.py`, `tests/test_torch_learn.py`.
5. **Learning on the compiled artifact.** Exact EM and gradient
   training provably share a fixed point (verified to ~1e-4 against
   the analytic MLE); failure priors recovered from alarm-only
   telemetry under partial observability; neural observation models
   trained end-to-end through the circuit with **no labels for the
   observables**, grounded by hard evidence elsewhere in the structure
   — with the two degeneracies of ungrounded/unnormalized training
   documented and reproduced.
   *Evidence*: `neximode/torch_learn.py`, `tests/test_learning.py`,
   `tests/test_observation.py`, `tests/test_implicit_em.py`; theory
   connection via Proposal 1.
6. **GPU evaluation.** Layered batched tensor lowering; one backward
   pass yields all posterior marginals (differential semantics);
   batched MPE. *Evidence*: `neximode/torch_backend.py`,
   `tests/test_torch.py`.

## Narrative frame

The hook is architectural continuity, with attribution per the record:
the neg-log-probability reading of compiled diagnosis weights is due to
Darwiche & Marquis (weighted bases, 2002) and, in its full
probability-measure form, Darwiche (2003) and Chavira & Darwiche
(2005–08); the deployed compiled-diagnosis systems of that era (MEXEC,
Barrett 2005) evaluated min-sum only. The paper's contribution is where
the probability reading is carried: the diagnosis engine as a
differentiable likelihood, with learned priors and neural observation
models.
Related work: Darwiche's DNNF/differential-semantics line, dsharp/D4
compilers, probabilistic circuits, DeepProbLog/semantic-loss
neurosymbolics, Livingstone-era mode estimation.

## Status of the previously missing pieces

- End-to-end case study: DONE — `drafts/02/case_study.md` (+ .py).
- Honest limitations: DONE — `drafts/02/limitations.md` (11 items).
- Real telemetry: PARTIALLY DONE — `drafts/02/real_data_study.md`
  (UCI hydraulic rig, 2,205 real cycles; calibration measured:
  confident calls perfect, mid-range overconfident 2-3x). Open: real
  sequences/tracking and rate learning on real data.
- External compiler: DONE — `drafts/02/d4_comparison.md` (counts
  match; d4 4-6x faster, up to 4.3x smaller circuits; conclusion:
  reference implementation + interop, not a compiler race).
