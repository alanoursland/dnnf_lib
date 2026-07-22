# Compiled Model-Based Diagnosis, Made Differentiable

Target: short journal/workshop article, ~8 pages + references. The
evidence already exists (`drafts/02/*.md`, `docs/`, `tests/`); every
section points at its artifacts and quotes at most its 2–3 strongest
numbers. Discipline: one idea per section, no capability catalogs —
the library has more features than this paper mentions.

## Sections (one file per section under md/)

- **00_abstract** (~150 words) — architecture revisited; the one
  artifact serving diagnose/track/learn; case-study headline (5 KiB /
  4 ms / day-14 detection / priors recovered); real-data calibration
  finding; d4 validation.
- **01_introduction** (~1 page) — the 2000s compiled-diagnosis
  architecture (offline/online split, neg-log costs) and what it could
  not do: true posteriors, learning. Claim: carrying the probability
  reading (Darwiche & Marquis 2004; Chavira & Darwiche 2008) onto the
  deployed architecture yields a diagnosis engine that is a
  differentiable likelihood — and the old offline/online split is the
  modern CPU/GPU split. Attribution per the record; contribution list
  (4 bullets, matching sections 3–6).
- **02_architecture** (~1 page) — native finite-domain decision-DNNF
  in brief: leaves are `var=value`, semiring queries, evidence as
  weight surgery, noisy sensors as desugared glitch variables. One
  figure: the lifecycle ring. Pointer to library docs for everything
  else.
- **03_queries** (~1.25 pages) — the query family on one circuit:
  ranked MPE, exact marginal MAP via modes-first compilation *with the
  measured structure-sensitivity result* (free on monitoring shapes,
  exponential on gate networks — the paper's sharpest empirical
  claim), min-cardinality, VOI. Table, not prose, for the family.
- **04_time_and_learning** (~1.25 pages) — beam tracking (exact-when-
  covering, verified vs HMM filtering; command-conditioned, correlated,
  and compiled joint transitions in one sentence each); learning: EM
  and SGD share the fixed point (~1e-4 vs analytic MLE); neural
  observation models trained without observable labels, with the two
  degeneracies (gauge, collapse) as *predictions confirmed* — cite
  companion paper [Oursland, implicit EM] rather than re-deriving.
- **05_case_study** (~1.5 pages) — condensed from `case_study.md`:
  the solar+battery lifecycle table, ambiguity + VOI resolution,
  tracked month (detection day 14), fleet EM (multi-start 3e-4).
  One figure: belief timeline.
- **06_real_data_and_baseline** (~1 page) — condensed from
  `real_data_study.md` + `d4_comparison.md`: hydraulic rig accuracy
  and the calibration split (confident bins perfect, mid-range
  overconfident 2–3x — misspecification measured); d4 count agreement
  (independent validation) and size/time gap (conclusion: reference
  implementation + interop). One figure: calibration bars.
- **07_related_work** (~0.5 page) — Darwiche line; compilers
  (c2d/dsharp/D4); Livingstone/MEXEC; probabilistic circuits;
  semantic-loss/DeepProbLog; algebraic model counting. One sentence
  each, positioned not surveyed.
- **08_limitations_and_conclusion** (~0.75 page) — compress
  `limitations.md` to the five that matter for the claims made
  (worst-case structure dependence; marginal-MAP fragility; beam;
  remaining real-data gap: sequences and rates; single-designer
  ergonomics), then 3-sentence conclusion ending on the lifecycle
  closing: model → compile → diagnose → track → learn → better model,
  one artifact throughout.
- **references** — reuse `drafts/01/md/references.md` entries +
  Barrett 2005, Helwig et al. (hydraulic dataset), D4, dsharp, PC
  survey; ~15 entries.

## Figures (3 total, all regenerable)

1. lifecycle ring (`tutorial/images/00a_lifecycle.png`, restyled)
2. tracked-month belief timeline (from `case_study.py`)
3. real-data calibration bars (from `real_data_study.py`)

## Length budget check

0.2 + 1 + 1 + 1.25 + 1.25 + 1.5 + 1 + 0.5 + 0.75 ≈ 8.5 pages before
references. Cut order if over: shrink 02 (point to library docs),
then 07.

## Style notes

- Every number in the paper is produced by a script in the repo;
  footnote the script path on first use.
- Claims about MEXEC/heritage cite Barrett 2005 only for what that
  paper states (min-sum evaluation, offline/online, hypergraph
  min-cut compilation); no institutional anecdotes.
- The scope fence from paper 1 applies here too: neural front-end
  results are demonstrations on synthetic grounding, not deployment
  claims.
