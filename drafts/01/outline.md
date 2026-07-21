# Exact Responsibilities: Verifying Implicit EM in Compiled Probabilistic Circuits

Target: ~6 pages. Role: standalone companion supporting *Gradient
Descent as Implicit EM* [Oursland2025] with an exactly solvable
instantiation. All experiments live in `dnnf_lib` CI
(`tests/test_implicit_em.py`); numbers from
`reports/implicit_em_lab_report.md`.

## Sections (one file per section under md/)

- **00_abstract** — thesis in 150 words; the 2.2e-16 number.
- **01_introduction** — parent claim; the just-so risk; what an exact
  testbed contributes; the rediscovery framing; contribution list.
- **02_background** — d-DNNF, weighted model counting as semiring
  evaluation, neg-log weights as distances, differential semantics.
- **03_correspondence** — the identity; composition through the DAG;
  Darwiche's differential semantics = Fisher's identity recursively;
  EM/SGD stationarity r̄ = π under softmax parameterization.
- **04_experiments** — E1 gradient=responsibility at depth (machine
  epsilon); E2 shared fixed point (six-decimal table); E3 condition
  violations (gauge explosion; posterior collapse). Reproduction.
- **05_regimes** — the paper's three regimes as literal system
  operations (table).
- **06_discussion** — what transfers to neural nets (design
  checklist); relation to circuit-flows EM; limitations and scope
  fence (explicit vs interpretive structure).
- **07_conclusion** — 2 paragraphs.
- **references** — bibliography, [AuthorYear] keys used across
  sections.

## Style notes

- First person plural, plain declarative sentences.
- Every empirical claim carries its number and its test-file pointer.
- The scope fence is explicit: circuits prove the mechanism; neural
  networks remain the interpretive case argued in [Oursland2025].
