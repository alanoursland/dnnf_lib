# Contributing

- Run `python -m pytest` before submitting; every inference path is
  cross-validated against brute force or closed forms, and changes to
  the compiler or evaluators must keep those checks passing unchanged.
- New queries should come with a brute-force reference test on small
  randomized instances (see `tests/helpers.py` / `tests/test_fd.py`
  for the pattern).
- Benchmark-affecting changes: run `python bench/run.py` and
  `python bench/scale.py` and report node counts (they must not change
  for pure-performance work) and timings in the PR.
- Design context lives in `docs/DESIGN.md`; exactness, invariant, and
  complexity claims live in `CONTRACTS.md`; roadmap in
  `docs/FUTURE_WORK.md`.
