# Paper proposals

Three publication candidates arising from this project, in descending
order of confidence and ascending order of effort. Each proposal file
maps its claims onto evidence that already exists in the repository
(tests, benchmarks, docs), so the writing task is mostly narration.

| # | Proposal | Venue | Effort | Status of evidence |
|---|---|---|---|---|
| 1 | [Standalone supporting paper: *Exact Responsibilities* — compiled circuits verifying implicit EM](01-implicit-em-companion.md) | arXiv, then TPM/UAI-workshop or KR audience | days–week | complete (`tests/test_implicit_em.py`) |
| 2 | [The compiled-diagnosis architecture revisited, differentiable](02-dx-differentiable-diagnosis.md) | DX (International Workshop on Principles of Diagnosis); journal later | weeks | complete at workshop scale; one real case study short of journal scale |
| 3 | [dnnf_lib software paper](03-joss-library.md) | JOSS | days | complete; needs packaging polish |

Suggested sequencing: write #1 now as a standalone paper supporting
(and citing) *Gradient Descent as Implicit EM* — its experiments exist
and are in CI, so it is mostly narration; submit #3 whenever, so #1 and
#2 can cite the library by DOI; write #2 when ready to tell the full
twenty-year arc — it is the flagship narrative.

The proposals cross-reference: #1 gives #2 its learning-theory
grounding; #2 gives #1 an applied instantiation; #3 makes both
reproducible with a citable artifact.
