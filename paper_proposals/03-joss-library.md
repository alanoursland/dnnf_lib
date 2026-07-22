# Proposal 3: neximode software paper (JOSS)

**Form**: short software paper (`paper.md` + metadata per JOSS format).
**Venue**: Journal of Open Source Software. **Purpose**: peer-reviewed,
DOI-citable artifact so Proposals 1 and 2 can cite the software
properly; JOSS review doubles as an external code/API review — the
first non-author eyes on the DSL, which the project has lacked.

## Statement of need (draft)

Existing open-source tooling covers fragments of the compiled-reasoning
stack: d-DNNF compilers without query/learning layers (c2d, dsharp,
D4), SDD/BDD packages without diagnosis semantics (PySDD, dd),
probabilistic-circuit learners without logical modeling front ends
(SPFlow, PyJuice). neximode integrates the full lifecycle behind one
representation: a finite-domain modeling DSL (modes with priors, noisy
sensors, quantized continuous ranges), a native multi-valued
decision-DNNF compiler, exact queries (WMC, MPE, marginal MAP, ordered
enumeration, sampling), temporal mode tracking, prior learning (EM and
SGD), neural observation models trained through the circuit, and a
batched differentiable PyTorch evaluator — with every inference path
cross-validated against brute-force enumeration (270+ tests).

## Submission checklist

- [ ] `paper.md` (~1000 words: summary, statement of need, key
      design points, acknowledgement of the JPL lineage) + `paper.bib`
- [ ] Version tag + archived release (Zenodo DOI)
- [ ] `CONTRIBUTING.md`, `LICENSE` file (pyproject declares MIT; the
      file itself must exist), issue templates
- [ ] API docs generated from docstrings (docstrings are already
      substantial; a small Sphinx/mkdocs config suffices)
- [ ] CI badge in README (workflow exists)
- [ ] Community guidelines + at least one tagged "good first issue"

Estimated effort: 1–2 days of packaging; the substance exists.

## Review risks

- Single-author-plus-agent development history: fine for JOSS, but the
  reviewers will exercise the API fresh — which is precisely the
  ergonomics validation the project needs (see
  `docs/MODELING_NOTES.md` on the circularity of self-validation).
- Scope question ("is this research software with a clear need?") —
  addressed by the statement of need above and the docs/DESIGN.md
  theory grounding.
