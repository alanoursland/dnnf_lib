# Tutorial plan: *Compiled Reasoning with neximode*

**Audience**: a CS undergraduate who knows Python, basic propositional
logic, and intro probability. No SAT, no graphical models, no ML
assumed. **Goal**: by the end, the reader can model *their own* system,
compile it, query it, track it over time, and learn its parameters —
and knows how to debug the two things that actually go wrong (circuit
blowup and model misspecification).

**Format**: one markdown file per chapter in `tutorial/`, numbered
`00_…` to `12_…`. Every chapter has: *Learning goals* (3–5 bullets), a
*runnable* code block sequence (copy-paste works, total runtime < 10 s
per chapter, CPU only), 2–3 *exercises* (solutions in
`tutorial/solutions/`), and a *checkpoint* — one question testing the
chapter's key idea. Every claim about the library links to the test
file that proves it.

**Running example**: a **smart greenhouse**, grown incrementally across
chapters. Components: water pump (`ok/weak/dead`), drip valve
(`ok/stuck_open/stuck_closed`), grow light (`ok/burnt_out`), moisture
sensor (noisy), light sensor (noisy), tank level (quantized). It is
relatable, small enough to brute-force check in early chapters, rich
enough for tracking/planning/learning later, and deliberately *not* one
of the repo's existing examples, so readers see a model built from
zero. (Repo examples become "further reading" at chapter ends.)

---

## Chapter sequence

### 00 — What this tutorial builds (and setup)
The pitch in one page: you will build a program that *knows* your
system — it can count its configurations, diagnose it from sensors,
watch it over time, plan commands, and learn its failure rates — all
from one compiled object. Install (`pip install -e .[torch,dev]`),
verify with `pytest -q`, run one 5-line teaser (greenhouse diagnosis).
*Image 00a*: the lifecycle ring — model → compile → {count, diagnose,
track, plan, learn} → better model.

### 01 — Worlds: logic as possibility
Propositions and finite-domain variables; constraints as "which worlds
survive"; a model/world = one complete assignment. Brute-force
enumeration in 10 lines of Python (itertools.product) over a 3-variable
greenhouse fragment — the reader *sees* the world table. Model counting
as the most basic question.
*Image 01a*: the world table for 3 variables with surviving rows
highlighted. *Image 01b*: the 2^n cliff — worlds vs variable count,
log scale, with "your laptop dies here" annotation.

### 02 — The compile trick: circuits instead of tables
Why enumerate every time when the *structure* of the answer is fixed?
NNF circuits: leaves are `var=value`, AND/OR gates above. The two
properties, taught as pictures not definitions: **decomposability**
(AND children talk about different variables — so their counts
*multiply*) and **determinism** (OR children can't both happen — so
their counts *add*). Multiply-and-add over a DAG = counting without
enumerating. Smoothness in one paragraph (every branch must mention
every variable, padded by "don't care" gadgets).
*Image 02a*: the same tiny formula as (left) an exponential decision
tree and (right) a shared DAG with cache hits circled. *Image 02b*:
decomposability/determinism as two annotated gate close-ups ("disjoint
vars → ×", "exclusive branches → +"). *Image 02c*: a real compiled
greenhouse-fragment circuit, rendered, small enough to trace by hand.

### 03 — First contact: model, compile, count
`SystemModel`: `bool`, `finite`, `add`, formula operators (`&`, `|`,
`~`, `>>`, `iff`). Compile; `circuit.stats()`; `fd.model_count`;
`fd.is_satisfiable`. Validate against chapter 1's brute force — the
tutorial's standing discipline: *at small scale, always check the
circuit against enumeration* (exactly what the library's own tests do).
Exercises: add a constraint, predict the count, verify; make the model
unsatisfiable and find the culprit constraint.
*Image 03a*: annotated code-to-circuit diagram — each `add()` call
pointing at the clauses it became.

### 04 — One circuit, many questions: weights and semirings
Put a weight on every `(variable, value)` leaf. Sweep the circuit with
(+, ×) → weighted counts and probabilities; with (min, +) over
−log-weights → most probable world. The semiring table, presented as
"swap the two operators, keep the circuit." Neg-log weights introduced
properly: probabilities multiply, costs add, `min` finds the likeliest.
This chapter is the conceptual heart; take it slow.
*Image 04a*: the same small circuit evaluated three ways side by side
(count / probability / min-cost), values written on every wire.
*Image 04b*: the neg-log number line (p=1 → 0, p=0.5 → 0.69, p→0 → ∞).

### 05 — Diagnosis I: modes, priors, evidence
`mode()` with priors; observables; evidence dicts; `log_evidence`,
`posteriors`, `diagnoses` (ranked by best supporting state) vs
`map_diagnoses` (ranked by summed posterior) — taught with a concrete
2-mode case where the rankings *differ*, so the distinction lands.
Greenhouse: pump vs valve ambiguity when the drip line is dry.
*Image 05a*: greenhouse schematic with modes and sensors labeled.
*Image 05b*: posterior bar charts before/after one observation.
*Image 05c*: MPE-vs-MAP: two mode assignments, one with a taller single
bar, one with more total area.

### 06 — Real sensors lie: noise and soft evidence
`sensor(expr, false_positive, false_negative)`; why deterministic
sensors "snap" posteriors 0→1 and why that's wrong; virtual evidence
(likelihood vectors) for detector confidences. Quantized continuous
variables (`quantized`, threshold atoms, numeric evidence bucketing).
*Image 06a*: posterior-vs-days plot, hard sensor (step) vs noisy sensor
(gradual conviction) — regenerate the home-battery day-4 contrast.
*Image 06b*: hard vs soft evidence as weight-vector diagrams (0/1 mask
vs 0.8/0.2 tilt).

### 07 — Time: tracking a changing system
`ModeTracker`: transition matrices (absorbing faults via zero rows),
per-step command-conditioned overrides, `transition_fn` for correlated
dynamics, `prev()` for hard joint constraints. Beam semantics stated
honestly: exact when the beam covers the mode space, best-first
approximation otherwise. Greenhouse: pump wearing out over a simulated
month.
*Image 07a*: HMM trellis with the circuit as the per-step observation
model. *Image 07b*: belief timeline (stacked area of P(pump mode) over
30 days with a fault injected on day 12).

### 08 — Explanations: ranked worlds and what to check next
`enumerate_models` (most-probable-first, lazily), minimum-cardinality
diagnoses ("all single faults first"), and `value_of_information`
("which sensor reading would help most"). Frame as the troubleshooting
loop: diagnose → VOI → observe → repeat.
*Image 08a*: the troubleshooting loop diagram. *Image 08b*: entropy
bars before/after each candidate observation (VOI visualized).

### 09 — Planning: the same circuit, run backwards
`Planner`: modes, commands, transitions with costs, observables and
behaviors; `compile(horizon)`; `estimate()` vs `plan()` vs
plan-from-observations — all one artifact, different clamps. The
MEXEC siderostat as a worked historical example (with the story: this
is how spacecraft did it).
*Image 09a*: the unrolled time-slice circuit (3 slices, mode/command/
observable lanes). *Image 09b*: three copies of the same circuit with
different leaves clamped, labeled "estimate", "plan", "plan from
sensors".

### 10 — Learning: closing the loop with data
`sample_state` (simulate telemetry), `fit_priors` (EM: watch the
log-likelihood climb, recover a known failure rate from alarm-only
data — the reader reruns the library's own headline experiment),
identifiability in one honest page (when two rates can't be told
apart). Optional advanced section: `fit_priors_torch` and
gradients-as-marginals, with pointers to `tests/test_implicit_em.py`
and the lab report for the theory-curious.
*Image 10a*: the EM loop diagram (E: posteriors, M: average, repeat).
*Image 10b*: log-likelihood trace + recovered-prior-vs-truth plot.

### 11 — Under the hood (optional but recommended)
How compilation actually works, on a 4-variable example traced by hand:
unit propagation, component splitting (why AND nodes are decomposable),
d-way branching (why OR nodes are deterministic), the cache (why it's a
DAG). Then the practical payoffs: reading `stats()`, what makes
circuits blow up (dense coupling, modes_first on gate-like models —
show the adder numbers from `docs/SCALE.md`), heuristic options, and
the small-scale-brute-force debugging discipline.
*Image 11a*: full compilation trace of the 4-variable example as a
storyboard (5 panels: input clauses → BCP → split → branch → cache
hit). *Image 11b*: the SCALE.md chart — nodes vs size for adders and
process lines, linear fits, with the modes_first blowup inset.

### 12 — Your project: a workflow that won't betray you
The checklist, distilled from this repo's own history: (1) model the
smallest honest version; (2) brute-force check counts/posteriors at toy
size; (3) grow, watching `stats()`; (4) add noise before trusting
evidence; (5) simulate before learning; (6) learn only identifiable
things; (7) serialize (`save`/`load`) for deployment. Pitfall gallery
with error messages the reader will actually see. Project idea menu
(dorm-room hydroponics, bike drivetrain diagnosis, CI-pipeline health,
D&D encounter validator) sized to the library's measured envelope
(~200 components). Where everything lives: docs map, examples map,
which test file demonstrates which feature.
*Image 12a*: the workflow checklist as a one-page flowchart (printable).

### Appendices
- **A — API cheat sheet**: one table per layer (model / query / track /
  plan / learn / persist), signature + one-liner + test-file link.
- **B — Glossary**: world/model, NNF, decomposability, determinism,
  smoothness, semiring, WMC, MPE, marginal MAP, prior/posterior,
  virtual evidence, beam, horizon.
- **C — Theory further reading**: Darwiche 2001/2003, Darwiche &
  Marquis 2002/2004, Chavira & Darwiche 2008, Barrett 2005, the
  knowledge compilation map — one sentence each on *why* to read it.

---

## Image plan (17 images)

All images generated by scripts in `tutorial/images/` (matplotlib +
networkx/graphviz; no hand-drawn assets, so they regenerate when the
library changes). One `generate_all.py` entry point; images committed
as PNG (and the two flowcharts as SVG). Style: consistent palette,
large fonts (readable at textbook width), no chartjunk.

| # | File | Content | Generator notes |
|---|---|---|---|
| 00a | `lifecycle.svg` | model→compile→query ring | graphviz circo |
| 01a | `world_table.png` | 8-row assignment table, survivors highlighted | matplotlib table |
| 01b | `explosion.png` | 2^n curve, log scale, annotations | matplotlib |
| 02a | `tree_vs_dag.png` | decision tree vs cached DAG, same formula | networkx, two panels |
| 02b | `two_properties.png` | annotated AND/OR gate close-ups | matplotlib patches |
| 02c | `first_circuit.png` | real compiled fragment (≤25 nodes) | render FDCircuit via graphviz (small helper to write DOT — *needs a `circuit_to_dot` utility, add to `neximode/viz.py`*) |
| 03a | `code_to_clauses.png` | add() calls → clauses mapping | matplotlib annotations |
| 04a | `three_sweeps.png` | one circuit, three semiring evaluations, wire values | graphviz ×3 panels |
| 04b | `neglog_line.png` | −log p number line | matplotlib |
| 05a | `greenhouse.svg` | component/sensor schematic | graphviz or hand-layout matplotlib |
| 05b | `posterior_bars.png` | prior vs posterior bars | matplotlib |
| 05c | `mpe_vs_map.png` | tall-bar vs big-area comparison | matplotlib |
| 06a | `snap_vs_gradual.png` | hard vs noisy sensor conviction over days | run tracker twice |
| 06b | `evidence_vectors.png` | mask vs tilt weight vectors | matplotlib |
| 07a | `trellis.png` | tracking trellis | matplotlib |
| 07b | `belief_timeline.png` | 30-day stacked-area belief | run tracker |
| 08a/b, 09a/b, 10a/b, 11a/b, 12a | as described above | mixed |

New library utility required: **`neximode/viz.py`** with
`circuit_to_dot(circuit, labels=...)` (used by images 02c, 04a, 09a and
generally useful — should ship with tests like everything else).

## Production order and effort

1. `neximode/viz.py` + image generator skeleton (half day).
2. Chapters 00–04 (the theory core; most images; ~2 sessions).
3. Chapters 05–09 (library tour on the greenhouse; ~2 sessions).
4. Chapters 10–12 + appendices (~1 session).
5. A CI job that executes every tutorial code block (doctest-style
   extraction) so the tutorial can never rot — same discipline as the
   rest of the repo.

## Review criteria (what "done" means per chapter)

- A reader who has only read *prior* chapters can run every block.
- Every new API name appears first in prose, then in code.
- Exercises are doable with only that chapter's content; solutions run.
- The checkpoint question has a one-sentence answer stated somewhere
  in the chapter.
- No claim without either a runnable demonstration or a link to the
  test that proves it.
