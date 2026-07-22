# Chapter 12 — Your project: a workflow that won't betray you

You know the theory (worlds → circuits → semirings) and the practice
(model → diagnose → track → plan → learn). This chapter is the
distilled *process* — every rule below was learned the hard way during
this library's own development.

## The workflow

1. **Model the smallest honest version.** One component, one sensor,
   one constraint. Resist completeness; you're validating the *shape*.
2. **Brute-force check it** (chapter 3). `assert fd.model_count(...) ==
   your_enumeration`. Also check one posterior by hand. Keep these as
   tests forever.
3. **Grow in steps, watching `stats()`.** Linear growth per component
   added = healthy. A bend in the curve = the last constraint you
   added is coupling things (chapter 11).
4. **Add noise before trusting evidence** (chapter 6). Every physical
   sensor gets `sensor(...)` rates — guessed is fine, zero is a lie
   that will manifest as a probability-0 crash on real data.
5. **Simulate before you learn** (chapter 10). Plant a rate with
   `sample_state`, recover it with `fit_priors`. If the pipeline can't
   recover planted truth, real telemetry won't save it.
6. **Learn only what's identifiable.** Multi-start; scatter = stop.
7. **Serialize for deployment.** `system.save(path)` /
   `CompiledSystem.load(path)` — compile on your laptop, ship the
   artifact; the loader needs one allocation pass (the header states
   all sizes up front — a habit this architecture inherited from
   flight software).

## Pitfall gallery (symptoms → causes)

| symptom | likely cause | chapter |
|---|---|---|
| posterior ≠ prior with no evidence | unweighted observable counting per value | 3 |
| P jumps 0→1 on one reading, later queries raise on "impossible" evidence | exact sensor that should be noisy | 6 |
| `ValueError: evidence inconsistent with all tracked trajectories` | beam too narrow, or a zero in the transition matrix that reality disagrees with | 7 |
| compile time explodes when you enable exact MAP | `modes_first` on a densely-coupled model — use MPE ranking | 11 |
| learned rate is confidently wrong | unidentifiable pair, or a wrong fixed rate absorbing the signal | 10 |
| `ValueError: ... not a quantization boundary` | threshold atom not aligned to your bucket edges | 6 |
| two mode assignments tie oddly / rankings differ between queries | MPE vs MAP semantics — know which you asked for | 5 |
| your test suite passed but the bug shipped | you piped pytest through `tail` and lost the exit code — `set -o pipefail` (yes, this repo did it) | — |

## Project menu (sized to the measured envelope)

Anything up to ~200 components / ~10 sensors per component is
comfortable (chapter 11 numbers). Ideas that have the right shape:

- **Dorm hydroponics / aquarium**: pumps, heaters, sensors — the
  greenhouse with your own hardware and *real telemetry to learn from*.
- **Bike drivetrain diagnosis**: modes for chain/cassette/derailleur
  wear, "sensors" are your own observations (skipping? noise?); VOI
  tells you what to inspect next. No electronics required.
- **CI pipeline health**: stages as components, flaky-test modes,
  build outcomes as sensors; tracking over commits; `prev()` for
  "a stage can't self-heal without a fix commit."
- **Board-game / puzzle validators**: pure constraint counting —
  chapter 3-4 machinery only, good for a weekend.
- **Home network diagnosis**: router/AP/ISP modes, ping/DNS checks as
  noisy sensors, min-cardinality for the "what do I reboot" ranking.

Pick one where *you* can generate evidence. The loop of chapter 8 —
diagnose, check, observe, repeat — is only fun with real observations.

## Where everything lives

| need | where |
|---|---|
| API by example | `examples/` (valves, home battery, car, tanks) |
| every feature's contract | `tests/` — each feature has a test named after it |
| theory & design rationale | `docs/DESIGN.md` |
| measured performance | `docs/SCALE.md` |
| modeling lessons learned | `docs/MODELING_NOTES.md` |
| this course's figures, regenerable | `tutorial/images/generate_*.py` |

Go build something. And when your model catches its first real fault —
or teaches you your mental model of your own system was wrong, which
is better — that's the whole point of exact reasoning: *the numbers
mean something, so being surprised by them is information.*

[Appendices: API cheat sheet, glossary, further reading](appendices.md)
