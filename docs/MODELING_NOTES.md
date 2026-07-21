# Modeling ergonomics: findings from a realistic system

The question behind `examples/home_battery.py`: **would a domain engineer
tolerate this DSL?** The modeling experience — not the inference — is the
real product risk for any diagnosis/monitoring application built on this
library. Findings from modeling a 5-component home solar+battery system
(and, earlier, valve networks and a serial diagnosis chain):

## What worked well

- **Modes with priors are one line each**, and exactly-one constraints are
  automatic. The component declarations read like a spec sheet.
- **Expectations as `iff(sensor, condition)` are natural** for anyone who
  can write a truth table. The five "physics" constraints of the battery
  system were each written on the first attempt and read back correctly.
- **Partial observations just work.** Scenario 3 omits two sensors; nothing
  special is needed — unobserved variables are summed over exactly.
- **Ambiguity is surfaced honestly.** "PV fine but not charging" splits
  posterior ~48/45 between dead battery and stuck controller — the model
  *knows* it can't tell, which is exactly what a technician needs to see
  (and what threshold-alarm systems structurally cannot express).
- **Compile time is a non-issue at this scale** (127 nodes, milliseconds),
  and the compiled artifact answers every query type without recompiling.

## Friction found (ordered by pain)

1. **Deterministic sensors make posteriors snap.** With exact expectations,
   one bad reading convicts a component outright (P jumps 0→1). Real
   sensors glitch. Hand-rolling noise (hidden XOR'd glitch variables)
   worked but was boilerplate nobody should write twice — so this
   experiment directly produced the `SystemModel.sensor(expr,
   false_positive=, false_negative=)` primitive, which fixed the
   day-4 posterior snap in the tracking demo into a gradual, believable
   accumulation of suspicion. *Lesson: the DSL needs probabilistic
   primitives, not just logical ones.*
2. **No quantitative abstraction.** "Degraded PV produces *some* current"
   forced a coarse boolean abstraction where `degraded` and `ok` are
   indistinguishable to the `pv_current` sensor. Real modeling wants
   ordered domains and threshold atoms (`level >= low`), compiled to
   one-hot ranges. This is the next DSL gap after noise.
3. **Mixed operator/function syntax.** `&`, `|`, `~`, `>>` are operators
   but equivalence is `iff(a, b)` — the seam shows. Minor, but a domain
   engineer would notice.
4. **No causal/directional sugar.** Everything is a constraint; there is
   no `when(mode == "ok").expect(sensor)` idiom. The undirected semantics
   are *right* (evidence flows backward through `iff` automatically), but
   a directional surface syntax would read closer to how engineers think
   about component behavior.
5. **Transition matrices are verbose** — nested dicts with every
   row spelled out. A `persist(0.995, decay={"weak": 0.004, ...})`
   helper would collapse the common "mostly stay, small decay" pattern.

## Implications

The inference core is done enough to stop being the bottleneck; the
leverage is now all in the modeling layer. Priority order suggested by
this experiment: quantitative/ordered domains, transition-matrix sugar,
directional behavior syntax. Each should be driven the same way this
round was: model a real system, feel the friction, fix the sharpest edge.
