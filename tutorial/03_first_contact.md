# Chapter 3 — First contact: model, compile, count

## Learning goals

- Build the chapter-1 fragment in the library and compile it.
- Read `circuit.stats()` and know which numbers matter.
- Adopt the discipline this repo lives by: **check the circuit against
  brute force while the model is still small.**

## The model, in the library

```python
from dnnf import SystemModel, iff, fd

m = SystemModel()
pump = m.mode("pump", ("ok", "weak", "dead"), priors=(0.90, 0.07, 0.03))
drip = m.bool("drip")
moist = m.bool("moist")

m.add(iff(drip, pump != "dead"))   # constraint 1: drip ⟺ pump ≠ dead
m.add(moist >> drip)               # constraint 2: moist ⟹ drip

system = m.compile()
print(system.circuit)
```

```
FDCircuit(nodes=12, edges=14, and=3, or=2, lit=7, vars=3)
```

What just happened, line by line:

- `mode(...)` declares a finite-domain variable and marks it as a
  *diagnosable component* with prior fault probabilities. (Priors do
  nothing in this chapter; they become weights in chapter 4.)
- `bool(...)` declares a boolean and returns an **atom** — a formula
  meaning "this variable is true" — so `drip` and `moist` can be
  combined with operators: `&` (and), `|` (or), `~` (not),
  `>>` (implies), and the function `iff(a, b)`.
- `pump != "dead"` is also an atom: "pump's value is in {ok, weak}".
  Finite-domain comparisons compile to *single* leaves, not
  encodings — this library's circuits speak `pump=ok` natively.
- `compile()` runs the whole pipeline (constraint encoding →
  compilation → smoothing) and returns a `CompiledSystem`, which owns
  the circuit plus your variable names.

(Sometimes `vars` exceeds what you declared: complex nested formulas
make the encoder introduce weightless internal helper variables. This
model's constraints were simple enough to avoid any — chapter 11 shows
when they appear.)

## Counting, and checking

```python
print(fd.model_count(system.circuit))     # -> 5

from itertools import product
survivors = [
    (p, d, mo)
    for p, d, mo in product(("ok", "weak", "dead"), (False, True), (False, True))
    if (d == (p != "dead")) and (not mo or d)
]
assert fd.model_count(system.circuit) == len(survivors)   # 5 == 5
```

Circuit and brute force agree. **Write that assert every time you
model something new** — it costs nothing while the model is small, and
it is precisely how this library's own test suite validates every
feature (hundreds of times, against exhaustive enumeration). When the
two counters disagree, one of them is answering a different question
than you think, and finding out *which* is always worth the hour.

## A subtlety worth meeting early

Ask for the pump's posterior with *no evidence at all*:

```python
print(system.posteriors({}, names=["pump"]))
# {'pump': {'ok': 0.9137, 'weak': 0.0711, 'dead': 0.0152}}
```

They sum to 1 — good — but look closer: `dead` came out at **0.0152,
below its declared prior of 0.03**, and you observed *nothing*. Why?

Count each mode's surviving worlds in the table: `ok` and `weak` each
appear in **two** worlds (soil moist or not), `dead` in only **one**
(dead forces dry). The unweighted observables act like a uniform
distribution over their consistent values, so modes compatible with
more observable-worlds soak up proportionally more mass:
0.03×1 vs 0.90×2 vs 0.07×2 — normalize and you get exactly the numbers
above. The *structure itself* shifted the posterior.

Whether that's right depends on what `moist` means in your model: if
it's a passive reading with no prior of its own, give it one
(chapter 6), or condition on it — once you observe `moist`, this
effect vanishes. The lesson generalizes and is worth memorizing now:

> **Unobserved, unweighted variables are not neutral.** Every value
> they can take counts once. Either observe them, weight them, or
> understand what uniform-over-consistent-values does to your
> posteriors.

## Reading stats

- `nodes` / `edges` — circuit size; every query is linear in this.
  Watch it as your model grows (chapter 11 has the scaling data).
- `and` / `or` / `lit` — structure; a healthy model has ORs near your
  case splits and ANDs where subsystems are independent.
- `vars` — declared + helper variables.

## Exercises

1. Add the grow-light subsystem from chapter 1's exercise (`bulb` mode,
   `light` bool, `iff(light, bulb == "ok")`). Predict `nodes` — will it
   roughly double, or grow by a constant? Why? (Think: is the light
   subsystem connected to the pump subsystem by any constraint?)
2. Break the model: add `m.add(drip & ~drip)`. Compile, and inspect
   `fd.is_satisfiable(system.circuit)` and `system.circuit`. What does
   an unsatisfiable system compile to?
3. Use your chapter-1 `count()` function to cross-check exercise 1's
   model. Automate the comparison in 5 lines. Congratulations: you now
   test like this repo does.

## Checkpoint

*Your circuit's model count is exactly 2× your brute-force count. Name
the two most likely explanations, and the one-line probe for each.*

Next: [Chapter 4 — One circuit, many questions](04_semirings.md)
