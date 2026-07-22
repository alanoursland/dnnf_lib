# Chapter 10 — Learning: closing the loop with data

## Learning goals

- Simulate telemetry from a model (`sample_state`) and learn priors
  back from it (`fit_priors`).
- State what EM does here in two sentences, and read a log-likelihood
  trace.
- Know when a rate is *identifiable* and what to check before
  trusting a learned number.

## Where do priors come from?

Every chapter so far typed priors in by hand — engineering guesses.
The honest ones (pump fails 3%/year?) are folklore. But a deployed
greenhouse produces something better than folklore: months of sensor
readings. This chapter learns failure rates *from telemetry alone*,
never observing the hidden modes directly.

```python
import random
from dnnf import SystemModel

def build(p_leaky):
    m = SystemModel()
    v = m.mode("valve", ("ok", "leaky"), priors=(1 - p_leaky, p_leaky))
    m.sensor("puddle", v == "leaky", false_positive=0.05, false_negative=0.10)
    return m.compile()

# The "real world": a fleet where 20% of valves are leaky.
rng = random.Random(7)
true_world = build(0.20)
telemetry = [{"puddle": true_world.sample_state(rng)["puddle"]}
             for _ in range(800)]
```

`sample_state` draws exact worlds from the model — priors, constraints,
sensor noise and all — so we can practice on data whose ground truth we
know. (In your project this list is just your logged readings.)

## Learn

```python
learner = build(0.5)                     # start from a WRONG prior: 50%
history = learner.fit_priors(telemetry, names=["valve"])
spec = learner.circuit.spec
fitted = learner._weights[spec.mvlit(learner.vars["valve"].fd_var, 1)]
print(round(fitted, 4), len(history))    # 0.2089, 7 iterations
```

From 800 puddle readings — *never once told which valves were
leaky* — the learner walks from 0.5 to **0.2089**. Is that good? Check
against the best any method could do: the observed puddle frequency
was 0.2275, and inverting the sensor model analytically,
`p̂ = (0.2275 − 0.05) / 0.85 = 0.2088`. The learner matched the exact
maximum-likelihood estimate to four decimals; the remaining gap to the
true 0.20 is sampling noise (800 flips), not algorithm error. More
telemetry shrinks it; no algorithm could do better from this data.

The `history` it returns is the average log-likelihood per iteration
(−0.667 → −0.536 here, converged in 7): **it never decreases** — a
mathematical guarantee of the method, so a decreasing trace means a
bug, full stop.

What is this method? **Expectation–Maximization**, and on this
library it's transparent: each iteration computes, for every telemetry
record, the posterior over the hidden mode given current priors
(chapter 5 machinery, nothing new), then sets the new prior to the
average of those posteriors. Guess → explain → average → repeat.

And the learned numbers immediately serve every earlier chapter:
`learner.posteriors({"puddle": True})["valve"]["leaky"]` now says
0.826 — diagnosis with priors the *fleet* voted on.

## The one big caveat: identifiability

Learning inverts the sensor model, which only works if the data can
tell the hypotheses apart. Two rates that produce the same observable
statistics are **unidentifiable** — EM will converge, the likelihood
will look fine, and the split between them is fiction. Classic trap:
learning the valve's leak rate *and* the sensor's false-positive rate
from the same single reading stream (a leakier fleet and a jumpier
sensor explain the same puddle frequency).

Practical checks before trusting a learned rate:
run the learner from several different starting priors — if they land
on different answers with equal likelihood, you're unidentifiable;
and simulate-then-recover (exactly this chapter's setup) *before*
feeding real data, so you know the pipeline can recover a rate you
planted.

Also honest: `fit_priors` learns priors of declared variables. It
will not invent a failure mode you didn't model — a model missing
`stuck_half_open` explains that data as noise, confidently. Learning
sharpens a correct structure; it cannot repair a wrong one.

*(GPU footnote: `fit_priors_torch` trains the same numbers by gradient
descent through the differentiable circuit — same fixed point, proven
in this repo's tests to four decimals. If you're curious why gradient
descent and EM agree so exactly here, the repo's
`reports/implicit_em_lab_report.md` is the rabbit hole, and it's a
good one.)*

## Exercises

1. Reduce telemetry to 50 records and rerun from three different
   starting priors. How much do the answers scatter? Now 5,000
   records. Plot fitted-vs-N. (Solution: `solutions/ch10.py`.)
2. Make it unidentifiable on purpose: `fit_priors(telemetry,
   names=["valve", "_puddle_fp"])`. Run from starts (0.5, 0.5) and
   (0.3, 0.01) and compare final likelihoods and final rates. Explain
   what you see in two sentences.
3. Full loop: sample 30 days from a *tracked* pump (chapter 7 model)
   with the true wear rate, then treat `ok→weak` as unknown and grid-
   search it by likelihood of the observed sequence. (Transition rates
   aren't `fit_priors`-learnable yet — a grid over `ModeTracker` runs
   is the honest workaround, and writing it teaches you exactly what
   the likelihood of a sequence *is*.)

## Checkpoint

*Your learned leak rate came out 0.02 with a beautiful likelihood
curve, but the true fleet rate is 0.20. The sensor's false-positive
rate in the model was set to 0.25 (way too high). Explain, in one
sentence, how the wrong sensor model absorbed the signal.*

Next: [Chapter 11 — Under the hood](11_under_the_hood.md)
