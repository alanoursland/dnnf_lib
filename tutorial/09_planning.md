# Chapter 9 — Planning: the same circuit, run backwards

## Learning goals

- Declare a transition system (modes, commands, transitions) and
  compile a fixed-horizon plan/estimate circuit.
- Explain "one artifact, three queries": estimation, planning, and
  planning-from-sensors are the same evaluation with different leaves
  clamped.
- Read plan costs as neg-log probabilities of success.

## The historical heart

This chapter rebuilds, in miniature, the exact architecture this
library descends from: a spacecraft "micro executive" that compiled
its device model once, then answered *what state am I in?* and *what
commands reach the target state?* with the same structure evaluated
two ways (Barrett, IJCAI 2005 — the running example below, a
star-tracking siderostat, is that paper's own).

```python
from modenexus import Planner
from modenexus.formula import iff

p = Planner()
p.mode("sw", ("Tracking", "Idling"), priors=(0.2, 0.8))
p.command("cmd", ("idle", "track", "none"))
p.observable("valid")                       # sensor: star lock valid?
p.behavior(lambda v: iff(v["valid"] == True, v["sw"] == "Tracking"))
p.transition("sw", "Tracking", "Idling", command=("cmd", "idle"))
p.transition("sw", "Idling", "Tracking", command=("cmd", "track"))

cp = p.compile(horizon=2)
print(cp.system.circuit)
# FDCircuit(nodes=74, edges=101, and=11, or=7, lit=56, vars=30)
```

`compile(horizon=2)` *unrolls time*: it lays down variables for the
mode at steps 0, 1, 2, the command at steps 0 and 1, the observable at
every step, plus per-step transition selectors — then compiles the lot
into one circuit. The `behavior` constraint is stamped into every time
slice. Thirty variables, 74 nodes: time is just more variables.

## Three questions, one artifact

**Estimation** — clamp the observation leaves, read the mode variables
of the cheapest world:

```python
print(cp.estimate([{"valid": True}, {"valid": False}],
                  commands=[{"cmd": "idle"}]))
# (1.6094, [{'sw': 'Tracking'}, {'sw': 'Idling'}])
```

Lock was valid, then invalid, and we know we commanded `idle`: the
most likely trajectory is Tracking → Idling. The cost is exactly
`−log 0.2` — the prior of starting in Tracking — because everything
else in that trajectory was certain. Costs stay meaningful all the way
through: `exp(−cost)` is the trajectory's probability.

**Planning** — clamp mode variables at both ends instead, read the
*command* variables:

```python
print(cp.plan({"sw": "Tracking"}, {"sw": "Idling"}))
# (1.6094..., [{'cmd': 'idle'}, {'cmd': 'idle'}])
```

The plan is `idle` at step 0. (Step 1's command is arbitrary — the
target is already reached, every zero-cost filler is equally good;
real executives re-plan each tick anyway.) An *unreachable* target
returns `None`, not a bad plan — within this horizon the circuit
provably contains every feasible trajectory, so absence is proof.

**Planning from sensors** — the fusion move, and the reason this
chapter exists. Don't clamp the current mode at all; clamp what the
*sensor said*:

```python
print(cp.plan(target={"sw": "Tracking"}, observations=[{"valid": False}]))
# (0.2231, [{'cmd': 'track'}, {'cmd': 'track'}])
```

We never asserted the current state. The circuit inferred it from
`valid=False` (must be Idling), then planned `track` — estimation and
planning in a single evaluation. Cost `0.2231 = −log 0.8`: the prior
probability of the inferred start. There is no hand-off between an
estimator module and a planner module to get out of sync; there is
one model of reality and two kinds of clamps.

Transitions can also carry costs (`cost=` is a neg-log likelihood of
the transition *working*), and then `plan` returns the *most reliable*
command sequence, not just any feasible one — chapter 4's semiring
doing mission planning.

## Exercises

1. Add a third mode `Safe` reachable only from `Idling` via a `safe`
   command. Show that `plan({"sw": "Tracking"}, {"sw": "Safe"})`
   returns None at horizon 1 and a 2-step plan at horizon 2.
2. Give the `Idling → Tracking` transition `cost=0.7` and add an
   alternative two-step route (e.g. via a `Warmup` mode) whose summed
   cost is lower. Verify `plan` switches routes, and check the
   returned cost against your hand sum. (Solution: `solutions/ch09.py`.)
3. Greenhouse planner: modes for a `valve` (`open`/`closed`), commands
   to actuate it, observable `moist` with a behavior constraint. Plan
   your way from "soil dry" (observation, not state!) to "valve open."

## Checkpoint

*Why does `plan(target=..., observations=...)` — with no `current`
argument — still produce a sound plan, and what would a traditional
"estimate first, then plan from the point estimate" pipeline get wrong
when the estimate is ambiguous?* (Hint: the circuit keeps *both*
possible starts alive and plans against their weighted mixture.)

Next: [Chapter 10 — Learning](10_learning.md)
