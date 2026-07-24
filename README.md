# modenexus

Compilation of propositional theories to **Decomposable Negation Normal
Form (DNNF)**, tractable weighted reasoning on the compiled circuits, and
model-based diagnosis — with an optional **PyTorch backend** for batched
GPU evaluation and autograd-powered marginals.

The architecture follows the knowledge-compilation approach used in
DNNF-based diagnosis at NASA JPL: encode a system model (component modes
with priors, observables, expectations) into logic, compile **once**
offline into a small circuit, then answer observation queries **online** in
time linear in circuit size — including enumerating complete system states
ordered from most to least probable, with leaf weights read as negative
log probabilities. See [docs/DESIGN.md](docs/DESIGN.md) for the full
technical treatment and [docs/FUTURE_WORK.md](docs/FUTURE_WORK.md) for the
roadmap from next steps to blue sky.

**New to the library?** Start with the [tutorial](tutorial/README.md) —
thirteen chapters from propositional logic to GPU-backed learning, written
for a CS-undergrad level, with exercises and worked solutions.

## Install

```bash
pip install modenexus            # core: pure Python, no dependencies
pip install modenexus[torch]     # + PyTorch backend
```

From a clone (development):

```bash
pip install -e .[dev]           # editable, + pytest
```

## Quick start: compile and query

```python
import modenexus

cnf = modenexus.CNF(num_vars=3, clauses=[(1, 2), (-1, 3)])
circuit = modenexus.compile_cnf(cnf, smooth=True)   # decision-DNNF: decomposable,
                                                    # deterministic, smooth
modenexus.model_count(circuit)                      # 4
modenexus.wmc(circuit, modenexus.weights_from_probs(3, {1: 0.9, 2: 0.5, 3: 0.5}))

# MPE / most probable model under neg-log costs
costs = modenexus.costs_from_probs(3, {1: 0.9, 2: 0.5, 3: 0.5})
cost, best = modenexus.mpe(circuit, costs)

# Models ordered most-probable-first (lazy k-best)
for cost, model in modenexus.enumerate_models(circuit, costs, k=5):
    print(cost, model)
```

Long offline compiles can be bounded and observed cooperatively:

```python
control = modenexus.CompileControl(
    timeout_seconds=60,
    max_nodes=100_000,
    cancel=lambda: operator_cancelled(),
    progress=lambda stats: print(stats),
)
circuit = modenexus.compile_cnf(cnf, smooth=True, control=control)
```

Cancellation and budget exceptions carry the last `CompilationStats`
snapshot, making an expensive variable order measurable and safe to abandon
without terminating the worker process. `compile_fd`, `SystemModel.compile`,
and `Planner.compile` accept the same control object.

DIMACS CNF (`modenexus.CNF.from_dimacs`) and the c2d `.nnf` circuit format
(`modenexus.nnf_io`) are supported, so circuits from external compilers
(c2d, dsharp, D4) plug into the same evaluators.

## Diagnosis: modes, priors, ranked explanations

```python
from modenexus import SystemModel, iff

m = SystemModel()
v1 = m.mode("valve1", ("ok", "stuck_open", "stuck_closed"), priors=(0.98, 0.01, 0.01))
v2 = m.mode("valve2", ("ok", "stuck_open", "stuck_closed"), priors=(0.98, 0.01, 0.01))
flow1, flow2 = m.bool("flow1"), m.bool("flow2")
m.add(iff(flow1, v1 != "stuck_closed"))
m.add(iff(flow2, v2 != "stuck_closed"))

system = m.compile()                                   # offline
for d in system.diagnoses({"flow1": False, "flow2": True}, k=3):  # online
    print(d)                # ranked by best supporting state (MPE)
for d in system.map_diagnoses({"flow1": False}, k=3):
    print(d)                # ranked by exact summed posterior (marginal MAP)
system.mode_posteriors({"flow1": False})   # exact P(mode=value | evidence)
```

`map_diagnoses` is exact marginal MAP — mode variables are branched first
during compilation (`modes_first=True`, the default), which constrains the
circuit so that max-over-modes / sum-over-everything-else is a single
sweep plus lazy k-best enumeration.

The diagnosis stack runs on a **native finite-domain core** (`modenexus.fd`):
variables carry their domains directly, circuit leaves are atomic
assignments like `valve1=stuck_closed`, and decision nodes branch d-ways —
no one-hot encoding, no exactly-one clauses, ~40% smaller circuits on
mode-heavy models than the boolean lowering. Continuous quantities can be
**quantized into bounded ranges** with threshold atoms and automatic
bucketing of numeric evidence:

```python
level = m.quantized("level", (0.0, 10.0, 50.0, 100.0), priors=(0.2, 0.5, 0.3))
m.sensor("low_alarm", level.below(10.0))
m.add((leak == "large") >> level.below(10.0))
system.log_evidence({"level": 37.2})  # numeric evidence, bucketed for you
```

Compiled systems are also **generative and learnable**:

```python
state = system.sample_state(rng)         # exact simulation from the model
telemetry = [{"alarm": system.sample_state(rng)["alarm"]}
             for _ in range(4000)]
system.fit_priors(telemetry)             # EM: learn failure rates from
                                         # partially observed telemetry
system.posteriors(evidence, names=[...]) # exact marginals for any variable
system.save("plant.json")                # single-allocation reload:
                                         # header states all bounds up front
```

`fit_priors` is exact expectation-maximization on the circuit (E-step:
WMC-ratio posteriors; M-step: average), with a guaranteed non-decreasing
likelihood — failure priors estimated from fleet data instead of
engineering guesses, with the logical model as a hard constraint.

Sensors can be noisy (`m.sensor("alarm", expr, false_positive=0.1,
false_negative=0.2)`), and `modenexus.ModeTracker` runs the monitoring loop:
per-mode transition matrices, observations each timestep, and a
beam-filtered belief over joint mode assignments (exact HMM filtering when
the beam covers the mode space — see `examples/home_battery.py` for a
degrading-battery week of telemetry, and `docs/MODELING_NOTES.md` for
ergonomics findings from that experiment). `ModeTracker(system, exact=True)`
computes the required joint-state capacity automatically, guarded by
`max_exact_states`; approximate trackers expose `last_step_info` with
expansion/beam truncation and retained-mass diagnostics.
`ModeTracker.belief()` returns a list-compatible `TrackedBelief` carrying
exactness and retained-mass metadata, so downstream planners can distinguish
an exact posterior from a normalized truncated beam. `copy()` and slicing
preserve that metadata, in-place mutation invalidates it, and `list(belief)`
is the explicit way to drop it — a routine container operation can no longer
flip a certificate.
For adaptive deployments, construct the tracker with
`retain_history=True`. `tracker.refine(beam=..., expand=...)` returns a
fresh tracker rebuilt from the original prior and all retained evidence and
per-step transition overrides; it does not mutate a truncated belief or
pretend discarded trajectories can be recovered in place.
`tracker.refine_until(evaluate, accept, ...)` evaluates the current belief,
replays at an explicit or geometrically increasing resource schedule, and
stops when the caller's certificate or regret predicate succeeds. Its result
records every attempted beam, replayed-step and generated-candidate counts,
replay and evaluation time, retained mass, and any downstream
`certificate_scope` and `maximum_regret`.
The default final attempt is exact, subject to `max_exact_states`.

Beyond ranked diagnoses: `value_of_information` scores which sensor to
read next (expected entropy reduction, in nats),
`diagnoses_min_cardinality` ranks by fewest broken components, and
`modenexus.Planner` unrolls the same model over an n-step horizon to
estimate hidden state from a command/observation history and to plan
command sequences that reach a goal — MEXEC-style, one circuit for both.
The existing `plan()` and `estimate()` tuple APIs remain compact;
`plan_detailed()` and `estimate_detailed()` additionally return the state
trajectory and a breakdown of initial-state, per-transition, hard-evidence,
and other model costs.

For one-step decisions under uncertainty, `plan_belief()` accepts the
correlated joint distribution returned by `ModeTracker.belief()` and ranks
actions by exact expected goal probability or expected utility. Applications
may supply stochastic action outcomes and separate operational costs;
outcome and observation callback probabilities must sum to 1 (within
`1e-6`), so a forgotten branch fails loudly instead of being silently
renormalized. An optional `PlanControl` adds cooperative
timeout/deadline limits, a cancellation callback, and progress snapshots
(`PlanningStats`) mirroring `CompileControl`; interrupted searches raise
`PlanningCancelled`/`PlanningBudgetExceeded` carrying the partial counts
and best root action found so far.
On planners compiled with a longer horizon, it performs bounded lookahead
over command sequences, propagates stochastic outcome branches, and returns
the best `BeliefPolicyResult`; sequence and branch budgets make the
exponential work explicit. Without an observation model this result has
`observation_branching=False`: execute its first action and replan after new
evidence.

Supplying both `outcome_model` and
`observation_model(next_state, command)` instead returns a
`ConditionalBeliefPolicyResult`. Each `BeliefPolicyNode` selects an action,
and its `BeliefPolicyBranch` edges combine states with the same observation
into a posterior belief before optimizing the next action. Observation
callbacks may be deterministic mappings or probability-weighted mappings,
so the same surface represents perfect, partial, or noisy sensing.
`max_policy_nodes`, `max_outcome_branches`, and
`max_observation_branches` make conditional-policy expansion explicit; the
result reports all three realized expansion counts. Observations after the
final action are not expanded because no decision remains.

Each retained `BeliefPolicyBranch` exposes the normalized posterior the
planner optimized against (plus `posterior_marginals()`), so an operator
can inspect which hidden states justified a branch's action without
re-deriving the Bayes update. At execution time,
`BeliefPolicyNode.route(observation)` projects telemetry onto the node's
`observation_schema` (extra sensor fields are ignored, missing schema keys
raise) and reports exact, fallback, terminal, or unmatched routing;
`continuation()` remains the compact accessor on top of it.

For noisy sensors with many low-probability readings, set
`min_observation_probability` and/or `max_observations_per_node`. Pruned
readings are merged into an optimized fallback posterior rather than
dropped and exposed as `fallback_branch` with its aggregate posterior and
contributing observations; well-formed readings matching no retained branch
route through that fallback. Results report generated and pruned branches,
retained/discarded observation probability, whether action ranking is
heuristic or certified, and whether the returned utility is a lower bound
on the exact full-observation optimum. Every evaluated root action exposes
utility lower and upper bounds. The overall result reports the unrestricted
optimal-utility upper bound and maximum root-action regret; the root action
is certified when its lower bound dominates every alternative upper bound.
For hard reliability requirements, `min_goal_probability` turns the floor
into a first-class chance constraint: feasibility is decided before utility
ranks the survivors, and in conditional planning the constraint is enforced
across the whole policy tree by Pareto-frontier lookahead — reliability
bought in one observation branch can compensate for another branch's
ceiling, which no per-node threshold can express.
`min_branch_goal_probability` adds the stricter per-branch safety variant.
An unreachable floor is reported (`feasible=False` with
`best_achievable_goal_probability`), never silently degraded, and
`constraint_certification` composes pruning and tracker mass into
certified-feasible / certified-infeasible / indeterminate.

When the input is a `TrackedBelief`, the planner composes tracker uncertainty
into separate end-to-end action bounds. Results preserve the policy-only
certificate, report certificate scope, and refuse to present a beam-only
certificate as end-to-end when tracker mass is unknown.

## GPU / batched evaluation (PyTorch)

```python
import torch
from modenexus.torch_backend import TorchCircuit

tc = TorchCircuit(circuit, semiring="logprob", device="cuda")
w = tc.weights_from_probs({1: 0.9}, batch=1024)   # (B, 2n) log-weights
log_z = tc(tc.condition(w, {2: True}))            # batched log-WMC
marginals = tc.marginals(w)                       # (B, 2n) posteriors:
                                                  # one backward pass computes
                                                  # every P(lit | evidence)

mp = TorchCircuit(circuit, semiring="neglog")     # min-sum semiring
costs, states = mp.mpe(cost_tensor)               # batched MPE
```

Because evaluation is ordinary autograd-friendly tensor code, the circuit
composes with other PyTorch models — e.g. neural observation models
producing leaf weights, trained end-to-end through exact inference.

## Layout

| Module | Contents |
|---|---|
| `modenexus.cnf` | CNF container, DIMACS I/O |
| `modenexus.formula` | propositional AST, Tseitin encoding |
| `modenexus.compiler` | CNF → decision-DNNF (DPLL + components + caching) |
| `modenexus.circuit` | circuit arrays, property checks, smooth/condition |
| `modenexus.eval` | semiring sweeps: SAT, counting, WMC, log-WMC, MPE |
| `modenexus.kbest` | lazy ordered model enumeration |
| `modenexus.torch_backend` | layered batched tensor evaluation, marginals |
| `modenexus.fd` | native finite-domain core: FD-CNF, compiler, queries, MAP |
| `modenexus.diagnosis` | `SystemModel` / ranked diagnoses / posteriors / quantized vars / EM / save-load |
| `modenexus.tracking` | `ModeTracker` temporal filtering |
| `modenexus.planning` | `Planner`: n-step estimation and command planning |
| `modenexus.torch_learn` | SGD prior learning, neural observation front-ends |
| `modenexus.viz` | Graphviz / matplotlib circuit rendering |
| `modenexus.external` | c2d compiler driver |
| `modenexus.nnf_io` | c2d `.nnf` interop |

## Tests

```bash
python -m pytest
```

All evaluators, the compiler, enumeration, the torch backend, and the
diagnosis layer are cross-validated against brute-force model enumeration
on randomized instances.
