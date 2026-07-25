# modenexus

Compilation of propositional theories to **Decomposable Negation Normal
Form (DNNF)**, tractable weighted reasoning on the compiled circuits, and
model-based diagnosis — with an optional **PyTorch backend** for batched
GPU evaluation and autograd-powered marginals.

The architecture follows the knowledge-compilation approach used in
DNNF-based diagnosis at NASA JPL: encode a system model (component modes
with priors, observables, expectations) into logic, compile **once**
offline into a circuit, then answer observation queries **online** over that
materialized structure — including enumerating complete system states
ordered from most to least probable. Compilation can be exponential on
densely coupled models; controls and statistics make that cost visible.
See [docs/DESIGN.md](docs/DESIGN.md) for the full
technical treatment, [CONTRACTS.md](CONTRACTS.md) for exactness, invariant,
certificate, and complexity claims, and
[docs/FUTURE_WORK.md](docs/FUTURE_WORK.md) for the roadmap from next steps to
blue sky.

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

For acyclic unary/binary CNFs, the default compiler uses a specialized tree
dynamic-programming path. Wider or cyclic theories use the general
component-caching DPLL compiler. `Circuit.condition(evidence)`
asserts the evidence in its returned circuit, so later `smooth()`, counting,
MPE, and enumeration cannot re-free conditioned variables.

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
    print(d)                # ranked by summed posterior (marginal MAP)
system.mode_posteriors({"flow1": False})   # P(mode=value | evidence)
```

`map_diagnoses` uses marginal-MAP semantics. Mode variables are branched
first during compilation (`modes_first=True`, the default), which may make
the offline circuit much larger on densely coupled models.

The diagnosis stack runs on a **native finite-domain core** (`modenexus.fd`):
variables carry their domains directly, circuit leaves are atomic
assignments like `valve1=stuck_closed`, and decision nodes branch d-ways —
without a public one-hot encoding. Continuous quantities can be
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
state = system.sample_state(rng)         # sample from the compiled model
telemetry = [{"alarm": system.sample_state(rng)["alarm"]}
             for _ in range(4000)]
system.fit_priors(telemetry)             # EM: learn failure rates from
                                         # partially observed telemetry
system.posteriors(evidence, names=[...]) # variable marginals
system.save("plant.json")                # save a reloadable artifact
```

`fit_priors` updates categorical priors from partial telemetry and returns
the likelihood trace for convergence monitoring.

Sensors can be noisy (`m.sensor("alarm", expr, false_positive=0.1,
false_negative=0.2)`), and `modenexus.ModeTracker` maintains a beam-filtered
belief as observations arrive. `TrackedBelief` carries approximation
metadata, while `retain_history`, `refine`, and `refine_until` support
measured replay at larger resource settings. See
[`CONTRACTS.md`](CONTRACTS.md) for exactness and retained-mass semantics.

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

For decisions under uncertainty, `plan_belief()` accepts a correlated joint
belief and optional outcome, observation, and action-cost callbacks.
`PlanControl` adds cooperative limits, cancellation, and progress snapshots.
`estimate_belief_work()` provides a callback-free preflight estimate, and
completed results expose measured work counters.

```python
estimate = compiled_planner.estimate_belief_work(
    initial_belief_states=len(belief),
    actions=("wait", "repair"),
    conditional=True,
    outcome_branching_hint=2,
    observation_branching_hint=3,
    max_policy_nodes=100_000,
)
result = compiled_planner.plan_belief(
    belief,
    target,
    outcome_model=outcomes,
    observation_model=observations,
)
print(estimate.estimated_policy_nodes)  # preflight structural estimate
print(result.work)                      # measured completed-search work
```

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
`BeliefPolicyNode.route(observation)` intersects telemetry with the node's
`observation_schema` and reports exact, fallback, terminal, or unmatched
routing. Extra sensor fields are ignored; callback branches may have
heterogeneous key sets because absence is part of observation identity.
`continuation()` remains the compact accessor on top of it.
Certificate-bearing policy trees are recursively immutable: nested action,
observation, posterior-state, and certificate mappings cannot be changed
without constructing a new result. Convenience properties such as
`result.action` return mutable defensive copies for execution.

For noisy or highly branching observations, use
`min_observation_probability`, `max_observations_per_node`, and
`max_frontier_points`. Reliability requirements use
`min_goal_probability` and `min_branch_goal_probability`. The result groups
the relevant fields under `approximation_details`, `certificate`, and
`diagnostics`, while retaining the original flat attributes for
compatibility. Detailed bound and certification relationships live in
[`CONTRACTS.md`](CONTRACTS.md).

For a finite uncertainty set, use `outcome_scenarios` with
`robust_objective="maximin"`. `robust_weight_resolution` and
`max_robust_candidates` bound the scenario-weight search; results include
per-scenario audits and the selected search scope.

For real execution, create a stateful runner from a conditional or one-step
belief planning result:

```python
execution = result.execution()
step = execution.advance(
    outcome={"battery": "ready"},
    observation={"reserve_meter": True},
)
next_belief = step.posterior
regime_marginals = step.posterior_marginals()
current_execution_belief = execution.belief()
```

The runner applies outcome and observation evidence with the same callbacks
used for planning, routes the policy, stops on physical success, and reports
the updated posterior and accumulated action cost. It updates posterior
belief even on terminal paths, where no observation branch was constructed.
For robust results, pass `outcome_scenario="name"` (or an explicit
`outcome_model`) to select the physical model used during execution.

When the input is a `TrackedBelief`, result metadata records how tracker
approximation participates in planning scope.

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
