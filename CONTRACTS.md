# ModeNexus technical contracts

This file collects semantic contracts, algorithmic properties, and numerical
invariants that are useful when auditing or extending ModeNexus. API
docstrings intentionally focus on how to call the library; this document is
the detailed reference for claims about exactness, bounds, representations,
and complexity.

These statements describe the behavior the implementation is intended to
satisfy. They are not a claim of formal verification or independent
certification. Coverage comes from a mixture of runtime validation,
construction rules, regression tests, and small-instance reference oracles.

## Terminology and numerical conventions

- A **probability** is finite and lies in `[0, 1]`.
- A **probability distribution** has finite, non-negative entries whose total
  is one. Caller-supplied categorical rows use a relative tolerance of
  `1e-6` where the API requires an already-normalized distribution.
- A **weight** need not be normalized. Weighted model counts are proportional
  masses unless the surrounding API explicitly divides by evidence mass.
- An **exact** result is exact for the supplied compiled model, evidence,
  weights, and declared search scope. It does not imply that the model is an
  exact representation of the physical system.
- An **approximate** result must identify the approximation source and must
  not present a policy-only or retained-mass certificate as a full-belief
  certificate.
- Small floating-point excursions may be tolerated at comparison boundaries.
  Materially invalid probabilities, distributions, or bounds must not be
  silently reinterpreted as valid values.

## Runtime invariant enforcement

`ModeNexusInvariantError` identifies an internal postcondition failure. It is
exported at the package root and is distinct from `ValueError`, which remains
the usual signal for invalid caller input. The checks are ordinary runtime
checks, not Python `assert` statements, so optimized interpreter mode does not
remove them.

Always-on checks cover the high-leverage public result boundaries:

- conditioned circuits retain their evidence literals;
- posterior and tracker belief rows are finite, normalized distributions;
- a posterior queried for a hard-observed variable is a point mass;
- EM log-likelihood steps do not materially decrease;
- value-of-information results remain between zero and the available entropy;
- planner probabilities, costs, interval endpoints, and certificate bounds
  satisfy their local numeric relationships; and
- explicit target evidence is intersected with branch evidence rather than
  replacing it.

These checks are defense in depth. They detect a broken implementation before
an invalid value becomes a trusted certificate, but they do not independently
prove the underlying algorithm or validate the physical model.

## Boolean NNF and DNNF circuits

An NNF circuit is a rooted DAG with literal or constant leaves and AND/OR
internal nodes. Children precede parents in the array representation.

The circuit properties used by evaluators are:

- **Decomposability:** children of an AND node mention pairwise-disjoint
  variable sets.
- **Determinism:** children of an OR node are pairwise inconsistent.
- **Smoothness:** children of an OR node mention the same variables, and the
  root mentions every declared variable unless the root is false.

For circuits satisfying the relevant properties:

- satisfiability and minimum-cost extraction use a forward circuit traversal;
- deterministic, decomposable, smooth circuits support additive weighted
  counting without double counting or missing variables;
- query time is linear in the materialized circuit size, although the circuit
  itself may be exponentially large.

`Circuit.is_deterministic()` is a sufficient syntactic check, not a complete
logical decision procedure. It recognizes contradictions exposed by asserted
literals, including compiler decision nodes and smoothing gadgets.

`Circuit.condition({var: value})` returns a circuit for the conjunction of the
original circuit and the supplied evidence. Evidence literals remain asserted
at the returned root, so later smoothing, counting, MPE, and enumeration
cannot make conditioned variables free again.

`Circuit.smooth()` preserves the represented models while adding missing
variable gadgets. It preserves decomposability and determinism when those
properties held on input. A false root remains false.

`CircuitBuilder` hash-conses nodes and emits children before parents. Its
simplifications preserve Boolean meaning. Nested OR nodes are not flattened
because doing so can invalidate determinism relied upon by the caller.

## Compilation and resource behavior

The Boolean and finite-domain compilers construct decision-DNNF circuits by
unit propagation, connected-component decomposition, variable branching, and
component caching:

- residual connected components become decomposable AND children;
- mutually exclusive variable values become deterministic OR branches;
- cached residual problems share circuit nodes;
- requesting smoothing makes the output suitable for counting evaluators.

The default Boolean compiler has a specialized tree-DP path for acyclic
unary/binary CNF. On that path, work and output size are linear in the primal
forest. Cyclic or wider theories use the general component-caching compiler.

No polynomial-size guarantee applies to general compilation. Variable order
can make runtime and output size exponential, particularly for densely coupled
models. This is an accepted design property: compilation controls and
statistics exist so expensive exact work can be bounded, observed, and
provisioned.

`minfill_order()` and `dtree_order()` are heuristics, not optimality
guarantees. Both return full variable orders. Variables not represented in
clauses remain present in the returned order.

`CompileControl` limits are cooperative. Timeout, cancellation, node, and
cache limits are checked during compilation, and interruption exceptions
carry the most recent `CompilationStats`. Statistics describe work observed
by the implementation; they are not a pre-run upper bound.

## Native finite-domain representation

Each finite-domain variable has one selected value. An FD leaf denotes
`variable=value`; an FD set literal denotes membership in a set of values.
Singleton domains are valid. Empty domains are invalid.

FD formula encoding is closed under complement, so negating an FD literal
does not require an auxiliary. AND/OR formula nodes receive hash-consed
Boolean-domain Tseitin auxiliaries with biconditional clauses. Those
auxiliaries are functionally determined by the user variables; neutral
auxiliary weights therefore do not change WMC or MPE rankings.

`FDCircuit` counting operations require a smooth deterministic decomposable
circuit mentioning all declared variables, except that a false root is valid.

`fd.sample()` samples a model with probability proportional to the product of
its value weights. It returns no assignment when total mass is zero.

The CNF and FD-CNF `models()` helpers are exponential reference enumerators
intended for tests and small instances.

## Formula encoding

Boolean Tseitin encoding introduces one auxiliary per shared non-literal
subformula and constrains the auxiliary biconditionally. Each assignment to
the original variables therefore has at most one compatible assignment to
the auxiliaries. Neutral auxiliary weights preserve counts and weighted
queries over the original variables.

## Evaluation and ranked enumeration

`model_count`, `wmc`, and their log-space FD equivalents rely on smooth
deterministic decomposable input. Public evaluators that require this structure
reject circuits that fail the available structural checks.

MPE uses additive literal or value costs and returns a minimum-cost satisfying
assignment. Ranked enumeration is ordered by the same additive cost.

Marginal-MAP enumeration sums over non-MAP variables and ranks assignments to
the requested MAP variables by that summed mass. The required variable order
must place MAP variables above variables being summed out.

## Diagnosis

Declared Boolean priors are probabilities. Finite-domain priors are finite,
non-negative relative weights with positive total and are normalized when the
variable is declared.

For a compiled system:

- `log_evidence(e)` is the log weighted mass of models consistent with `e`;
- `diagnoses(e)` uses MPE semantics and ranks complete supporting states;
- `map_diagnoses(e)` ranks joint mode assignments by mass summed over their
  compatible non-mode states;
- `posteriors(e)` and `mode_posteriors(e)` represent
  `P(variable=value | e)` and each returned row is a probability
  distribution;
- hard evidence on a queried variable produces a point-mass posterior;
- inconsistent hard evidence raises rather than returning a normalized row.

Exact marginal MAP requires a compatible compilation order. The default
`modes_first=True` order supplies that structure. Mode-first compilation may
be exponentially larger on densely coupled networks; this is an explicit
offline cost rather than a promise of compact compilation.

Soft evidence is a per-value likelihood vector. It is multiplied into value
weights and need not sum to one because only likelihood ratios matter.

Noisy Boolean sensors use:

```
P(sensor=True | expression=True)  = 1 - false_negative
P(sensor=True | expression=False) = false_positive
```

Hidden sensor-noise variables are excluded from decoded public states.

`fit_priors()` performs circuit-based expectation-maximization for independent
categorical priors. Its E-step uses posterior marginals and its M-step uses
average responsibilities. Under those model assumptions, the average
log-likelihood trace is expected to be non-decreasing up to numerical
tolerance.

`value_of_information()` reports expected reduction in the sum of marginal
mode entropies. For a single mode variable this is ordinary entropy reduction;
for multiple mode variables the summed marginal entropy is an upper bound on
joint entropy. A candidate fixed by hard evidence has zero remaining
information and is omitted.

## Temporal tracking

`ModeTracker.step()` returns a normalized belief over joint mode assignments.
The result raises on total belief collapse.

Tracking is exact for the compiled model when expansion covers every
pre-transition state and no beam truncation occurs. Otherwise the belief is a
normalized approximation to retained trajectories.

`TrackedBelief` carries exactness and retained-mass metadata:

- copying and slicing preserve the metadata;
- in-place mutation invalidates exactness and retained-mass claims;
- conversion with `list(belief)` intentionally discards the metadata.

Per-variable transition rows are probability distributions. Joint transition
constraints may prune otherwise possible transitions; surviving transition
mass is interpreted conditionally on a legal transition.

History refinement reconstructs a new tracker from its original prior and
replays retained successful evidence and transition overrides. It does not
recover discarded trajectories by mutating the current truncated beam.
Stateful transition callbacks are reproducible only if the callback itself can
be replayed consistently.

`refine_until()` evaluates the current belief before replaying larger resource
settings. An explicit resource schedule is deterministic; the default schedule
grows the beam geometrically and may attempt exact tracking subject to
`max_exact_states`.

## Planning

Planning probabilities are always scoped to the compiled transition model,
initial belief, supplied outcome model, observation model, and target.

Caller-supplied initial beliefs, outcome rows, scenario rows, and observation
rows must contain finite non-negative probabilities and, where documented as
distributions, must sum to one within tolerance. Missing outcome branches are
not silently filled or normalized.

When an explicit outcome model supplies next-state evidence, that evidence
defines the branch reached by the outcome. The conditional target query
intersects the target with existing branch evidence. Conflicting target
evidence has probability zero; compiled transition weights do not reweight an
explicitly declared outcome branch a second time.

`plan()` ranks command sequences using additive negative-log costs.
`plan_detailed()` decomposes the selected cost into initial-state,
per-transition, and remaining model contributions.

Belief planning has three execution scopes:

- horizon one chooses an action by expectation over the supplied joint belief;
- longer-horizon open-loop planning integrates stochastic outcomes but does
  not branch future commands on observations;
- conditional planning returns a policy tree whose future commands may depend
  on observations.

Observation routing projects telemetry onto the node's observation schema.
Absent keys remain distinct from present keys whose value is `None`.
Observations removed by pruning route through the recorded fallback policy.

### Conditional-policy approximation

Observation pruning merges omitted branches into fallback posteriors rather
than dropping their probability mass. The returned policy value is the value
of that executable coarsened policy. It is not, by itself, the unrestricted
full-observation optimum.

Frontier dominance is defined by no-lower goal probability and no-higher cost.
Removing a dominated frontier point is lossless for ancestor combinations.
Capping a nondominated frontier is an approximation and must be reflected in
the result metadata.

### Certificate relationships

For every root action certificate:

```
policy_utility_lower_bound <= policy_utility_upper_bound
utility_lower_bound        <= utility_upper_bound
policy_goal_probability    <= policy_goal_probability_upper_bound
goal_probability_lower_bound <= goal_probability_upper_bound
```

Policy-prefixed fields are scoped to the supplied normalized belief.
Unprefixed fields compose known tracker-retained mass with adversarial bounds
on missing mass. Unknown retained mass cannot produce a full-belief exactness
claim.

The selected root action is certified only when its lower bound dominates
every alternative root action's upper bound. Maximum regret is the
non-negative gap between the strongest alternative upper bound and the
selected lower bound.

A feasibility certificate incorporates every active whole-policy and branch
constraint. Observation pruning or approximate tracker belief may reduce a
branch certificate to indeterminate. The result fields
`certificate_scope`, `constraint_certification`,
`observation_partition_optimality`, and the retained-mass metadata state that
scope explicitly.

The planner's certificates are internal mathematical certificates under the
declared model and search scope. They are not an external certification of the
model, controller, or physical plant.

### Planning resource controls

`max_action_sequences`, `max_outcome_branches`, `max_policy_nodes`,
`max_observation_branches`, and frontier caps are explicit work or
approximation controls. `PlanControl` adds cooperative timeout,
cancellation, and progress snapshots.

Exact computation may still be exponential in horizon, action count,
observation branching, state count, or frontier width. High computation is
allowed; the contract is that configured limits and reported statistics make
the chosen computation visible.

`CompiledPlanner.estimate_belief_work()` provides a preflight
`PlanningWorkEstimate`. It uses the declared horizon, actions, belief size,
scenario grid, and optional outcome/observation branch hints. It is a planning
estimate, not an upper bound or a duration prediction. The configured hard
caps remain authoritative.

Belief-planning results expose a `PlanningWorkReport` through `result.work`.
It pairs that preflight estimate with elapsed time and measured action
evaluations, goal queries, policy nodes, callback calls, branch counts, and
frontier work. These counters make an expensive run explainable after the
fact without pretending that the per-decision cost is constant.

`ConditionalBeliefPolicyResult` retains its flat fields for compatibility and
also groups them into `approximation_details`, `certificate`, and
`diagnostics` views. The views reorganize the same immutable result data; they
do not alter certificate scope.

## Torch evaluation and learning

`TorchCircuit` lowers a circuit into depth-ordered tensor operations.
Log-probability evaluation requires smooth deterministic decomposable input;
negative-log minimum evaluation requires decomposability. Structural
preconditions remain the caller's responsibility unless checked before
construction.

In the `"logprob"` semiring, the root is log WMC. Autograd derivatives with
respect to literal log weights give posterior literal marginals, so
`marginals()` obtains all requested literal marginals from one backward pass.
In the `"neglog"` semiring, the root is the minimum additive cost and `mpe()`
decodes a minimizing assignment. A leading batch dimension evaluates multiple
weight or evidence rows through the same layered circuit.

Neural observation likelihood blocks are log-softmax normalized before
circuit evaluation. Without this normalization their scale is unidentifiable
and a network could improve the objective by increasing every likelihood.
Hard or otherwise informative evidence elsewhere in the model is required to
ground weakly supervised observation learning.

Gradient-based prior learning optimizes the same categorical parameters as
the diagnosis model but does not inherit EM's monotonic-update property.

## Verification locations

The principal regression and property checks live in:

- `tests/test_compiler.py` and `tests/test_fd.py`;
- `tests/test_diagnosis.py`;
- `tests/test_belief_planning.py`;
- `tests/planning_oracle.py` and `tests/test_planning_oracle.py`, whose
  deliberately small horizon-two exhaustive evaluator does not call the
  planner or ModeNexus inference routines;
- `tests/test_tracking.py` and tracker refinement tests;
- the adversarial suite under `modenexus_stress/exercises`.

When a new exactness, bound, normalization, or complexity claim is added here,
it should be paired with a runtime check or an independent regression/property
test where practical.
