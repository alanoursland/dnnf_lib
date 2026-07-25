# Changelog

Notable user-facing changes are recorded here for inclusion in release notes.

## Unreleased

### Fixed

- **GAP-001 — observed-candidate VOI:** `CompiledSystem.value_of_information`
  now omits explicitly supplied candidates that are already fixed by the
  evidence. This prevents negative values for tests whose result is known.
- **GAP-002 — mode-prior validation:** categorical priors now reject
  non-finite values, negative values, length mismatches, and vectors without
  a positive total. Valid positive vectors remain relative weights and are
  normalized automatically. Validation also covers Boolean priors, planner
  initial-state priors, dynamic mode-prior overrides, learning with empty
  observations, and loaded compiled-system weights.
- **GAP-003 — sensor-probability validation:** `false_positive` and
  `false_negative` now reject non-finite values and values outside the closed
  interval `[0, 1]`, with errors naming the sensor and parameter.
- **GAP-007 — static planner modes:** planner modes without explicit
  transition rules now persist automatically across the planning horizon.
  Static persistence is lowered directly without creating an invalid
  one-value transition selector, and detailed planner results report the
  implicit transition as a zero-cost noop.
- **GAP-015 — tracked-belief metadata preservation:** `TrackedBelief.copy()`
  and slicing now return metadata-preserving `TrackedBelief` instances, so a
  routine copy can no longer silently upgrade a beam-scoped certificate to
  `caller-supplied-belief`. In-place mutation (`belief[:] = ...`, `append`,
  `sort`, and the other mutating list methods) invalidates exactness and
  retained mass instead of reporting stale guarantees. `list(belief)` remains
  the explicit way to obtain an unscoped plain list.
- **GAP-016 — callback probability contract:** `outcome_model` and
  `observation_model` probabilities must now sum to 1 within `1e-6` relative
  tolerance. Subnormalized, supernormalized, and zero totals raise a
  `ValueError` naming the state, action, and observed total instead of being
  silently rescaled, which previously masked forgotten outcome branches.
  Input belief masses are unchanged: they remain validated, normalized
  relative weights.
- **GAP-018 — policy observation routing:** `BeliefPolicyNode.continuation()`
  now projects the supplied observation onto the node's observation schema,
  so telemetry mappings carrying extra sensor fields match their retained
  branch instead of silently routing to the pruned-observation fallback. New
  `BeliefPolicyNode.observation_schema` exposes the required keys, and
  `route()` returns a structured `BeliefPolicyRouting` reporting exact,
  fallback, terminal, or unmatched routing with the projected observation.
- **GAP-021 — certificate-bearing policy immutability:** action,
  observation, posterior-state, contributing-observation, projected-routing,
  and certificate mappings reachable from a conditional result are now
  recursively immutable. Public convenience properties still return mutable
  defensive copies for execution.
- **GAP-022 — heterogeneous observation routing:** policy routing now treats
  key absence as part of observation identity. Callback branches with
  different key sets route exactly as returned, unrelated telemetry fields
  are ignored, and a present `None` remains distinct from an absent key.
- **GAP-023 — branch-constraint certification:** every active whole-policy
  and per-branch reliability constraint now participates in
  `constraint_certification`; branch-only constraints no longer receive a
  missing status, and an infeasible branch constraint cannot be reported
  certified feasible.
- **GAP-024 — branch safety under observation pruning:** a per-branch floor
  is no longer certified feasible after raw observations are merged into a
  fallback posterior. Pruned branch certificates are `indeterminate` unless
  exact, unaggregated analysis certifies the result; exact raw-branch
  infeasibility remains `certified-infeasible`.
- **GAP-026 — best-achievable reliability scope:** constrained conditional
  results now distinguish Pareto-frontier exactness from observation
  partition optimality. `best_achievable_goal_probability_scope` identifies
  full-observation versus selected-coarsening ceilings, and pruned results
  expose an unrestricted reliability upper bound instead of presenting the
  coarsened value as a global ceiling.
- **GAP-028 — robust scenario execution costs:** native robust policy audits
  now stop a path when the compiled target is already satisfied. Scenario
  action costs and utilities no longer include continuation actions after
  physical success and match independent policy enumeration.
- **GAP-029 — inherited robust metric scope:** robust results now expose
  `inherited_metric_scope="scenario-weighted-candidate-generation"` and the
  selected candidate's `selected_scenario_weights`. Callers can distinguish
  inherited scalarization metrics from worst-case and per-scenario values
  without matching policy objects into the candidate portfolio.
- **GAP-031 — mixed terminal/nonterminal continuation costs:** ordinary
  conditional planning now charges a continuation action only to unresolved
  posterior mass. Noisy observations that mix target-satisfied and failed
  states preserve all goal probability without overcharging satisfied paths
  or understating policy utility.

### Added

- **GAP-004 — compilation controls:** added `CompileControl`,
  `CompilationStats`, `CompilationCancelled`, and
  `CompilationBudgetExceeded`. Boolean and finite-domain compilation now
  support cooperative time, node, and cache budgets; cancellation callbacks;
  and progress callbacks. The same control object is accepted by
  `compile_cnf`, `compile_fd`, `SystemModel.compile`, and `Planner.compile`.
  Interrupted exceptions carry the latest statistics snapshot.
- **GAP-005 — planner cost explanations:** added `plan_detailed()` and
  `estimate_detailed()`. Their structured results include the selected state
  trajectory, commands, normalized initial-state prior cost, each selected
  transition and its cost, hard-evidence cost, other model cost, and total
  cost. Existing `plan()` and `estimate()` tuple return values are unchanged.
- **GAP-006 — exact tracking ergonomics:** added
  `ModeTracker(..., exact=True)` with automatic joint-mode state-space sizing
  and a configurable `max_exact_states` guard. Trackers expose
  `joint_state_count`, `is_exact`, and `last_step_info`, including expansion
  truncation, beam truncation, and retained probability mass when measurable.
- **GAP-008 — belief-aware planning:** added one-step
  `CompiledPlanner.plan_belief()`. It optimizes exact expected goal
  probability or expected utility over a correlated joint belief such as
  `ModeTracker.belief()`, supports explicit stochastic outcome models,
  separates action costs from physical outcome probabilities, validates and
  normalizes belief/outcome masses, and returns every action evaluation.
  The original `BeliefPlanResult` remains the horizon-one result type.
- **GAP-009 — multi-step belief lookahead:** `plan_belief()` now supports
  planners compiled with horizons greater than one. It enumerates bounded
  command sequences, propagates explicit stochastic outcome branches,
  accumulates action costs, and maximizes terminal expected utility. New
  `BeliefPolicyResult` and `BeliefSequenceEvaluation` types expose the best
  sequence, every alternative, per-step goal probabilities, and the explicit
  `observation_branching=False` limitation. Sequence and outcome-branch
  budgets fail early with the required resource count.
- **GAP-010 — observation-contingent belief policies:** `plan_belief()` now
  accepts an `observation_model` alongside explicit stochastic outcomes and
  returns a bounded `ConditionalBeliefPolicyResult`. `BeliefPolicyNode` and
  `BeliefPolicyBranch` expose the selected action, observation probabilities,
  posterior-specific continuations, expected terminal goal probability,
  expected action cost, and utility. Perfect, partial, and stochastic
  observations use the same callback surface. Explicit policy-node,
  outcome-branch, and observation-branch budgets bound the potentially
  exponential policy tree, and successful results report the corresponding
  expansion counts.
- **GAP-011 — conditional observation pruning:** conditional planning no
  longer expands observations after the final action, where no decision
  remains. New `min_observation_probability` and
  `max_observations_per_node` controls merge rare readings into an optimized
  fallback posterior instead of dropping their mass. Policy nodes route
  unmatched readings through `continuation()`, while results expose
  generated/pruned branch counts, retained/discarded probability,
  exact-versus-heuristic action ranking, and lower-bound status. Existing
  hard node and branch budgets remain available.
- **GAP-012 — approximate action certificates:** every conditional root
  evaluation now exposes utility lower and upper bounds. Approximate results
  report the unrestricted optimal-utility upper bound, maximum possible
  root-action regret, and `root_action_certified`. A pruned result uses
  `action_ranking="certified"` when its selected lower bound dominates every
  alternative upper bound; otherwise it remains explicitly `"heuristic"`.
  Bounds use the maximum remaining reward on collapsed observation branches
  and require no exact reference run.
- **GAP-013 — tracker/planner certificate composition:** `ModeTracker.belief`
  now returns a list-compatible `TrackedBelief` with exactness and
  retained-mass metadata. Conditional planning composes known retained mass
  adversarially into per-action bounds and reports certificate scope,
  policy-only certification, and end-to-end certification separately.
  Unknown tracker mass receives conservative missing-state bounds, preventing
  a certificate valid only for a normalized beam from being presented as an
  exact-posterior guarantee.
- **GAP-014 — adaptive tracker refinement:** `ModeTracker` can now retain
  successful evidence and per-step transition overrides on request.
  `refine()` reconstructs a fresh larger tracker by replaying that history,
  while `from_history()` supports explicit checkpoints. `refine_until()`
  evaluates a downstream predicate over successively larger beliefs and
  returns every attempt with its beam, expansion width, replayed steps,
  generated candidates, replay/evaluation time, retained mass, certificate
  scope, and regret.
  Exact fallback remains deliberate and respects `max_exact_states`.
- **GAP-017 — planning cooperative controls:** `plan_belief()` accepts an
  optional `PlanControl` with cooperative `timeout_seconds`/`deadline`
  limits, a cancellation callback, and progress callbacks receiving
  `PlanningStats` snapshots (elapsed time, policy-node/outcome/observation
  counts, and the best fully evaluated root action so far), mirroring
  `CompileControl` for compilation. Cancellation raises `PlanningCancelled`;
  time budgets raise `PlanningBudgetExceeded`. The existing hard
  `max_action_sequences`/`max_outcome_branches`/`max_policy_nodes`/
  `max_observation_branches` caps now also raise `PlanningBudgetExceeded`
  carrying partial statistics; it subclasses `ValueError`, so existing
  handlers keep working.
- **GAP-020 — policy-branch posterior explainability:** every retained
  `BeliefPolicyBranch` now exposes the normalized observation-conditioned
  `posterior` the planner optimized its continuation against, plus a
  `posterior_marginals()` convenience. The aggregated pruned-observation
  fallback is exposed as `BeliefPolicyNode.fallback_branch` with its
  aggregate posterior and the `contributing_observations` merged into it;
  `route()` returns that branch for fallback routing.
- **GAP-019 — chance-constrained belief planning:** `plan_belief()` accepts
  `min_goal_probability`, a whole-policy reliability floor: feasibility is
  decided first and utility ranks only the feasible candidates. Conditional
  planning enforces the constraint by Pareto-frontier dynamic programming
  over (goal probability, expected cost) pairs — a chance constraint cannot
  be enforced per node, since the reliability one observation branch must
  deliver depends on what the others deliver.
  `min_branch_goal_probability` adds the stricter per-branch safety
  variant. Infeasibility is reported, not raised: the most reliable policy
  returns with `feasible=False` and `best_achievable_goal_probability`.
  `max_frontier_points` bounds each frontier with endpoints preserved, so
  feasibility and best-achievable stay exact within the selected observation
  partition under frontier truncation, while cost-optimality can degrade
  (`constraint_optimality`). Action
  certificates gain goal-probability lower/upper bounds, and
  `constraint_certification` composes observation pruning and
  tracker-retained mass into certified-feasible, certified-infeasible, or
  indeterminate, scoped like the utility certificates. The same floor works
  on the one-step and open-loop paths.
- **GAP-025 — frontier provisioning diagnostics:** constrained conditional
  results and `PlanningStats` now expose generated and retained frontier
  points, largest pre-cap frontier, truncated-node count, saturation by
  depth and root action, and a conservative feasible-utility upper bound and
  optimality gap. Callers can measure progress as `max_frontier_points`
  increases without weakening exact feasibility endpoints.
- **GAP-027 — scenario-robust outcome planning:** `plan_belief()` now accepts
  named `outcome_scenarios` with `robust_objective="maximin"`. A bounded,
  positive scenario-weight grid generates common executable policy trees,
  which are independently audited under every scenario. Results expose the
  candidate portfolio, per-scenario probability/cost/utility, worst-case
  metrics, robust feasibility, and the explicit
  `weight-grid-heuristic`/`robust-scenario-weight-grid` scope.
- **GAP-030 — robust branch goal constraints:** scenario-robust planning now
  accepts `min_branch_goal_probability`. Candidate audits expose per-scenario
  minimum branch probability and observation path, the limiting scenario,
  robust branch feasibility/certification, and the best minimum branch floor
  generated by the bounded weight grid.

### Documentation and examples

- Documented compilation controls, detailed planner results, and exact
  tracking in the README and API docstrings.
- Updated the recovery-planner example to display its cost decomposition.
- Updated the service-reliability example to request exact tracking without
  manually calculating beam capacity.
- Updated the microgrid belief-control experiment to use `plan_belief`
  instead of its application-local expected-utility loop.
- Updated the microgrid receding-horizon controller and policy trial to use
  two-step belief lookahead for prerequisite recovery actions.
- Updated the conditional microgrid experiment to validate the public policy
  tree against its independent exhaustive evaluator.
- Updated the noisy-observation scaling experiment and conditional
  controller to use explicit observation pruning with fallback behavior.
- Updated the pruning-tradeoff experiment to compare retrospective exact
  regret with the public worst-case regret certificate.
- Updated the tracker-certificate experiment to distinguish policy-only and
  end-to-end root-action guarantees.
- Updated the microgrid controller and tracker-refinement experiment to
  retain history, replay an uncertified beam at explicit exact capacity, and
  expose the resulting computational work and certificate transition.
- Documented the tracked-belief copy/mutation contract, the strict callback
  probability contract, `PlanControl` cooperative planning controls, policy
  observation routing, and branch posteriors in the README and docstrings.
- Updated the belief-metadata, validation-edge, operational-control,
  observation-routing, and branch-explainability experiments and their lab
  reports to validate the fixed behavior.
- Updated the microgrid risk-constraint experiment to exercise the public
  `min_goal_probability` chance constraint, including infeasibility
  reporting for an unreachable floor.
- Updated the mutation, heterogeneous-observation, branch-constraint,
  branch-pruning, and islanded-frontier experiments to validate GAP-021
  through GAP-025 and report the new diagnostic surface.
- Updated the best-achievable counterexample and adaptive island controller
  to validate scoped reliability ceilings and bounded scenario-robust
  planning.
- Updated the robust metric-scope, branch-safety, terminal-cost, and regime
  phase-diagram experiments to validate GAP-029 through GAP-031.
- Updated the black-box edge-case experiment and archived GAP-001 through
  GAP-031 under `modenexus_apps/gaps/fixed`.

### Verification

- ModeNexus test suite: **388 passed, 1 skipped**.
- Black-box API edge-case checks: all fixed cases report `OK`.
- Companion applications: all **70 tests passed** in bounded partitions.
  The 62-test non-islanded pass completed with one stale tied-scenario label;
  after making that assertion tie-safe, its 61 other tests and the corrected
  native audit all passed. The adaptive controller passed in 93.24 seconds,
  and the final seven islanded/regime tests passed in 173.12 seconds.
