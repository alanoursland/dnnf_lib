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
- Updated the black-box edge-case experiment and archived GAP-001 through
  GAP-010 under `modenexus_apps/gaps/fixed`.

### Verification

- ModeNexus test suite: **307 passed, 7 skipped**.
- Black-box API edge-case checks: all fixed cases report `OK`.
- Companion applications: all **33 non-PyTorch tests passed**. The optional
  PyTorch verification was not run because PyTorch was unavailable in the
  verification environment.
