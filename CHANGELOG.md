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

### Documentation and examples

- Documented compilation controls, detailed planner results, and exact
  tracking in the README and API docstrings.
- Updated the recovery-planner example to display its cost decomposition.
- Updated the service-reliability example to request exact tracking without
  manually calculating beam capacity.
- Updated the black-box edge-case experiment and archived GAP-001 through
  GAP-006 under `modenexus_apps/gaps/fixed`.

### Verification

- ModeNexus test suite: **295 passed, 7 skipped**.
- Black-box API edge-case checks: all fixed cases report `OK`.
- Companion applications: all eight non-PyTorch tests passed. The optional
  PyTorch verification was not run because PyTorch was unavailable in the
  verification environment.
