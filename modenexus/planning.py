"""Unified planning and estimation on one compiled circuit (MEXEC-style).

Declare a transition system — modes (with priors), per-step commands,
observables with behavior constraints, and command-conditioned
transitions with neg-log costs — and compile an n-step unrolling into a
single FD circuit.  The same compiled structure answers, by tropical
evaluation with different leaves clamped (Barrett 2005; Darwiche &
Marquis 2004):

* **mode estimation** — clamp observed sensor values (and known
  commands) over the first k steps; the min-cost model's mode variables
  are the most likely trajectory;
* **reconfiguration planning** — clamp current modes at step 0 and
  target modes at step n; the min-cost model's command variables are
  the most probable plan;
* **planning under observations** — clamp observations instead of (or
  in addition to) the current mode: the plan is computed from what the
  sensors say the state is, without a separate estimation pass.

v1 simplifications: transition preconditions are command values;
``noop`` (persistence) always available at cost 0; observables are
exact (wrap noisy sensing in the behavior constraints if needed).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from . import fd
from .compile_control import CompileControl
from .diagnosis import SystemModel, _normalize_categorical_weights
from .formula import Formula


@dataclass
class _Transition:
    mode: str
    frm: str
    to: str
    command: Optional[Tuple[str, object]]
    cost: float


@dataclass(frozen=True)
class TransitionCost:
    """One selected per-mode transition and its cost contribution."""

    step: int
    mode: str
    transition: str
    frm: object
    to: object
    command: Optional[Tuple[str, object]]
    cost: float


@dataclass(frozen=True)
class PlannerCostBreakdown:
    """Auditable decomposition of a planner's tropical objective."""

    initial_state_cost: float
    transition_costs: Tuple[TransitionCost, ...]
    evidence_cost: float
    other_model_cost: float
    total_cost: float

    @property
    def action_cost(self) -> float:
        return sum(item.cost for item in self.transition_costs)


@dataclass(frozen=True)
class PlanResult:
    total_cost: float
    commands: List[Dict[str, object]]
    trajectory: List[Dict[str, object]]
    costs: PlannerCostBreakdown


@dataclass(frozen=True)
class EstimateResult:
    total_cost: float
    trajectory: List[Dict[str, object]]
    commands: List[Dict[str, object]]
    costs: PlannerCostBreakdown


@dataclass(frozen=True)
class BeliefActionEvaluation:
    """Expected one-step value of one command under a joint belief."""

    action: Dict[str, object]
    expected_goal_probability: float
    action_cost: float
    expected_utility: float


@dataclass(frozen=True)
class BeliefPlanResult:
    """Best one-step action and all evaluated alternatives."""

    action: Dict[str, object]
    expected_goal_probability: float
    action_cost: float
    expected_utility: float
    evaluations: Tuple[BeliefActionEvaluation, ...]

    @property
    def commands(self) -> List[Dict[str, object]]:
        """Single-command list for symmetry with :class:`PlanResult`."""
        return [dict(self.action)]


@dataclass(frozen=True)
class BeliefSequenceEvaluation:
    """Expected terminal value of one open-loop command sequence."""

    commands: Tuple[Dict[str, object], ...]
    expected_goal_probability: float
    action_cost: float
    expected_utility: float
    expected_goal_probabilities: Tuple[float, ...]


@dataclass(frozen=True)
class BeliefPolicyResult:
    """Best bounded-lookahead sequence under a joint initial belief.

    ``observation_branching`` is false for this initial multi-step surface:
    stochastic outcomes are integrated exactly, but future commands do not
    branch on observations.
    """

    commands: Tuple[Dict[str, object], ...]
    expected_goal_probability: float
    action_cost: float
    expected_utility: float
    expected_goal_probabilities: Tuple[float, ...]
    evaluations: Tuple[BeliefSequenceEvaluation, ...]
    observation_branching: bool = False

    @property
    def action(self) -> Dict[str, object]:
        """First action, suitable for receding-horizon execution."""
        return dict(self.commands[0])


class Planner:
    """Build a transition system, then ``compile(horizon)``."""

    def __init__(self) -> None:
        self._modes: Dict[str, Tuple[Sequence, Optional[Sequence[float]]]] = {}
        self._commands: Dict[str, Sequence] = {}
        self._observables: Dict[str, Sequence] = {}
        self._behaviors: List[Callable[[Dict[str, object]], Formula]] = []
        self._transitions: List[_Transition] = []

    def mode(
        self, name: str, values: Sequence,
        priors: Optional[Sequence[float]] = None,
    ) -> None:
        """A state variable; ``priors`` (if given) weight step 0 —
        the initial-mode likelihood used by estimation.  A mode with no
        declared transition rules is static and persists across every step."""
        values = tuple(values)
        if priors is not None:
            priors = tuple(_normalize_categorical_weights(name, values, priors))
        self._modes[name] = (values, priors)

    def command(self, name: str, values: Sequence) -> None:
        """A per-step command input; include an inert value (e.g.
        ``"none"``) if doing nothing must be expressible."""
        self._commands[name] = tuple(values)

    def observable(self, name: str, values: Sequence = (False, True)) -> None:
        """A per-step sensed value, constrained via :meth:`behavior`."""
        self._observables[name] = tuple(values)

    def behavior(
        self, fn: Callable[[Dict[str, object]], Formula]
    ) -> None:
        """A constraint holding at every step.  ``fn`` receives a dict of
        this step's variables by name (modes and observables at every
        step; commands only at non-final steps, where they appear under
        their names as well) and returns a formula, e.g.::

            p.behavior(lambda v: iff(v["valid"] == True, v["sw"] == "Tracking"))
        """
        self._behaviors.append(fn)

    def transition(
        self,
        mode: str,
        frm: str,
        to: str,
        command: Optional[Tuple[str, object]] = None,
        cost: float = 0.0,
    ) -> None:
        """Allow ``mode`` to move ``frm -> to`` on steps where the command
        condition holds; ``cost`` is neg-log likelihood."""
        self._transitions.append(_Transition(mode, frm, to, command, cost))

    # ------------------------------------------------------------------
    def compile(
        self, horizon: int, control: Optional[CompileControl] = None
    ) -> "CompiledPlanner":
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        m = SystemModel()
        step_vars: List[Dict[str, object]] = [dict() for _ in range(horizon + 1)]
        spec = m.cnf.spec
        for t in range(horizon + 1):
            for name, (values, priors) in self._modes.items():
                v = m.finite(f"{name}@{t}", values)
                step_vars[t][name] = v
                if t == 0 and priors is not None:
                    for i, p in enumerate(priors):
                        m._prior_weights[spec.mvlit(v.fd_var, i)] = p
            for name, values in self._observables.items():
                step_vars[t][name] = m.finite(f"{name}@{t}", values)
        for t in range(horizon):
            for name, values in self._commands.items():
                step_vars[t][name] = m.finite(f"{name}@{t}", values)

        for t in range(horizon + 1):
            for fn in self._behaviors:
                m.add(fn(dict(step_vars[t])))

        for t in range(horizon):
            for name, (values, _) in self._modes.items():
                trans = [tr for tr in self._transitions if tr.mode == name]
                cur, nxt = step_vars[t][name], step_vars[t + 1][name]
                if not trans:
                    # A mode with no transition rules is static.  Lower
                    # persistence directly instead of creating a one-value
                    # selector, which is not a valid finite-domain variable.
                    for value in values:
                        m.add((cur != value) | (nxt == value))
                    continue
                labels = ["noop"] + [f"t{i}" for i in range(len(trans))]
                tv = m.finite(f"_{name}@{t}#trans", labels)
                for i, tr in enumerate(trans):
                    w = math.exp(-tr.cost)
                    if w != 1.0:
                        m._prior_weights[spec.mvlit(tv.fd_var, i + 1)] = w
                for v in values:  # noop: mode persists
                    m.add((tv == "noop") >> ((cur != v) | (nxt == v)))
                for i, tr in enumerate(trans):
                    effect = (cur == tr.frm) & (nxt == tr.to)
                    if tr.command is not None:
                        cname, cval = tr.command
                        effect = effect & (step_vars[t][cname] == cval)
                    m.add((tv == f"t{i}") >> effect)
        system = m.compile(modes_first=False, control=control)
        return CompiledPlanner(
            system, list(self._modes), list(self._commands),
            list(self._observables), horizon, list(self._transitions),
        )


class CompiledPlanner:
    """One compiled circuit; estimation and planning are evaluations."""

    def __init__(
        self, system, mode_names, command_names, obs_names, horizon, transitions
    ):
        self.system = system
        self.mode_names = mode_names
        self.command_names = command_names
        self.obs_names = obs_names
        self.horizon = horizon
        self.transitions = transitions

    # -- shared machinery ----------------------------------------------
    def _solve(self, evidence: Dict[str, object]):
        costs = self.system._conditioned_costs(evidence)
        return fd.mpe(self.system.circuit, costs)

    def _step_evidence(
        self, per_step: Optional[Sequence[Dict[str, object]]]
    ) -> Dict[str, object]:
        evidence: Dict[str, object] = {}
        if per_step:
            if len(per_step) > self.horizon + 1:
                raise ValueError(
                    f"{len(per_step)} steps of observations exceed "
                    f"horizon {self.horizon}"
                )
            for t, obs in enumerate(per_step):
                for name, value in obs.items():
                    evidence[f"{name}@{t}"] = value
        return evidence

    def _decode(self, assignment, names, t) -> Dict[str, object]:
        out = {}
        for name in names:
            var = self.system.vars[f"{name}@{t}"]
            out[name] = var.values[assignment[var.fd_var]]
        return out

    def _trajectory(self, assignment, steps: int) -> List[Dict[str, object]]:
        return [
            self._decode(assignment, self.mode_names, t)
            for t in range(steps)
        ]

    def _commands(self, assignment, steps: int) -> List[Dict[str, object]]:
        return [
            self._decode(assignment, self.command_names, t)
            for t in range(steps)
        ]

    def _cost_breakdown(
        self, total_cost: float, assignment
    ) -> PlannerCostBreakdown:
        spec = self.system.circuit.spec
        initial_state_cost = 0.0
        for name in self.mode_names:
            var = self.system.vars[f"{name}@0"]
            initial_state_cost += self.system._costs[
                spec.mvlit(var.fd_var, assignment[var.fd_var])
            ]

        selected: List[TransitionCost] = []
        by_mode = {
            name: [tr for tr in self.transitions if tr.mode == name]
            for name in self.mode_names
        }
        for step in range(self.horizon):
            for name in self.mode_names:
                selector_name = f"_{name}@{step}#trans"
                if selector_name not in self.system.vars:
                    current = self._decode(assignment, [name], step)[name]
                    following = self._decode(
                        assignment, [name], step + 1
                    )[name]
                    selected.append(
                        TransitionCost(
                            step,
                            name,
                            "noop",
                            current,
                            following,
                            None,
                            0.0,
                        )
                    )
                    continue
                selector = self.system.vars[selector_name]
                selected_index = assignment[selector.fd_var]
                label = selector.values[selected_index]
                contribution = self.system._costs[
                    spec.mvlit(selector.fd_var, selected_index)
                ]
                if label == "noop":
                    current = self._decode(assignment, [name], step)[name]
                    following = self._decode(
                        assignment, [name], step + 1
                    )[name]
                    item = TransitionCost(
                        step, name, "noop", current, following, None,
                        contribution,
                    )
                else:
                    transition = by_mode[name][int(str(label)[1:])]
                    item = TransitionCost(
                        step,
                        name,
                        str(label),
                        transition.frm,
                        transition.to,
                        transition.command,
                        contribution,
                    )
                selected.append(item)

        action_cost = sum(item.cost for item in selected)
        other_model_cost = total_cost - initial_state_cost - action_cost
        if abs(other_model_cost) < 1e-12:
            other_model_cost = 0.0
        return PlannerCostBreakdown(
            initial_state_cost=initial_state_cost,
            transition_costs=tuple(selected),
            evidence_cost=0.0,
            other_model_cost=other_model_cost,
            total_cost=total_cost,
        )

    # -- queries ---------------------------------------------------------
    def plan(
        self,
        current: Optional[Dict[str, object]] = None,
        target: Dict[str, object] = None,
        observations: Optional[Sequence[Dict[str, object]]] = None,
    ) -> Optional[Tuple[float, List[Dict[str, object]]]]:
        """Most probable command sequence reaching ``target`` within the
        horizon, or None if unreachable.  ``current`` (mode clamp at
        step 0) is optional when ``observations`` determine the state —
        planning directly from sensor readings, no separate estimation
        pass.  Returns ``(cost, [per-step {command: value}])``.  The cost is
        the selected normalized initial-state neg-log prior plus transition
        costs; use :meth:`plan_detailed` for the decomposition."""
        result = self.plan_detailed(current, target, observations)
        if result is None:
            return None
        return result.total_cost, result.commands

    def plan_detailed(
        self,
        current: Optional[Dict[str, object]] = None,
        target: Dict[str, object] = None,
        observations: Optional[Sequence[Dict[str, object]]] = None,
    ) -> Optional[PlanResult]:
        """Return a plan with its state trajectory and cost decomposition.

        Hard evidence contributes zero finite cost; it removes inconsistent
        assignments.  ``initial_state_cost`` is the selected normalized
        step-0 prior, and each selected transition contributes its declared
        neg-log cost.  ``other_model_cost`` makes any future weighted model
        terms explicit instead of silently folding them into the total.
        """
        evidence = self._step_evidence(observations)
        if current:
            for name, value in current.items():
                evidence[f"{name}@0"] = value
        for name, value in (target or {}).items():
            evidence[f"{name}@{self.horizon}"] = value
        cost, assignment = self._solve(evidence)
        if assignment is None:
            return None
        return PlanResult(
            total_cost=cost,
            commands=self._commands(assignment, self.horizon),
            trajectory=self._trajectory(assignment, self.horizon + 1),
            costs=self._cost_breakdown(cost, assignment),
        )

    def plan_belief(
        self,
        belief: Sequence[Tuple[Mapping[str, object], float]],
        target: Mapping[str, object],
        *,
        action_costs: Optional[
            Mapping[object, float] | Callable[[Dict[str, object]], float]
        ] = None,
        outcome_model: Optional[
            Callable[
                [Dict[str, object], Dict[str, object]],
                Sequence[Tuple[Mapping[str, object], float]],
            ]
        ] = None,
        actions: Optional[Sequence[object]] = None,
        goal_reward: float = 1.0,
        cost_weight: float = 1.0,
        max_action_sequences: int = 100_000,
        max_outcome_branches: int = 100_000,
    ) -> BeliefPlanResult | BeliefPolicyResult:
        """Choose one action by exact expectation over a correlated belief.

        With ``horizon=1`` this returns :class:`BeliefPlanResult`.  Longer
        horizons perform bounded open-loop lookahead and return
        :class:`BeliefPolicyResult`: stochastic outcomes are integrated
        exactly, but the selected future commands do not branch on future
        observations.  Execute its first ``action`` and replan after the next
        observation for receding-horizon control.

        ``belief`` has the shape returned by
        :meth:`modenexus.ModeTracker.belief`: ``[(joint_state, mass), ...]``.
        Masses are validated and normalized; correlations between modes are
        preserved.  State keys not used by this planner are ignored, and
        omitted planner modes remain latent.

        By default, transition-selector weights in the compiled planner
        define ``P(target at step 1 | state, action)``.  ``outcome_model`` can
        instead supply explicit stochastic outcomes as
        ``[(next_state_updates, probability), ...]`` for each state/action.
        Updates may be partial; unspecified modes persist from the belief
        state.  This separates physical success probabilities from action
        costs when planner transition costs represent operational effort.

        Utility is ``goal_reward * P(target) - cost_weight * action_cost``.
        For a single command variable, ``action_costs`` is either a mapping
        from command values to costs or a callable receiving the command
        dictionary.  Evaluations are returned best-first.  The explicit
        sequence and outcome-branch limits make exponential lookahead
        inspectable and provisionable.
        """
        if len(self.command_names) != 1:
            raise NotImplementedError(
                "plan_belief currently supports exactly one command variable"
            )
        if not math.isfinite(goal_reward) or goal_reward < 0.0:
            raise ValueError("goal_reward must be finite and non-negative")
        if not math.isfinite(cost_weight) or cost_weight < 0.0:
            raise ValueError("cost_weight must be finite and non-negative")
        if max_action_sequences < 1:
            raise ValueError("max_action_sequences must be at least 1")
        if max_outcome_branches < 1:
            raise ValueError("max_outcome_branches must be at least 1")
        if not target:
            raise ValueError("target must not be empty")
        valid_targets = set(self.mode_names) | set(self.obs_names)
        unknown_targets = set(target) - valid_targets
        if unknown_targets:
            raise KeyError(
                f"unknown planner target variables: {sorted(unknown_targets)}"
            )

        weighted_states = list(belief)
        if not weighted_states:
            raise ValueError("belief must not be empty")
        total_mass = 0.0
        checked_states: List[Tuple[Dict[str, object], float]] = []
        for index, (state, mass) in enumerate(weighted_states):
            if not isinstance(state, Mapping):
                raise ValueError(f"belief state {index} is not a mapping")
            try:
                numeric_mass = float(mass)
            except (TypeError, ValueError):
                raise ValueError(
                    f"belief mass at index {index} must be finite and "
                    f"non-negative; got {mass!r}"
                ) from None
            if not math.isfinite(numeric_mass) or numeric_mass < 0.0:
                raise ValueError(
                    f"belief mass at index {index} must be finite and "
                    f"non-negative; got {mass!r}"
                )
            state_dict = dict(state)
            if self.mode_names and not any(
                name in state_dict for name in self.mode_names
            ):
                raise ValueError(
                    f"belief state {index} contains no modes used by planner"
                )
            checked_states.append((state_dict, numeric_mass))
            total_mass += numeric_mass
        if not math.isfinite(total_mass) or total_mass <= 0.0:
            raise ValueError("belief must have positive finite total mass")
        checked_states = [
            (state, mass / total_mass) for state, mass in checked_states
        ]

        command_name = self.command_names[0]
        command_var = self.system.vars[f"{command_name}@0"]
        action_values = list(command_var.values if actions is None else actions)
        if not action_values:
            raise ValueError("actions must not be empty")
        for action in action_values:
            command_var._index(action)

        def step_evidence(
            state: Mapping[str, object],
            step: int,
            *,
            include_observables: bool = False,
        ) -> Dict[str, object]:
            names = set(self.mode_names)
            if include_observables:
                names.update(self.obs_names)
            return {
                f"{name}@{step}": value
                for name, value in state.items()
                if name in names
            }

        def conditional_goal_probability(
            evidence: Dict[str, object],
            target_step: Optional[int] = None,
        ) -> float:
            if target_step is None:
                target_step = self.horizon
            log_denominator = self.system.log_evidence(evidence)
            if log_denominator == -math.inf:
                raise ValueError(
                    f"belief/action outcome is inconsistent with planner: "
                    f"{evidence}"
                )
            with_goal = dict(evidence)
            with_goal.update(
                {
                    f"{name}@{target_step}": value
                    for name, value in target.items()
                }
            )
            log_numerator = self.system.log_evidence(with_goal)
            if log_numerator == -math.inf:
                return 0.0
            return min(1.0, math.exp(log_numerator - log_denominator))

        def action_cost(action: object, command: Dict[str, object]) -> float:
            if action_costs is None:
                value = 0.0
            elif callable(action_costs):
                value = action_costs(command)
            else:
                if action not in action_costs:
                    raise KeyError(f"missing action cost for {action!r}")
                value = action_costs[action]
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                raise ValueError(
                    f"action cost for {action!r} must be finite and "
                    f"non-negative; got {value!r}"
                ) from None
            if not math.isfinite(numeric) or numeric < 0.0:
                raise ValueError(
                    f"action cost for {action!r} must be finite and "
                    f"non-negative; got {value!r}"
                )
            return numeric

        def normalized_outcomes(
            state: Dict[str, object], command: Dict[str, object]
        ) -> List[Tuple[Dict[str, object], float]]:
            if outcome_model is None:
                raise RuntimeError("normalized_outcomes needs outcome_model")
            outcomes = list(outcome_model(dict(state), dict(command)))
            if not outcomes:
                raise ValueError(
                    f"outcome_model returned no outcomes for "
                    f"state={state}, action={command}"
                )
            checked = []
            total = 0.0
            for outcome, probability in outcomes:
                try:
                    numeric_probability = float(probability)
                except (TypeError, ValueError):
                    raise ValueError(
                        "outcome probabilities must be finite and "
                        f"non-negative; got {probability!r}"
                    ) from None
                if (
                    not math.isfinite(numeric_probability)
                    or numeric_probability < 0.0
                ):
                    raise ValueError(
                        "outcome probabilities must be finite and "
                        f"non-negative; got {probability!r}"
                    )
                checked.append((dict(outcome), numeric_probability))
                total += numeric_probability
            if not math.isfinite(total) or total <= 0:
                raise ValueError(
                    "outcome probabilities must have positive finite "
                    "total mass"
                )
            return [
                (outcome, probability / total)
                for outcome, probability in checked
                if probability > 0.0
            ]

        if self.horizon > 1:
            sequence_count = len(action_values) ** self.horizon
            if sequence_count > max_action_sequences:
                raise ValueError(
                    f"belief lookahead requires {sequence_count} action "
                    f"sequences, exceeding "
                    f"max_action_sequences={max_action_sequences}"
                )
            sequence_evaluations: List[BeliefSequenceEvaluation] = []
            for action_sequence in product(
                action_values, repeat=self.horizon
            ):
                commands = tuple(
                    {command_name: action} for action in action_sequence
                )
                total_cost = sum(
                    action_cost(action, command)
                    for action, command in zip(action_sequence, commands)
                )
                if outcome_model is None:
                    expected_by_step = [0.0] * self.horizon
                    for state, state_mass in checked_states:
                        evidence = step_evidence(state, 0)
                        for step, action in enumerate(action_sequence):
                            evidence[f"{command_name}@{step}"] = action
                            expected_by_step[step] += (
                                state_mass
                                * conditional_goal_probability(
                                    evidence, step + 1
                                )
                            )
                else:
                    # Branches retain path evidence so outcomes are checked
                    # against the compiled transition relation at every step.
                    branches = [
                        (dict(state), mass, step_evidence(state, 0))
                        for state, mass in checked_states
                        if mass > 0.0
                    ]
                    expected_by_step = []
                    for step, command in enumerate(commands):
                        expanded = []
                        for state, branch_mass, path_evidence in branches:
                            base = dict(path_evidence)
                            base[f"{command_name}@{step}"] = command[
                                command_name
                            ]
                            for updates, probability in normalized_outcomes(
                                state, command
                            ):
                                next_state = dict(state)
                                next_state.update(updates)
                                outcome_evidence = dict(base)
                                outcome_evidence.update(
                                    step_evidence(
                                        next_state,
                                        step + 1,
                                    )
                                )
                                outcome_evidence.update(
                                    {
                                        f"{name}@{step + 1}": value
                                        for name, value in updates.items()
                                        if name in self.obs_names
                                    }
                                )
                                # Validate the supplied physical outcome
                                # against the planner's hard model now,
                                # before terminal utility is evaluated.
                                if (
                                    self.system.log_evidence(outcome_evidence)
                                    == -math.inf
                                ):
                                    raise ValueError(
                                        "outcome_model produced a state "
                                        "inconsistent with planner: "
                                        f"{outcome_evidence}"
                                    )
                                expanded.append(
                                    (
                                        next_state,
                                        branch_mass * probability,
                                        outcome_evidence,
                                    )
                                )
                                if len(expanded) > max_outcome_branches:
                                    raise ValueError(
                                        "belief lookahead exceeded "
                                        f"max_outcome_branches="
                                        f"{max_outcome_branches}"
                                    )
                        branches = expanded
                        expected_by_step.append(
                            sum(
                                branch_mass
                                * conditional_goal_probability(
                                    path_evidence, step + 1
                                )
                                for _, branch_mass, path_evidence in branches
                            )
                        )
                expected_goal = expected_by_step[-1]
                utility = (
                    goal_reward * expected_goal
                    - cost_weight * total_cost
                )
                sequence_evaluations.append(
                    BeliefSequenceEvaluation(
                        commands=commands,
                        expected_goal_probability=expected_goal,
                        action_cost=total_cost,
                        expected_utility=utility,
                        expected_goal_probabilities=tuple(expected_by_step),
                    )
                )
            sequence_evaluations.sort(
                key=lambda item: (
                    -item.expected_utility,
                    -sum(item.expected_goal_probabilities),
                )
            )
            best_sequence = sequence_evaluations[0]
            return BeliefPolicyResult(
                commands=tuple(
                    dict(command) for command in best_sequence.commands
                ),
                expected_goal_probability=(
                    best_sequence.expected_goal_probability
                ),
                action_cost=best_sequence.action_cost,
                expected_utility=best_sequence.expected_utility,
                expected_goal_probabilities=(
                    best_sequence.expected_goal_probabilities
                ),
                evaluations=tuple(sequence_evaluations),
            )

        evaluations: List[BeliefActionEvaluation] = []
        for action in action_values:
            command = {command_name: action}
            expected_goal = 0.0
            for state, state_mass in checked_states:
                base = step_evidence(state, 0)
                base[f"{command_name}@0"] = action
                if outcome_model is None:
                    state_goal = conditional_goal_probability(base)
                else:
                    outcomes = list(outcome_model(dict(state), dict(command)))
                    if not outcomes:
                        raise ValueError(
                            f"outcome_model returned no outcomes for "
                            f"state={state}, action={command}"
                        )
                    checked_outcomes = []
                    outcome_total = 0.0
                    for outcome, probability in outcomes:
                        try:
                            numeric_probability = float(probability)
                        except (TypeError, ValueError):
                            raise ValueError(
                                "outcome probabilities must be finite and "
                                f"non-negative; got {probability!r}"
                            ) from None
                        if (
                            not math.isfinite(numeric_probability)
                            or numeric_probability < 0.0
                        ):
                            raise ValueError(
                                "outcome probabilities must be finite and "
                                f"non-negative; got {probability!r}"
                            )
                        checked_outcomes.append(
                            (dict(outcome), numeric_probability)
                        )
                        outcome_total += numeric_probability
                    if not math.isfinite(outcome_total) or outcome_total <= 0:
                        raise ValueError(
                            "outcome probabilities must have positive "
                            "finite total mass"
                        )
                    state_goal = 0.0
                    for updates, probability in checked_outcomes:
                        next_state = {
                            name: value for name, value in state.items()
                            if name in self.mode_names
                        }
                        next_state.update(
                            {
                                name: value for name, value in updates.items()
                                if name in valid_targets
                            }
                        )
                        outcome_evidence = dict(base)
                        outcome_evidence.update(
                            step_evidence(
                                next_state, 1, include_observables=True
                            )
                        )
                        state_goal += (
                            probability
                            / outcome_total
                            * conditional_goal_probability(outcome_evidence)
                        )
                expected_goal += state_mass * state_goal
            cost = action_cost(action, command)
            utility = goal_reward * expected_goal - cost_weight * cost
            evaluations.append(
                BeliefActionEvaluation(
                    action=command,
                    expected_goal_probability=expected_goal,
                    action_cost=cost,
                    expected_utility=utility,
                )
            )

        evaluations.sort(key=lambda item: -item.expected_utility)
        best = evaluations[0]
        return BeliefPlanResult(
            action=dict(best.action),
            expected_goal_probability=best.expected_goal_probability,
            action_cost=best.action_cost,
            expected_utility=best.expected_utility,
            evaluations=tuple(evaluations),
        )

    def estimate(
        self,
        observations: Sequence[Dict[str, object]],
        commands: Optional[Sequence[Dict[str, object]]] = None,
    ) -> Optional[Tuple[float, List[Dict[str, object]]]]:
        """MEXEC mode estimation: given k steps of observations (and,
        optionally, the commands that were issued), return the most
        likely mode trajectory ``(cost, [per-step {mode: value}])`` over
        those steps, or None if inconsistent.  Use :meth:`estimate_detailed`
        for the trajectory's cost decomposition."""
        result = self.estimate_detailed(observations, commands)
        if result is None:
            return None
        return result.total_cost, result.trajectory

    def estimate_detailed(
        self,
        observations: Sequence[Dict[str, object]],
        commands: Optional[Sequence[Dict[str, object]]] = None,
    ) -> Optional[EstimateResult]:
        """Return an estimated trajectory with an auditable cost breakdown."""
        evidence = self._step_evidence(observations)
        if commands:
            for t, cmd in enumerate(commands):
                for name, value in cmd.items():
                    evidence[f"{name}@{t}"] = value
        cost, assignment = self._solve(evidence)
        if assignment is None:
            return None
        command_steps = min(self.horizon, max(0, len(observations) - 1))
        return EstimateResult(
            total_cost=cost,
            trajectory=self._trajectory(assignment, len(observations)),
            commands=self._commands(assignment, command_steps),
            costs=self._cost_breakdown(cost, assignment),
        )
