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

    ``observation_branching`` is false for this open-loop result: stochastic
    outcomes are integrated exactly, but future commands do not branch on
    observations. Supply an observation model to receive a
    :class:`ConditionalBeliefPolicyResult`.
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


@dataclass(frozen=True)
class BeliefPolicyBranch:
    """One observation edge in a conditional belief-policy tree."""

    observation: Dict[str, object]
    probability: float
    expected_goal_probability: float
    expected_action_cost: float
    expected_utility: float
    policy: Optional["BeliefPolicyNode"]


@dataclass(frozen=True)
class BeliefPolicyNode:
    """One action and its observation-contingent continuations.

    ``fallback_policy`` receives observations omitted by threshold or top-k
    pruning. ``retained_observation_probability`` includes all later levels;
    ``discarded_observation_probability`` is local to this node.
    """

    action: Dict[str, object]
    immediate_action_cost: float
    expected_goal_probability: float
    expected_action_cost: float
    expected_utility: float
    utility_upper_bound: float
    branches: Tuple[BeliefPolicyBranch, ...]
    fallback_policy: Optional["BeliefPolicyNode"]
    retained_observation_probability: float
    discarded_observation_probability: float

    def continuation(
        self, observation: Mapping[str, object]
    ) -> Optional["BeliefPolicyNode"]:
        """Return the matching continuation or the pruned-observation fallback."""
        observed = dict(observation)
        for branch in self.branches:
            if branch.observation == observed:
                return branch.policy
        return self.fallback_policy

    @property
    def utility_lower_bound(self) -> float:
        """Value of this executable policy under its observation fallback."""
        return self.expected_utility


@dataclass(frozen=True)
class BeliefActionCertificate:
    """Policy-only and belief-composed utility bounds for one root action."""

    action: Dict[str, object]
    policy_utility_lower_bound: float
    policy_utility_upper_bound: float
    utility_lower_bound: float
    utility_upper_bound: float


@dataclass(frozen=True)
class ConditionalBeliefPolicyResult:
    """Best bounded policy tree and all alternative root actions.

    Approximate results preserve all probability mass by merging pruned
    observations into fallback posteriors. Their utility is the exact value
    of that coarsened policy and a lower bound on the unrestricted
    full-observation optimum. Per-action upper bounds use the maximum
    remaining reward on collapsed branches. Root ranking is certified when
    the selected lower bound dominates every alternative upper bound.
    ``action_certificates`` additionally compose tracker-retained mass; the
    ``policy_*`` fields preserve the certificate scoped only to the supplied
    normalized belief.
    """

    policy: BeliefPolicyNode
    evaluations: Tuple[BeliefPolicyNode, ...]
    policy_node_count: int
    outcome_branch_count: int
    observation_branch_count: int
    generated_observation_branch_count: int
    pruned_observation_branch_count: int
    retained_observation_probability: float
    approximation: str
    action_ranking: str
    utility_is_lower_bound: bool
    optimal_utility_upper_bound: float
    maximum_regret: float
    root_action_certified: bool
    action_certificates: Tuple[BeliefActionCertificate, ...]
    policy_maximum_regret: float
    policy_root_action_certified: bool
    belief_exact: Optional[bool]
    belief_retained_probability_mass: Optional[float]
    certificate_scope: str
    observation_branching: bool = True

    @property
    def action(self) -> Dict[str, object]:
        """Root action selected after valuing conditional continuations."""
        return dict(self.policy.action)

    @property
    def expected_goal_probability(self) -> float:
        return self.policy.expected_goal_probability

    @property
    def action_cost(self) -> float:
        """Expected total action cost under the selected policy."""
        return self.policy.expected_action_cost

    @property
    def expected_utility(self) -> float:
        return self.policy.expected_utility

    @property
    def utility_lower_bound(self) -> float:
        return self.action_certificates[0].utility_lower_bound

    @property
    def utility_upper_bound(self) -> float:
        """Belief-composed upper bound for the selected root action."""
        return self.action_certificates[0].utility_upper_bound

    @property
    def discarded_observation_probability(self) -> float:
        """Selected-policy probability routed through a fallback branch."""
        return max(0.0, 1.0 - self.retained_observation_probability)


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
        observation_model: Optional[
            Callable[
                [Dict[str, object], Dict[str, object]],
                Mapping[str, object]
                | Sequence[Tuple[Mapping[str, object], float]],
            ]
        ] = None,
        actions: Optional[Sequence[object]] = None,
        goal_reward: float = 1.0,
        cost_weight: float = 1.0,
        max_action_sequences: int = 100_000,
        max_outcome_branches: int = 100_000,
        max_policy_nodes: int = 100_000,
        max_observation_branches: int = 100_000,
        min_observation_probability: float = 0.0,
        max_observations_per_node: Optional[int] = None,
    ) -> (
        BeliefPlanResult
        | BeliefPolicyResult
        | ConditionalBeliefPolicyResult
    ):
        """Choose one action by exact expectation over a correlated belief.

        With ``horizon=1`` this returns :class:`BeliefPlanResult`.  Longer
        horizons perform bounded lookahead.  Without ``observation_model``
        they return an open-loop :class:`BeliefPolicyResult`.  With an
        observation callback they return
        :class:`ConditionalBeliefPolicyResult`, whose future actions branch
        on the observations produced after each stochastic outcome.

        ``belief`` has the shape returned by
        :meth:`modenexus.ModeTracker.belief`: ``[(joint_state, mass), ...]``.
        Masses are validated and normalized; correlations between modes are
        preserved.  State keys not used by this planner are ignored, and
        omitted planner modes remain latent. Tracker beliefs also carry
        exactness and retained-mass metadata. Conditional-policy certificates
        compose that metadata adversarially; unknown tracker mass prevents a
        beam-only certificate from being presented as end-to-end.

        By default, transition-selector weights in the compiled planner
        define ``P(target at step 1 | state, action)``.  ``outcome_model`` can
        instead supply explicit stochastic outcomes as
        ``[(next_state_updates, probability), ...]`` for each state/action.
        Updates may be partial; unspecified modes persist from the belief
        state.  This separates physical success probabilities from action
        costs when planner transition costs represent operational effort.
        Conditional lookahead requires this explicit outcome model.

        ``observation_model(next_state, command)`` returns either one
        observation mapping or a probability-weighted sequence of observation
        mappings.  States producing the same observation are combined into a
        posterior belief before the next action is optimized.  Returning the
        state itself implements perfect observation; returning an empty
        mapping implements no information.

        Approximate conditional planning can set
        ``min_observation_probability`` and/or
        ``max_observations_per_node``. Rare observation groups are merged
        into one fallback posterior rather than removed from the expected
        value. The returned policy routes unmatched observations through
        ``BeliefPolicyNode.fallback_policy`` and reports retained probability
        mass, pruning counts, per-action utility bounds, maximum regret, and
        exact, certified, or heuristic root-action ranking.

        Utility is ``goal_reward * P(target) - cost_weight * action_cost``.
        For a single command variable, ``action_costs`` is either a mapping
        from command values to costs or a callable receiving the command
        dictionary.  Evaluations are returned best-first.  The explicit
        sequence, outcome-branch, policy-node, and observation-branch limits
        make exponential lookahead inspectable and provisionable.
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
        if max_policy_nodes < 1:
            raise ValueError("max_policy_nodes must be at least 1")
        if max_observation_branches < 1:
            raise ValueError("max_observation_branches must be at least 1")
        if (
            not math.isfinite(min_observation_probability)
            or min_observation_probability < 0.0
            or min_observation_probability >= 1.0
        ):
            raise ValueError(
                "min_observation_probability must be finite and in [0, 1)"
            )
        if (
            max_observations_per_node is not None
            and max_observations_per_node < 1
        ):
            raise ValueError(
                "max_observations_per_node must be at least 1"
            )
        if observation_model is not None and self.horizon < 2:
            raise ValueError(
                "observation branching requires a planner horizon of at "
                "least 2"
            )
        if observation_model is not None and outcome_model is None:
            raise ValueError(
                "observation branching requires an explicit outcome_model"
            )
        if not target:
            raise ValueError("target must not be empty")
        valid_targets = set(self.mode_names) | set(self.obs_names)
        unknown_targets = set(target) - valid_targets
        if unknown_targets:
            raise KeyError(
                f"unknown planner target variables: {sorted(unknown_targets)}"
            )

        belief_exact = getattr(belief, "exact", None)
        if belief_exact not in (True, False, None):
            raise ValueError("belief exactness metadata must be boolean")
        supplied_retained_mass = getattr(
            belief,
            "retained_probability_mass",
            None,
        )
        if supplied_retained_mass is not None:
            try:
                supplied_retained_mass = float(supplied_retained_mass)
            except (TypeError, ValueError):
                raise ValueError(
                    "belief retained probability mass must be finite and "
                    "in [0, 1]"
                ) from None
            if (
                not math.isfinite(supplied_retained_mass)
                or supplied_retained_mass < 0.0
                or supplied_retained_mass > 1.0
            ):
                raise ValueError(
                    "belief retained probability mass must be finite and "
                    "in [0, 1]"
                )
        if belief_exact is True:
            supplied_retained_mass = 1.0

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

        def normalized_observations(
            state: Dict[str, object], command: Dict[str, object]
        ) -> List[Tuple[Dict[str, object], float]]:
            if observation_model is None:
                raise RuntimeError(
                    "normalized_observations needs observation_model"
                )
            supplied = observation_model(dict(state), dict(command))
            if isinstance(supplied, Mapping):
                observations = [(supplied, 1.0)]
            else:
                observations = list(supplied)
            if not observations:
                raise ValueError(
                    "observation_model returned no observations for "
                    f"state={state}, action={command}"
                )
            checked = []
            total = 0.0
            for index, item in enumerate(observations):
                try:
                    observation, probability = item
                except (TypeError, ValueError):
                    raise ValueError(
                        "observation_model entries must be "
                        "(observation, probability) pairs; "
                        f"entry {index} is {item!r}"
                    ) from None
                if not isinstance(observation, Mapping):
                    raise ValueError(
                        f"observation {index} is not a mapping"
                    )
                try:
                    numeric_probability = float(probability)
                except (TypeError, ValueError):
                    raise ValueError(
                        "observation probabilities must be finite and "
                        f"non-negative; got {probability!r}"
                    ) from None
                if (
                    not math.isfinite(numeric_probability)
                    or numeric_probability < 0.0
                ):
                    raise ValueError(
                        "observation probabilities must be finite and "
                        f"non-negative; got {probability!r}"
                    )
                checked.append((dict(observation), numeric_probability))
                total += numeric_probability
            if not math.isfinite(total) or total <= 0.0:
                raise ValueError(
                    "observation probabilities must have positive finite "
                    "total mass"
                )
            return [
                (observation, probability / total)
                for observation, probability in checked
                if probability > 0.0
            ]

        def frozen_mapping(
            value: Mapping[str, object],
            *,
            label: str,
        ) -> Tuple[Tuple[str, object], ...]:
            try:
                frozen = tuple(sorted(value.items()))
                hash(frozen)
                return frozen
            except TypeError:
                raise ValueError(
                    f"{label} keys and values must be orderable and hashable"
                ) from None

        if observation_model is not None:
            counters = {
                "policy_nodes": 0,
                "outcome_branches": 0,
                "observation_branches": 0,
                "generated_observation_branches": 0,
                "pruned_observation_branches": 0,
            }

            def solve_conditional_policy(
                branch_belief: Sequence[
                    Tuple[Dict[str, object], float]
                ],
                step: int,
            ) -> Tuple[BeliefPolicyNode, Tuple[BeliefPolicyNode, ...]]:
                candidates: List[BeliefPolicyNode] = []
                for action in action_values:
                    counters["policy_nodes"] += 1
                    if counters["policy_nodes"] > max_policy_nodes:
                        raise ValueError(
                            "conditional belief lookahead exceeded "
                            f"max_policy_nodes={max_policy_nodes}"
                        )
                    command = {command_name: action}
                    expanded_outcomes: List[
                        Tuple[Dict[str, object], float]
                    ] = []
                    for state, state_mass in branch_belief:
                        base = step_evidence(state, step)
                        base[f"{command_name}@{step}"] = action
                        for updates, outcome_probability in (
                            normalized_outcomes(state, command)
                        ):
                            counters["outcome_branches"] += 1
                            if (
                                counters["outcome_branches"]
                                > max_outcome_branches
                            ):
                                raise ValueError(
                                    "conditional belief lookahead exceeded "
                                    f"max_outcome_branches="
                                    f"{max_outcome_branches}"
                                )
                            next_state = dict(state)
                            next_state.update(updates)
                            transition_evidence = dict(base)
                            transition_evidence.update(
                                step_evidence(next_state, step + 1)
                            )
                            transition_evidence.update(
                                {
                                    f"{name}@{step + 1}": value
                                    for name, value in updates.items()
                                    if name in self.obs_names
                                }
                            )
                            if (
                                self.system.log_evidence(
                                    transition_evidence
                                )
                                == -math.inf
                            ):
                                raise ValueError(
                                    "outcome_model produced a state "
                                    "inconsistent with planner: "
                                    f"{transition_evidence}"
                                )
                            expanded_outcomes.append(
                                (
                                    next_state,
                                    state_mass * outcome_probability,
                                )
                            )

                    immediate_cost = action_cost(action, command)
                    if step + 1 == self.horizon:
                        expected_goal = sum(
                            probability
                            * conditional_goal_probability(
                                step_evidence(
                                    next_state,
                                    step + 1,
                                    include_observables=True,
                                ),
                                step + 1,
                            )
                            for next_state, probability
                            in expanded_outcomes
                        )
                        candidates.append(
                            BeliefPolicyNode(
                                action=command,
                                immediate_action_cost=immediate_cost,
                                expected_goal_probability=expected_goal,
                                expected_action_cost=immediate_cost,
                                expected_utility=(
                                    goal_reward * expected_goal
                                    - cost_weight * immediate_cost
                                ),
                                utility_upper_bound=(
                                    goal_reward * expected_goal
                                    - cost_weight * immediate_cost
                                ),
                                branches=(),
                                fallback_policy=None,
                                retained_observation_probability=1.0,
                                discarded_observation_probability=0.0,
                            )
                        )
                        continue

                    grouped: Dict[
                        Tuple[Tuple[str, object], ...],
                        Dict[str, object],
                    ] = {}
                    for next_state, outcome_mass in expanded_outcomes:
                        for (
                            observation,
                            observation_probability,
                        ) in normalized_observations(
                            next_state, command
                        ):
                            observation_key = frozen_mapping(
                                observation,
                                label="observation",
                            )
                            state_key = frozen_mapping(
                                next_state,
                                label="state",
                            )
                            group = grouped.setdefault(
                                observation_key,
                                {
                                    "observation": observation,
                                    "states": {},
                                    "mass": 0.0,
                                },
                            )
                            branch_mass = (
                                outcome_mass
                                * observation_probability
                            )
                            states = group["states"]
                            if state_key in states:
                                states[state_key][1] += branch_mass
                            else:
                                states[state_key] = [
                                    next_state,
                                    branch_mass,
                                ]
                            group["mass"] += branch_mass

                    groups = sorted(
                        (
                            group for group in grouped.values()
                            if float(group["mass"]) > 0.0
                        ),
                        key=lambda group: -float(group["mass"]),
                    )
                    counters["generated_observation_branches"] += len(
                        groups
                    )
                    retained_groups = [
                        group for group in groups
                        if (
                            float(group["mass"])
                            >= min_observation_probability
                        )
                    ]
                    discarded_groups = [
                        group for group in groups
                        if (
                            float(group["mass"])
                            < min_observation_probability
                        )
                    ]
                    if (
                        max_observations_per_node is not None
                        and len(retained_groups)
                        > max_observations_per_node
                    ):
                        discarded_groups.extend(
                            retained_groups[max_observations_per_node:]
                        )
                        retained_groups = retained_groups[
                            :max_observations_per_node
                        ]
                    counters["pruned_observation_branches"] += len(
                        discarded_groups
                    )

                    branches: List[BeliefPolicyBranch] = []
                    expected_goal = 0.0
                    expected_continuation_cost = 0.0
                    continuation_utility_upper_bound = 0.0
                    retained_path_probability = 0.0

                    def account_observation_branch() -> None:
                        counters["observation_branches"] += 1
                        if (
                            counters["observation_branches"]
                            > max_observation_branches
                        ):
                            raise ValueError(
                                "conditional belief lookahead exceeded "
                                f"max_observation_branches="
                                f"{max_observation_branches}"
                            )

                    for group in retained_groups:
                        probability = float(group["mass"])
                        account_observation_branch()
                        posterior = [
                            (state, mass / probability)
                            for state, mass in group["states"].values()
                        ]
                        child, _ = solve_conditional_policy(
                            posterior, step + 1
                        )
                        branch_goal = child.expected_goal_probability
                        branch_cost = child.expected_action_cost
                        branch_utility = child.expected_utility
                        expected_goal += probability * branch_goal
                        expected_continuation_cost += (
                            probability * branch_cost
                        )
                        continuation_utility_upper_bound += (
                            probability * child.utility_upper_bound
                        )
                        retained_path_probability += (
                            probability
                            * child.retained_observation_probability
                        )
                        branches.append(
                            BeliefPolicyBranch(
                                observation=dict(group["observation"]),
                                probability=probability,
                                expected_goal_probability=branch_goal,
                                expected_action_cost=branch_cost,
                                expected_utility=branch_utility,
                                policy=child,
                            )
                        )

                    fallback_policy: Optional[BeliefPolicyNode] = None
                    discarded_probability = sum(
                        float(group["mass"])
                        for group in discarded_groups
                    )
                    if discarded_probability > 0.0:
                        account_observation_branch()
                        fallback_states: Dict[
                            Tuple[Tuple[str, object], ...],
                            List[object],
                        ] = {}
                        for group in discarded_groups:
                            for state_key, (
                                state,
                                mass,
                            ) in group["states"].items():
                                if state_key in fallback_states:
                                    fallback_states[state_key][1] += mass
                                else:
                                    fallback_states[state_key] = [
                                        state,
                                        mass,
                                    ]
                        fallback_belief = [
                            (state, mass / discarded_probability)
                            for state, mass in fallback_states.values()
                        ]
                        fallback_policy, _ = solve_conditional_policy(
                            fallback_belief, step + 1
                        )
                        expected_goal += (
                            discarded_probability
                            * fallback_policy.expected_goal_probability
                        )
                        expected_continuation_cost += (
                            discarded_probability
                            * fallback_policy.expected_action_cost
                        )
                        continuation_utility_upper_bound += (
                            discarded_probability * goal_reward
                        )

                    expected_cost = (
                        immediate_cost + expected_continuation_cost
                    )
                    candidates.append(
                        BeliefPolicyNode(
                            action=command,
                            immediate_action_cost=immediate_cost,
                            expected_goal_probability=expected_goal,
                            expected_action_cost=expected_cost,
                            expected_utility=(
                                goal_reward * expected_goal
                                - cost_weight * expected_cost
                            ),
                            utility_upper_bound=(
                                continuation_utility_upper_bound
                                - cost_weight * immediate_cost
                            ),
                            branches=tuple(
                                sorted(
                                    branches,
                                    key=lambda branch: -branch.probability,
                                )
                            ),
                            fallback_policy=fallback_policy,
                            retained_observation_probability=(
                                retained_path_probability
                            ),
                            discarded_observation_probability=(
                                discarded_probability
                            ),
                        )
                    )

                candidates.sort(
                    key=lambda item: (
                        -round(item.expected_utility, 12),
                        -round(item.expected_goal_probability, 12),
                        round(item.expected_action_cost, 12),
                    )
                )
                return candidates[0], tuple(candidates)

            best_policy, root_evaluations = solve_conditional_policy(
                checked_states, 0
            )
            is_approximate = (
                counters["pruned_observation_branches"] > 0
            )
            policy_alternative_upper_bound = max(
                (
                    evaluation.utility_upper_bound
                    for evaluation in root_evaluations[1:]
                ),
                default=-math.inf,
            )
            policy_maximum_regret = max(
                0.0,
                policy_alternative_upper_bound
                - best_policy.utility_lower_bound,
            )
            if policy_maximum_regret <= 1e-12:
                policy_maximum_regret = 0.0
            policy_root_action_certified = (
                policy_maximum_regret == 0.0
            )

            if belief_exact is True:
                retained_mass_for_bounds = 1.0
                certificate_scope = "exact-tracker-belief"
            elif belief_exact is False:
                retained_mass_for_bounds = (
                    0.0
                    if supplied_retained_mass is None
                    else supplied_retained_mass
                )
                certificate_scope = (
                    "tracked-belief-mass-bound"
                    if supplied_retained_mass is not None
                    else "tracked-belief-mass-unknown"
                )
            else:
                # Backward-compatible caller-supplied beliefs have no source
                # metadata. Bounds remain explicitly scoped to that supplied
                # distribution rather than claiming tracker exactness.
                retained_mass_for_bounds = 1.0
                certificate_scope = "caller-supplied-belief"

            max_action_cost = max(
                evaluation.immediate_action_cost
                for evaluation in root_evaluations
            )
            missing_future_cost = (
                max(0, self.horizon - 1) * max_action_cost
            )
            action_certificates = tuple(
                BeliefActionCertificate(
                    action=dict(evaluation.action),
                    policy_utility_lower_bound=(
                        evaluation.utility_lower_bound
                    ),
                    policy_utility_upper_bound=(
                        evaluation.utility_upper_bound
                    ),
                    utility_lower_bound=(
                        retained_mass_for_bounds
                        * evaluation.utility_lower_bound
                        + (1.0 - retained_mass_for_bounds)
                        * (
                            -cost_weight
                            * (
                                evaluation.immediate_action_cost
                                + missing_future_cost
                            )
                        )
                    ),
                    utility_upper_bound=(
                        retained_mass_for_bounds
                        * evaluation.utility_upper_bound
                        + (1.0 - retained_mass_for_bounds)
                        * (
                            goal_reward
                            - cost_weight
                            * evaluation.immediate_action_cost
                        )
                    ),
                )
                for evaluation in root_evaluations
            )
            selected_certificate = action_certificates[0]
            alternative_upper_bound = max(
                (
                    certificate.utility_upper_bound
                    for certificate in action_certificates[1:]
                ),
                default=-math.inf,
            )
            maximum_regret = max(
                0.0,
                alternative_upper_bound
                - selected_certificate.utility_lower_bound,
            )
            if maximum_regret <= 1e-12:
                maximum_regret = 0.0
            root_action_certified = maximum_regret == 0.0
            belief_is_approximate = belief_exact is False
            return ConditionalBeliefPolicyResult(
                policy=best_policy,
                evaluations=root_evaluations,
                policy_node_count=counters["policy_nodes"],
                outcome_branch_count=counters["outcome_branches"],
                observation_branch_count=counters[
                    "observation_branches"
                ],
                generated_observation_branch_count=counters[
                    "generated_observation_branches"
                ],
                pruned_observation_branch_count=counters[
                    "pruned_observation_branches"
                ],
                retained_observation_probability=(
                    best_policy.retained_observation_probability
                ),
                approximation=(
                    "belief-and-observation-approximate"
                    if belief_is_approximate and is_approximate
                    else (
                        "belief-approximate"
                        if belief_is_approximate
                        else (
                            "observation-pruned"
                            if is_approximate
                            else "exact"
                        )
                    )
                ),
                action_ranking=(
                    "certified"
                    if root_action_certified and (
                        belief_is_approximate or is_approximate
                    )
                    else (
                        "exact"
                        if (
                            root_action_certified
                            and not belief_is_approximate
                            and not is_approximate
                        )
                        else "heuristic"
                    )
                ),
                utility_is_lower_bound=(
                    is_approximate or belief_is_approximate
                ),
                optimal_utility_upper_bound=max(
                    certificate.utility_upper_bound
                    for certificate in action_certificates
                ),
                maximum_regret=maximum_regret,
                root_action_certified=root_action_certified,
                action_certificates=action_certificates,
                policy_maximum_regret=policy_maximum_regret,
                policy_root_action_certified=(
                    policy_root_action_certified
                ),
                belief_exact=belief_exact,
                belief_retained_probability_mass=(
                    supplied_retained_mass
                ),
                certificate_scope=certificate_scope,
            )

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
