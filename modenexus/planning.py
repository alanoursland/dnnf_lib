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
import time
from dataclasses import dataclass, field, replace
from itertools import product
from typing import (
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from . import fd
from .compile_control import CompileControl
from .diagnosis import SystemModel, _normalize_categorical_weights
from .formula import Formula


class _ImmutableMapping(Mapping[str, object]):
    """Small recursively immutable mapping used by certified results."""

    __slots__ = ("_items",)

    def __init__(self, value: Mapping[str, object]):
        self._items = tuple(
            (key, _immutable_value(item)) for key, item in value.items()
        )

    def __getitem__(self, key: str) -> object:
        for candidate, value in self._items:
            if candidate == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (key for key, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        return repr(dict(self._items))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Mapping):
            return dict(self.items()) == dict(other.items())
        return False

    def copy(self) -> Dict[str, object]:
        """Return a mutable shallow copy for execution integrations."""
        return dict(self._items)


def _immutable_value(value: object) -> object:
    if isinstance(value, _ImmutableMapping):
        return value
    if isinstance(value, Mapping):
        return _ImmutableMapping(value)
    if isinstance(value, (list, tuple)):
        return tuple(_immutable_value(item) for item in value)
    if isinstance(value, set):
        return frozenset(_immutable_value(item) for item in value)
    return value


def _immutable_mapping(
    value: Mapping[str, object],
) -> Mapping[str, object]:
    frozen = _immutable_value(value)
    assert isinstance(frozen, Mapping)
    return frozen


@dataclass(frozen=True)
class PlanningStats:
    """Snapshot of an in-progress or completed belief-planning search.

    Chance-constrained searches additionally report aggregate frontier
    generation/retention counts, the largest nondominated pre-cap frontier,
    truncated-node count, and saturation grouped by depth and root action.
    """

    elapsed_seconds: float
    policy_nodes: int
    outcome_branches: int
    observation_branches: int
    best_action: Optional[Dict[str, object]]
    best_expected_utility: Optional[float]
    generated_frontier_points: int = 0
    retained_frontier_points: int = 0
    maximum_frontier_size: int = 0
    truncated_frontier_nodes: int = 0
    frontier_saturation_by_depth: Tuple[Tuple[int, int], ...] = ()
    frontier_saturated_root_actions: Tuple[object, ...] = ()
    complete: bool = False


class PlanningInterrupted(RuntimeError):
    """Base class for cooperative belief-planning termination."""

    def __init__(self, message: str, stats: PlanningStats):
        super().__init__(message)
        self.stats = stats


class PlanningCancelled(PlanningInterrupted):
    """Raised when a plan_belief cancellation callback requests a stop."""


class PlanningBudgetExceeded(PlanningInterrupted, ValueError):
    """Raised when a plan_belief time, deadline, or resource budget is
    exceeded.  Also a ``ValueError`` so callers already catching the plain
    hard-cap errors keep working; ``stats`` carries the partial search
    counts and best root action found before the limit tripped."""


@dataclass(frozen=True)
class PlanControl:
    """Optional cooperative timeout, cancellation, and progress reporting
    for :meth:`CompiledPlanner.plan_belief`'s conditional-policy search.

    This is independent of the existing hard ``max_action_sequences``,
    ``max_outcome_branches``, ``max_policy_nodes``, and
    ``max_observation_branches`` counts, which remain plain resource caps
    raising ``ValueError``.  ``PlanControl`` instead lets an interactive or
    real-time caller bound wall-clock time, cancel promptly, and observe
    progress without changing those caps.

    Parameters
    ----------
    timeout_seconds:
        Maximum elapsed wall-clock time for this search.
    deadline:
        Absolute :func:`time.monotonic` deadline.  When both deadline and
        timeout are supplied, the earlier limit wins.
    cancel:
        Zero-argument callback returning true when cancellation is requested.
    progress:
        Callback receiving :class:`PlanningStats` snapshots, including the
        best root action found so far.  A final snapshot with
        ``complete=True`` is always emitted on success.
    progress_interval:
        Minimum seconds between non-final progress callbacks.
    """

    timeout_seconds: Optional[float] = None
    deadline: Optional[float] = None
    cancel: Optional[Callable[[], bool]] = None
    progress: Optional[Callable[[PlanningStats], None]] = None
    progress_interval: float = 0.25

    def __post_init__(self) -> None:
        if self.timeout_seconds is not None and (
            not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds < 0
        ):
            raise ValueError("timeout_seconds must be finite and non-negative")
        if self.deadline is not None and not math.isfinite(self.deadline):
            raise ValueError("deadline must be finite")
        if (
            not math.isfinite(self.progress_interval)
            or self.progress_interval < 0
        ):
            raise ValueError("progress_interval must be finite and non-negative")

    def _start(self) -> "_PlanSession":
        return _PlanSession(self)


class _PlanSession:
    """Mutable per-invocation state owned by :meth:`CompiledPlanner.plan_belief`."""

    def __init__(self, control: PlanControl):
        self.control = control
        self.started = time.monotonic()
        timeout_deadline = (
            self.started + control.timeout_seconds
            if control.timeout_seconds is not None
            else None
        )
        if timeout_deadline is None:
            self.deadline = control.deadline
        elif control.deadline is None:
            self.deadline = timeout_deadline
        else:
            self.deadline = min(timeout_deadline, control.deadline)
        self.last_progress = self.started
        self.policy_nodes = 0
        self.outcome_branches = 0
        self.observation_branches = 0
        self.frontier_generated_points = 0
        self.frontier_retained_points = 0
        self.frontier_maximum_size = 0
        self.frontier_truncated_nodes = 0
        self.frontier_saturation_by_depth: Tuple[
            Tuple[int, int], ...
        ] = ()
        self.frontier_saturated_root_actions: Tuple[object, ...] = ()
        self.best_action: Optional[Dict[str, object]] = None
        self.best_expected_utility: Optional[float] = None

    def note_root_candidate(
        self, action: Dict[str, object], expected_utility: float
    ) -> None:
        if (
            self.best_expected_utility is None
            or expected_utility > self.best_expected_utility
        ):
            self.best_action = dict(action)
            self.best_expected_utility = expected_utility

    def snapshot(self, complete: bool = False) -> PlanningStats:
        return PlanningStats(
            elapsed_seconds=time.monotonic() - self.started,
            policy_nodes=self.policy_nodes,
            outcome_branches=self.outcome_branches,
            observation_branches=self.observation_branches,
            best_action=self.best_action,
            best_expected_utility=self.best_expected_utility,
            generated_frontier_points=self.frontier_generated_points,
            retained_frontier_points=self.frontier_retained_points,
            maximum_frontier_size=self.frontier_maximum_size,
            truncated_frontier_nodes=self.frontier_truncated_nodes,
            frontier_saturation_by_depth=(
                self.frontier_saturation_by_depth
            ),
            frontier_saturated_root_actions=(
                self.frontier_saturated_root_actions
            ),
            complete=complete,
        )

    def check(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if self.control.cancel is not None and self.control.cancel():
            raise PlanningCancelled(
                "belief planning cancelled", self.snapshot()
            )
        if self.deadline is not None and now >= self.deadline:
            raise PlanningBudgetExceeded(
                "belief planning time budget exceeded", self.snapshot()
            )
        if self.control.progress is not None and (
            force
            or now - self.last_progress >= self.control.progress_interval
        ):
            self.control.progress(self.snapshot())
            self.last_progress = now

    def finish(self) -> PlanningStats:
        stats = self.snapshot(complete=True)
        if self.control.progress is not None:
            self.control.progress(stats)
        return stats


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
    """Best one-step action and all evaluated alternatives.

    When planning used an explicit ``outcome_model``, :meth:`execution`
    creates a one-action execution state that updates the latent belief from
    the observed physical outcome.
    """

    action: Dict[str, object]
    expected_goal_probability: float
    action_cost: float
    expected_utility: float
    evaluations: Tuple[BeliefActionEvaluation, ...]
    goal_probability_constraint: Optional[float] = None
    feasible: Optional[bool] = None
    best_achievable_goal_probability: Optional[float] = None
    _execution_context: Optional[
        "_BeliefPolicyExecutionContext"
    ] = field(default=None, repr=False, compare=False)
    _execution_certificate_scope: str = field(
        default="caller-supplied-belief",
        repr=False,
        compare=False,
    )

    @property
    def commands(self) -> List[Dict[str, object]]:
        """Single-command list for symmetry with :class:`PlanResult`."""
        return [dict(self.action)]

    def execution(
        self,
        belief=None,
        *,
        outcome_model: Optional[Callable] = None,
    ) -> "BeliefPolicyExecution":
        """Create a one-action state that updates belief from real outcome.

        The explicit outcome callback supplied during planning is reused by
        default. Callers may replace it for execution. A result planned only
        from compiled transition weights has no physical outcome callback
        and therefore requires ``outcome_model`` here.
        """
        context = self._execution_context
        if context is None:
            raise RuntimeError(
                "this result does not carry belief execution context"
            )
        selected_model = (
            context.outcome_model
            if outcome_model is None
            else outcome_model
        )
        if selected_model is None:
            raise RuntimeError(
                "belief execution requires the planning outcome_model "
                "or an explicit outcome_model"
            )
        policy = BeliefPolicyNode(
            action=dict(self.action),
            immediate_action_cost=self.action_cost,
            expected_goal_probability=self.expected_goal_probability,
            expected_action_cost=self.action_cost,
            expected_utility=self.expected_utility,
            utility_upper_bound=self.expected_utility,
            branches=(),
            fallback_policy=None,
            retained_observation_probability=1.0,
            discarded_observation_probability=0.0,
            goal_probability_upper_bound=self.expected_goal_probability,
        )
        return BeliefPolicyExecution(
            policy=policy,
            belief=(
                context.initial_belief if belief is None else belief
            ),
            outcome_model=selected_model,
            observation_model=None,
            goal_probability=context.goal_probability,
            action_cost=context.action_cost,
            certificate_scope=self._execution_certificate_scope,
        )


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
    goal_probability_constraint: Optional[float] = None
    feasible: Optional[bool] = None
    best_achievable_goal_probability: Optional[float] = None

    @property
    def action(self) -> Dict[str, object]:
        """First action, suitable for receding-horizon execution."""
        return dict(self.commands[0])


@dataclass(frozen=True)
class BeliefPolicyBranch:
    """One observation edge in a conditional belief-policy tree.

    ``posterior`` is the normalized observation-conditioned joint belief the
    planner optimized the continuation against — the hidden states that
    justify the branch's action, exposed instead of forcing applications to
    re-derive outcome propagation and Bayes weighting themselves.  For the
    aggregated pruned-observation fallback branch on
    :attr:`BeliefPolicyNode.fallback_policy`, ``contributing_observations``
    lists the pruned observations merged into that posterior. Observation,
    posterior-state, and contributing-observation mappings are recursively
    immutable so execution cannot diverge from the certified snapshot.
    """

    observation: Mapping[str, object]
    probability: float
    expected_goal_probability: float
    expected_action_cost: float
    expected_utility: float
    policy: Optional["BeliefPolicyNode"]
    posterior: Tuple[Tuple[Mapping[str, object], float], ...] = ()
    contributing_observations: Tuple[Mapping[str, object], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "observation", _immutable_mapping(self.observation)
        )
        object.__setattr__(
            self,
            "posterior",
            tuple(
                (_immutable_mapping(state), probability)
                for state, probability in self.posterior
            ),
        )
        object.__setattr__(
            self,
            "contributing_observations",
            tuple(
                _immutable_mapping(observation)
                for observation in self.contributing_observations
            ),
        )

    def posterior_marginals(self) -> Dict[str, Dict[object, float]]:
        """Per-variable marginals of :attr:`posterior`."""
        out: Dict[str, Dict[object, float]] = {}
        for state, probability in self.posterior:
            for name, value in state.items():
                out.setdefault(name, {})
                out[name][value] = out[name].get(value, 0.0) + probability
        return out


@dataclass(frozen=True)
class BeliefPolicyRouting:
    """Structured result of routing an observation through a policy node.

    ``kind`` is one of:

    - ``"exact"``: the observation, projected onto the node's
      :attr:`~BeliefPolicyNode.observation_schema`, matched a retained
      branch (``branch`` and ``policy`` are set);
    - ``"fallback"``: a well-formed observation matched no retained branch
      and routed to the pruned-observation fallback policy (``branch`` is
      the aggregate :attr:`~BeliefPolicyNode.fallback_branch`);
    - ``"terminal"``: the node has no continuations (end of the policy);
    - ``"unmatched"``: no retained branch matched and no fallback exists.

    ``projected_observation`` is the schema projection actually compared,
    so callers can see exactly which fields participated in matching.
    """

    kind: str
    policy: Optional["BeliefPolicyNode"]
    branch: Optional["BeliefPolicyBranch"]
    projected_observation: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "projected_observation",
            _immutable_mapping(self.projected_observation),
        )


@dataclass(frozen=True)
class BeliefPolicyNode:
    """One action and its observation-contingent continuations.

    ``fallback_policy`` receives observations omitted by threshold or top-k
    pruning; ``fallback_branch`` carries that fallback's aggregate posterior
    and the pruned observations contributing to it.
    ``retained_observation_probability`` includes all later levels;
    ``discarded_observation_probability`` is local to this node. The action
    mapping and every mapping reachable through the policy tree are
    recursively immutable; use ``dict(node.action)`` for a mutable execution
    copy.
    """

    action: Mapping[str, object]
    immediate_action_cost: float
    expected_goal_probability: float
    expected_action_cost: float
    expected_utility: float
    utility_upper_bound: float
    branches: Tuple[BeliefPolicyBranch, ...]
    fallback_policy: Optional["BeliefPolicyNode"]
    retained_observation_probability: float
    discarded_observation_probability: float
    fallback_branch: Optional[BeliefPolicyBranch] = None
    goal_probability_upper_bound: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "action", _immutable_mapping(self.action))

    @property
    def observation_schema(self) -> Tuple[str, ...]:
        """Union of keys this node's retained branch observations use.

        Branches may use heterogeneous key sets. :meth:`route` intersects
        execution-time telemetry with this union, so unrelated extra keys
        are ignored while absence of a callback-produced key remains part
        of the branch observation's identity.
        """
        keys = set()
        for branch in self.branches:
            keys.update(branch.observation)
        return tuple(sorted(keys))

    def route(
        self, observation: Mapping[str, object]
    ) -> BeliefPolicyRouting:
        """Route an observation and report how it was matched.

        The observation is intersected with :attr:`observation_schema`, so
        unrelated telemetry fields are ignored. The intersection is matched
        exactly: callback mappings may have heterogeneous key sets, and an
        absent key is distinct from a present key whose value is ``None``.
        """
        if not self.branches and self.fallback_policy is None:
            return BeliefPolicyRouting(
                kind="terminal",
                policy=None,
                branch=None,
                projected_observation=dict(observation),
            )
        schema = self.observation_schema
        supplied = dict(observation)
        projected = {
            key: supplied[key] for key in schema if key in supplied
        }
        for branch in self.branches:
            if branch.observation == projected:
                return BeliefPolicyRouting(
                    kind="exact",
                    policy=branch.policy,
                    branch=branch,
                    projected_observation=projected,
                )
        if self.fallback_policy is not None:
            return BeliefPolicyRouting(
                kind="fallback",
                policy=self.fallback_policy,
                branch=self.fallback_branch,
                projected_observation=projected,
            )
        return BeliefPolicyRouting(
            kind="unmatched",
            policy=None,
            branch=None,
            projected_observation=projected,
        )

    def continuation(
        self, observation: Mapping[str, object]
    ) -> Optional["BeliefPolicyNode"]:
        """Return the continuation for an observation via :meth:`route`.

        Extra observation keys are projected away before matching. Absence
        is part of observation identity, so heterogeneous callback mappings
        route exactly with only the keys they originally supplied. An
        observation matching no retained branch returns the pruned-
        observation fallback (or ``None`` when the policy is terminal or
        has no fallback). Use
        :meth:`route` to distinguish exact, fallback, terminal, and
        unmatched routing explicitly.
        """
        return self.route(observation).policy

    @property
    def utility_lower_bound(self) -> float:
        """Value of this executable policy under its observation fallback."""
        return self.expected_utility


@dataclass(frozen=True)
class BeliefPolicyExecutionStep:
    """One observed execution step and its updated latent belief."""

    action: Mapping[str, object]
    outcome: Optional[Mapping[str, object]]
    observation: Optional[Mapping[str, object]]
    route: Optional[BeliefPolicyRouting]
    posterior: Tuple[Tuple[Mapping[str, object], float], ...]
    goal_reached: bool
    terminal: bool
    accumulated_cost: float
    certificate_scope: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "action", _immutable_mapping(self.action))
        if self.outcome is not None:
            object.__setattr__(
                self, "outcome", _immutable_mapping(self.outcome)
            )
        if self.observation is not None:
            object.__setattr__(
                self,
                "observation",
                _immutable_mapping(self.observation),
            )
        object.__setattr__(
            self,
            "posterior",
            tuple(
                (_immutable_mapping(state), probability)
                for state, probability in self.posterior
            ),
        )

    def posterior_marginals(self) -> Dict[str, Dict[object, float]]:
        """Per-variable marginals of :attr:`posterior`."""
        out: Dict[str, Dict[object, float]] = {}
        for state, probability in self.posterior:
            for name, value in state.items():
                out.setdefault(name, {})
                out[name][value] = out[name].get(value, 0.0) + probability
        return out


@dataclass(frozen=True)
class _BeliefPolicyExecutionContext:
    initial_belief: Tuple[Tuple[Mapping[str, object], float], ...]
    outcome_model: Optional[Callable]
    outcome_scenarios: Tuple[Tuple[str, Callable], ...]
    observation_model: Optional[Callable]
    goal_probability: Callable
    action_cost: Callable


class BeliefPolicyExecution:
    """Advance a returned conditional policy with real observed evidence."""

    def __init__(
        self,
        *,
        policy: BeliefPolicyNode,
        belief,
        outcome_model,
        observation_model,
        goal_probability,
        action_cost,
        certificate_scope: str,
    ) -> None:
        weighted = []
        total = 0.0
        for state, mass in belief:
            numeric = float(mass)
            if not math.isfinite(numeric) or numeric < 0.0:
                raise ValueError(
                    "execution belief masses must be finite and "
                    "non-negative"
                )
            weighted.append((dict(state), numeric))
            total += numeric
        if total <= 0.0:
            raise ValueError(
                "execution belief must have positive total mass"
            )
        self._belief = tuple(
            (state, mass / total) for state, mass in weighted if mass > 0.0
        )
        self._policy: Optional[BeliefPolicyNode] = policy
        self._outcome_model = outcome_model
        self._observation_model = observation_model
        self._goal_probability = goal_probability
        self._action_cost = action_cost
        self._certificate_scope = certificate_scope
        self._accumulated_cost = 0.0
        self._step = 0
        self._terminal = self._goal_reached()

    def belief(self) -> Tuple[Tuple[Mapping[str, object], float], ...]:
        """Return the current normalized latent belief.

        This is a method, matching :meth:`ModeTracker.belief` in the
        monitor-plan-execute workflow.
        """
        return tuple(
            (_immutable_mapping(state), probability)
            for state, probability in self._belief
        )

    @property
    def action(self) -> Optional[Dict[str, object]]:
        if self._terminal or self._policy is None:
            return None
        return dict(self._policy.action)

    @property
    def accumulated_cost(self) -> float:
        return self._accumulated_cost

    @property
    def terminal(self) -> bool:
        return self._terminal

    @property
    def requires_observation(self) -> bool:
        """Whether the current action has an observation continuation."""
        return bool(
            not self._terminal
            and self._policy is not None
            and (
                self._policy.branches
                or self._policy.fallback_policy is not None
            )
        )

    def _goal_reached(self) -> bool:
        return bool(self._belief) and all(
            self._goal_probability(state, self._step)
            >= 1.0 - 1e-12
            for state, _ in self._belief
        )

    @staticmethod
    def _distribution(entries, *, label):
        checked = []
        total = 0.0
        for value, probability in entries:
            numeric = float(probability)
            if not math.isfinite(numeric) or numeric < 0.0:
                raise ValueError(
                    f"{label} probabilities must be finite and "
                    f"non-negative; got {probability!r}"
                )
            checked.append((dict(value), numeric))
            total += numeric
        if not checked or not math.isclose(
            total, 1.0, rel_tol=1e-6, abs_tol=1e-9
        ):
            raise ValueError(
                f"{label} probabilities must sum to 1; got {total!r}"
            )
        return [
            (value, probability)
            for value, probability in checked
            if probability > 0.0
        ]

    @staticmethod
    def _normalize_particles(particles):
        combined = {}
        total = 0.0
        for state, mass in particles:
            key = tuple(sorted(state.items()))
            if key in combined:
                combined[key][1] += mass
            else:
                combined[key] = [state, mass]
            total += mass
        if total <= 0.0:
            raise ValueError(
                "observed execution evidence has zero probability"
            )
        return tuple(
            (state, mass / total)
            for state, mass in combined.values()
            if mass > 0.0
        )

    def advance(
        self,
        *,
        outcome: Optional[Mapping[str, object]] = None,
        observation: Optional[Mapping[str, object]] = None,
    ) -> BeliefPolicyExecutionStep:
        """Consume observed next-state fields and optional telemetry."""
        if self._terminal or self._policy is None:
            raise RuntimeError("policy execution is already terminal")
        node = self._policy
        command = dict(node.action)
        self._accumulated_cost += float(self._action_cost(command))

        outcome_evidence = dict(outcome) if outcome is not None else None
        particles = []
        for state, state_mass in self._belief:
            entries = self._distribution(
                list(self._outcome_model(dict(state), dict(command))),
                label="outcome",
            )
            for updates, probability in entries:
                next_state = dict(state)
                next_state.update(updates)
                if (
                    outcome_evidence is not None
                    and any(
                        key not in next_state
                        or next_state[key] != value
                        for key, value in outcome_evidence.items()
                    )
                ):
                    continue
                particles.append(
                    (next_state, state_mass * probability)
                )
        particles = self._normalize_particles(particles)

        if observation is not None:
            if self._observation_model is None:
                raise ValueError(
                    "this execution result has no observation model"
                )
            generated = []
            schema = set()
            for state, state_mass in particles:
                supplied = self._observation_model(
                    dict(state), dict(command)
                )
                entries = (
                    [(supplied, 1.0)]
                    if isinstance(supplied, Mapping)
                    else list(supplied)
                )
                checked = self._distribution(
                    entries, label="observation"
                )
                for emitted, probability in checked:
                    schema.update(emitted)
                    generated.append(
                        (
                            state,
                            state_mass * probability,
                            emitted,
                        )
                    )
            observed = dict(observation)
            projected = {
                key: observed[key] for key in schema if key in observed
            }
            particles = self._normalize_particles(
                [
                    (state, mass)
                    for state, mass, emitted in generated
                    if emitted == projected
                ]
            )

        self._belief = particles
        self._step += 1
        goal_reached = self._goal_reached()
        route = None
        if goal_reached:
            route = BeliefPolicyRouting(
                kind="terminal",
                policy=None,
                branch=None,
                projected_observation=(
                    dict(observation) if observation is not None else {}
                ),
            )
            self._policy = None
            self._terminal = True
        elif not node.branches and node.fallback_policy is None:
            route = node.route(observation or {})
            self._policy = None
            self._terminal = True
        else:
            if observation is None:
                raise ValueError(
                    "a nonterminal policy step requires observation"
                )
            route = node.route(observation)
            if route.policy is None:
                raise ValueError(
                    "observation did not route to a continuation policy"
                )
            self._policy = route.policy
            self._terminal = False

        return BeliefPolicyExecutionStep(
            action=command,
            outcome=outcome_evidence,
            observation=(
                dict(observation) if observation is not None else None
            ),
            route=route,
            posterior=self._belief,
            goal_reached=goal_reached,
            terminal=self._terminal,
            accumulated_cost=self._accumulated_cost,
            certificate_scope=self._certificate_scope,
        )


@dataclass(frozen=True)
class BeliefActionCertificate:
    """Policy-only and belief-composed utility and goal-probability bounds
    for one root action.

    The ``policy_*`` fields are scoped to the supplied normalized belief;
    the unprefixed bounds compose tracker-retained mass adversarially.
    ``policy_goal_probability`` is exact for the executable (possibly
    observation-pruned) policy; ``policy_goal_probability_upper_bound``
    bounds the unrestricted full-observation optimum for this root action.
    """

    action: Mapping[str, object]
    policy_utility_lower_bound: float
    policy_utility_upper_bound: float
    utility_lower_bound: float
    utility_upper_bound: float
    policy_goal_probability: float = 0.0
    policy_goal_probability_upper_bound: float = 1.0
    goal_probability_lower_bound: float = 0.0
    goal_probability_upper_bound: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "action", _immutable_mapping(self.action))


@dataclass(frozen=True)
class RobustScenarioEvaluation:
    """Execution metrics for one policy under one outcome scenario."""

    scenario: str
    expected_goal_probability: float
    expected_action_cost: float
    expected_utility: float
    minimum_branch_goal_probability: Optional[float] = None
    minimum_branch_observation_path: Tuple[
        Mapping[str, object], ...
    ] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "minimum_branch_observation_path",
            tuple(
                _immutable_mapping(observation)
                for observation in self.minimum_branch_observation_path
            ),
        )


@dataclass(frozen=True)
class RobustPolicyEvaluation:
    """One common policy generated by a scenario-weight scalarization."""

    scenario_weights: Tuple[Tuple[str, float], ...]
    policy: BeliefPolicyNode
    scenario_evaluations: Tuple[RobustScenarioEvaluation, ...]
    worst_case_scenario: str
    worst_case_expected_utility: float
    worst_case_goal_probability: float
    worst_case_branch_scenario: Optional[str] = None
    worst_case_minimum_branch_goal_probability: Optional[float] = None
    worst_case_branch_observation_path: Tuple[
        Mapping[str, object], ...
    ] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "worst_case_branch_observation_path",
            tuple(
                _immutable_mapping(observation)
                for observation in self.worst_case_branch_observation_path
            ),
        )

    @property
    def action(self) -> Dict[str, object]:
        return dict(self.policy.action)


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

    Under a reliability constraint, ``feasible`` reports whether the
    selected policy meets ``goal_probability_constraint`` (and the optional
    ``branch_goal_probability_constraint``) against the supplied belief.
    ``best_achievable_goal_probability_scope`` states whether the reported
    best probability covers the full observation-policy space or only the
    selected observation coarsening.  Under pruning, the latter is paired
    with ``best_achievable_goal_probability_upper_bound`` for the
    unrestricted space.  ``constraint_optimality`` describes frontier
    truncation, while ``observation_partition_optimality`` separately
    describes observation coarsening. ``constraint_certification`` scopes
    feasibility through pruning and tracker mass like the utility
    certificates. Every active whole-policy and per-branch floor
    participates in ``constraint_certification``. Observation pruning makes
    branch-floor feasibility ``indeterminate`` unless infeasibility is
    established without aggregation.

    Frontier diagnostics report aggregate generated/retained point counts,
    the largest nondominated pre-cap frontier, truncation counts by depth and
    root action, and a conservative feasible-utility upper bound/gap. The
    same search counters appear in :class:`PlanningStats`.

    Scenario-robust results expose independently audited per-scenario
    metrics and all deduplicated generated candidates. Their
    ``robust_optimality`` and ``certificate_scope`` explicitly identify the
    bounded weight-grid search. ``inherited_metric_scope`` and
    ``selected_scenario_weights`` identify scalarized policy metrics as those
    of the mixture that generated the selected common policy. Robust branch
    fields report the minimum continuation probability, limiting
    scenario/observation path, feasibility, and pruning-aware certification.
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
    goal_probability_constraint: Optional[float] = None
    branch_goal_probability_constraint: Optional[float] = None
    feasible: Optional[bool] = None
    best_achievable_goal_probability: Optional[float] = None
    constraint_certification: Optional[str] = None
    constraint_optimality: Optional[str] = None
    generated_frontier_points: int = 0
    retained_frontier_points: int = 0
    maximum_frontier_size: int = 0
    truncated_frontier_nodes: int = 0
    frontier_saturation_by_depth: Tuple[Tuple[int, int], ...] = ()
    frontier_saturated_root_actions: Tuple[object, ...] = ()
    constraint_utility_upper_bound: Optional[float] = None
    constraint_utility_optimality_gap: Optional[float] = None
    best_achievable_goal_probability_upper_bound: Optional[float] = None
    best_achievable_goal_probability_scope: Optional[str] = None
    observation_partition_optimality: Optional[str] = None
    robust_objective: Optional[str] = None
    robust_scenario_evaluations: Tuple[
        RobustScenarioEvaluation, ...
    ] = ()
    robust_policy_evaluations: Tuple[RobustPolicyEvaluation, ...] = ()
    worst_case_scenario: Optional[str] = None
    worst_case_expected_utility: Optional[float] = None
    worst_case_goal_probability: Optional[float] = None
    robust_candidate_count: int = 0
    robust_weight_resolution: Optional[int] = None
    robust_optimality: Optional[str] = None
    robust_feasible: Optional[bool] = None
    robust_best_achievable_min_goal_probability: Optional[float] = None
    inherited_metric_scope: Optional[str] = None
    selected_scenario_weights: Tuple[Tuple[str, float], ...] = ()
    worst_case_branch_scenario: Optional[str] = None
    worst_case_minimum_branch_goal_probability: Optional[float] = None
    worst_case_branch_observation_path: Tuple[
        Mapping[str, object], ...
    ] = ()
    robust_branch_feasible: Optional[bool] = None
    robust_branch_constraint_certification: Optional[str] = None
    robust_best_achievable_min_branch_goal_probability: Optional[
        float
    ] = None
    _execution_context: Optional[
        _BeliefPolicyExecutionContext
    ] = field(default=None, repr=False, compare=False)

    @property
    def action(self) -> Dict[str, object]:
        """Root action selected after valuing conditional continuations."""
        return dict(self.policy.action)

    def execution(
        self,
        belief=None,
        *,
        outcome_scenario: Optional[str] = None,
        outcome_model: Optional[Callable] = None,
    ) -> BeliefPolicyExecution:
        """Create execution state that updates belief from real evidence.

        Ordinary results reuse their planning outcome callback. Robust
        results require either a named ``outcome_scenario`` or an explicit
        realized ``outcome_model``.
        """
        context = self._execution_context
        if context is None:
            raise RuntimeError(
                "this result does not carry policy execution context"
            )
        if outcome_scenario is not None and outcome_model is not None:
            raise ValueError(
                "supply either outcome_scenario or outcome_model, not both"
            )
        scenarios = dict(context.outcome_scenarios)
        selected_model = outcome_model
        if scenarios:
            if selected_model is None:
                if outcome_scenario is None:
                    raise ValueError(
                        "robust policy execution requires "
                        "outcome_scenario or outcome_model"
                    )
                if outcome_scenario not in scenarios:
                    raise ValueError(
                        f"unknown outcome scenario {outcome_scenario!r}; "
                        f"expected one of {tuple(scenarios)!r}"
                    )
                selected_model = scenarios[outcome_scenario]
        else:
            if outcome_scenario is not None:
                raise ValueError(
                    "outcome_scenario is only valid for robust results"
                )
            if selected_model is None:
                selected_model = context.outcome_model
        if selected_model is None:
            raise RuntimeError("policy execution has no outcome model")
        return BeliefPolicyExecution(
            policy=self.policy,
            belief=(
                context.initial_belief if belief is None else belief
            ),
            outcome_model=selected_model,
            observation_model=context.observation_model,
            goal_probability=context.goal_probability,
            action_cost=context.action_cost,
            certificate_scope=self.certificate_scope,
        )

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
        if not values:
            raise ValueError(
                f"mode {name!r} needs at least 1 value; got {values!r}"
            )
        if priors is not None:
            priors = tuple(_normalize_categorical_weights(name, values, priors))
        self._modes[name] = (values, priors)

    def command(self, name: str, values: Sequence) -> None:
        """A per-step command input; include an inert value (e.g.
        ``"none"``) if doing nothing must be expressible."""
        values = tuple(values)
        if not values:
            raise ValueError(
                f"command {name!r} needs at least 1 value; got {values!r}"
            )
        self._commands[name] = values

    def observable(self, name: str, values: Sequence = (False, True)) -> None:
        """A per-step sensed value, constrained via :meth:`behavior`."""
        values = tuple(values)
        if not values:
            raise ValueError(
                f"observable {name!r} needs at least 1 value; got "
                f"{values!r}"
            )
        self._observables[name] = values

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
                    # persistence directly; no selector variable is needed.
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
        outcome_scenarios: Optional[Mapping[str, Callable]] = None,
        robust_objective: Optional[str] = None,
        robust_weight_resolution: int = 8,
        max_robust_candidates: int = 100,
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
        min_goal_probability: Optional[float] = None,
        min_branch_goal_probability: Optional[float] = None,
        max_frontier_points: int = 256,
        control: Optional[PlanControl] = None,
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
        One-step and conditional results expose ``execution()`` to consume
        real physical evidence and carry the updated belief into replanning.

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
        mapping implements no information. Observation mappings at one node
        may use heterogeneous key sets. Policy routing treats absence as part
        of observation identity while ignoring telemetry keys never emitted
        by any retained branch at that node.

        ``outcome_scenarios`` enables bounded robust conditional planning.
        It maps scenario names to outcome callbacks with the same contract as
        ``outcome_model``. With ``robust_objective="maximin"``, the planner
        generates common executable policies from a positive scenario-weight
        grid, evaluates every policy under every scenario, and selects the
        greatest worst-case utility (subject to ``min_goal_probability`` in
        every scenario when supplied). If no generated candidate meets that
        floor, the most reliable candidate is returned with
        ``robust_feasible=False``. ``robust_weight_resolution`` controls grid
        density and ``max_robust_candidates`` is a hard generation cap.
        Results expose every scenario/candidate evaluation and label search
        optimality ``"weight-grid-heuristic"`` rather than claiming global
        robust optimality. Inherited ``expected_*`` fields remain scoped to
        the selected candidate's generating mixture; the result exposes
        ``inherited_metric_scope`` and ``selected_scenario_weights`` directly.
        ``min_branch_goal_probability`` additionally requires every audited
        observation continuation to meet the floor in every scenario and
        reports the limiting scenario/path. Observation fallback pruning
        keeps branch certification indeterminate.

        Unlike ``belief`` (validated and normalized above), the
        probabilities returned by ``outcome_model`` and ``observation_model``
        are a physical distribution over the branches of that one call and
        must already sum to 1 within ``1e-6`` relative tolerance; a total
        outside that tolerance raises ``ValueError`` naming the state,
        action, and observed total rather than being silently rescaled.
        This catches, for example, two outcomes summing to 0.5 because a
        third branch was left out.

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

        ``min_goal_probability`` adds a chance constraint: feasibility
        (policy goal probability at or above the floor) is decided first,
        and utility ranks only the feasible candidates.  In conditional
        planning the constraint is enforced over the whole policy — not per
        node — via Pareto-frontier dynamic programming over
        (goal probability, expected cost) pairs, so reliability spent in one
        observation branch can compensate for another branch's ceiling.
        ``min_branch_goal_probability`` optionally adds the stricter safety
        variant: every observation branch's continuation must individually
        satisfy the floor.  Infeasibility is reported, not raised: the most
        reliable policy is returned with ``feasible=False`` and
        ``best_achievable_goal_probability``.  ``max_frontier_points``
        bounds each Pareto frontier; when the cap actually binds, the result
        says ``constraint_optimality="frontier-truncated"`` (feasibility and
        best-achievable stay exact; only cost-optimality may be lost).
        Result and progress statistics expose generated/retained points,
        largest frontier, truncation by depth/root action, and a conservative
        utility upper bound/gap for provisioning.
        ``constraint_certification`` composes observation pruning and
        tracker-retained mass into ``certified-feasible``,
        ``certified-infeasible``, or ``indeterminate``, scoped exactly like
        the utility certificates. Every active whole-policy and branch floor
        participates. A merged observation fallback cannot certify every
        contributing raw branch and is therefore ``indeterminate`` unless an
        unaggregated result proves infeasibility. Under a constraint, the
        utility-regret fields still compare utilities across the reported
        per-action policies; feasibility governs selection.

        ``control`` optionally supplies cooperative wall-clock limits,
        a cancellation callback, and progress reporting, mirroring
        :class:`modenexus.CompileControl` for compilation.  Cancellations
        raise :class:`PlanningCancelled`; time and resource budgets raise
        :class:`PlanningBudgetExceeded` (also a ``ValueError``).  Both carry
        a :class:`PlanningStats` snapshot with the partial search counts and
        the best fully evaluated root action so far.
        """
        if outcome_scenarios is not None:
            return self._plan_belief_robust(
                belief=belief,
                target=target,
                outcome_scenarios=outcome_scenarios,
                robust_objective=robust_objective,
                robust_weight_resolution=robust_weight_resolution,
                max_robust_candidates=max_robust_candidates,
                observation_model=observation_model,
                action_costs=action_costs,
                actions=actions,
                goal_reward=goal_reward,
                cost_weight=cost_weight,
                max_action_sequences=max_action_sequences,
                max_outcome_branches=max_outcome_branches,
                max_policy_nodes=max_policy_nodes,
                max_observation_branches=max_observation_branches,
                min_observation_probability=min_observation_probability,
                max_observations_per_node=max_observations_per_node,
                min_goal_probability=min_goal_probability,
                min_branch_goal_probability=min_branch_goal_probability,
                max_frontier_points=max_frontier_points,
                control=control,
                supplied_outcome_model=outcome_model,
            )
        if robust_objective is not None:
            raise ValueError(
                "robust_objective requires outcome_scenarios"
            )
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
        for constraint_name, constraint_value in (
            ("min_goal_probability", min_goal_probability),
            ("min_branch_goal_probability", min_branch_goal_probability),
        ):
            if constraint_value is not None and (
                not math.isfinite(constraint_value)
                or constraint_value < 0.0
                or constraint_value > 1.0
            ):
                raise ValueError(
                    f"{constraint_name} must be finite and in [0, 1]"
                )
        if (
            min_branch_goal_probability is not None
            and observation_model is None
        ):
            raise ValueError(
                "min_branch_goal_probability requires an observation_model: "
                "without observation branching there are no per-branch "
                "continuations to constrain"
            )
        if max_frontier_points < 2:
            raise ValueError("max_frontier_points must be at least 2")
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

        # A session always exists so budget errors can carry partial stats;
        # with the default control it has no deadline, cancel, or progress.
        session = (control or PlanControl())._start()
        session.check(force=control is not None and control.progress is not None)

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
        if self.horizon > 1:
            sequence_count = len(action_values) ** self.horizon
            if sequence_count > max_action_sequences:
                raise PlanningBudgetExceeded(
                    f"belief lookahead requires {sequence_count} action "
                    f"sequences, exceeding "
                    f"max_action_sequences={max_action_sequences}",
                    session.snapshot(),
                )

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
            for name, value in target.items():
                key = f"{name}@{target_step}"
                if key in with_goal and with_goal[key] != value:
                    return 0.0
                with_goal[key] = value
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

        def checked_probability_distribution(
            entries: Sequence[Tuple[Mapping[str, object], object]],
            *,
            kind: str,
            context: str,
        ) -> List[Tuple[Dict[str, object], float]]:
            """Validate a callback's (item, probability) pairs and require
            the total to already be normalized to one within tolerance.

            Silent normalization of an arbitrary positive total would accept
            e.g. two outcomes summing to 0.5 as if that were the complete
            distribution, masking a forgotten branch; instead any total
            outside a small tolerance of 1.0 is a contextual ValueError.
            """
            checked = []
            total = 0.0
            for index, item in enumerate(entries):
                try:
                    value, probability = item
                except (TypeError, ValueError):
                    raise ValueError(
                        f"{kind} entries must be (value, probability) "
                        f"pairs; entry {index} is {item!r}"
                    ) from None
                if not isinstance(value, Mapping):
                    raise ValueError(f"{kind} {index} is not a mapping")
                try:
                    numeric_probability = float(probability)
                except (TypeError, ValueError):
                    raise ValueError(
                        f"{kind} probability must be finite and "
                        f"non-negative; got {probability!r}"
                    ) from None
                if (
                    not math.isfinite(numeric_probability)
                    or numeric_probability < 0.0
                ):
                    raise ValueError(
                        f"{kind} probability must be finite and "
                        f"non-negative; got {probability!r}"
                    )
                checked.append((dict(value), numeric_probability))
                total += numeric_probability
            if not checked:
                raise ValueError(f"{kind} must not be empty for {context}")
            if not math.isfinite(total) or not math.isclose(
                total, 1.0, rel_tol=1e-6, abs_tol=1e-9
            ):
                raise ValueError(
                    f"{kind} for {context} must sum to 1 (within 1e-6); "
                    f"got total {total!r}"
                )
            return [
                (value, probability)
                for value, probability in checked
                if probability > 0.0
            ]

        def normalized_outcomes(
            state: Dict[str, object], command: Dict[str, object]
        ) -> List[Tuple[Dict[str, object], float]]:
            if outcome_model is None:
                raise RuntimeError("normalized_outcomes needs outcome_model")
            outcomes = list(outcome_model(dict(state), dict(command)))
            return checked_probability_distribution(
                outcomes,
                kind="outcome probabilities",
                context=f"state={state}, action={command}",
            )

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
            return checked_probability_distribution(
                observations,
                kind="observation probabilities",
                context=f"state={state}, action={command}",
            )

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
                "frontier_truncations": 0,
                "frontier_generated_points": 0,
                "frontier_retained_points": 0,
                "frontier_maximum_size": 0,
            }
            frontier_truncations_by_depth: Dict[int, int] = {}
            frontier_saturated_root_actions = set()

            def sync_session() -> None:
                session.policy_nodes = counters["policy_nodes"]
                session.outcome_branches = counters["outcome_branches"]
                session.observation_branches = counters[
                    "observation_branches"
                ]
                session.frontier_generated_points = counters[
                    "frontier_generated_points"
                ]
                session.frontier_retained_points = counters[
                    "frontier_retained_points"
                ]
                session.frontier_maximum_size = counters[
                    "frontier_maximum_size"
                ]
                session.frontier_truncated_nodes = counters[
                    "frontier_truncations"
                ]
                session.frontier_saturation_by_depth = tuple(
                    sorted(frontier_truncations_by_depth.items())
                )
                session.frontier_saturated_root_actions = tuple(
                    sorted(
                        frontier_saturated_root_actions,
                        key=repr,
                    )
                )

            def expand_action(
                branch_belief: Sequence[Tuple[Dict[str, object], float]],
                step: int,
                action: object,
                command: Dict[str, object],
            ) -> List[Tuple[Dict[str, object], float]]:
                """Count one policy node and expand this action's validated
                stochastic outcomes over the belief."""
                counters["policy_nodes"] += 1
                sync_session()
                session.check()
                if counters["policy_nodes"] > max_policy_nodes:
                    raise PlanningBudgetExceeded(
                        "conditional belief lookahead exceeded "
                        f"max_policy_nodes={max_policy_nodes}",
                        session.snapshot(),
                    )
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
                            sync_session()
                            raise PlanningBudgetExceeded(
                                "conditional belief lookahead exceeded "
                                f"max_outcome_branches="
                                f"{max_outcome_branches}",
                                session.snapshot(),
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
                            self.system.log_evidence(transition_evidence)
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
                return expanded_outcomes

            def split_terminal_belief(branch_belief, step):
                terminal_mass = 0.0
                active_belief = []
                for state, probability in branch_belief:
                    state_goal = conditional_goal_probability(
                        step_evidence(
                            state,
                            step,
                            include_observables=True,
                        ),
                        step,
                    )
                    if state_goal >= 1.0 - 1e-12:
                        terminal_mass += probability
                    else:
                        active_belief.append((state, probability))
                return terminal_mass, active_belief

            def terminal_node(
                command: Dict[str, object],
                immediate_cost: float,
                expanded_outcomes: Sequence[
                    Tuple[Dict[str, object], float]
                ],
                step: int,
                *,
                prior_goal_probability: float = 0.0,
                execution_probability: float = 1.0,
            ) -> BeliefPolicyNode:
                expected_goal = (
                    prior_goal_probability
                    + sum(
                        probability
                        * conditional_goal_probability(
                            step_evidence(
                                next_state,
                                step + 1,
                                include_observables=True,
                            ),
                            step + 1,
                        )
                        for next_state, probability in expanded_outcomes
                    )
                )
                expected_cost = execution_probability * immediate_cost
                value = (
                    goal_reward * expected_goal
                    - cost_weight * expected_cost
                )
                return BeliefPolicyNode(
                    action=command,
                    immediate_action_cost=immediate_cost,
                    expected_goal_probability=expected_goal,
                    expected_action_cost=expected_cost,
                    expected_utility=value,
                    utility_upper_bound=value,
                    branches=(),
                    fallback_policy=None,
                    retained_observation_probability=1.0,
                    discarded_observation_probability=0.0,
                    goal_probability_upper_bound=expected_goal,
                )

            def observation_groups(expanded_outcomes, command):
                """Group outcomes by observation, then split retained from
                pruned groups under the observation-pruning controls."""
                grouped: Dict[
                    Tuple[Tuple[str, object], ...],
                    Dict[str, object],
                ] = {}
                for next_state, outcome_mass in expanded_outcomes:
                    for (
                        observation,
                        observation_probability,
                    ) in normalized_observations(next_state, command):
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
                            outcome_mass * observation_probability
                        )
                        states = group["states"]
                        if state_key in states:
                            states[state_key][1] += branch_mass
                        else:
                            states[state_key] = [next_state, branch_mass]
                        group["mass"] += branch_mass

                groups = sorted(
                    (
                        group for group in grouped.values()
                        if float(group["mass"]) > 0.0
                    ),
                    key=lambda group: -float(group["mass"]),
                )
                counters["generated_observation_branches"] += len(groups)
                retained_groups = [
                    group for group in groups
                    if float(group["mass"]) >= min_observation_probability
                ]
                discarded_groups = [
                    group for group in groups
                    if float(group["mass"]) < min_observation_probability
                ]
                if (
                    max_observations_per_node is not None
                    and len(retained_groups) > max_observations_per_node
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
                discarded_probability = sum(
                    float(group["mass"]) for group in discarded_groups
                )
                return retained_groups, discarded_groups, discarded_probability

            def account_observation_branch() -> None:
                counters["observation_branches"] += 1
                if (
                    counters["observation_branches"]
                    > max_observation_branches
                ):
                    sync_session()
                    raise PlanningBudgetExceeded(
                        "conditional belief lookahead exceeded "
                        f"max_observation_branches="
                        f"{max_observation_branches}",
                        session.snapshot(),
                    )

            def group_posterior(group):
                probability = float(group["mass"])
                return [
                    (state, mass / probability)
                    for state, mass in group["states"].values()
                ]

            def merged_fallback_belief(
                discarded_groups, discarded_probability
            ):
                fallback_states: Dict[
                    Tuple[Tuple[str, object], ...],
                    List[object],
                ] = {}
                for group in discarded_groups:
                    for state_key, (state, mass) in group["states"].items():
                        if state_key in fallback_states:
                            fallback_states[state_key][1] += mass
                        else:
                            fallback_states[state_key] = [state, mass]
                return [
                    (state, mass / discarded_probability)
                    for state, mass in fallback_states.values()
                ]

            def retained_branch(group, probability, posterior, child):
                return BeliefPolicyBranch(
                    observation=dict(group["observation"]),
                    probability=probability,
                    expected_goal_probability=(
                        child.expected_goal_probability
                    ),
                    expected_action_cost=child.expected_action_cost,
                    expected_utility=child.expected_utility,
                    policy=child,
                    posterior=tuple(
                        (dict(state), state_probability)
                        for state, state_probability in posterior
                    ),
                )

            def fallback_branch_for(
                discarded_groups, discarded_probability, posterior, child
            ):
                return BeliefPolicyBranch(
                    observation={},
                    probability=discarded_probability,
                    expected_goal_probability=(
                        child.expected_goal_probability
                    ),
                    expected_action_cost=child.expected_action_cost,
                    expected_utility=child.expected_utility,
                    policy=child,
                    posterior=tuple(
                        (dict(state), state_probability)
                        for state, state_probability in posterior
                    ),
                    contributing_observations=tuple(
                        dict(group["observation"])
                        for group in discarded_groups
                    ),
                )

            def solve_conditional_policy(
                branch_belief: Sequence[
                    Tuple[Dict[str, object], float]
                ],
                step: int,
            ) -> Tuple[BeliefPolicyNode, Tuple[BeliefPolicyNode, ...]]:
                terminal_mass, active_belief = split_terminal_belief(
                    branch_belief, step
                )
                active_mass = sum(
                    probability for _, probability in active_belief
                )
                candidates: List[BeliefPolicyNode] = []
                for action in action_values:
                    command = {command_name: action}
                    expanded_outcomes = expand_action(
                        active_belief, step, action, command
                    )
                    immediate_cost = action_cost(action, command)
                    expected_immediate_cost = (
                        active_mass * immediate_cost
                    )
                    if step + 1 == self.horizon:
                        candidates.append(
                            terminal_node(
                                command,
                                immediate_cost,
                                expanded_outcomes,
                                step,
                                prior_goal_probability=terminal_mass,
                                execution_probability=active_mass,
                            )
                        )
                        if step == 0:
                            session.note_root_candidate(
                                candidates[-1].action,
                                candidates[-1].expected_utility,
                            )
                        continue

                    (
                        retained_groups,
                        discarded_groups,
                        discarded_probability,
                    ) = observation_groups(expanded_outcomes, command)

                    branches: List[BeliefPolicyBranch] = []
                    expected_goal = terminal_mass
                    expected_continuation_cost = 0.0
                    continuation_utility_upper_bound = (
                        terminal_mass * goal_reward
                    )
                    continuation_goal_upper_bound = terminal_mass
                    retained_path_probability = terminal_mass

                    for group in retained_groups:
                        probability = float(group["mass"])
                        account_observation_branch()
                        posterior = group_posterior(group)
                        child, _ = solve_conditional_policy(
                            posterior, step + 1
                        )
                        expected_goal += (
                            probability * child.expected_goal_probability
                        )
                        expected_continuation_cost += (
                            probability * child.expected_action_cost
                        )
                        continuation_utility_upper_bound += (
                            probability * child.utility_upper_bound
                        )
                        continuation_goal_upper_bound += (
                            probability
                            * child.goal_probability_upper_bound
                        )
                        retained_path_probability += (
                            probability
                            * child.retained_observation_probability
                        )
                        branches.append(
                            retained_branch(
                                group, probability, posterior, child
                            )
                        )

                    fallback_policy: Optional[BeliefPolicyNode] = None
                    fallback_branch: Optional[BeliefPolicyBranch] = None
                    if discarded_probability > 0.0:
                        account_observation_branch()
                        fallback_belief = merged_fallback_belief(
                            discarded_groups, discarded_probability
                        )
                        fallback_policy, _ = solve_conditional_policy(
                            fallback_belief, step + 1
                        )
                        fallback_branch = fallback_branch_for(
                            discarded_groups,
                            discarded_probability,
                            fallback_belief,
                            fallback_policy,
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
                        continuation_goal_upper_bound += (
                            discarded_probability
                        )

                    expected_cost = (
                        expected_immediate_cost
                        + expected_continuation_cost
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
                                - cost_weight
                                * expected_immediate_cost
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
                            fallback_branch=fallback_branch,
                            goal_probability_upper_bound=min(
                                1.0, continuation_goal_upper_bound
                            ),
                        )
                    )
                    if step == 0:
                        session.note_root_candidate(
                            candidates[-1].action,
                            candidates[-1].expected_utility,
                        )

                candidates.sort(
                    key=lambda item: (
                        -round(item.expected_utility, 12),
                        -round(item.expected_goal_probability, 12),
                        round(item.expected_action_cost, 12),
                    )
                )
                return candidates[0], tuple(candidates)

            def prune_frontier(
                points,
                *,
                depth: int,
                root_action: Optional[object],
            ):
                """Keep the nondominated (goal desc, cost asc) frontier,
                capped at ``max_frontier_points`` with endpoints preserved.

                Dominance pruning is lossless for the tree DP: any ancestor
                combination using a dominated point can swap in the
                dominating point with goal probability no worse and cost no
                higher.  Capping only drops interior points, so feasibility
                detection and the maximum achievable goal probability stay
                exact; only cost-optimality can degrade, which the result
                reports as ``frontier-truncated``.
                """
                counters["frontier_generated_points"] += len(points)
                points.sort(key=lambda point: (-point[0], point[1]))
                kept = []
                best_cost = math.inf
                for point in points:
                    if point[1] < best_cost - 1e-15:
                        kept.append(point)
                        best_cost = point[1]
                counters["frontier_maximum_size"] = max(
                    counters["frontier_maximum_size"], len(kept)
                )
                if len(kept) > max_frontier_points:
                    counters["frontier_truncations"] += 1
                    frontier_truncations_by_depth[depth] = (
                        frontier_truncations_by_depth.get(depth, 0) + 1
                    )
                    if root_action is not None:
                        frontier_saturated_root_actions.add(root_action)
                    elif depth == 0:
                        frontier_saturated_root_actions.update(
                            point[2].action[command_name]
                            for point in kept
                        )
                    span = len(kept) - 1
                    picks = sorted({
                        round(index * span / (max_frontier_points - 1))
                        for index in range(max_frontier_points)
                    })
                    kept = [kept[index] for index in picks]
                counters["frontier_retained_points"] += len(kept)
                return kept

            def solve_policy_frontier(
                branch_belief: Sequence[
                    Tuple[Dict[str, object], float]
                ],
                step: int,
                branch_floor: Optional[float],
                root_action: Optional[object] = None,
            ) -> List[Tuple[float, float, BeliefPolicyNode]]:
                """Pareto frontier of (goal probability, expected cost,
                policy) points for this belief.

                A chance constraint cannot be enforced per node: the
                reliability one observation branch must deliver depends on
                what the other branches deliver, so whole frontiers
                propagate upward and the floor is applied only at the root.
                ``branch_floor`` is the stricter per-branch variant and is
                applied to every child frontier.
                """
                terminal_mass, active_belief = split_terminal_belief(
                    branch_belief, step
                )
                active_mass = sum(
                    probability for _, probability in active_belief
                )
                points: List[Tuple[float, float, BeliefPolicyNode]] = []
                for action in action_values:
                    action_root = action if step == 0 else root_action
                    command = {command_name: action}
                    expanded_outcomes = expand_action(
                        active_belief, step, action, command
                    )
                    immediate_cost = action_cost(action, command)
                    expected_immediate_cost = (
                        active_mass * immediate_cost
                    )
                    if step + 1 == self.horizon:
                        node = terminal_node(
                            command,
                            immediate_cost,
                            expanded_outcomes,
                            step,
                            prior_goal_probability=terminal_mass,
                            execution_probability=active_mass,
                        )
                        points.append(
                            (
                                node.expected_goal_probability,
                                expected_immediate_cost,
                                node,
                            )
                        )
                        continue

                    (
                        retained_groups,
                        discarded_groups,
                        discarded_probability,
                    ) = observation_groups(expanded_outcomes, command)

                    # (probability, posterior, group); group None marks the
                    # aggregated pruned-observation fallback.
                    branch_specs = []
                    for group in retained_groups:
                        account_observation_branch()
                        branch_specs.append(
                            (
                                float(group["mass"]),
                                group_posterior(group),
                                group,
                            )
                        )
                    if discarded_probability > 0.0:
                        account_observation_branch()
                        branch_specs.append(
                            (
                                discarded_probability,
                                merged_fallback_belief(
                                    discarded_groups,
                                    discarded_probability,
                                ),
                                None,
                            )
                        )

                    child_frontiers = []
                    action_allowed = True
                    goal_upper = terminal_mass
                    utility_upper = terminal_mass * goal_reward
                    for probability, posterior, group in branch_specs:
                        frontier = solve_policy_frontier(
                            posterior,
                            step + 1,
                            branch_floor,
                            action_root,
                        )
                        if group is None:
                            # Pruned mass keeps the same maximum-credit
                            # bound the unconstrained certificates use.
                            goal_upper += probability
                            utility_upper += probability * goal_reward
                        else:
                            goal_upper += probability * max(
                                point[2].goal_probability_upper_bound
                                for point in frontier
                            )
                            utility_upper += probability * max(
                                point[2].utility_upper_bound
                                for point in frontier
                            )
                        if branch_floor is not None:
                            frontier = [
                                point for point in frontier
                                if point[0] >= branch_floor - 1e-9
                            ]
                        if not frontier:
                            action_allowed = False
                            break
                        child_frontiers.append(frontier)
                    if not action_allowed:
                        continue

                    combos = [(terminal_mass, 0.0, ())]
                    for (probability, _, _), frontier in zip(
                        branch_specs, child_frontiers
                    ):
                        merged = []
                        for goal_sum, cost_sum, chosen in combos:
                            for point in frontier:
                                merged.append(
                                    (
                                        goal_sum + probability * point[0],
                                        cost_sum + probability * point[1],
                                        chosen + (point,),
                                    )
                                )
                        combos = prune_frontier(
                            merged,
                            depth=step,
                            root_action=action_root,
                        )

                    action_points = []
                    for expected_goal, continuation_cost, chosen in combos:
                        branches: List[BeliefPolicyBranch] = []
                        fallback_policy: Optional[BeliefPolicyNode] = None
                        fallback_branch: Optional[
                            BeliefPolicyBranch
                        ] = None
                        retained_path_probability = 0.0
                        for (probability, posterior, group), point in zip(
                            branch_specs, chosen
                        ):
                            child = point[2]
                            if group is None:
                                fallback_policy = child
                                fallback_branch = fallback_branch_for(
                                    discarded_groups,
                                    discarded_probability,
                                    posterior,
                                    child,
                                )
                            else:
                                retained_path_probability += (
                                    probability
                                    * child.retained_observation_probability
                                )
                                branches.append(
                                    retained_branch(
                                        group,
                                        probability,
                                        posterior,
                                        child,
                                    )
                                )
                        expected_cost = (
                            expected_immediate_cost
                            + continuation_cost
                        )
                        node = BeliefPolicyNode(
                            action=command,
                            immediate_action_cost=immediate_cost,
                            expected_goal_probability=expected_goal,
                            expected_action_cost=expected_cost,
                            expected_utility=(
                                goal_reward * expected_goal
                                - cost_weight * expected_cost
                            ),
                            utility_upper_bound=(
                                utility_upper
                                - cost_weight
                                * expected_immediate_cost
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
                            fallback_branch=fallback_branch,
                            goal_probability_upper_bound=min(
                                1.0, goal_upper
                            ),
                        )
                        action_points.append(
                            (expected_goal, expected_cost, node)
                        )
                    if step == 0 and action_points:
                        best_for_action = max(
                            action_points,
                            key=lambda point: point[2].expected_utility,
                        )
                        session.note_root_candidate(
                            command, best_for_action[2].expected_utility
                        )
                    points.extend(action_points)
                return prune_frontier(
                    points,
                    depth=step,
                    root_action=root_action,
                )

            constraint_extras: Dict[str, object] = {}
            branch_floor_satisfiable: Optional[bool] = None
            if (
                min_goal_probability is not None
                or min_branch_goal_probability is not None
            ):
                root_points = solve_policy_frontier(
                    checked_states, 0, min_branch_goal_probability
                )
                branch_floor_satisfiable = bool(root_points)
                if not root_points:
                    # The per-branch floor eliminated every policy;
                    # re-solve without it so infeasibility is reported with
                    # a concrete best-effort policy, not an empty result.
                    root_points = solve_policy_frontier(
                        checked_states, 0, None
                    )
                if not branch_floor_satisfiable:
                    feasible_points = []
                elif min_goal_probability is not None:
                    feasible_points = [
                        point for point in root_points
                        if point[0] >= min_goal_probability - 1e-9
                    ]
                else:
                    feasible_points = list(root_points)
                if feasible_points:
                    feasible = True
                    selected_point = max(
                        feasible_points,
                        key=lambda point: point[2].expected_utility,
                    )
                else:
                    feasible = False
                    selected_point = max(
                        root_points, key=lambda point: point[0]
                    )
                best_policy = selected_point[2]

                by_action: Dict[object, List] = {}
                for point in root_points:
                    by_action.setdefault(
                        point[2].action[command_name], []
                    ).append(point)
                per_action_nodes = []
                for action_points in by_action.values():
                    if min_goal_probability is not None:
                        action_feasible = [
                            point for point in action_points
                            if point[0] >= min_goal_probability - 1e-9
                        ]
                    else:
                        action_feasible = action_points
                    if action_feasible:
                        choice = max(
                            action_feasible,
                            key=lambda point: point[2].expected_utility,
                        )
                    else:
                        choice = max(
                            action_points, key=lambda point: point[0]
                        )
                    per_action_nodes.append(choice[2])
                alternatives = [
                    node for node in per_action_nodes
                    if node is not best_policy
                ]
                alternatives.sort(
                    key=lambda item: (
                        -round(item.expected_utility, 12),
                        -round(item.expected_goal_probability, 12),
                        round(item.expected_action_cost, 12),
                    )
                )
                root_evaluations = tuple([best_policy] + alternatives)
                constraint_extras = {
                    "goal_probability_constraint": min_goal_probability,
                    "branch_goal_probability_constraint": (
                        min_branch_goal_probability
                    ),
                    "feasible": feasible,
                    "best_achievable_goal_probability": max(
                        point[0] for point in root_points
                    ),
                    "constraint_optimality": (
                        "frontier-truncated"
                        if counters["frontier_truncations"]
                        else "exact"
                    ),
                    "generated_frontier_points": counters[
                        "frontier_generated_points"
                    ],
                    "retained_frontier_points": counters[
                        "frontier_retained_points"
                    ],
                    "maximum_frontier_size": counters[
                        "frontier_maximum_size"
                    ],
                    "truncated_frontier_nodes": counters[
                        "frontier_truncations"
                    ],
                    "frontier_saturation_by_depth": tuple(
                        sorted(frontier_truncations_by_depth.items())
                    ),
                    "frontier_saturated_root_actions": tuple(
                        sorted(
                            frontier_saturated_root_actions,
                            key=repr,
                        )
                    ),
                }
                minimum_immediate_cost = min(
                    action_cost(
                        action,
                        {command_name: action},
                    )
                    for action in action_values
                )
                constraint_utility_upper_bound = (
                    goal_reward
                    - cost_weight * minimum_immediate_cost
                    if counters["frontier_truncations"]
                    else best_policy.expected_utility
                )
                constraint_extras[
                    "constraint_utility_upper_bound"
                ] = constraint_utility_upper_bound
                constraint_extras[
                    "constraint_utility_optimality_gap"
                ] = max(
                    0.0,
                    constraint_utility_upper_bound
                    - best_policy.expected_utility,
                )
            else:
                best_policy, root_evaluations = solve_conditional_policy(
                    checked_states, 0
                )
            sync_session()
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
                    policy_goal_probability=(
                        evaluation.expected_goal_probability
                    ),
                    policy_goal_probability_upper_bound=(
                        evaluation.goal_probability_upper_bound
                    ),
                    goal_probability_lower_bound=(
                        retained_mass_for_bounds
                        * evaluation.expected_goal_probability
                    ),
                    goal_probability_upper_bound=min(
                        1.0,
                        retained_mass_for_bounds
                        * evaluation.goal_probability_upper_bound
                        + (1.0 - retained_mass_for_bounds),
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
            if (
                min_goal_probability is not None
                or min_branch_goal_probability is not None
            ):
                # Every active reliability constraint participates in the
                # combined certificate. Whole-policy feasibility composes
                # tracker mass through the action certificates. A branch
                # floor is certifiable only when raw observations were not
                # merged and the tracked belief itself is not approximate.
                statuses = []
            if min_goal_probability is not None:
                composed_goal_lower = (
                    selected_certificate.goal_probability_lower_bound
                )
                composed_goal_upper_max = max(
                    certificate.goal_probability_upper_bound
                    for certificate in action_certificates
                )
                if composed_goal_lower >= min_goal_probability - 1e-9:
                    statuses.append("satisfied")
                elif composed_goal_upper_max < min_goal_probability - 1e-9:
                    statuses.append("impossible")
                else:
                    statuses.append("unknown")
            if min_branch_goal_probability is not None:
                branch_certificate_exact = (
                    not is_approximate
                    and belief_exact is not False
                )
                if not branch_floor_satisfiable:
                    statuses.append(
                        "impossible"
                        if branch_certificate_exact
                        else "unknown"
                    )
                elif branch_certificate_exact:
                    statuses.append("satisfied")
                else:
                    statuses.append("unknown")
            if (
                min_goal_probability is not None
                or min_branch_goal_probability is not None
            ):
                if "impossible" in statuses:
                    certification = "certified-infeasible"
                elif statuses and all(
                    status == "satisfied" for status in statuses
                ):
                    certification = "certified-feasible"
                else:
                    certification = "indeterminate"
                constraint_extras["constraint_certification"] = (
                    certification
                )
                constraint_extras[
                    "best_achievable_goal_probability_upper_bound"
                ] = max(
                    certificate.goal_probability_upper_bound
                    for certificate in action_certificates
                )
                constraint_extras[
                    "best_achievable_goal_probability_scope"
                ] = (
                    "selected-observation-coarsening"
                    if is_approximate
                    else "full-observation-policy-space"
                )
                constraint_extras[
                    "observation_partition_optimality"
                ] = (
                    "heuristic-pruned"
                    if is_approximate
                    else "exact"
                )
            session.finish()
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
                _execution_context=_BeliefPolicyExecutionContext(
                    initial_belief=tuple(
                        (dict(state), probability)
                        for state, probability in checked_states
                    ),
                    outcome_model=outcome_model,
                    outcome_scenarios=(),
                    observation_model=observation_model,
                    goal_probability=(
                        lambda state, step: (
                            conditional_goal_probability(
                                step_evidence(
                                    state,
                                    step,
                                    include_observables=True,
                                ),
                                step,
                            )
                        )
                    ),
                    action_cost=(
                        lambda command: action_cost(
                            command[command_name], command
                        )
                    ),
                ),
                **constraint_extras,
            )

        if self.horizon > 1:
            sequence_evaluations: List[BeliefSequenceEvaluation] = []
            for action_sequence in product(
                action_values, repeat=self.horizon
            ):
                session.check()
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
                                session.outcome_branches += 1
                                if len(expanded) > max_outcome_branches:
                                    raise PlanningBudgetExceeded(
                                        "belief lookahead exceeded "
                                        f"max_outcome_branches="
                                        f"{max_outcome_branches}",
                                        session.snapshot(),
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
                session.note_root_candidate(commands[0], utility)
            sequence_evaluations.sort(
                key=lambda item: (
                    -item.expected_utility,
                    -sum(item.expected_goal_probabilities),
                )
            )
            sequence_extras: Dict[str, object] = {}
            if min_goal_probability is not None:
                feasible_sequences = [
                    evaluation for evaluation in sequence_evaluations
                    if (
                        evaluation.expected_goal_probability
                        >= min_goal_probability - 1e-9
                    )
                ]
                if feasible_sequences:
                    best_sequence = feasible_sequences[0]
                    sequence_feasible = True
                else:
                    best_sequence = max(
                        sequence_evaluations,
                        key=lambda item: item.expected_goal_probability,
                    )
                    sequence_feasible = False
                sequence_extras = {
                    "goal_probability_constraint": min_goal_probability,
                    "feasible": sequence_feasible,
                    "best_achievable_goal_probability": max(
                        evaluation.expected_goal_probability
                        for evaluation in sequence_evaluations
                    ),
                }
            else:
                best_sequence = sequence_evaluations[0]
            session.finish()
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
                **sequence_extras,
            )

        evaluations: List[BeliefActionEvaluation] = []
        for action in action_values:
            session.check()
            command = {command_name: action}
            expected_goal = 0.0
            for state, state_mass in checked_states:
                base = step_evidence(state, 0)
                base[f"{command_name}@0"] = action
                if outcome_model is None:
                    state_goal = conditional_goal_probability(base)
                else:
                    state_goal = 0.0
                    for updates, probability in normalized_outcomes(
                        state, command
                    ):
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
            session.note_root_candidate(command, utility)

        evaluations.sort(key=lambda item: -item.expected_utility)
        action_extras: Dict[str, object] = {}
        if min_goal_probability is not None:
            feasible_actions = [
                evaluation for evaluation in evaluations
                if (
                    evaluation.expected_goal_probability
                    >= min_goal_probability - 1e-9
                )
            ]
            if feasible_actions:
                best = feasible_actions[0]
                action_feasible = True
            else:
                best = max(
                    evaluations,
                    key=lambda item: item.expected_goal_probability,
                )
                action_feasible = False
            action_extras = {
                "goal_probability_constraint": min_goal_probability,
                "feasible": action_feasible,
                "best_achievable_goal_probability": max(
                    evaluation.expected_goal_probability
                    for evaluation in evaluations
                ),
            }
        else:
            best = evaluations[0]
        if belief_exact is True:
            execution_certificate_scope = "exact-tracker-belief"
        elif belief_exact is False:
            execution_certificate_scope = (
                "tracked-belief-mass-bound"
                if supplied_retained_mass is not None
                else "tracked-belief-mass-unknown"
            )
        else:
            execution_certificate_scope = "caller-supplied-belief"
        session.finish()
        return BeliefPlanResult(
            action=dict(best.action),
            expected_goal_probability=best.expected_goal_probability,
            action_cost=best.action_cost,
            expected_utility=best.expected_utility,
            evaluations=tuple(evaluations),
            _execution_context=_BeliefPolicyExecutionContext(
                initial_belief=tuple(
                    (dict(state), probability)
                    for state, probability in checked_states
                ),
                outcome_model=outcome_model,
                outcome_scenarios=(),
                observation_model=None,
                goal_probability=(
                    lambda state, step: conditional_goal_probability(
                        step_evidence(
                            state,
                            step,
                            include_observables=True,
                        ),
                        step,
                    )
                ),
                action_cost=(
                    lambda command: action_cost(
                        command[command_name], command
                    )
                ),
            ),
            _execution_certificate_scope=execution_certificate_scope,
            **action_extras,
        )

    def _plan_belief_robust(
        self,
        *,
        belief,
        target,
        outcome_scenarios,
        robust_objective,
        robust_weight_resolution,
        max_robust_candidates,
        observation_model,
        action_costs,
        actions,
        goal_reward,
        cost_weight,
        max_action_sequences,
        max_outcome_branches,
        max_policy_nodes,
        max_observation_branches,
        min_observation_probability,
        max_observations_per_node,
        min_goal_probability,
        min_branch_goal_probability,
        max_frontier_points,
        control,
        supplied_outcome_model,
    ) -> ConditionalBeliefPolicyResult:
        """Generate and audit common policies over finite outcome scenarios."""
        if supplied_outcome_model is not None:
            raise ValueError(
                "supply either outcome_model or outcome_scenarios, not both"
            )
        if robust_objective != "maximin":
            raise ValueError(
                "outcome_scenarios currently requires "
                "robust_objective='maximin'"
            )
        if observation_model is None or self.horizon < 2:
            raise ValueError(
                "scenario-robust planning requires an observation_model "
                "and horizon of at least 2"
            )
        if not isinstance(outcome_scenarios, Mapping):
            raise ValueError("outcome_scenarios must be a mapping")
        scenarios = tuple(outcome_scenarios.items())
        if len(scenarios) < 2:
            raise ValueError(
                "outcome_scenarios must contain at least two scenarios"
            )
        for name, callback in scenarios:
            if not isinstance(name, str) or not name:
                raise ValueError(
                    "outcome scenario names must be non-empty strings"
                )
            if not callable(callback):
                raise ValueError(
                    f"outcome scenario {name!r} must be callable"
                )
        if (
            not isinstance(robust_weight_resolution, int)
            or robust_weight_resolution < len(scenarios)
        ):
            raise ValueError(
                "robust_weight_resolution must be an integer at least "
                "the number of scenarios"
            )
        if (
            not isinstance(max_robust_candidates, int)
            or max_robust_candidates < 1
        ):
            raise ValueError("max_robust_candidates must be at least 1")

        original_belief = list(belief)
        if not original_belief:
            raise ValueError("belief must not be empty")
        scenario_key = "__modenexus_outcome_scenario__"
        if any(scenario_key in state for state, _ in original_belief):
            raise ValueError(
                f"belief state key {scenario_key!r} is reserved for "
                "scenario-robust planning"
            )
        scenario_names = tuple(name for name, _ in scenarios)
        scenario_models = dict(scenarios)

        def positive_compositions(total: int, parts: int):
            if parts == 1:
                yield (total,)
                return
            for first in range(1, total - parts + 2):
                for rest in positive_compositions(
                    total - first, parts - 1
                ):
                    yield (first,) + rest

        weight_counts = tuple(
            positive_compositions(
                robust_weight_resolution, len(scenarios)
            )
        )
        has_constraint = (
            min_goal_probability is not None
            or min_branch_goal_probability is not None
        )
        call_multiplier = 2 if has_constraint else 1
        required_calls = len(weight_counts) * call_multiplier
        if required_calls > max_robust_candidates:
            raise ValueError(
                "scenario-robust planning requires "
                f"{required_calls} scalarized candidates for "
                f"robust_weight_resolution={robust_weight_resolution}; "
                f"increase max_robust_candidates={max_robust_candidates} "
                "or lower the resolution"
            )

        def visible_state(state):
            return {
                key: value
                for key, value in state.items()
                if key != scenario_key
            }

        def scenario_outcomes(state, command):
            scenario = state[scenario_key]
            return scenario_models[scenario](
                visible_state(state), dict(command)
            )

        def scenario_observations(state, command):
            return observation_model(
                visible_state(state), dict(command)
            )

        def policy_signature(node: BeliefPolicyNode):
            return (
                tuple(
                    sorted(
                        (key, repr(value))
                        for key, value in node.action.items()
                    )
                ),
                tuple(
                    (
                        tuple(
                            sorted(
                                (key, repr(value))
                                for key, value in branch.observation.items()
                            )
                        ),
                        policy_signature(branch.policy)
                        if branch.policy is not None
                        else None,
                    )
                    for branch in node.branches
                ),
                policy_signature(node.fallback_policy)
                if node.fallback_policy is not None
                else None,
            )

        candidate_sources = {}
        candidate_weights = {}
        for counts in weight_counts:
            weights = tuple(
                (
                    name,
                    count / robust_weight_resolution,
                )
                for name, count in zip(scenario_names, counts)
            )
            weighted_belief = []
            for scenario, scenario_weight in weights:
                for state, mass in original_belief:
                    tagged = dict(state)
                    tagged[scenario_key] = scenario
                    weighted_belief.append(
                        (tagged, scenario_weight * float(mass))
                    )
            constraint_settings = (
                (
                    (None, None),
                    (
                        min_goal_probability,
                        min_branch_goal_probability,
                    ),
                )
                if has_constraint
                else ((None, None),)
            )
            for scalar_goal_floor, scalar_branch_floor in (
                constraint_settings
            ):
                result = self.plan_belief(
                    weighted_belief,
                    target,
                    action_costs=action_costs,
                    outcome_model=scenario_outcomes,
                    observation_model=scenario_observations,
                    actions=actions,
                    goal_reward=goal_reward,
                    cost_weight=cost_weight,
                    max_action_sequences=max_action_sequences,
                    max_outcome_branches=max_outcome_branches,
                    max_policy_nodes=max_policy_nodes,
                    max_observation_branches=max_observation_branches,
                    min_observation_probability=(
                        min_observation_probability
                    ),
                    max_observations_per_node=max_observations_per_node,
                    min_goal_probability=scalar_goal_floor,
                    min_branch_goal_probability=scalar_branch_floor,
                    max_frontier_points=max_frontier_points,
                    control=control,
                )
                if not isinstance(result, ConditionalBeliefPolicyResult):
                    raise RuntimeError(
                        "scenario-robust planning expected a conditional "
                        "policy result"
                    )
                signature = policy_signature(result.policy)
                candidate_sources.setdefault(signature, result)
                candidate_weights.setdefault(signature, weights)

        command_name = self.command_names[0]

        def checked_distribution(entries, *, label):
            checked = []
            total = 0.0
            for value, probability in entries:
                numeric = float(probability)
                if not math.isfinite(numeric) or numeric < 0.0:
                    raise ValueError(
                        f"{label} probabilities must be finite and "
                        f"non-negative; got {probability!r}"
                    )
                checked.append((dict(value), numeric))
                total += numeric
            if not checked or not math.isclose(
                total, 1.0, rel_tol=1e-6, abs_tol=1e-9
            ):
                raise ValueError(
                    f"{label} probabilities must sum to 1; got {total!r}"
                )
            return [
                (value, probability)
                for value, probability in checked
                if probability > 0.0
            ]

        def scenario_action_cost(action, command):
            if action_costs is None:
                value = 0.0
            elif callable(action_costs):
                value = action_costs(dict(command))
            else:
                if action not in action_costs:
                    raise KeyError(f"missing action cost for {action!r}")
                value = action_costs[action]
            numeric = float(value)
            if not math.isfinite(numeric) or numeric < 0.0:
                raise ValueError(
                    f"action cost for {action!r} must be finite and "
                    f"non-negative; got {value!r}"
                )
            return numeric

        def state_evidence(state, step, *, observables=False):
            evidence = {
                f"{name}@{step}": state[name]
                for name in self.mode_names
                if name in state
            }
            if observables:
                evidence.update(
                    {
                        f"{name}@{step}": state[name]
                        for name in self.obs_names
                        if name in state
                    }
                )
            return evidence

        def goal_probability(state, step):
            evidence = state_evidence(state, step, observables=True)
            denominator = self.system.log_evidence(evidence)
            if denominator == -math.inf:
                raise ValueError(
                    "robust policy reached a state inconsistent with "
                    f"planner: {state}"
                )
            numerator_evidence = dict(evidence)
            numerator_evidence.update(
                {
                    f"{name}@{step}": value
                    for name, value in target.items()
                }
            )
            numerator = self.system.log_evidence(numerator_evidence)
            if numerator == -math.inf:
                return 0.0
            return min(1.0, math.exp(numerator - denominator))

        def normalized_observations(state, command):
            supplied = observation_model(dict(state), dict(command))
            entries = (
                [(supplied, 1.0)]
                if isinstance(supplied, Mapping)
                else list(supplied)
            )
            return checked_distribution(
                entries, label="observation"
            )

        def audit_policy(policy, model):
            weighted = []
            total = 0.0
            for state, mass in original_belief:
                numeric = float(mass)
                if not math.isfinite(numeric) or numeric < 0.0:
                    raise ValueError(
                        "belief masses must be finite and non-negative"
                    )
                weighted.append((dict(state), numeric))
                total += numeric
            if total <= 0.0:
                raise ValueError("belief must have positive total mass")
            weighted = [
                (state, mass / total) for state, mass in weighted
            ]

            def execute(node, branch_belief, step, observation_path=()):
                reached_mass = 0.0
                active_belief = []
                for state, state_mass in branch_belief:
                    reached = goal_probability(state, step)
                    if reached >= 1.0 - 1e-12:
                        reached_mass += state_mass
                    else:
                        active_belief.append((state, state_mass))
                if not active_belief:
                    return reached_mass, 0.0, []

                command = dict(node.action)
                action = command[command_name]
                active_mass = sum(
                    state_mass for _, state_mass in active_belief
                )
                immediate_cost = (
                    active_mass * scenario_action_cost(action, command)
                )
                expanded = []
                for state, state_mass in active_belief:
                    outcomes = checked_distribution(
                        list(model(dict(state), dict(command))),
                        label="outcome",
                    )
                    for updates, probability in outcomes:
                        next_state = dict(state)
                        next_state.update(updates)
                        base = state_evidence(state, step)
                        base[f"{command_name}@{step}"] = action
                        base.update(
                            state_evidence(
                                next_state,
                                step + 1,
                                observables=True,
                            )
                        )
                        base.update(
                            {
                                f"{name}@{step + 1}": value
                                for name, value in updates.items()
                                if name in self.obs_names
                            }
                        )
                        if self.system.log_evidence(base) == -math.inf:
                            raise ValueError(
                                "outcome scenario produced a state "
                                f"inconsistent with planner: {base}"
                            )
                        expanded.append(
                            (
                                next_state,
                                state_mass * probability,
                            )
                        )
                if step + 1 == self.horizon:
                    expected_goal = (
                        reached_mass
                        + sum(
                            mass * goal_probability(state, step + 1)
                            for state, mass in expanded
                        )
                    )
                    return expected_goal, immediate_cost, []

                groups = {}
                for state, outcome_mass in expanded:
                    for observation, probability in (
                        normalized_observations(state, command)
                    ):
                        key = tuple(sorted(observation.items()))
                        group = groups.setdefault(
                            key,
                            {
                                "observation": observation,
                                "states": {},
                                "mass": 0.0,
                            },
                        )
                        state_key = tuple(sorted(state.items()))
                        branch_mass = outcome_mass * probability
                        if state_key in group["states"]:
                            group["states"][state_key][1] += branch_mass
                        else:
                            group["states"][state_key] = [
                                state,
                                branch_mass,
                            ]
                        group["mass"] += branch_mass

                expected_goal = reached_mass
                continuation_cost = 0.0
                branch_evaluations = []
                for group in groups.values():
                    probability = float(group["mass"])
                    posterior = [
                        (state, mass / probability)
                        for state, mass in group["states"].values()
                    ]
                    routed = node.route(group["observation"])
                    if routed.policy is None:
                        return None
                    child_path = observation_path + (
                        dict(group["observation"]),
                    )
                    child_goal, child_cost, descendants = execute(
                        routed.policy,
                        posterior,
                        step + 1,
                        child_path,
                    )
                    expected_goal += probability * child_goal
                    continuation_cost += probability * child_cost
                    branch_evaluations.append(
                        (child_path, child_goal)
                    )
                    branch_evaluations.extend(descendants)
                return (
                    expected_goal,
                    immediate_cost + continuation_cost,
                    branch_evaluations,
                )

            return execute(policy, weighted, 0)

        robust_candidates = []
        valid_sources = {}
        for signature, source in candidate_sources.items():
            scenario_evaluations = []
            valid = True
            for name, model in scenarios:
                audited = audit_policy(source.policy, model)
                if audited is None:
                    valid = False
                    break
                goal, cost, branch_evaluations = audited
                minimum_branch = (
                    min(
                        branch_evaluations,
                        key=lambda item: item[1],
                    )
                    if branch_evaluations
                    else None
                )
                scenario_evaluations.append(
                    RobustScenarioEvaluation(
                        scenario=name,
                        expected_goal_probability=goal,
                        expected_action_cost=cost,
                        expected_utility=(
                            goal_reward * goal - cost_weight * cost
                        ),
                        minimum_branch_goal_probability=(
                            minimum_branch[1]
                            if minimum_branch is not None
                            else None
                        ),
                        minimum_branch_observation_path=(
                            minimum_branch[0]
                            if minimum_branch is not None
                            else ()
                        ),
                    )
                )
            if not valid:
                continue
            worst_utility = min(
                scenario_evaluations,
                key=lambda item: item.expected_utility,
            )
            branch_scenarios = [
                evaluation
                for evaluation in scenario_evaluations
                if evaluation.minimum_branch_goal_probability is not None
            ]
            worst_branch = (
                min(
                    branch_scenarios,
                    key=lambda item: (
                        item.minimum_branch_goal_probability
                    ),
                )
                if branch_scenarios
                else None
            )
            robust_candidates.append(
                RobustPolicyEvaluation(
                    scenario_weights=candidate_weights[signature],
                    policy=source.policy,
                    scenario_evaluations=tuple(scenario_evaluations),
                    worst_case_scenario=worst_utility.scenario,
                    worst_case_expected_utility=(
                        worst_utility.expected_utility
                    ),
                    worst_case_goal_probability=min(
                        item.expected_goal_probability
                        for item in scenario_evaluations
                    ),
                    worst_case_branch_scenario=(
                        worst_branch.scenario
                        if worst_branch is not None
                        else None
                    ),
                    worst_case_minimum_branch_goal_probability=(
                        worst_branch.minimum_branch_goal_probability
                        if worst_branch is not None
                        else None
                    ),
                    worst_case_branch_observation_path=(
                        worst_branch.minimum_branch_observation_path
                        if worst_branch is not None
                        else ()
                    ),
                )
            )
            valid_sources[policy_signature(source.policy)] = source
        if not robust_candidates:
            raise ValueError(
                "no generated robust policy routed every scenario "
                "observation; enable observation fallback pruning or "
                "increase scenario weight coverage"
            )

        robust_best_goal = max(
            candidate.worst_case_goal_probability
            for candidate in robust_candidates
        )
        branch_values = [
            candidate.worst_case_minimum_branch_goal_probability
            for candidate in robust_candidates
            if (
                candidate.worst_case_minimum_branch_goal_probability
                is not None
            )
        ]
        robust_best_branch = (
            max(branch_values) if branch_values else None
        )

        def meets_goal_constraint(candidate):
            return (
                min_goal_probability is None
                or candidate.worst_case_goal_probability
                >= min_goal_probability - 1e-9
            )

        def meets_branch_constraint(candidate):
            return (
                min_branch_goal_probability is None
                or (
                    candidate.worst_case_minimum_branch_goal_probability
                    is not None
                    and (
                        candidate
                        .worst_case_minimum_branch_goal_probability
                        >= min_branch_goal_probability - 1e-9
                    )
                )
            )

        feasible_candidates = [
            candidate
            for candidate in robust_candidates
            if (
                meets_goal_constraint(candidate)
                and meets_branch_constraint(candidate)
            )
        ]
        robust_feasible = (
            bool(feasible_candidates) if has_constraint else None
        )
        if feasible_candidates:
            selected = max(
                feasible_candidates,
                key=lambda candidate: (
                    candidate.worst_case_expected_utility,
                    candidate.worst_case_goal_probability,
                ),
            )
        elif min_branch_goal_probability is not None:
            branch_feasible_candidates = [
                candidate
                for candidate in robust_candidates
                if meets_branch_constraint(candidate)
            ]
            if branch_feasible_candidates:
                selected = max(
                    branch_feasible_candidates,
                    key=lambda candidate: (
                        candidate.worst_case_goal_probability,
                        candidate.worst_case_expected_utility,
                    ),
                )
            else:
                selected = max(
                    robust_candidates,
                    key=lambda candidate: (
                        candidate
                        .worst_case_minimum_branch_goal_probability
                        if candidate
                        .worst_case_minimum_branch_goal_probability
                        is not None
                        else -math.inf,
                        candidate.worst_case_goal_probability,
                        candidate.worst_case_expected_utility,
                    ),
                )
        else:
            selected = max(
                robust_candidates,
                key=lambda candidate: (
                    candidate.worst_case_goal_probability,
                    candidate.worst_case_expected_utility,
                ),
            )
        selected_source = valid_sources[
            policy_signature(selected.policy)
        ]

        def policy_uses_fallback(node):
            return (
                node.fallback_policy is not None
                or any(
                    branch.policy is not None
                    and policy_uses_fallback(branch.policy)
                    for branch in node.branches
                )
            )

        selected_branch_feasible = (
            meets_branch_constraint(selected)
            if min_branch_goal_probability is not None
            else None
        )
        branch_certification = None
        if min_branch_goal_probability is not None:
            branch_certification = (
                "certified-feasible"
                if (
                    selected_branch_feasible
                    and not policy_uses_fallback(selected.policy)
                )
                else "indeterminate"
            )
        constraint_updates = {}
        if has_constraint:
            constraint_updates = {
                "goal_probability_constraint": min_goal_probability,
                "branch_goal_probability_constraint": (
                    min_branch_goal_probability
                ),
                "feasible": robust_feasible,
                "best_achievable_goal_probability": robust_best_goal,
                "best_achievable_goal_probability_upper_bound": 1.0,
                "best_achievable_goal_probability_scope": (
                    "robust-weight-grid-candidates"
                ),
                "constraint_certification": (
                    "certified-feasible"
                    if (
                        robust_feasible
                        and branch_certification
                        != "indeterminate"
                    )
                    else "indeterminate"
                ),
            }
        return replace(
            selected_source,
            action_ranking="heuristic",
            optimal_utility_upper_bound=goal_reward,
            maximum_regret=math.inf,
            root_action_certified=False,
            policy_maximum_regret=math.inf,
            policy_root_action_certified=False,
            certificate_scope="robust-scenario-weight-grid",
            robust_objective="maximin",
            robust_scenario_evaluations=(
                selected.scenario_evaluations
            ),
            robust_policy_evaluations=tuple(robust_candidates),
            inherited_metric_scope=(
                "scenario-weighted-candidate-generation"
            ),
            selected_scenario_weights=selected.scenario_weights,
            worst_case_scenario=selected.worst_case_scenario,
            worst_case_expected_utility=(
                selected.worst_case_expected_utility
            ),
            worst_case_goal_probability=(
                selected.worst_case_goal_probability
            ),
            robust_candidate_count=len(robust_candidates),
            robust_weight_resolution=robust_weight_resolution,
            robust_optimality="weight-grid-heuristic",
            robust_feasible=robust_feasible,
            robust_best_achievable_min_goal_probability=(
                robust_best_goal
            ),
            worst_case_branch_scenario=(
                selected.worst_case_branch_scenario
            ),
            worst_case_minimum_branch_goal_probability=(
                selected.worst_case_minimum_branch_goal_probability
            ),
            worst_case_branch_observation_path=(
                selected.worst_case_branch_observation_path
            ),
            robust_branch_feasible=selected_branch_feasible,
            robust_branch_constraint_certification=(
                branch_certification
            ),
            robust_best_achievable_min_branch_goal_probability=(
                robust_best_branch
            ),
            _execution_context=_BeliefPolicyExecutionContext(
                initial_belief=tuple(
                    (dict(state), float(mass))
                    for state, mass in original_belief
                ),
                outcome_model=None,
                outcome_scenarios=scenarios,
                observation_model=observation_model,
                goal_probability=goal_probability,
                action_cost=(
                    lambda command: scenario_action_cost(
                        command[command_name], command
                    )
                ),
            ),
            **constraint_updates,
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
