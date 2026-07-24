"""Temporal mode tracking: filtering a belief over mode assignments.

This is the monitoring loop that snapshot diagnosis lacks — the
Livingstone-style capability: modes evolve stochastically between
timesteps, observations arrive each step, and the tracker maintains a
belief over joint mode assignments.

Semantics.  Mode variables evolve independently between steps with
per-variable transition matrices ``T[name][from_value][to_value]``; the
system's constraints and observation structure are the compiled circuit,
evaluated fresh each step.  One filtering step from belief ``B_t``:

    B_{t+1}(M') proportional-to
        sum_{M in beam} B_t(M) * [ sum over non-mode vars of
            product of literal weights, with mode priors set to
            T[.][M[.]][.] and evidence e_{t+1} applied, at modes M' ]

The inner bracket is exactly a marginal-MAP-style summed mass, so each
step is |beam| conditioned circuit sweeps plus lazy ranked enumeration —
no new inference machinery.

Exactness.  Belief propagation is **beam-limited**: only the ``beam``
most probable mode assignments survive each step, and each survivor
proposes its ``expand`` best successors.  When ``beam`` and ``expand``
cover the full joint mode space the recursion is the exact HMM forward
algorithm (verified in tests); with smaller beams it is the standard
best-first approximation used by tracking diagnosis engines, and mass
outside the beam is dropped (renormalized away).
"""

from __future__ import annotations

import math
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .diagnosis import CompiledSystem, EvidenceValue

ModeAssignment = Tuple[Tuple[str, str], ...]  # sorted ((var, value), ...)
Transitions = Dict[str, Dict[str, Dict[str, float]]]


class TrackedBelief(list):
    """Normalized tracker belief carrying approximation metadata.

    This remains a ``list`` for compatibility with existing consumers.
    ``retained_probability_mass`` is a lower bound relative to the exact
    posterior when known; ``None`` means no nontrivial mass certificate is
    available.
    """

    def __init__(
        self,
        values,
        *,
        exact: bool,
        retained_probability_mass: Optional[float],
    ) -> None:
        super().__init__(values)
        self.exact = exact
        self.retained_probability_mass = retained_probability_mass
        self.source = "ModeTracker"


@dataclass(frozen=True)
class TrackingStepInfo:
    """Diagnostics for the most recent filtering step."""

    timestep: int
    joint_state_count: int
    previous_states: int
    generated_candidates: int
    retained_states: int
    expansion_truncated: bool
    beam_truncated: bool
    retained_probability_mass: Optional[float]
    exact: bool


@dataclass(frozen=True)
class TrackingHistoryStep:
    """One replayable filtering input retained by :class:`ModeTracker`."""

    evidence: Dict[str, EvidenceValue]
    transitions: Optional[Transitions]


@dataclass(frozen=True)
class TrackingReplayInfo:
    """Work performed while rebuilding a tracker at higher resources."""

    source_beam: int
    source_expand: int
    target_beam: int
    target_expand: int
    steps_replayed: int
    generated_candidates: int
    replay_seconds: float
    exact: bool
    retained_probability_mass: Optional[float]


@dataclass(frozen=True)
class TrackingRefinementAttempt:
    """One downstream evaluation in an adaptive refinement run."""

    beam: int
    expand: int
    steps_replayed: int
    generated_candidates: int
    replay_seconds: float
    evaluation_seconds: float
    exact: bool
    retained_probability_mass: Optional[float]
    accepted: bool
    certificate_scope: Optional[str]
    maximum_regret: Optional[float]
    evaluation: Any


@dataclass(frozen=True)
class TrackingRefinementResult:
    """Final tracker, evaluation, and measured adaptive-refinement work."""

    tracker: "ModeTracker"
    evaluation: Any
    accepted: bool
    attempts: Tuple[TrackingRefinementAttempt, ...]

    @property
    def total_steps_replayed(self) -> int:
        return sum(attempt.steps_replayed for attempt in self.attempts)

    @property
    def total_generated_candidates(self) -> int:
        return sum(
            attempt.generated_candidates for attempt in self.attempts
        )

    @property
    def total_replay_seconds(self) -> float:
        return sum(attempt.replay_seconds for attempt in self.attempts)

    @property
    def total_evaluation_seconds(self) -> float:
        return sum(attempt.evaluation_seconds for attempt in self.attempts)


def _logsumexp(values: List[float]) -> float:
    m = max(values)
    if m == -math.inf:
        return -math.inf
    return m + math.log(sum(math.exp(v - m) for v in values))


class ModeTracker:
    """Beam-filtered belief over joint mode assignments across time.

    Parameters
    ----------
    system:
        A compiled system (``modes_first`` compilation, the default).
    transitions:
        ``{mode_var: {from_value: {to_value: prob}}}``.  Mode variables
        without an entry are *resampled from their static priors* each
        step (i.e. treated as memoryless).  Rows should sum to 1.
    transition_fn:
        Optional callable ``prev_modes -> {mode_var: {value: prob}}``
        giving each variable's next-value distribution as a function of
        the **entire** previous joint mode assignment — this is how
        correlated dynamics are expressed (e.g. a pump failure raising
        the valve's failure rate).  Variables missing from its result
        fall back to ``transitions`` (then to static priors).  A per-step
        ``transitions`` override passed to :meth:`step` takes precedence
        over both.
    beam:
        Maximum number of mode assignments kept in the belief.
    expand:
        Successors proposed per belief particle per step (defaults to
        ``beam``).
    exact:
        Size ``beam`` and ``expand`` to the full Cartesian product of mode
        domains.  This gives exact filtering while keeping the exponential
        state-space requirement explicit in ``joint_state_count``.
    max_exact_states:
        Safety limit for ``exact=True``.  Raise it deliberately when the
        computed state-space size is acceptable for the deployment.
    retain_history:
        Retain successful step evidence and per-step transition overrides so
        :meth:`refine` can rebuild the posterior with a larger beam. History
        retention is opt-in because evidence volume can grow without bound.

    ``belief()`` returns a list-compatible :class:`TrackedBelief` carrying
    exactness and retained-mass metadata for downstream certificate
    composition. ``last_step_info`` reports whether expansion or final beam
    selection truncated an update. Retained posterior mass is reported only
    when the predecessor belief was exact and all successor expansions were
    enumerated; otherwise it is conservatively unknown.
    """

    def __init__(
        self,
        system: CompiledSystem,
        transitions: Optional[Transitions] = None,
        beam: Optional[int] = None,
        expand: Optional[int] = None,
        transition_fn=None,
        exact: bool = False,
        max_exact_states: int = 100_000,
        retain_history: bool = False,
    ):
        self.system = system
        self.joint_state_count = math.prod(
            len(system.finites[name].values) for name in system.mode_vars
        )
        if exact:
            if self.joint_state_count > max_exact_states:
                raise ValueError(
                    f"exact tracking needs {self.joint_state_count} joint "
                    f"states, exceeding max_exact_states={max_exact_states}"
                )
            if beam is not None and beam < self.joint_state_count:
                raise ValueError(
                    f"exact tracking requires beam >= "
                    f"{self.joint_state_count}"
                )
            if expand is not None and expand < self.joint_state_count:
                raise ValueError(
                    f"exact tracking requires expand >= "
                    f"{self.joint_state_count}"
                )
            beam = self.joint_state_count
            expand = self.joint_state_count
        self.beam = 10 if beam is None else beam
        self.expand = self.beam if expand is None else expand
        if self.beam < 1 or self.expand < 1:
            raise ValueError("beam and expand must be positive")
        self.is_exact = (
            self.beam >= self.joint_state_count
            and self.expand >= self.joint_state_count
        )
        self.transition_fn = transition_fn
        transitions = transitions or {}
        for name, matrix in transitions.items():
            fv = system.finites[name]
            for from_value, row in matrix.items():
                if from_value not in fv.values:
                    raise KeyError(f"{name} has no value {from_value!r}")
                total = sum(row.values())
                if not math.isclose(total, 1.0, rel_tol=1e-6):
                    raise ValueError(
                        f"transition row {name}[{from_value}] sums to {total}"
                    )
        self.transitions = deepcopy(transitions)
        self.retain_history = bool(retain_history)
        self._history: List[TrackingHistoryStep] = []
        self.last_replay_info: Optional[TrackingReplayInfo] = None
        # Belief: {mode assignment: log mass}, unnormalized.
        self._belief: Dict[ModeAssignment, float] = {}
        self._belief_exact = False
        self._retained_probability_mass: Optional[float] = None
        self.t = 0
        self.last_step_info: Optional[TrackingStepInfo] = None
        self._init_belief()

    def _init_belief(self) -> None:
        log_w = self.system.log_weights_for({})
        ranked = self.system.ranked_map(log_w, self.beam)
        for cost, modes in ranked:
            self._belief[tuple(sorted(modes.items()))] = -cost
        retained_log_mass = _logsumexp([-cost for cost, _ in ranked])
        total_log_mass = self.system.log_evidence({})
        self._retained_probability_mass = min(
            1.0,
            math.exp(retained_log_mass - total_log_mass),
        )
        # ``expand`` affects the next update, not whether the current
        # initialization enumerated the complete mode space.
        self._belief_exact = self.beam >= self.joint_state_count

    # ------------------------------------------------------------------
    def step(
        self,
        evidence: Dict[str, EvidenceValue],
        transitions: Optional[Transitions] = None,
    ) -> List[Tuple[Dict[str, str], float]]:
        """Advance one timestep with the given observations; returns the
        updated (normalized) belief as ``[(modes, prob), ...]``, most
        probable first.  Raises ValueError if the evidence is inconsistent
        with every tracked trajectory (belief collapse — enlarge the beam
        or check the model).

        ``transitions``, if given, overrides the tracker's transition
        matrices *for this step only* (per mode variable; unlisted
        variables keep their defaults).  This is how command-conditioned
        dynamics work: pass the matrix matching what was commanded this
        tick — e.g. a valve only risks transitioning to ``stuck_open``
        on a step where it was actually commanded to open."""
        step_transitions = dict(self.transitions)
        if transitions:
            step_transitions.update(transitions)
        candidates: Dict[ModeAssignment, List[float]] = {}
        previous_states = len(self._belief)
        previous_belief_exact = self._belief_exact
        expansion_truncated = False
        for modes_key, log_mass in self._belief.items():
            prev = dict(modes_key)
            mode_priors = {
                name: matrix[prev[name]]
                for name, matrix in step_transitions.items()
            }
            if self.transition_fn is not None:
                correlated = self.transition_fn(prev)
                for name, dist in correlated.items():
                    if transitions is None or name not in transitions:
                        mode_priors[name] = dist
            # Joint transition constraints: condition the compiled
            # prev-slice variables to this particle's modes, so hard
            # relations between consecutive slices prune candidates.
            step_evidence = evidence
            if self.system.prev_map:
                step_evidence = dict(evidence)
                for mode_name, prev_name in self.system.prev_map.items():
                    step_evidence[prev_name] = prev[mode_name]
            log_w = self.system.log_weights_for(step_evidence, mode_priors)
            probe = (
                self.expand
                if self.expand >= self.joint_state_count
                else self.expand + 1
            )
            successors = self.system.ranked_map(log_w, probe)
            if len(successors) > self.expand:
                expansion_truncated = True
                successors = successors[:self.expand]
            for cost, modes in successors:
                key = tuple(sorted(modes.items()))
                candidates.setdefault(key, []).append(log_mass - cost)
        merged = {
            key: _logsumexp(parts) for key, parts in candidates.items()
        }
        merged = {k: v for k, v in merged.items() if v > -math.inf}
        if not merged:
            raise ValueError(
                "evidence inconsistent with all tracked trajectories; "
                "increase beam or revisit the model"
            )
        ranked = sorted(merged.items(), key=lambda kv: -kv[1])
        beam_truncated = len(ranked) > self.beam
        top = ranked[: self.beam]
        # Renormalize to keep log masses well-scaled over long runs.
        z = _logsumexp([v for _, v in top])
        retained_probability_mass = None
        if not expansion_truncated and previous_belief_exact:
            all_z = _logsumexp(list(merged.values()))
            retained_probability_mass = math.exp(z - all_z)
        self._belief = {k: v - z for k, v in top}
        self._belief_exact = (
            previous_belief_exact
            and not expansion_truncated
            and not beam_truncated
        )
        self._retained_probability_mass = (
            1.0 if self._belief_exact else retained_probability_mass
        )
        self.t += 1
        self.last_step_info = TrackingStepInfo(
            timestep=self.t,
            joint_state_count=self.joint_state_count,
            previous_states=previous_states,
            generated_candidates=len(merged),
            retained_states=len(top),
            expansion_truncated=expansion_truncated,
            beam_truncated=beam_truncated,
            retained_probability_mass=retained_probability_mass,
            exact=self._belief_exact,
        )
        if self.retain_history:
            self._history.append(
                TrackingHistoryStep(
                    evidence=deepcopy(evidence),
                    transitions=deepcopy(transitions),
                )
            )
        return self.belief()

    # ------------------------------------------------------------------
    def belief(self) -> TrackedBelief:
        """Current normalized beam plus exactness and retained-mass metadata."""
        z = _logsumexp(list(self._belief.values()))
        ranked = sorted(self._belief.items(), key=lambda kv: -kv[1])
        return TrackedBelief(
            [(dict(k), math.exp(v - z)) for k, v in ranked],
            exact=self._belief_exact,
            retained_probability_mass=self._retained_probability_mass,
        )

    def marginals(self) -> Dict[str, Dict[str, float]]:
        """Per-mode-variable marginals of the current belief."""
        out: Dict[str, Dict[str, float]] = {
            name: {v: 0.0 for v in self.system.finites[name].values}
            for name in self.system.mode_vars
        }
        for modes, prob in self.belief():
            for name, value in modes.items():
                out[name][value] += prob
        return out

    def most_probable(self) -> Tuple[Dict[str, str], float]:
        return self.belief()[0]

    @property
    def history(self) -> Tuple[TrackingHistoryStep, ...]:
        """A defensive copy of retained successful filtering inputs."""
        return tuple(deepcopy(self._history))

    @classmethod
    def from_history(
        cls,
        system: CompiledSystem,
        history: Sequence[TrackingHistoryStep],
        transitions: Optional[Transitions] = None,
        *,
        beam: Optional[int] = None,
        expand: Optional[int] = None,
        transition_fn=None,
        exact: bool = False,
        max_exact_states: int = 100_000,
        retain_history: bool = True,
    ) -> "ModeTracker":
        """Build a tracker and replay an explicit retained history.

        Only successful steps belong in ``history``. The replay uses the
        supplied base transitions and transition function, plus the
        per-step overrides captured in each :class:`TrackingHistoryStep`.
        A stateful ``transition_fn`` must itself be replay-safe.
        """
        tracker = cls(
            system,
            transitions=deepcopy(transitions),
            beam=beam,
            expand=expand,
            transition_fn=transition_fn,
            exact=exact,
            max_exact_states=max_exact_states,
            retain_history=retain_history,
        )
        generated_candidates = 0
        replay_started = time.perf_counter()
        for entry in history:
            if not isinstance(entry, TrackingHistoryStep):
                raise TypeError(
                    "history entries must be TrackingHistoryStep instances"
                )
            tracker.step(
                deepcopy(entry.evidence),
                transitions=deepcopy(entry.transitions),
            )
            generated_candidates += tracker.last_step_info.generated_candidates
        replay_seconds = time.perf_counter() - replay_started
        tracker.last_replay_info = TrackingReplayInfo(
            source_beam=tracker.beam,
            source_expand=tracker.expand,
            target_beam=tracker.beam,
            target_expand=tracker.expand,
            steps_replayed=len(history),
            generated_candidates=generated_candidates,
            replay_seconds=replay_seconds,
            exact=tracker.belief().exact,
            retained_probability_mass=(
                tracker.belief().retained_probability_mass
            ),
        )
        return tracker

    def refine(
        self,
        *,
        beam: Optional[int] = None,
        expand: Optional[int] = None,
        exact: bool = False,
        max_exact_states: int = 100_000,
    ) -> "ModeTracker":
        """Return a freshly replayed tracker with greater resources.

        Refinement never mutates this tracker and never pretends discarded
        trajectories can be recovered in place. For a tracker that has
        advanced, ``retain_history=True`` must have been selected at
        construction. With no explicit target, both resources double up to
        the complete joint state count.
        """
        if self.t and not self.retain_history:
            raise RuntimeError(
                "tracker refinement requires retain_history=True before "
                "filtering steps are recorded"
            )
        if len(self._history) != self.t:
            raise RuntimeError("retained tracker history is incomplete")

        if exact:
            target_beam = self.joint_state_count
            target_expand = self.joint_state_count
        else:
            target_beam = (
                min(
                    self.joint_state_count,
                    max(self.beam + 1, self.beam * 2),
                )
                if beam is None
                else beam
            )
            target_expand = (
                max(self.expand, target_beam)
                if expand is None
                else expand
            )
            if target_beam < self.beam or target_expand < self.expand:
                raise ValueError(
                    "refinement resources cannot be smaller than the "
                    "current beam and expand"
                )
            if (
                target_beam == self.beam
                and target_expand == self.expand
            ):
                raise ValueError(
                    "refinement must increase beam or expand"
                )

        refined = type(self).from_history(
            self.system,
            self._history,
            transitions=self.transitions,
            beam=None if exact else target_beam,
            expand=None if exact else target_expand,
            transition_fn=self.transition_fn,
            exact=exact,
            max_exact_states=max_exact_states,
            retain_history=True,
        )
        replay = refined.last_replay_info
        refined.last_replay_info = TrackingReplayInfo(
            source_beam=self.beam,
            source_expand=self.expand,
            target_beam=refined.beam,
            target_expand=refined.expand,
            steps_replayed=replay.steps_replayed,
            generated_candidates=replay.generated_candidates,
            replay_seconds=replay.replay_seconds,
            exact=replay.exact,
            retained_probability_mass=replay.retained_probability_mass,
        )
        return refined

    def refine_until(
        self,
        evaluate: Callable[[TrackedBelief], Any],
        accept: Callable[[Any], bool],
        *,
        beams: Optional[Sequence[int]] = None,
        max_beam: Optional[int] = None,
        growth_factor: float = 2.0,
        exact_fallback: bool = True,
        max_exact_states: int = 100_000,
    ) -> TrackingRefinementResult:
        """Replay at increasing resources until ``accept(evaluate(...))``.

        The current belief is evaluated first without replay. By default,
        the beam grows geometrically and the last permitted attempt is exact.
        Explicit ``beams`` make the resource schedule fully deterministic.
        The returned attempts expose replay work and, when the evaluation
        provides them, ``certificate_scope`` and ``maximum_regret``.
        """
        if growth_factor <= 1.0:
            raise ValueError("growth_factor must be greater than 1")
        limit = self.joint_state_count if max_beam is None else max_beam
        if limit < self.beam:
            raise ValueError("max_beam cannot be smaller than current beam")

        if beams is None:
            targets = []
            candidate = self.beam
            while candidate < min(limit, self.joint_state_count):
                candidate = min(
                    min(limit, self.joint_state_count),
                    max(candidate + 1, math.ceil(candidate * growth_factor)),
                )
                targets.append(candidate)
        else:
            targets = list(beams)
            if any(target <= self.beam for target in targets):
                raise ValueError(
                    "refinement beams must be greater than current beam"
                )
            if any(
                right <= left
                for left, right in zip(targets, targets[1:])
            ):
                raise ValueError(
                    "refinement beams must be strictly increasing"
                )
            if any(target > limit for target in targets):
                raise ValueError("refinement beam exceeds max_beam")

        if (
            exact_fallback
            and limit >= self.joint_state_count
            and self.joint_state_count > self.beam
            and self.joint_state_count not in targets
        ):
            targets.append(self.joint_state_count)

        attempts = []
        current = self
        evaluation_started = time.perf_counter()
        evaluation = evaluate(current.belief())
        evaluation_seconds = time.perf_counter() - evaluation_started

        def record(
            tracker: "ModeTracker",
            result: Any,
            accepted: bool,
            result_seconds: float,
        ) -> TrackingRefinementAttempt:
            replay = tracker.last_replay_info
            belief = tracker.belief()
            return TrackingRefinementAttempt(
                beam=tracker.beam,
                expand=tracker.expand,
                steps_replayed=0 if replay is None else replay.steps_replayed,
                generated_candidates=(
                    0 if replay is None else replay.generated_candidates
                ),
                replay_seconds=(
                    0.0 if replay is None else replay.replay_seconds
                ),
                evaluation_seconds=result_seconds,
                exact=belief.exact,
                retained_probability_mass=(
                    belief.retained_probability_mass
                ),
                accepted=accepted,
                certificate_scope=getattr(
                    result, "certificate_scope", None
                ),
                maximum_regret=getattr(result, "maximum_regret", None),
                evaluation=result,
            )

        accepted = bool(accept(evaluation))
        attempts.append(
            record(current, evaluation, accepted, evaluation_seconds)
        )
        for target in targets:
            if accepted:
                break
            current = self.refine(
                beam=target,
                expand=target,
                exact=target >= self.joint_state_count,
                max_exact_states=max_exact_states,
            )
            evaluation_started = time.perf_counter()
            evaluation = evaluate(current.belief())
            evaluation_seconds = time.perf_counter() - evaluation_started
            accepted = bool(accept(evaluation))
            attempts.append(
                record(current, evaluation, accepted, evaluation_seconds)
            )

        return TrackingRefinementResult(
            tracker=current,
            evaluation=evaluation,
            accepted=accepted,
            attempts=tuple(attempts),
        )
