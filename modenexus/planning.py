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
from typing import Callable, Dict, List, Optional, Sequence, Tuple

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
        the initial-mode likelihood used by estimation."""
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
                labels = ["noop"] + [f"t{i}" for i in range(len(trans))]
                tv = m.finite(f"_{name}@{t}#trans", labels)
                for i, tr in enumerate(trans):
                    w = math.exp(-tr.cost)
                    if w != 1.0:
                        m._prior_weights[spec.mvlit(tv.fd_var, i + 1)] = w
                cur, nxt = step_vars[t][name], step_vars[t + 1][name]
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
                selector = self.system.vars[f"_{name}@{step}#trans"]
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
