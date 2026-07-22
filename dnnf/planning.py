"""Reconfiguration planning on compiled circuits (MEXEC-style).

Declare a transition system — modes, commands, and command-conditioned
transitions with neg-log costs — and compile an n-step unrolling into
one FD circuit.  Planning is then a tropical (min-sum) evaluation:
clamp step 0 to the current modes and step n to the target, minimize,
and decode the command variables; the same compiled structure answers
"what commands reach the target" the way the diagnosis circuits answer
"what is broken" (Barrett 2005; Darwiche & Marquis 2004).

Kept deliberately simple in v1: transition preconditions are command
values; ``noop`` (persistence) is always available at cost 0; costs are
additive neg-log likelihoods, so the returned plan is the most probable
one within the horizon.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from . import fd
from .diagnosis import SystemModel


@dataclass
class _Transition:
    mode: str
    frm: str
    to: str
    command: Optional[Tuple[str, object]]  # (command var, required value)
    cost: float


class Planner:
    """Build a transition system, then ``compile(horizon)`` and ``plan``."""

    def __init__(self) -> None:
        self._modes: Dict[str, Sequence] = {}
        self._commands: Dict[str, Sequence] = {}
        self._transitions: List[_Transition] = []

    def mode(self, name: str, values: Sequence) -> None:
        self._modes[name] = tuple(values)

    def command(self, name: str, values: Sequence) -> None:
        """A per-step command input; include an inert value (e.g.
        ``"none"``) if doing nothing must be expressible."""
        self._commands[name] = tuple(values)

    def transition(
        self,
        mode: str,
        frm: str,
        to: str,
        command: Optional[Tuple[str, object]] = None,
        cost: float = 0.0,
    ) -> None:
        """Allow ``mode`` to move ``frm -> to`` on steps where the
        command condition holds; ``cost`` is neg-log likelihood."""
        self._transitions.append(_Transition(mode, frm, to, command, cost))

    # ------------------------------------------------------------------
    def compile(self, horizon: int) -> "CompiledPlanner":
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        m = SystemModel()
        mode_vars: Dict[Tuple[str, int], object] = {}
        cmd_vars: Dict[Tuple[str, int], object] = {}
        for t in range(horizon + 1):
            for name, values in self._modes.items():
                mode_vars[(name, t)] = m.finite(f"{name}@{t}", values)
        for t in range(horizon):
            for name, values in self._commands.items():
                cmd_vars[(name, t)] = m.finite(f"{name}@{t}", values)

        trans_weight: Dict[int, float] = {}
        for t in range(horizon):
            for name, values in self._modes.items():
                trans = [
                    tr for tr in self._transitions if tr.mode == name
                ]
                labels = ["noop"] + [f"t{i}" for i in range(len(trans))]
                weights = [1.0] + [math.exp(-tr.cost) for tr in trans]
                tv = m.finite(f"_{name}@{t}#trans", labels)
                spec = m.cnf.spec
                for i, w in enumerate(weights):
                    if w != 1.0:
                        m._prior_weights[spec.mvlit(tv.fd_var, i)] = w
                cur, nxt = mode_vars[(name, t)], mode_vars[(name, t + 1)]
                for v in values:  # noop: mode persists
                    m.add((tv == "noop") >> ((cur != v) | (nxt == v)))
                for i, tr in enumerate(trans):
                    effect = (cur == tr.frm) & (nxt == tr.to)
                    if tr.command is not None:
                        cname, cval = tr.command
                        effect = effect & (cmd_vars[(cname, t)] == cval)
                    m.add((tv == f"t{i}") >> effect)
        system = m.compile(modes_first=False)
        return CompiledPlanner(system, list(self._modes), list(self._commands), horizon)


class CompiledPlanner:
    def __init__(self, system, mode_names, command_names, horizon):
        self.system = system
        self.mode_names = mode_names
        self.command_names = command_names
        self.horizon = horizon

    def plan(
        self, current: Dict[str, object], target: Dict[str, object]
    ) -> Optional[Tuple[float, List[Dict[str, object]]]]:
        """Most probable command sequence from ``current`` modes to
        ``target`` modes within the horizon, or None if unreachable.
        Returns ``(cost, [per-step {command: value}])``."""
        evidence = {}
        for name, value in current.items():
            evidence[f"{name}@0"] = value
        for name, value in target.items():
            evidence[f"{name}@{self.horizon}"] = value
        costs = self.system._conditioned_costs(evidence)
        cost, assignment = fd.mpe(self.system.circuit, costs)
        if assignment is None:
            return None
        steps: List[Dict[str, object]] = []
        for t in range(self.horizon):
            step = {}
            for cname in self.command_names:
                var = self.system.vars[f"{cname}@{t}"]
                step[cname] = var.values[assignment[var.fd_var]]
            steps.append(step)
        return cost, steps
