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
from typing import Dict, List, Optional, Tuple

from .diagnosis import CompiledSystem, EvidenceValue

ModeAssignment = Tuple[Tuple[str, str], ...]  # sorted ((var, value), ...)
Transitions = Dict[str, Dict[str, Dict[str, float]]]


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
    beam:
        Maximum number of mode assignments kept in the belief.
    expand:
        Successors proposed per belief particle per step (defaults to
        ``beam``).
    """

    def __init__(
        self,
        system: CompiledSystem,
        transitions: Transitions,
        beam: int = 10,
        expand: Optional[int] = None,
    ):
        self.system = system
        self.beam = beam
        self.expand = expand or beam
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
        self.transitions = transitions
        # Belief: {mode assignment: log mass}, unnormalized.
        self._belief: Dict[ModeAssignment, float] = {}
        self.t = 0
        self._init_belief()

    def _init_belief(self) -> None:
        log_w = self.system.log_weights_for({})
        for cost, modes in self.system.ranked_map(log_w, self.beam):
            self._belief[tuple(sorted(modes.items()))] = -cost

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
        for modes_key, log_mass in self._belief.items():
            prev = dict(modes_key)
            mode_priors = {
                name: matrix[prev[name]]
                for name, matrix in step_transitions.items()
            }
            log_w = self.system.log_weights_for(evidence, mode_priors)
            for cost, modes in self.system.ranked_map(log_w, self.expand):
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
        top = sorted(merged.items(), key=lambda kv: -kv[1])[: self.beam]
        # Renormalize to keep log masses well-scaled over long runs.
        z = _logsumexp([v for _, v in top])
        self._belief = {k: v - z for k, v in top}
        self.t += 1
        return self.belief()

    # ------------------------------------------------------------------
    def belief(self) -> List[Tuple[Dict[str, str], float]]:
        """Current belief, normalized over the beam, most probable first."""
        z = _logsumexp(list(self._belief.values()))
        ranked = sorted(self._belief.items(), key=lambda kv: -kv[1])
        return [(dict(k), math.exp(v - z)) for k, v in ranked]

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
