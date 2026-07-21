"""Model-based diagnosis on compiled DNNF circuits.

This layer mirrors the architecture of the JPL DNNF diagnosis engines: a
system is described as components with discrete *modes* (carrying prior
probabilities) plus *observables*, connected by propositional constraints.
The description is encoded to CNF, compiled once (offline) to a smooth
d-DNNF, and then queried online:

* :meth:`CompiledSystem.diagnoses` — the k most probable complete system
  states consistent with the observations, projected onto mode variables
  and ordered from most to least probable (best-first enumeration over the
  circuit with neg-log-probability leaf weights);
* :meth:`CompiledSystem.mode_posteriors` — exact posterior marginals
  ``P(mode = value | evidence)`` via weighted-model-count ratios;
* :meth:`CompiledSystem.log_evidence` — ``log P(evidence)``.

Priors: mode values are one-hot encoded with exactly-one constraints; the
positive literal of value ``m`` gets weight ``P(m)`` (cost ``-log P(m)``)
and every other literal weight 1 (cost 0), the standard literal-weighted
WMC encoding of discrete priors.  Boolean observables default to an
uninformative weight of 1 per polarity, which cancels in every posterior
ratio.  Tseitin auxiliaries are weight-neutral and functionally determined,
so they affect nothing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union

from . import eval as _eval
from .circuit import Circuit, lit_index
from .cnf import CNF
from .compiler import compile_cnf
from .formula import Formula, Prop, encode, exactly_one
from .kbest import enumerate_models

EvidenceValue = Union[bool, str]


class FiniteVar:
    """A finite-domain variable, one-hot encoded over propositional vars."""

    def __init__(self, name: str, values: Sequence[str], var_ids: Sequence[int]):
        self.name = name
        self.values = tuple(values)
        self.var_ids = tuple(var_ids)
        self._props = {
            v: Prop(i, f"{name}={v}") for v, i in zip(values, var_ids)
        }

    def __eq__(self, value: str) -> Formula:  # type: ignore[override]
        if value not in self._props:
            raise KeyError(f"{self.name} has no value {value!r}")
        return self._props[value]

    def __ne__(self, value: str) -> Formula:  # type: ignore[override]
        return ~(self == value)

    def __hash__(self):
        return hash((self.name, self.values, self.var_ids))

    def props(self) -> List[Prop]:
        return [self._props[v] for v in self.values]


@dataclass
class Diagnosis:
    """One ranked system state."""

    modes: Dict[str, str]
    state: Dict[str, EvidenceValue]
    cost: float  # neg-log joint probability (up to observable weighting)
    posterior: float  # normalized P(state | evidence)

    def __repr__(self) -> str:  # pragma: no cover
        modes = ", ".join(f"{k}={v}" for k, v in sorted(self.modes.items()))
        return f"Diagnosis({modes}; p={self.posterior:.4g})"


class SystemModel:
    """Declarative system description: modes, observables, constraints."""

    def __init__(self) -> None:
        self.cnf = CNF()
        self._constraints: List[Formula] = []
        self._bools: Dict[str, Prop] = {}
        self._finites: Dict[str, FiniteVar] = {}
        self._priors: Dict[int, float] = {}  # positive-literal weight per var
        self._mode_vars: List[str] = []

    # -- variable declaration ------------------------------------------
    def bool(self, name: str, prior: Optional[float] = None) -> Prop:
        """Declare a boolean variable.  ``prior`` (if given) is
        ``P(name = True)``; omitted means uninformative weight 1/1."""
        if name in self._bools or name in self._finites:
            raise ValueError(f"duplicate variable {name!r}")
        p = Prop(self.cnf.add_var(), name)
        self._bools[name] = p
        if prior is not None:
            self._priors[p.var] = prior
        return p

    def finite(
        self,
        name: str,
        values: Sequence[str],
        priors: Optional[Sequence[float]] = None,
        mode: bool = False,
    ) -> FiniteVar:
        """Declare a finite-domain variable (one-hot + exactly-one).

        With ``mode=True`` (or when ``priors`` are given) the variable is
        treated as a component mode: it appears in diagnoses and its priors
        weight the enumeration.
        """
        if name in self._bools or name in self._finites:
            raise ValueError(f"duplicate variable {name!r}")
        ids = [self.cnf.add_var() for _ in values]
        fv = FiniteVar(name, values, ids)
        self._finites[name] = fv
        self._constraints.append(exactly_one(fv.props()))
        if priors is not None:
            if len(priors) != len(values):
                raise ValueError("priors length must match values")
            total = sum(priors)
            for i, p in zip(ids, priors):
                self._priors[i] = p / total
        if mode or priors is not None:
            self._mode_vars.append(name)
        return fv

    def mode(
        self, name: str, values: Sequence[str], priors: Sequence[float]
    ) -> FiniteVar:
        """Shorthand for a component mode variable with priors."""
        return self.finite(name, values, priors=priors, mode=True)

    def add(self, formula: Formula) -> None:
        self._constraints.append(formula)

    # ------------------------------------------------------------------
    def compile(self, var_order: Optional[Sequence[int]] = None) -> "CompiledSystem":
        num_original = self.cnf.num_vars
        encode(self._constraints, self.cnf)
        circuit = compile_cnf(self.cnf, var_order=var_order, smooth=True)
        return CompiledSystem(
            circuit=circuit,
            bools=dict(self._bools),
            finites=dict(self._finites),
            priors=dict(self._priors),
            mode_vars=list(self._mode_vars),
            num_original=num_original,
        )


class CompiledSystem:
    def __init__(
        self,
        circuit: Circuit,
        bools: Dict[str, Prop],
        finites: Dict[str, FiniteVar],
        priors: Dict[int, float],
        mode_vars: List[str],
        num_original: int,
    ):
        self.circuit = circuit
        self.bools = bools
        self.finites = finites
        self.priors = priors
        self.mode_vars = mode_vars
        self.num_original = num_original
        n = circuit.num_vars
        # Probability weights (for WMC) and neg-log costs (for MPE/k-best).
        self._weights = [1.0] * (2 * n)
        for var, p in priors.items():
            self._weights[lit_index(var)] = p
        self._costs = [
            0.0 if w == 1.0 else (math.inf if w <= 0 else -math.log(w))
            for w in self._weights
        ]

    # -- evidence -------------------------------------------------------
    def _evidence_assignment(
        self, evidence: Dict[str, EvidenceValue]
    ) -> Dict[int, bool]:
        assign: Dict[int, bool] = {}
        for name, value in evidence.items():
            if name in self.bools:
                if not isinstance(value, bool):
                    raise TypeError(f"{name} is boolean; got {value!r}")
                assign[self.bools[name].var] = value
            elif name in self.finites:
                fv = self.finites[name]
                if value not in fv.values:
                    raise KeyError(f"{name} has no value {value!r}")
                for v, i in zip(fv.values, fv.var_ids):
                    assign[i] = v == value
            else:
                raise KeyError(f"unknown variable {name!r}")
        return assign

    def _conditioned(self, evidence: Dict[str, EvidenceValue]):
        assign = self._evidence_assignment(evidence)
        weights = list(self._weights)
        costs = list(self._costs)
        for var, val in assign.items():
            forbidden = -var if val else var
            weights[lit_index(forbidden)] = 0.0
            costs[lit_index(forbidden)] = math.inf
        return weights, costs

    # -- queries --------------------------------------------------------
    def log_evidence(self, evidence: Dict[str, EvidenceValue]) -> float:
        """``log P(evidence)`` up to the constant weighting of unweighted
        observables (exact when all non-mode vars are observed or
        deterministic given modes)."""
        weights, _ = self._conditioned(evidence)
        log_w = [
            -math.inf if w <= 0 else math.log(w) for w in weights
        ]
        return _eval.log_wmc(self.circuit, log_w)

    def diagnoses(
        self,
        evidence: Dict[str, EvidenceValue],
        k: int = 5,
        project_to_modes: bool = True,
    ) -> List[Diagnosis]:
        """The ``k`` most probable system states consistent with the
        evidence, most probable first.

        With ``project_to_modes=True`` states whose mode projection repeats
        an earlier (more probable) state are skipped, so the result is the
        k best *distinct* mode assignments by best-state probability.  Note
        this ranks mode assignments by their single best supporting state
        (pure MPE semantics), not by summing over states — exact marginal
        posteriors per mode are available from :meth:`mode_posteriors`.
        """
        weights, costs = self._conditioned(evidence)
        log_z = self.log_evidence(evidence)
        if log_z == -math.inf:
            return []
        out: List[Diagnosis] = []
        seen_modes: set = set()
        # Enumerate generously; projection may collapse states.
        for cost, assignment in enumerate_models(
            self.circuit, costs, k=None
        ):
            modes = {
                name: self._decode_finite(name, assignment)
                for name in self.mode_vars
            }
            key = tuple(sorted(modes.items()))
            if project_to_modes:
                if key in seen_modes:
                    continue
                seen_modes.add(key)
            state = self._decode_state(assignment)
            out.append(
                Diagnosis(
                    modes=modes,
                    state=state,
                    cost=cost,
                    posterior=math.exp(-cost - log_z),
                )
            )
            if len(out) >= k:
                break
        return out

    def mode_posteriors(
        self, evidence: Dict[str, EvidenceValue]
    ) -> Dict[str, Dict[str, float]]:
        """Exact ``P(mode = value | evidence)`` for every mode variable,
        via WMC ratios."""
        log_z = self.log_evidence(evidence)
        if log_z == -math.inf:
            raise ValueError("evidence is inconsistent with the model")
        out: Dict[str, Dict[str, float]] = {}
        for name in self.mode_vars:
            fv = self.finites[name]
            dist: Dict[str, float] = {}
            for value in fv.values:
                ev = dict(evidence)
                ev[name] = value
                try:
                    log_num = self.log_evidence(ev)
                except KeyError:  # pragma: no cover
                    log_num = -math.inf
                dist[value] = math.exp(log_num - log_z)
            out[name] = dist
        return out

    # -- decoding -------------------------------------------------------
    def _decode_finite(self, name: str, assignment: Dict[int, bool]) -> str:
        fv = self.finites[name]
        for value, var in zip(fv.values, fv.var_ids):
            if assignment.get(var, False):
                return value
        return "?"  # unreachable on smooth circuits with exactly-one

    def _decode_state(
        self, assignment: Dict[int, bool]
    ) -> Dict[str, EvidenceValue]:
        state: Dict[str, EvidenceValue] = {}
        for name, prop in self.bools.items():
            if prop.var in assignment:
                state[name] = assignment[prop.var]
        for name in self.finites:
            state[name] = self._decode_finite(name, assignment)
        return state
