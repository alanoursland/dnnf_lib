"""Model-based diagnosis on natively multi-valued compiled circuits.

A system is described as components with discrete *modes* (carrying prior
probabilities), *observables* (boolean, finite-domain, or quantized
continuous), and propositional constraints.  The description is compiled
once (offline) to a smooth finite-domain d-DNNF — leaves are atomic
assignments like ``valve=stuck_closed`` — and then queried online:

* :meth:`CompiledSystem.diagnoses` — ranked complete system states (MPE
  semantics), most probable first;
* :meth:`CompiledSystem.map_diagnoses` — ranked joint mode assignments by
  **exact summed posterior** (marginal MAP);
* :meth:`CompiledSystem.mode_posteriors` — exact per-mode marginals;
* :meth:`CompiledSystem.log_evidence` — ``log P(evidence)``.

Weights live directly on ``(variable, value)`` leaves: a mode's prior is
the weight of its value, evidence masks the weights of ruled-out values,
and Tseitin auxiliaries are neutral.  There is no one-hot encoding and no
exactly-one clauses — finite domains are native (see :mod:`neximode.fd`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

from . import fd
from .fd import FDAtom, FDCircuit, FDCnf
from .formula import Formula, Not, iff

EvidenceValue = Union[bool, str, int, float]


class FiniteVar:
    """A named finite-domain variable; ``var == value`` yields an atom."""

    def __init__(self, name: str, values: Sequence, fd_var: int):
        self.name = name
        self.values = tuple(values)
        self.fd_var = fd_var

    def _index(self, value) -> int:
        try:
            return self.values.index(value)
        except ValueError:
            raise KeyError(f"{self.name} has no value {value!r}") from None

    def __eq__(self, value) -> Formula:  # type: ignore[override]
        return FDAtom(
            self.fd_var, frozenset((self._index(value),)),
            f"{self.name}={value}",
        )

    def __ne__(self, value) -> Formula:  # type: ignore[override]
        idx = self._index(value)
        rest = frozenset(range(len(self.values))) - {idx}
        return FDAtom(self.fd_var, rest, f"{self.name}!={value}")

    def in_(self, values) -> Formula:
        """Atom: this variable's value is one of ``values``."""
        idxs = frozenset(self._index(v) for v in values)
        return FDAtom(self.fd_var, idxs, f"{self.name}in{list(values)}")

    def __hash__(self):
        return hash((self.name, self.fd_var))


class QuantizedVar(FiniteVar):
    """A continuous quantity quantized into bounded intervals.

    ``boundaries = [b0, b1, ..., bn]`` defines n buckets ``[b_i, b_{i+1})``;
    threshold atoms (:meth:`below`, :meth:`at_least`, :meth:`between`) are
    single set-literals over bucket indices, and numeric evidence is
    bucketed automatically.
    """

    def __init__(self, name: str, boundaries: Sequence[float], fd_var: int):
        bs = list(boundaries)
        if len(bs) < 3 or any(a >= b for a, b in zip(bs, bs[1:])):
            raise ValueError(
                "boundaries must be strictly increasing with >= 2 buckets"
            )
        labels = [f"[{a},{b})" for a, b in zip(bs, bs[1:])]
        super().__init__(name, labels, fd_var)
        self.boundaries = bs

    def _boundary_index(self, x: float) -> int:
        try:
            return self.boundaries.index(x)
        except ValueError:
            raise ValueError(
                f"{x} is not a quantization boundary of {self.name}; "
                f"boundaries are {self.boundaries}"
            ) from None

    def below(self, x: float) -> Formula:
        """Atom: value < x (x must be a boundary)."""
        j = self._boundary_index(x)
        return FDAtom(self.fd_var, frozenset(range(j)), f"{self.name}<{x}")

    def at_least(self, x: float) -> Formula:
        j = self._boundary_index(x)
        return FDAtom(
            self.fd_var,
            frozenset(range(j, len(self.values))),
            f"{self.name}>={x}",
        )

    def between(self, lo: float, hi: float) -> Formula:
        """Atom: lo <= value < hi (both boundaries)."""
        i, j = self._boundary_index(lo), self._boundary_index(hi)
        return FDAtom(
            self.fd_var, frozenset(range(i, j)), f"{lo}<={self.name}<{hi}"
        )

    def bucket_of(self, x: float) -> int:
        for i in range(len(self.boundaries) - 1):
            if self.boundaries[i] <= x < self.boundaries[i + 1]:
                return i
        raise ValueError(
            f"{x} outside the quantized range "
            f"[{self.boundaries[0]}, {self.boundaries[-1]}) of {self.name}"
        )


@dataclass
class Diagnosis:
    """One ranked explanation."""

    modes: Dict[str, str]
    state: Dict[str, EvidenceValue]
    cost: float  # neg-log (unnormalized) mass
    posterior: float  # normalized P(. | evidence)

    def __repr__(self) -> str:  # pragma: no cover
        modes = ", ".join(f"{k}={v}" for k, v in sorted(self.modes.items()))
        return f"Diagnosis({modes}; p={self.posterior:.4g})"


class SystemModel:
    """Declarative system description over native finite domains."""

    def __init__(self) -> None:
        self.cnf = FDCnf()
        self._constraints: List[Formula] = []
        self.vars: Dict[str, FiniteVar] = {}
        # mvlit -> weight (only non-default entries stored)
        self._prior_weights: Dict[int, float] = {}
        self._mode_vars: List[str] = []
        self._prev_map: Dict[str, str] = {}  # mode name -> prev-var name

    def _register(self, var: FiniteVar) -> FiniteVar:
        if var.name in self.vars:
            raise ValueError(f"duplicate variable {var.name!r}")
        self.vars[var.name] = var
        return var

    # -- variable declaration ------------------------------------------
    def bool(self, name: str, prior: Optional[float] = None) -> Formula:
        """Declare a boolean variable; returns the atom "name is true".
        ``prior`` (if given) is ``P(true)``; omitted means uninformative
        weight 1 per polarity."""
        fd_var = self.cnf.spec.add_var(2)
        v = self._register(FiniteVar(name, (False, True), fd_var))
        if prior is not None:
            self._prior_weights[self.cnf.spec.mvlit(fd_var, 0)] = 1.0 - prior
            self._prior_weights[self.cnf.spec.mvlit(fd_var, 1)] = prior
        return v == True  # noqa: E712

    def finite(
        self,
        name: str,
        values: Sequence,
        priors: Optional[Sequence[float]] = None,
        mode: bool = False,
    ) -> FiniteVar:
        """Declare a finite-domain variable.  With ``mode=True`` (or when
        ``priors`` are given) it is a component mode: it appears in
        diagnoses and its priors weight the enumeration."""
        fd_var = self.cnf.spec.add_var(len(values))
        v = self._register(FiniteVar(name, values, fd_var))
        if priors is not None:
            if len(priors) != len(values):
                raise ValueError("priors length must match values")
            total = sum(priors)
            for i, p in enumerate(priors):
                self._prior_weights[self.cnf.spec.mvlit(fd_var, i)] = p / total
        if mode or priors is not None:
            self._mode_vars.append(name)
        return v

    def mode(
        self, name: str, values: Sequence, priors: Sequence[float]
    ) -> FiniteVar:
        """Shorthand for a component mode variable with priors."""
        return self.finite(name, values, priors=priors, mode=True)

    def quantized(
        self,
        name: str,
        boundaries: Sequence[float],
        priors: Optional[Sequence[float]] = None,
        mode: bool = False,
    ) -> QuantizedVar:
        """Declare a continuous quantity quantized into the bounded
        intervals ``[b0,b1), [b1,b2), ...``.  Threshold atoms:
        ``v.below(x)``, ``v.at_least(x)``, ``v.between(lo, hi)`` (x at
        boundaries).  Evidence may be given as a raw number and is
        bucketed automatically."""
        fd_var = self.cnf.spec.add_var(len(boundaries) - 1)
        v = QuantizedVar(name, boundaries, fd_var)
        self._register(v)
        if priors is not None:
            if len(priors) != len(v.values):
                raise ValueError("priors length must match bucket count")
            total = sum(priors)
            for i, p in enumerate(priors):
                self._prior_weights[self.cnf.spec.mvlit(fd_var, i)] = p / total
        if mode or priors is not None:
            self._mode_vars.append(name)
        return v

    def add(self, formula: Formula) -> None:
        self._constraints.append(formula)

    def prev(self, name: str) -> FiniteVar:
        """The previous-timestep copy of mode variable ``name``, for use
        in **joint transition constraints** — hard relations between
        consecutive slices that per-variable transition matrices cannot
        express, e.g.::

            m.add(~((m.prev("a") == "ok") & (m.prev("b") == "ok")
                    & (a == "bad") & (b == "bad")))   # no common-cause pair failure

        During tracking, :class:`neximode.tracking.ModeTracker` conditions
        the prev variables to each belief particle's modes, so these
        constraints prune illegal transitions (pruned mass is
        renormalized: probabilities are conditional on a legal
        transition).  At t=0 prev variables are unconstrained.
        """
        if name not in self._mode_vars:
            raise KeyError(f"{name!r} is not a mode variable")
        if name in self._prev_map:
            return self.vars[self._prev_map[name]]
        cur = self.vars[name]
        fd_var = self.cnf.spec.add_var(len(cur.values))
        pv = self._register(FiniteVar(f"{name}@prev", cur.values, fd_var))
        self._prev_map[name] = pv.name
        return pv

    def sensor(
        self,
        name: str,
        expr: Formula,
        false_positive: float = 0.0,
        false_negative: float = 0.0,
    ) -> Formula:
        """Declare an observable that noisily reports ``expr``:
        ``P(name=True | expr) = 1 - false_negative``, ``P(name=True |
        ~expr) = false_positive``.  Hidden fault-injection variables
        (``_<name>_fp`` / ``_<name>_fn``) are excluded from reported
        states."""
        s = self.bool(name)
        true_when: Formula = expr
        if false_negative > 0.0:
            g_fn = self.bool(f"_{name}_fn", prior=false_negative)
            true_when = expr & ~g_fn
        if false_positive > 0.0:
            g_fp = self.bool(f"_{name}_fp", prior=false_positive)
            self._constraints.append(iff(s, true_when | (Not(expr) & g_fp)))
        else:
            self._constraints.append(iff(s, true_when))
        return s

    # ------------------------------------------------------------------
    def compile(
        self,
        var_order: Optional[Sequence[int]] = None,
        modes_first: bool = True,
    ) -> "CompiledSystem":
        """Compile to a smooth finite-domain d-DNNF.  With ``modes_first``
        (default) mode variables are branched above all others, enabling
        exact marginal MAP (:meth:`CompiledSystem.map_diagnoses`)."""
        fd.encode(self._constraints, self.cnf)
        if var_order is None and modes_first:
            var_order = [self.vars[n].fd_var for n in self._mode_vars]
        circuit = fd.compile_fd(self.cnf, var_order=var_order, smooth=True)
        return CompiledSystem(
            circuit=circuit,
            variables=dict(self.vars),
            prior_weights=dict(self._prior_weights),
            mode_vars=list(self._mode_vars),
            prev_map=dict(self._prev_map),
        )


class CompiledSystem:
    def __init__(
        self,
        circuit: FDCircuit,
        variables: Dict[str, FiniteVar],
        prior_weights: Dict[int, float],
        mode_vars: List[str],
        prev_map: Optional[Dict[str, str]] = None,
    ):
        self.circuit = circuit
        self.vars = variables
        self.mode_vars = mode_vars
        self.prev_map = prev_map or {}
        self._weights = [1.0] * circuit.spec.total
        for mvlit, w in prior_weights.items():
            self._weights[mvlit] = w
        self._costs = [
            0.0 if w == 1.0 else (math.inf if w <= 0 else -math.log(w))
            for w in self._weights
        ]

    # Backwards-compatible view: finite variables by name.
    @property
    def finites(self) -> Dict[str, FiniteVar]:
        return self.vars

    # -- evidence -------------------------------------------------------
    def _value_index(self, name: str, value: EvidenceValue) -> int:
        var = self.vars.get(name)
        if var is None:
            raise KeyError(f"unknown variable {name!r}")
        if isinstance(var, QuantizedVar) and isinstance(value, (int, float)) \
                and not isinstance(value, bool):
            return var.bucket_of(float(value))
        return var._index(value)

    def log_weights_for(
        self,
        evidence: Dict[str, EvidenceValue],
        mode_priors: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> List[float]:
        """Log value weights with evidence applied and, optionally, some
        variables' static priors replaced by ``{value: prob}`` rows."""
        weights = list(self._weights)
        spec = self.circuit.spec
        if mode_priors:
            for name, dist in mode_priors.items():
                var = self.vars[name]
                for i, value in enumerate(var.values):
                    weights[spec.mvlit(var.fd_var, i)] = dist.get(value, 0.0)
        for name, value in evidence.items():
            var = self.vars[name]
            lik = self._soft_likelihoods(var, value)
            if lik is not None:
                for i, l in enumerate(lik):
                    weights[spec.mvlit(var.fd_var, i)] *= l
                continue
            chosen = self._value_index(name, value)
            for i in range(len(var.values)):
                if i != chosen:
                    weights[spec.mvlit(var.fd_var, i)] = 0.0
        return [-math.inf if w <= 0 else math.log(w) for w in weights]

    @staticmethod
    def _soft_likelihoods(var: FiniteVar, value) -> Optional[List[float]]:
        """Soft (virtual) evidence: an observation may be a per-value
        likelihood vector instead of a hard value — ``(0.9, 0.1)`` (in
        the variable's value order) or ``{value: likelihood}``.  These
        are Pearl virtual-evidence likelihoods ``P(reading | var=v)``,
        multiplied into the value weights; they need not sum to 1 (only
        ratios matter)."""
        if isinstance(value, dict):
            return [float(value.get(v, 0.0)) for v in var.values]
        if isinstance(value, (tuple, list)):
            if len(value) != len(var.values):
                raise ValueError(
                    f"likelihood vector for {var.name} needs "
                    f"{len(var.values)} entries"
                )
            return [float(x) for x in value]
        return None

    def _conditioned_costs(
        self, evidence: Dict[str, EvidenceValue]
    ) -> List[float]:
        costs = list(self._costs)
        spec = self.circuit.spec
        for name, value in evidence.items():
            var = self.vars[name]
            lik = self._soft_likelihoods(var, value)
            if lik is not None:
                for i, l in enumerate(lik):
                    costs[spec.mvlit(var.fd_var, i)] += (
                        math.inf if l <= 0 else -math.log(l)
                    )
                continue
            chosen = self._value_index(name, value)
            for i in range(len(var.values)):
                if i != chosen:
                    costs[spec.mvlit(var.fd_var, i)] = math.inf
        return costs

    # -- queries --------------------------------------------------------
    def log_evidence(self, evidence: Dict[str, EvidenceValue]) -> float:
        """``log P(evidence)`` up to the constant weighting of unweighted
        observables."""
        return fd.log_wmc(self.circuit, self.log_weights_for(evidence))

    def diagnoses(
        self,
        evidence: Dict[str, EvidenceValue],
        k: int = 5,
        project_to_modes: bool = True,
    ) -> List[Diagnosis]:
        """The ``k`` most probable complete system states consistent with
        the evidence (MPE semantics: mode assignments ranked by their best
        supporting state; see :meth:`map_diagnoses` for summed posteriors)."""
        costs = self._conditioned_costs(evidence)
        log_z = self.log_evidence(evidence)
        if log_z == -math.inf:
            return []
        out: List[Diagnosis] = []
        seen_modes: set = set()
        for cost, assignment in fd.enumerate_models(
            self.circuit, costs, k=None
        ):
            modes = {
                name: self.vars[name].values[assignment[self.vars[name].fd_var]]
                for name in self.mode_vars
            }
            key = tuple(sorted(modes.items()))
            if project_to_modes:
                if key in seen_modes:
                    continue
                seen_modes.add(key)
            out.append(
                Diagnosis(
                    modes=modes,
                    state=self._decode_state(assignment),
                    cost=cost,
                    posterior=math.exp(-cost - log_z),
                )
            )
            if len(out) >= k:
                break
        return out

    def map_diagnoses(
        self, evidence: Dict[str, EvidenceValue], k: int = 5
    ) -> List[Diagnosis]:
        """The ``k`` most probable **joint mode assignments** by exact
        summed posterior (marginal MAP), most probable first.  Requires
        the default ``modes_first`` compilation."""
        log_w = self.log_weights_for(evidence)
        log_z = fd.log_wmc(self.circuit, log_w)
        if log_z == -math.inf:
            return []
        return [
            Diagnosis(
                modes=modes,
                state=dict(modes),
                cost=cost,
                posterior=math.exp(-cost - log_z),
            )
            for cost, modes in self.ranked_map(log_w, k)
        ]

    def ranked_map(
        self, log_w: Sequence[float], k: Optional[int]
    ) -> List[Tuple[float, Dict[str, str]]]:
        """Ranked joint mode assignments under explicit log weights:
        ``(cost, {mode_var: value})``.  Building block for
        :meth:`map_diagnoses` and :class:`neximode.tracking.ModeTracker`."""
        map_fd_vars = [self.vars[n].fd_var for n in self.mode_vars]
        by_fd = {self.vars[n].fd_var: n for n in self.mode_vars}
        out: List[Tuple[float, Dict[str, str]]] = []
        for cost, assignment in fd.enumerate_map(
            self.circuit, log_w, map_fd_vars, k=k
        ):
            modes = {
                by_fd[v]: self.vars[by_fd[v]].values[val]
                for v, val in assignment.items()
                if v in by_fd
            }
            out.append((cost, modes))
        return out

    def posteriors(
        self,
        evidence: Dict[str, EvidenceValue],
        names: Optional[Sequence[str]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Exact ``P(var = value | evidence)`` via WMC ratios for the
        named variables (default: mode variables).  Works for any
        declared variable, including hidden sensor-noise variables."""
        log_z = self.log_evidence(evidence)
        if log_z == -math.inf:
            raise ValueError("evidence is inconsistent with the model")
        out: Dict[str, Dict[str, float]] = {}
        for name in (self.mode_vars if names is None else names):
            var = self.vars[name]
            dist: Dict[str, float] = {}
            for value in var.values:
                ev = dict(evidence)
                ev[name] = value
                dist[value] = math.exp(self.log_evidence(ev) - log_z)
            out[name] = dist
        return out

    def mode_posteriors(
        self, evidence: Dict[str, EvidenceValue]
    ) -> Dict[str, Dict[str, float]]:
        """Exact ``P(mode = value | evidence)`` via WMC ratios."""
        return self.posteriors(evidence)

    # -- learning -------------------------------------------------------
    def fit_priors(
        self,
        observations: Sequence[Dict[str, EvidenceValue]],
        names: Optional[Sequence[str]] = None,
        iterations: int = 25,
        tol: float = 1e-6,
    ) -> List[float]:
        """Learn value priors for the named variables (default: modes)
        from partially observed telemetry, by expectation-maximization.

        Each observation is an evidence dict (any subset of variables).
        E-step: exact posteriors of the fitted variables given each
        observation under the current priors (WMC ratios on the compiled
        circuit).  M-step: each fitted variable's prior becomes the
        average posterior.  This is exact EM for the model class
        (independent categorical priors + the compiled constraint/noise
        structure), so the returned per-iteration average log-likelihood
        is non-decreasing; iteration stops early when it improves by
        less than ``tol``.

        Priors are updated in place (subsequent queries use them).
        Returns the log-likelihood trace.
        """
        fit_names = list(self.mode_vars if names is None else names)
        spec = self.circuit.spec
        history: List[float] = []
        for _ in range(iterations):
            sums = {
                name: [0.0] * len(self.vars[name].values)
                for name in fit_names
            }
            log_lik = 0.0
            for obs in observations:
                log_lik += self.log_evidence(obs)
                post = self.posteriors(obs, fit_names)
                for name in fit_names:
                    values = self.vars[name].values
                    for i, value in enumerate(values):
                        sums[name][i] += post[name][value]
            history.append(log_lik / max(len(observations), 1))
            for name in fit_names:
                var = self.vars[name]
                total = sum(sums[name])
                for i in range(len(var.values)):
                    w = sums[name][i] / total if total > 0 else 0.0
                    mvlit = spec.mvlit(var.fd_var, i)
                    self._weights[mvlit] = w
                    self._costs[mvlit] = (
                        math.inf if w <= 0 else -math.log(w)
                    )
            if len(history) >= 2 and history[-1] - history[-2] < tol:
                break
        return history

    def fit_priors_torch(
        self,
        observations: Sequence[Dict[str, EvidenceValue]],
        names: Optional[Sequence[str]] = None,
        epochs: int = 300,
        lr: float = 0.05,
        device: str = "cpu",
    ) -> List[float]:
        """Gradient-based alternative to :meth:`fit_priors`: trains the
        same priors by Adam on the differentiable torch backend (batched
        masked log-WMC), writes them back, and returns the average
        log-likelihood trace.  Scales to large telemetry sets and GPU;
        requires torch."""
        from .torch_learn import PriorLearner

        return PriorLearner(self, names=names, device=device).fit(
            observations, epochs=epochs, lr=lr
        )

    def sample_state(self, rng) -> Dict[str, EvidenceValue]:
        """Draw one complete system state from the model's current
        weighted distribution (priors + constraints).  Useful for
        simulation and for generating synthetic telemetry."""
        assignment = fd.sample(
            self.circuit, self.log_weights_for({}), rng
        )
        if assignment is None:
            raise ValueError("model has zero total mass")
        return self._decode_state(assignment)

    def diagnoses_min_cardinality(
        self, evidence: Dict[str, EvidenceValue], k: int = 5
    ) -> List[Tuple[int, Dict[str, str]]]:
        """The ``k`` mode assignments with fewest faults consistent with
        the evidence, fewest first: ``(fault_count, modes)``.  A fault is
        any mode value other than the variable's nominal (highest-prior)
        value.  Classic minimum-cardinality diagnosis via the tropical
        semiring with unit fault costs."""
        spec = self.circuit.spec
        costs = [0.0] * spec.total
        nominal: Dict[str, int] = {}
        for name in self.mode_vars:
            var = self.vars[name]
            weights = [
                self._weights[spec.mvlit(var.fd_var, i)]
                for i in range(len(var.values))
            ]
            nom = max(range(len(var.values)), key=lambda i: weights[i])
            nominal[name] = nom
            for i in range(len(var.values)):
                if i != nom:
                    costs[spec.mvlit(var.fd_var, i)] = 1.0
        for name, value in evidence.items():
            var = self.vars[name]
            lik = self._soft_likelihoods(var, value)
            chosen = None if lik is not None else self._value_index(name, value)
            for i in range(len(var.values)):
                if lik is not None:
                    if lik[i] <= 0:
                        costs[spec.mvlit(var.fd_var, i)] = math.inf
                elif i != chosen:
                    costs[spec.mvlit(var.fd_var, i)] = math.inf
        out: List[Tuple[int, Dict[str, str]]] = []
        seen: set = set()
        for cost, assignment in fd.enumerate_models(self.circuit, costs):
            modes = {
                name: self.vars[name].values[assignment[self.vars[name].fd_var]]
                for name in self.mode_vars
            }
            key = tuple(sorted(modes.items()))
            if key in seen:
                continue
            seen.add(key)
            out.append((round(cost), modes))
            if len(out) >= k:
                break
        return out

    def value_of_information(
        self,
        evidence: Dict[str, EvidenceValue],
        candidates: Optional[Sequence[str]] = None,
    ) -> List[Tuple[str, float]]:
        """Rank unobserved variables by expected reduction in diagnosis
        uncertainty: for each candidate ``c``, ``VOI(c) = H(modes | e) -
        E_{v ~ P(c|e)}[H(modes | e, c=v)]`` where H is the sum of
        per-mode-variable marginal entropies (an upper bound on joint
        entropy; exact for a single mode variable).  Returns
        ``[(name, voi), ...]`` best first — "which sensor should I read
        next."  Candidates default to all unobserved non-hidden,
        non-mode variables."""
        if candidates is None:
            candidates = [
                n for n in self.vars
                if n not in evidence and not n.startswith("_")
                and n not in self.mode_vars and "@prev" not in n
            ]

        def entropy(ev) -> float:
            h = 0.0
            for name in self.mode_vars:
                for p in self.posteriors(ev, names=[name])[name].values():
                    if p > 0:
                        h -= p * math.log(p)
            return h

        h0 = entropy(evidence)
        out: List[Tuple[str, float]] = []
        for c in candidates:
            var = self.vars[c]
            dist = self.posteriors(evidence, names=[c])[c]
            expected = 0.0
            for value, p in dist.items():
                if p <= 0:
                    continue
                ev = dict(evidence)
                ev[c] = value
                expected += p * entropy(ev)
            out.append((c, h0 - expected))
        out.sort(key=lambda nv: -nv[1])
        return out

    # -- persistence ----------------------------------------------------
    def save(self, path: str) -> None:
        """Serialize the compiled system (circuit, variables, weights) to
        JSON so the offline/online split survives process boundaries:
        compile once, ship the file, load in the monitoring service."""
        import json

        c = self.circuit
        doc = {
            "format": "neximode.compiled_system.v1",
            # All resource requirements up front, flight-software style:
            # a loader can make one allocation pass from the header alone
            # before reading any body section (the original spacecraft
            # deployments of this architecture loaded with a single
            # memory allocation and ran for the mission duration).
            "header": {
                "num_nodes": len(c),
                "num_edges": c.num_edges,
                "max_children": max(
                    (len(ch) for ch in c.children), default=0
                ),
                "num_fd_vars": c.spec.num_vars,
                "num_weight_slots": c.spec.total,
                "max_domain": max(c.spec.sizes, default=0),
                "num_named_vars": len(self.vars),
            },
            "spec_sizes": list(c.spec.sizes),
            "kinds": list(c.kinds),
            "lits": list(c.lits),
            "children": [list(ch) for ch in c.children],
            "root": c.root,
            "weights": list(self._weights),
            "mode_vars": list(self.mode_vars),
            "prev_map": dict(self.prev_map),
            "vars": [
                {
                    "name": v.name,
                    "fd_var": v.fd_var,
                    "values": list(v.values),
                    "boundaries": getattr(v, "boundaries", None),
                }
                for v in self.vars.values()
            ],
        }
        with open(path, "w") as f:
            json.dump(doc, f)

    @classmethod
    def load(cls, path: str) -> "CompiledSystem":
        import json

        with open(path) as f:
            doc = json.load(f)
        if doc.get("format") != "neximode.compiled_system.v1":
            raise ValueError(f"unrecognized format in {path}")
        spec = fd.FDSpec()
        for size in doc["spec_sizes"]:
            spec.add_var(size)
        circuit = fd.FDCircuit(
            spec,
            doc["kinds"],
            doc["lits"],
            [tuple(ch) for ch in doc["children"]],
            doc["root"],
        )
        variables: Dict[str, FiniteVar] = {}
        for v in doc["vars"]:
            if v["boundaries"] is not None:
                var = QuantizedVar(v["name"], v["boundaries"], v["fd_var"])
            else:
                var = FiniteVar(v["name"], v["values"], v["fd_var"])
            variables[v["name"]] = var
        system = cls(
            circuit=circuit,
            variables=variables,
            prior_weights={},
            mode_vars=doc["mode_vars"],
            prev_map=doc["prev_map"],
        )
        system._weights = list(doc["weights"])
        system._costs = [
            0.0 if w == 1.0 else (math.inf if w <= 0 else -math.log(w))
            for w in system._weights
        ]
        return system

    # -- decoding -------------------------------------------------------
    def _decode_state(
        self, assignment: Dict[int, int]
    ) -> Dict[str, EvidenceValue]:
        state: Dict[str, EvidenceValue] = {}
        for name, var in self.vars.items():
            if name.startswith("_"):
                continue  # hidden noise-injection variables
            if var.fd_var in assignment:
                state[name] = var.values[assignment[var.fd_var]]
        return state
