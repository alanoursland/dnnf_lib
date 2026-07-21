"""Semiring evaluation of DNNF circuits (pure-Python reference backend).

A DNNF circuit evaluated over a commutative semiring computes a sum over
models of a product over literals — different semirings answer different
queries:

=================  ==========  =========  ==============================
Semiring           OR          AND        Query
=================  ==========  =========  ==============================
Boolean            or          and        satisfiability / consistency
Counting           ``+``       ``*``      model counting
Real               ``+``       ``*``      weighted model counting (WMC)
Log                logsumexp   ``+``      log-space WMC
Min-sum (tropical) min         ``+``      MPE over neg-log costs
=================  ==========  =========  ==============================

Counting-style semirings (Counting/Real/Log) require a **smooth d-DNNF**;
min-sum requires only decomposability but is usually run on the smoothed
circuit so every model assigns every variable.

The neg-log view is the one used in model-based diagnosis: give literal
``l`` the weight ``-log P(l)``; then the min-sum value of the circuit is
the cost of the most probable model consistent with the theory, and k-best
enumeration (see :mod:`dnnf.kbest`) yields models ordered from most to
least probable.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from .circuit import AND, FALSE, LIT, OR, TRUE, Circuit, lit_index

Assignment = Dict[int, bool]


# ----------------------------------------------------------------------
# Literal weight vectors
# ----------------------------------------------------------------------
def uniform_weights(num_vars: int, value: float = 1.0) -> List[float]:
    """A weight vector assigning ``value`` to every literal."""
    return [value] * (2 * num_vars)


def weights_from_probs(num_vars: int, probs: Dict[int, float]) -> List[float]:
    """Weights for WMC from ``P(v = true)`` per variable (default 0.5)."""
    w = []
    for v in range(1, num_vars + 1):
        p = probs.get(v, 0.5)
        w.extend([p, 1.0 - p])
    return w


def costs_from_probs(num_vars: int, probs: Dict[int, float]) -> List[float]:
    """Neg-log costs from ``P(v = true)`` per variable (default 0.5)."""

    def nl(p: float) -> float:
        return math.inf if p <= 0.0 else -math.log(p)

    w = []
    for v in range(1, num_vars + 1):
        p = probs.get(v, 0.5)
        w.extend([nl(p), nl(1.0 - p)])
    return w


def condition_weights(
    weights: Sequence[float], evidence: Assignment, annihilator: float
) -> List[float]:
    """Return a copy of ``weights`` with literals contradicting ``evidence``
    set to ``annihilator`` (``0.0`` for probability weights, ``math.inf``
    for neg-log costs)."""
    w = list(weights)
    for var, value in evidence.items():
        forbidden = -var if value else var
        w[lit_index(forbidden)] = annihilator
    return w


# ----------------------------------------------------------------------
# Core evaluation
# ----------------------------------------------------------------------
def _forward(
    circuit: Circuit,
    leaf,
    add,
    mul,
    zero,
    one,
) -> List:
    vals: List = [None] * len(circuit)
    for i, kind in enumerate(circuit.kinds):
        if kind == FALSE:
            vals[i] = zero
        elif kind == TRUE:
            vals[i] = one
        elif kind == LIT:
            vals[i] = leaf(circuit.lits[i])
        elif kind == AND:
            acc = one
            for c in circuit.children[i]:
                acc = mul(acc, vals[c])
            vals[i] = acc
        else:  # OR
            acc = zero
            for c in circuit.children[i]:
                acc = add(acc, vals[c])
            vals[i] = acc
    return vals


def is_satisfiable(circuit: Circuit) -> bool:
    """Consistency check; valid on any DNNF."""
    vals = _forward(
        circuit,
        leaf=lambda lit: True,
        add=lambda a, b: a or b,
        mul=lambda a, b: a and b,
        zero=False,
        one=True,
    )
    return vals[circuit.root]


def model_count(circuit: Circuit) -> int:
    """Exact model count over all ``num_vars`` variables.

    Requires a smooth d-DNNF (compile with ``smooth=True`` or call
    ``circuit.smooth()`` first).
    """
    _require_smooth_ddnnf(circuit)
    vals = _forward(
        circuit,
        leaf=lambda lit: 1,
        add=lambda a, b: a + b,
        mul=lambda a, b: a * b,
        zero=0,
        one=1,
    )
    return vals[circuit.root]


def wmc(circuit: Circuit, weights: Sequence[float]) -> float:
    """Weighted model count: sum over models of the product of literal
    weights.  Requires a smooth d-DNNF."""
    _require_smooth_ddnnf(circuit)
    vals = _forward(
        circuit,
        leaf=lambda lit: weights[lit_index(lit)],
        add=lambda a, b: a + b,
        mul=lambda a, b: a * b,
        zero=0.0,
        one=1.0,
    )
    return vals[circuit.root]


def log_wmc(circuit: Circuit, log_weights: Sequence[float]) -> float:
    """Log-space WMC.  ``log_weights`` holds log literal weights;
    returns ``log(WMC)`` (``-inf`` for an inconsistent circuit).
    Requires a smooth d-DNNF."""
    _require_smooth_ddnnf(circuit)

    def lse(a: float, b: float) -> float:
        if a == -math.inf:
            return b
        if b == -math.inf:
            return a
        m = max(a, b)
        return m + math.log(math.exp(a - m) + math.exp(b - m))

    vals = _forward(
        circuit,
        leaf=lambda lit: log_weights[lit_index(lit)],
        add=lse,
        mul=lambda a, b: a + b,
        zero=-math.inf,
        one=0.0,
    )
    return vals[circuit.root]


def mpe(
    circuit: Circuit, costs: Sequence[float]
) -> Tuple[float, Optional[Assignment]]:
    """Most probable explanation under additive (neg-log) literal costs.

    Returns ``(min_cost, assignment)`` where the assignment attains the
    minimum total cost among models, or ``(inf, None)`` if unsatisfiable.
    Runs on any DNNF; on a smoothed circuit the assignment is total.
    """
    vals = _forward(
        circuit,
        leaf=lambda lit: costs[lit_index(lit)],
        add=min,
        mul=lambda a, b: a + b,
        zero=math.inf,
        one=0.0,
    )
    best = vals[circuit.root]
    if best == math.inf:
        return math.inf, None

    assignment: Assignment = {}
    stack = [circuit.root]
    while stack:
        i = stack.pop()
        kind = circuit.kinds[i]
        if kind == LIT:
            lit = circuit.lits[i]
            assignment[abs(lit)] = lit > 0
        elif kind == AND:
            stack.extend(circuit.children[i])
        elif kind == OR:
            target = vals[i]
            chosen = None
            for c in circuit.children[i]:
                if vals[c] == target:
                    chosen = c
                    break
            if chosen is None:  # numeric slack fallback
                chosen = min(circuit.children[i], key=lambda c: vals[c])
            stack.append(chosen)
    return best, assignment


# ----------------------------------------------------------------------
def _require_smooth_ddnnf(circuit: Circuit) -> None:
    # Cheap cached validation; users compiling through dnnf.compile_cnf
    # always pass.  Re-verifying properties on every call would be O(n^2)
    # for determinism, so we only check smoothness structure lazily.
    cache = getattr(circuit, "_smooth_ddnnf_ok", None)
    if cache is True:
        return
    if cache is False:
        raise ValueError(
            "circuit is not a smooth d-DNNF; compile with smooth=True or "
            "call circuit.smooth()"
        )
    if circuit.kinds[circuit.root] == FALSE:
        circuit._smooth_ddnnf_ok = True  # type: ignore[attr-defined]
        return
    all_vars = frozenset(range(1, circuit.num_vars + 1))
    ok = (
        circuit.is_smooth()
        and circuit.is_deterministic()
        and circuit.mentioned_vars() == all_vars
    )
    circuit._smooth_ddnnf_ok = ok  # type: ignore[attr-defined]
    if not ok:
        raise ValueError(
            "circuit is not a smooth d-DNNF; compile with smooth=True or "
            "call circuit.smooth()"
        )
