"""Lazy cost-ordered enumeration over compiled circuits.

See ``CONTRACTS.md`` for structural requirements and marginal-MAP scope.
"""

from __future__ import annotations

import heapq
import math
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from .circuit import AND, FALSE, LIT, OR, TRUE, Circuit, lit_index
from .invariants import ModeNexusInvariantError

Derivation = Tuple[float, Tuple[int, ...]]  # (cost, sorted literal tuple)


class _NodeStream:
    """Ranked derivation stream for one circuit node, materialized on demand.

    ``get(i)`` returns the i-th cheapest derivation or None if fewer exist.
    """

    __slots__ = ("items", "_heap", "_seen", "_kind", "_children", "_counter")

    def __init__(self, kind: int, children: Sequence["_NodeStream"],
                 first: Optional[Derivation]):
        self.items: List[Derivation] = []
        self._kind = kind
        self._children = list(children)
        self._heap: List = []
        self._seen: set = set()
        self._counter = 0
        if kind in (LIT, TRUE):
            if first is not None:
                self.items.append(first)
        elif kind == OR:
            for ci, child in enumerate(self._children):
                d = child.get(0)
                if d is not None:
                    self._push(d[0], (ci, 0))
        elif kind == AND:
            vec = (0,) * len(self._children)
            self._try_push_vec(vec)

    # -- heap helpers ---------------------------------------------------
    def _push(self, cost: float, state) -> None:
        heapq.heappush(self._heap, (cost, self._counter, state))
        self._counter += 1

    def _try_push_vec(self, vec: Tuple[int, ...]) -> None:
        if vec in self._seen:
            return
        cost = 0.0
        for child, idx in zip(self._children, vec):
            d = child.get(idx)
            if d is None:
                return
            cost += d[0]
        self._seen.add(vec)
        self._push(cost, vec)

    def _materialize_next(self) -> bool:
        if not self._heap:
            return False
        cost, _, state = heapq.heappop(self._heap)
        if self._kind == OR:
            ci, idx = state
            d = self._children[ci].get(idx)
            if d is None:
                raise ModeNexusInvariantError(
                    "ranked OR heap referenced a missing child derivation"
                )
            self.items.append(d)
            nxt = self._children[ci].get(idx + 1)
            if nxt is not None:
                self._push(nxt[0], (ci, idx + 1))
        else:  # AND
            vec = state
            lits: List[int] = []
            for child, idx in zip(self._children, vec):
                d = child.get(idx)
                if d is None:
                    raise ModeNexusInvariantError(
                        "ranked AND heap referenced a missing child "
                        "derivation"
                    )
                lits.extend(d[1])
            self.items.append((cost, tuple(sorted(lits, key=abs))))
            for pos in range(len(vec)):
                nxt = list(vec)
                nxt[pos] += 1
                self._try_push_vec(tuple(nxt))
        return True

    def get(self, i: int) -> Optional[Derivation]:
        while len(self.items) <= i:
            if not self._materialize_next():
                return None
        return self.items[i]


def enumerate_map(
    circuit: Circuit,
    log_weights: Sequence[float],
    map_vars,
    k: Optional[int] = None,
) -> Iterator[Tuple[float, Dict[int, bool]]]:
    """Yield marginal-MAP assignments as ``(neg_log_mass, assignment)``.

    ``map_vars`` must have been placed first in the compilation order. A
    structurally incompatible circuit raises :class:`ValueError`.
    """
    from .eval import log_values

    map_vars = frozenset(map_vars)
    if not circuit.is_smooth() or (
        circuit.kinds[circuit.root] != FALSE
        and circuit.mentioned_vars()
        != frozenset(range(1, circuit.num_vars + 1))
    ):
        circuit = circuit.smooth()

    vals = log_values(circuit, log_weights)
    var_sets = circuit.var_sets()
    asserted = circuit.asserted_literals()
    has_map = [bool(var_sets[i] & map_vars) for i in range(len(circuit))]

    # Verify the constrained structure: every OR mentioning a map variable
    # must be a decision on one, i.e. each child asserts a literal of some
    # common map variable.
    for i, kind in enumerate(circuit.kinds):
        if kind != OR or not has_map[i]:
            continue
        candidates = map_vars & {abs(l) for l in asserted[circuit.children[i][0]]}
        for c in circuit.children[i][1:]:
            candidates = candidates & {abs(l) for l in asserted[c]}
            if not candidates:
                break
        if not candidates:
            raise ValueError(
                "circuit is not constrained for marginal MAP over these "
                "variables; compile with var_order listing the MAP "
                "variables first"
            )

    streams: List[Optional[_NodeStream]] = [None] * len(circuit)
    for i, kind in enumerate(circuit.kinds):
        if not has_map[i]:
            # Terminal: entire subtree is summed out.
            cost = -vals[i]
            first = None if cost == math.inf else (cost, ())
            streams[i] = _NodeStream(LIT, (), first)
        elif kind == LIT:
            lit = circuit.lits[i]
            cost = -log_weights[lit_index(lit)]
            first = None if cost == math.inf else (cost, (lit,))
            streams[i] = _NodeStream(LIT, (), first)
        else:  # AND / OR in the upper (map) region
            streams[i] = _NodeStream(
                kind, [streams[c] for c in circuit.children[i]], None
            )

    root = streams[circuit.root]
    emitted = 0
    seen: set = set()
    i = 0
    while k is None or emitted < k:
        d = root.get(i)
        i += 1
        if d is None:
            return
        cost, lits = d
        if cost == math.inf:
            return
        if lits in seen:
            continue
        seen.add(lits)
        yield cost, {abs(l): l > 0 for l in lits}
        emitted += 1


def enumerate_models(
    circuit: Circuit,
    costs: Sequence[float],
    k: Optional[int] = None,
    dedup: bool = True,
) -> Iterator[Tuple[float, Dict[int, bool]]]:
    """Yield up to ``k`` models as ``(cost, {var: bool})`` in nondecreasing
    cost order (all models when ``k`` is None).

    ``costs`` is a per-literal additive cost vector (index via
    :func:`modenexus.circuit.lit_index`); use neg-log probabilities to get
    most-probable-first enumeration.  Models with infinite cost (e.g. those
    contradicting evidence applied via
    :func:`modenexus.eval.condition_weights`) are suppressed.

    The circuit is smoothed automatically if needed so every yielded model
    assigns every variable.
    """
    if not circuit.is_smooth() or (
        circuit.kinds[circuit.root] != FALSE
        and circuit.mentioned_vars()
        != frozenset(range(1, circuit.num_vars + 1))
    ):
        circuit = circuit.smooth()

    streams: List[Optional[_NodeStream]] = [None] * len(circuit)
    for i, kind in enumerate(circuit.kinds):
        if kind == FALSE:
            streams[i] = _NodeStream(FALSE, (), None)
        elif kind == TRUE:
            streams[i] = _NodeStream(TRUE, (), (0.0, ()))
        elif kind == LIT:
            lit = circuit.lits[i]
            cost = costs[lit_index(lit)]
            first = None if cost == math.inf else (cost, (lit,))
            streams[i] = _NodeStream(LIT, (), first)
        else:
            streams[i] = _NodeStream(
                kind, [streams[c] for c in circuit.children[i]], None
            )

    root = streams[circuit.root]
    emitted = 0
    seen: set = set()
    i = 0
    while k is None or emitted < k:
        d = root.get(i)
        i += 1
        if d is None:
            return
        cost, lits = d
        if cost == math.inf:
            return
        if dedup:
            if lits in seen:
                continue
            seen.add(lits)
        yield cost, {abs(l): l > 0 for l in lits}
        emitted += 1
