"""Native finite-domain (multi-valued) DNNF.

This is the representation the classic DNNF diagnosis engines actually
used: variables carry finite domains (``shutters in {open, closed}``,
modes with several failure states), circuit leaves are atomic assignments
``var=value``, and decision OR nodes branch **d ways** — one child per
domain value.  Compared to the boolean core (:mod:`dnnf.circuit` /
:mod:`dnnf.compiler`), there is no one-hot encoding: no negative-literal
bookkeeping leaves, no pairwise exactly-one clauses, and evidence is
applied by masking value weights directly.

Vocabulary:

* **FD literal**: ``(var, values)`` meaning "the value of ``var`` is in
  ``values``" — closed under negation (complement against the domain),
  which is what lets threshold atoms like ``level < 50`` be single
  literals over quantized ranges.
* **FD clause**: a disjunction of FD literals, at most one per variable.
* **Leaf**: an ``mvlit`` — a dense index for one ``(var, value)`` pair;
  weight vectors are indexed by mvlit, so a variable's weights are a row
  of a categorical distribution.

The DNNF properties transfer directly: decomposability (AND children over
disjoint variables), determinism (OR children assert conflicting values
of a common variable), smoothness (OR children mention the same
variables; gadgets are ORs over a variable's full domain).  Queries are
the same semiring sweeps and lazy k-best machinery as the boolean core.
"""

from __future__ import annotations

import math
import sys
from collections import Counter
from itertools import product
from typing import Dict, FrozenSet, Iterable, Iterator, List, Optional, Sequence, Tuple

from .circuit import AND, FALSE, LIT, OR, TRUE
from .eval import _forward
from .kbest import _NodeStream

FDLit = Tuple[int, FrozenSet[int]]  # (var, allowed value indices)
FDClause = Tuple[FDLit, ...]


class FDSpec:
    """Variable table: domain sizes and dense mvlit numbering."""

    def __init__(self) -> None:
        self.sizes: List[int] = []
        self.offsets: List[int] = []
        self.total = 0

    @property
    def num_vars(self) -> int:
        return len(self.sizes)

    def add_var(self, size: int) -> int:
        if size < 2:
            raise ValueError("domains need at least 2 values")
        self.offsets.append(self.total)
        self.sizes.append(size)
        self.total += size
        return len(self.sizes) - 1

    def mvlit(self, var: int, val: int) -> int:
        if not 0 <= val < self.sizes[var]:
            raise ValueError(f"value {val} out of domain for var {var}")
        return self.offsets[var] + val

    def decode(self, mvlit: int) -> Tuple[int, int]:
        # Linear scan is fine for the sizes we handle; callers in hot loops
        # keep their own mapping.
        for var in range(len(self.offsets) - 1, -1, -1):
            if mvlit >= self.offsets[var]:
                return var, mvlit - self.offsets[var]
        raise ValueError(f"bad mvlit {mvlit}")

    def full(self, var: int) -> FrozenSet[int]:
        return frozenset(range(self.sizes[var]))


class FDCnf:
    """A conjunction of FD clauses over an :class:`FDSpec`."""

    def __init__(self, spec: Optional[FDSpec] = None):
        self.spec = spec or FDSpec()
        self.clauses: List[FDClause] = []

    def add_clause(self, lits: Iterable[Tuple[int, Iterable[int]]]) -> None:
        """Add a disjunction of ``(var, values)`` literals.  Same-variable
        literals are unioned; a literal covering the full domain makes the
        clause a tautology (dropped); empty-set literals are dropped."""
        merged: Dict[int, FrozenSet[int]] = {}
        for var, values in lits:
            vs = frozenset(values) & self.spec.full(var)
            merged[var] = merged.get(var, frozenset()) | vs
        clause: List[FDLit] = []
        for var, vs in sorted(merged.items()):
            if vs == self.spec.full(var):
                return  # tautology
            if vs:
                clause.append((var, vs))
        self.clauses.append(tuple(clause))

    # -- reference semantics (exponential; tests and tiny problems) ----
    def satisfied_by(self, assignment: Sequence[int]) -> bool:
        for clause in self.clauses:
            if not any(assignment[var] in vs for var, vs in clause):
                return False
        return True

    def models(self) -> Iterator[Tuple[int, ...]]:
        for assignment in product(*(range(s) for s in self.spec.sizes)):
            if self.satisfied_by(assignment):
                yield assignment


# ----------------------------------------------------------------------
# Circuit
# ----------------------------------------------------------------------
class FDCircuit:
    """Finite-domain NNF circuit; ``lits[i]`` holds the mvlit for leaves."""

    def __init__(self, spec: FDSpec, kinds, lits, children, root):
        self.spec = spec
        self.kinds = kinds
        self.lits = lits
        self.children = children
        self.root = root

    def __len__(self) -> int:
        return len(self.kinds)

    @property
    def num_edges(self) -> int:
        return sum(len(c) for c in self.children)

    # Duck-typed hooks used by shared evaluators / the torch backend.
    def leaf_index(self, lit: int) -> int:
        return lit

    @property
    def num_weight_slots(self) -> int:
        return self.spec.total

    def var_sets(self) -> List[frozenset]:
        out: List[frozenset] = []
        for i, kind in enumerate(self.kinds):
            if kind == LIT:
                out.append(frozenset((self.spec.decode(self.lits[i])[0],)))
            elif kind in (AND, OR):
                s: frozenset = frozenset()
                for c in self.children[i]:
                    s = s | out[c]
                out.append(s)
            else:
                out.append(frozenset())
        return out

    def mentioned_vars(self) -> frozenset:
        return self.var_sets()[self.root]

    def asserted_values(self) -> List[frozenset]:
        """Per-node sets of (var, value) pairs every model must take."""
        out: List[frozenset] = []
        for i, kind in enumerate(self.kinds):
            if kind == LIT:
                out.append(frozenset((self.spec.decode(self.lits[i]),)))
            elif kind == AND:
                s: frozenset = frozenset()
                for c in self.children[i]:
                    s = s | out[c]
                out.append(s)
            else:
                out.append(frozenset())
        return out

    def is_decomposable(self) -> bool:
        vs = self.var_sets()
        for i, kind in enumerate(self.kinds):
            if kind != AND:
                continue
            seen: set = set()
            for c in self.children[i]:
                if seen & vs[c]:
                    return False
                seen |= vs[c]
        return True

    def is_smooth(self) -> bool:
        vs = self.var_sets()
        for i, kind in enumerate(self.kinds):
            if kind != OR:
                continue
            ch = self.children[i]
            if ch and any(vs[c] != vs[ch[0]] for c in ch[1:]):
                return False
        return True

    def is_deterministic(self) -> bool:
        """Syntactic check: every OR-child pair asserts conflicting values
        of some common variable (sufficient, not necessary)."""
        asserted = self.asserted_values()
        for i, kind in enumerate(self.kinds):
            if kind != OR:
                continue
            ch = self.children[i]
            for a in range(len(ch)):
                da = dict(asserted[ch[a]])
                for b in range(a + 1, len(ch)):
                    if not any(
                        var in da and da[var] != val
                        for var, val in asserted[ch[b]]
                    ):
                        return False
        return True

    def smooth(self) -> "FDCircuit":
        """Equivalent smooth circuit mentioning every variable at the root;
        missing variables get full-domain gadgets ``OR(var=v for v in D)``."""
        if self.kinds[self.root] == FALSE:
            return FDCircuit(self.spec, [FALSE], [0], [()], 0)
        vs = self.var_sets()
        b = FDBuilder(self.spec)
        gadgets: Dict[int, int] = {}

        def gadget(var: int) -> int:
            if var not in gadgets:
                gadgets[var] = b.or_(
                    [b.leaf(var, v) for v in range(self.spec.sizes[var])]
                )
            return gadgets[var]

        def pad(node_id: int, missing) -> int:
            missing = sorted(missing)
            if not missing:
                return node_id
            return b.and_([node_id] + [gadget(v) for v in missing])

        new_id: List[int] = []
        for i, kind in enumerate(self.kinds):
            if kind == FALSE:
                new_id.append(b.false())
            elif kind == TRUE:
                new_id.append(b.true())
            elif kind == LIT:
                var, val = self.spec.decode(self.lits[i])
                new_id.append(b.leaf(var, val))
            elif kind == AND:
                new_id.append(b.and_([new_id[c] for c in self.children[i]]))
            else:
                union: frozenset = frozenset()
                for c in self.children[i]:
                    union = union | vs[c]
                new_id.append(
                    b.or_(
                        [pad(new_id[c], union - vs[c]) for c in self.children[i]]
                    )
                )
        root = pad(
            new_id[self.root],
            frozenset(range(self.spec.num_vars)) - vs[self.root],
        )
        return b.finish(root)

    def stats(self) -> Dict[str, int]:
        c = Counter(self.kinds)
        return {
            "nodes": len(self),
            "edges": self.num_edges,
            "and": c[AND],
            "or": c[OR],
            "lit": c[LIT],
            "vars": self.spec.num_vars,
        }

    def __repr__(self) -> str:  # pragma: no cover
        s = self.stats()
        return (
            f"FDCircuit(nodes={s['nodes']}, edges={s['edges']}, "
            f"and={s['and']}, or={s['or']}, lit={s['lit']}, vars={s['vars']})"
        )


class FDBuilder:
    """Hash-consing bottom-up builder (FD twin of CircuitBuilder)."""

    def __init__(self, spec: FDSpec):
        self.spec = spec
        self.kinds: List[int] = []
        self.lits: List[int] = []
        self.children: List[Tuple[int, ...]] = []
        self._unique: Dict[Tuple, int] = {}

    def _emit(self, kind: int, lit: int, children: Tuple[int, ...]) -> int:
        key = (kind, lit, children)
        idx = self._unique.get(key)
        if idx is None:
            idx = len(self.kinds)
            self.kinds.append(kind)
            self.lits.append(lit)
            self.children.append(children)
            self._unique[key] = idx
        return idx

    def false(self) -> int:
        return self._emit(FALSE, 0, ())

    def true(self) -> int:
        return self._emit(TRUE, 0, ())

    def leaf(self, var: int, val: int) -> int:
        return self._emit(LIT, self.spec.mvlit(var, val), ())

    def and_(self, child_ids: Sequence[int]) -> int:
        flat: List[int] = []
        for c in child_ids:
            kind = self.kinds[c]
            if kind == FALSE:
                return self.false()
            if kind == TRUE:
                continue
            if kind == AND:
                flat.extend(self.children[c])
            else:
                flat.append(c)
        flat = sorted(set(flat))
        if not flat:
            return self.true()
        if len(flat) == 1:
            return flat[0]
        return self._emit(AND, 0, tuple(flat))

    def or_(self, child_ids: Sequence[int]) -> int:
        kept: List[int] = []
        for c in child_ids:
            kind = self.kinds[c]
            if kind == TRUE:
                return self.true()
            if kind == FALSE:
                continue
            kept.append(c)
        kept = sorted(set(kept))
        if not kept:
            return self.false()
        if len(kept) == 1:
            return kept[0]
        return self._emit(OR, 0, tuple(kept))

    def finish(self, root: int) -> FDCircuit:
        reachable = [False] * len(self.kinds)
        stack = [root]
        reachable[root] = True
        while stack:
            n = stack.pop()
            for c in self.children[n]:
                if not reachable[c]:
                    reachable[c] = True
                    stack.append(c)
        remap = [-1] * len(self.kinds)
        kinds: List[int] = []
        lits: List[int] = []
        children: List[Tuple[int, ...]] = []
        for i in range(len(self.kinds)):
            if reachable[i]:
                remap[i] = len(kinds)
                kinds.append(self.kinds[i])
                lits.append(self.lits[i])
                children.append(tuple(remap[c] for c in self.children[i]))
        return FDCircuit(self.spec, kinds, lits, children, remap[root])


# ----------------------------------------------------------------------
# Compilation: exhaustive d-way DPLL with components and caching
# ----------------------------------------------------------------------
def _assign(clauses: FrozenSet[FDClause], var: int, val: int) -> FrozenSet[FDClause]:
    out = []
    for clause in clauses:
        keep: List[FDLit] = []
        satisfied = False
        for cvar, vs in clause:
            if cvar == var:
                if val in vs:
                    satisfied = True
                    break
            else:
                keep.append((cvar, vs))
        if not satisfied:
            out.append(tuple(keep))
    return frozenset(out)


def _bcp_fd(
    clauses: FrozenSet[FDClause],
) -> Tuple[Optional[Dict[int, int]], FrozenSet[FDClause]]:
    """Propagate forced assignments (unit clauses whose single literal has a
    single value).  Returns (assignments, residual); assignments None on
    conflict."""
    assigned: Dict[int, int] = {}
    current = clauses
    while True:
        if () in current:
            return None, frozenset()
        forced: Dict[int, int] = {}
        for clause in current:
            if len(clause) == 1 and len(clause[0][1]) == 1:
                var, vs = clause[0]
                (val,) = vs
                if forced.get(var, val) != val:
                    return None, frozenset()
                forced[var] = val
        if not forced:
            return assigned, current
        for var, val in forced.items():
            if assigned.get(var, val) != val:
                return None, frozenset()
            assigned[var] = val
            current = _assign(current, var, val)


def _components_fd(clauses: FrozenSet[FDClause]) -> List[FrozenSet[FDClause]]:
    clause_list = list(clauses)
    parent = list(range(len(clause_list)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    var_to_clause: Dict[int, int] = {}
    for idx, clause in enumerate(clause_list):
        for var, _ in clause:
            if var in var_to_clause:
                ri, rj = find(var_to_clause[var]), find(idx)
                if ri != rj:
                    parent[rj] = ri
            else:
                var_to_clause[var] = idx
    groups: Dict[int, List[FDClause]] = {}
    for idx, clause in enumerate(clause_list):
        groups.setdefault(find(idx), []).append(clause)
    return [frozenset(g) for g in groups.values()]


def _pick_var_fd(
    clauses: FrozenSet[FDClause], order: Optional[Sequence[int]]
) -> int:
    present = {var for clause in clauses for var, _ in clause}
    if order is not None:
        for v in order:
            if v in present:
                return v
    counts = Counter(var for clause in clauses for var, _ in clause)
    return min(counts, key=lambda v: (-counts[v], v))


def compile_fd(
    cnf: FDCnf,
    var_order: Optional[Sequence[int]] = None,
    smooth: bool = False,
    heuristic: str = "dynamic",
) -> FDCircuit:
    """Compile an FD-CNF to a finite-domain decision-DNNF.

    Same architecture as the boolean compiler — unit propagation,
    connected-component decomposition (decomposable ANDs), d-way branching
    (deterministic ORs), component caching — but decisions enumerate a
    variable's domain directly, so there are no encoding artifacts.
    ``heuristic``: ``"dynamic"`` (most occurrences, default) or
    ``"minfill"`` (static, see :func:`minfill_order`); ignored when
    ``var_order`` is given.
    """
    if var_order is None and heuristic == "minfill":
        var_order = minfill_order(cnf)
    elif var_order is None and heuristic != "dynamic":
        raise ValueError(f"unknown heuristic {heuristic!r}")
    spec = cnf.spec
    builder = FDBuilder(spec)
    if any(clause == () for clause in cnf.clauses):
        return builder.finish(builder.false())
    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(old_limit, 10000 + 50 * spec.num_vars))
    try:
        memo: Dict[FrozenSet[FDClause], int] = {}

        def solve(clauses: FrozenSet[FDClause]) -> int:
            cached = memo.get(clauses)
            if cached is not None:
                return cached
            assigned, residual = _bcp_fd(clauses)
            if assigned is None:
                node = builder.false()
            else:
                parts = [
                    builder.leaf(var, val)
                    for var, val in sorted(assigned.items())
                ]
                for comp in _components_fd(residual):
                    comp_node = memo.get(comp)
                    if comp_node is None:
                        var = _pick_var_fd(comp, var_order)
                        branches = [
                            builder.and_(
                                [
                                    builder.leaf(var, val),
                                    solve(_assign(comp, var, val)),
                                ]
                            )
                            for val in range(spec.sizes[var])
                        ]
                        comp_node = builder.or_(branches)
                        memo[comp] = comp_node
                    parts.append(comp_node)
                node = builder.and_(parts)
            memo[clauses] = node
            return node

        root = solve(frozenset(cnf.clauses))
    finally:
        sys.setrecursionlimit(old_limit)
    circuit = builder.finish(root)
    if smooth:
        circuit = circuit.smooth()
    return circuit


# ----------------------------------------------------------------------
# Queries (weights are vectors indexed by mvlit, length spec.total)
# ----------------------------------------------------------------------
def _require_smooth_ddnnf(circuit: FDCircuit) -> None:
    cached = getattr(circuit, "_smooth_ddnnf_ok", None)
    if cached is None:
        if circuit.kinds[circuit.root] == FALSE:
            cached = True
        else:
            cached = (
                circuit.is_smooth()
                and circuit.is_deterministic()
                and circuit.mentioned_vars()
                == frozenset(range(circuit.spec.num_vars))
            )
        circuit._smooth_ddnnf_ok = cached  # type: ignore[attr-defined]
    if not cached:
        raise ValueError(
            "circuit is not a smooth FD d-DNNF; compile with smooth=True"
        )


def is_satisfiable(circuit: FDCircuit) -> bool:
    vals = _forward(
        circuit, lambda ml: True, lambda a, b: a or b,
        lambda a, b: a and b, False, True,
    )
    return vals[circuit.root]


def model_count(circuit: FDCircuit) -> int:
    _require_smooth_ddnnf(circuit)
    vals = _forward(
        circuit, lambda ml: 1, lambda a, b: a + b,
        lambda a, b: a * b, 0, 1,
    )
    return vals[circuit.root]


def wmc(circuit: FDCircuit, weights: Sequence[float]) -> float:
    _require_smooth_ddnnf(circuit)
    vals = _forward(
        circuit, lambda ml: weights[ml], lambda a, b: a + b,
        lambda a, b: a * b, 0.0, 1.0,
    )
    return vals[circuit.root]


def log_values(circuit: FDCircuit, log_weights: Sequence[float]) -> List[float]:
    _require_smooth_ddnnf(circuit)

    def lse(a: float, b: float) -> float:
        if a == -math.inf:
            return b
        if b == -math.inf:
            return a
        m = max(a, b)
        return m + math.log(math.exp(a - m) + math.exp(b - m))

    return _forward(
        circuit, lambda ml: log_weights[ml], lse,
        lambda a, b: a + b, -math.inf, 0.0,
    )


def log_wmc(circuit: FDCircuit, log_weights: Sequence[float]) -> float:
    return log_values(circuit, log_weights)[circuit.root]


def mpe(
    circuit: FDCircuit, costs: Sequence[float]
) -> Tuple[float, Optional[Dict[int, int]]]:
    """Min-cost model under additive per-(var,value) costs; returns
    ``(cost, {var: value})`` or ``(inf, None)``."""
    vals = _forward(
        circuit, lambda ml: costs[ml], min, lambda a, b: a + b,
        math.inf, 0.0,
    )
    best = vals[circuit.root]
    if best == math.inf:
        return math.inf, None
    assignment: Dict[int, int] = {}
    stack = [circuit.root]
    while stack:
        i = stack.pop()
        kind = circuit.kinds[i]
        if kind == LIT:
            var, val = circuit.spec.decode(circuit.lits[i])
            assignment[var] = val
        elif kind == AND:
            stack.extend(circuit.children[i])
        elif kind == OR:
            stack.append(min(circuit.children[i], key=lambda c: vals[c]))
    return best, assignment


def sample(
    circuit: FDCircuit, log_weights: Sequence[float], rng
) -> Optional[Dict[int, int]]:
    """Draw one exact sample from the distribution the weighted circuit
    defines: ``P(model) proportional to product of value weights``.

    Top-down: at each OR node a child is chosen with probability
    proportional to its weighted model mass (one log-sum-exp sweep
    computes all masses); AND nodes take every child.  Requires a smooth
    d-DNNF; returns None if the circuit has zero mass.
    """
    circuit = _ensure_smooth(circuit)
    vals = log_values(circuit, log_weights)
    if vals[circuit.root] == -math.inf:
        return None
    assignment: Dict[int, int] = {}
    stack = [circuit.root]
    while stack:
        i = stack.pop()
        kind = circuit.kinds[i]
        if kind == LIT:
            var, val = circuit.spec.decode(circuit.lits[i])
            assignment[var] = val
        elif kind == AND:
            stack.extend(circuit.children[i])
        elif kind == OR:
            total = vals[i]
            u = rng.random()
            acc = 0.0
            chosen = circuit.children[i][-1]
            for c in circuit.children[i]:
                if vals[c] == -math.inf:
                    continue
                acc += math.exp(vals[c] - total)
                if u <= acc:
                    chosen = c
                    break
            stack.append(chosen)
    return assignment


def minfill_order(cnf: FDCnf) -> List[int]:
    """Min-fill elimination order over the FD primal graph, reversed for
    branch-first-on-central-variables (see the boolean
    :func:`dnnf.compiler.minfill_order`)."""
    adj: Dict[int, set] = {v: set() for v in range(cnf.spec.num_vars)}
    for clause in cnf.clauses:
        cvars = [var for var, _ in clause]
        for i in range(len(cvars)):
            for j in range(i + 1, len(cvars)):
                adj[cvars[i]].add(cvars[j])
                adj[cvars[j]].add(cvars[i])
    remaining = set(adj)
    elim: List[int] = []

    def fill_cost(v: int) -> int:
        nbrs = list(adj[v])
        return sum(
            1
            for i in range(len(nbrs))
            for j in range(i + 1, len(nbrs))
            if nbrs[j] not in adj[nbrs[i]]
        )

    while remaining:
        v = min(remaining, key=lambda u: (fill_cost(u), len(adj[u]), u))
        nbrs = list(adj[v])
        for i in range(len(nbrs)):
            for j in range(i + 1, len(nbrs)):
                adj[nbrs[i]].add(nbrs[j])
                adj[nbrs[j]].add(nbrs[i])
        for u in nbrs:
            adj[u].discard(v)
        del adj[v]
        remaining.remove(v)
        elim.append(v)
    return list(reversed(elim))


# ----------------------------------------------------------------------
# Ordered enumeration and marginal MAP
# ----------------------------------------------------------------------
def _ensure_smooth(circuit: FDCircuit) -> FDCircuit:
    if not circuit.is_smooth() or (
        circuit.kinds[circuit.root] != FALSE
        and circuit.mentioned_vars() != frozenset(range(circuit.spec.num_vars))
    ):
        return circuit.smooth()
    return circuit


def enumerate_models(
    circuit: FDCircuit, costs: Sequence[float], k: Optional[int] = None
) -> Iterator[Tuple[float, Dict[int, int]]]:
    """Yield models as ``(cost, {var: value})`` in nondecreasing additive
    cost, best first — infinite-cost (evidence-contradicting) models are
    suppressed."""
    circuit = _ensure_smooth(circuit)
    streams: List[Optional[_NodeStream]] = [None] * len(circuit)
    for i, kind in enumerate(circuit.kinds):
        if kind == FALSE:
            streams[i] = _NodeStream(FALSE, (), None)
        elif kind == TRUE:
            streams[i] = _NodeStream(TRUE, (), (0.0, ()))
        elif kind == LIT:
            cost = costs[circuit.lits[i]]
            first = None if cost == math.inf else (cost, (circuit.lits[i],))
            streams[i] = _NodeStream(LIT, (), first)
        else:
            streams[i] = _NodeStream(
                kind, [streams[c] for c in circuit.children[i]], None
            )
    yield from _drain(circuit, streams, k)


def enumerate_map(
    circuit: FDCircuit,
    log_weights: Sequence[float],
    map_vars,
    k: Optional[int] = None,
) -> Iterator[Tuple[float, Dict[int, int]]]:
    """Ranked marginal MAP over ``map_vars`` (see the boolean
    :func:`dnnf.kbest.enumerate_map`; identical semantics with ``(var,
    value)`` assignments).  Requires map variables decided above all
    others — compile with ``var_order=list(map_vars)``."""
    map_vars = frozenset(map_vars)
    circuit = _ensure_smooth(circuit)
    vals = log_values(circuit, log_weights)
    var_sets = circuit.var_sets()
    asserted = circuit.asserted_values()
    has_map = [bool(var_sets[i] & map_vars) for i in range(len(circuit))]

    for i, kind in enumerate(circuit.kinds):
        if kind != OR or not has_map[i]:
            continue
        candidates = map_vars & {v for v, _ in asserted[circuit.children[i][0]]}
        for c in circuit.children[i][1:]:
            candidates = candidates & {v for v, _ in asserted[c]}
            if not candidates:
                break
        if not candidates:
            raise ValueError(
                "circuit is not constrained for marginal MAP over these "
                "variables; compile with var_order listing them first"
            )

    streams: List[Optional[_NodeStream]] = [None] * len(circuit)
    for i, kind in enumerate(circuit.kinds):
        if not has_map[i]:
            cost = -vals[i]
            first = None if cost == math.inf else (cost, ())
            streams[i] = _NodeStream(LIT, (), first)
        elif kind == LIT:
            cost = -log_weights[circuit.lits[i]]
            first = None if cost == math.inf else (cost, (circuit.lits[i],))
            streams[i] = _NodeStream(LIT, (), first)
        else:
            streams[i] = _NodeStream(
                kind, [streams[c] for c in circuit.children[i]], None
            )
    yield from _drain(circuit, streams, k)


# ----------------------------------------------------------------------
# Formula atoms and Tseitin encoding over FD literals
# ----------------------------------------------------------------------
from .formula import And, Formula, Not, Or  # noqa: E402


class FDAtom(Formula):
    """The atom ``value(var) in values`` — e.g. ``shutters=open`` or a
    threshold set like ``level in {low, medium}``."""

    def __init__(self, var: int, values: FrozenSet[int], name: str = ""):
        self.var = var
        self.values = frozenset(values)
        self.name = name

    def __repr__(self) -> str:  # pragma: no cover
        return self.name or f"v{self.var}in{sorted(self.values)}"


def encode(formulas: Iterable[Formula], cnf: FDCnf) -> None:
    """Tseitin-encode constraints into ``cnf`` (mutated).  FD literals are
    closed under negation (domain complement), so ``Not`` never needs an
    auxiliary; ``And``/``Or`` get hash-consed boolean-domain auxiliaries
    with biconditional clauses (auxiliaries are functionally determined,
    so neutral weights leave WMC/MPE untouched)."""
    spec = cnf.spec
    cache: Dict[Tuple, FDLit] = {}

    def neg(lit: FDLit) -> FDLit:
        var, vs = lit
        return (var, spec.full(var) - vs)

    def enc(f: Formula) -> FDLit:
        if isinstance(f, FDAtom):
            return (f.var, f.values)
        if isinstance(f, Not):
            return neg(enc(f.child))
        if isinstance(f, (And, Or)):
            child_lits = tuple(
                sorted(
                    (enc(c) for c in f.children),
                    key=lambda l: (l[0], tuple(sorted(l[1]))),
                )
            )
            key = (type(f).__name__, child_lits)
            if key in cache:
                return cache[key]
            aux_var = spec.add_var(2)
            aux: FDLit = (aux_var, frozenset((1,)))
            if isinstance(f, And):
                for cl in child_lits:
                    cnf.add_clause((neg(aux), cl))
                cnf.add_clause((aux,) + tuple(neg(cl) for cl in child_lits))
            else:
                cnf.add_clause((neg(aux),) + child_lits)
                for cl in child_lits:
                    cnf.add_clause((aux, neg(cl)))
            cache[key] = aux
            return aux
        raise TypeError(f"unknown formula node: {f!r}")

    def literal_clause(f: Formula) -> Optional[List[FDLit]]:
        def lit_of(g: Formula) -> Optional[FDLit]:
            if isinstance(g, FDAtom):
                return (g.var, g.values)
            if isinstance(g, Not) and isinstance(g.child, FDAtom):
                return neg((g.child.var, g.child.values))
            return None

        if isinstance(f, (FDAtom, Not)):
            l = lit_of(f)
            return [l] if l is not None else None
        if isinstance(f, Or):
            out = []
            for c in f.children:
                l = lit_of(c)
                if l is None:
                    return None
                out.append(l)
            return out
        return None

    def top(f: Formula) -> None:
        if isinstance(f, And):
            for c in f.children:
                top(c)
            return
        clause = literal_clause(f)
        if clause is not None:
            cnf.add_clause(clause)
        else:
            cnf.add_clause((enc(f),))

    for f in formulas:
        top(f)


def _drain(
    circuit: FDCircuit, streams, k: Optional[int]
) -> Iterator[Tuple[float, Dict[int, int]]]:
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
        yield cost, dict(circuit.spec.decode(ml) for ml in lits)
        emitted += 1
