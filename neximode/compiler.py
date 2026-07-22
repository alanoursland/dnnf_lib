"""Top-down CNF -> decision-DNNF compilation.

This is an exhaustive DPLL trace compiler in the style of c2d / dsharp / D4:

* **Unit propagation** at every search node; implied literals become AND
  conjuncts.
* **Component decomposition**: the residual clause set is split into
  connected components (clauses connected when they share a variable), each
  compiled independently and conjoined.  This is what yields decomposable
  AND nodes.
* **Branching** on a decision variable inside each component produces
  ``(v AND C|v) OR (~v AND C|~v)`` decision nodes, which makes every OR node
  deterministic.
* **Component caching**: residual clause sets are memoized so identical
  subproblems reached along different branches share one sub-circuit.  This
  is the mechanism that produces *small* (not minimal) circuits: circuit
  size is governed by the branching heuristic, mirroring the search +
  heuristics approach used in the JPL model-based diagnosis compiler.

The output is a decision-DNNF: decomposable and deterministic, suitable for
weighted model counting after :meth:`neximode.circuit.Circuit.smooth`.
"""

from __future__ import annotations

import sys
from collections import Counter
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

from .circuit import Circuit, CircuitBuilder
from .cnf import CNF

Clause = Tuple[int, ...]
ClauseSet = FrozenSet[Clause]


def _preprocess(cnf: CNF) -> Optional[List[Clause]]:
    """Deduplicate literals, drop tautologies. Returns None on empty clause."""
    out: List[Clause] = []
    for clause in cnf.clauses:
        lits = set(clause)
        if any(-l in lits for l in lits):
            continue  # tautology
        if not lits:
            return None
        out.append(tuple(sorted(lits, key=lambda l: (abs(l), l < 0))))
    return out


def _assign(clauses: ClauseSet, lit: int) -> ClauseSet:
    """Condition a clause set on a literal being true."""
    out = []
    for c in clauses:
        if lit in c:
            continue
        if -lit in c:
            out.append(tuple(l for l in c if l != -lit))
        else:
            out.append(c)
    return frozenset(out)


def _bcp(clauses: ClauseSet) -> Tuple[Optional[FrozenSet[int]], ClauseSet]:
    """Unit propagation.  Returns (implied literals, residual clauses);
    implied is None on conflict."""
    implied: set = set()
    current = clauses
    while True:
        if () in current:
            return None, frozenset()
        units = {c[0] for c in current if len(c) == 1}
        if not units:
            return frozenset(implied), current
        if any(-l in units for l in units):
            return None, frozenset()
        implied |= units
        for l in units:
            current = _assign(current, l)


def _components(clauses: ClauseSet) -> List[ClauseSet]:
    """Split clauses into connected components (shared-variable relation)."""
    clause_list = list(clauses)
    parent = list(range(len(clause_list)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    var_to_clause: Dict[int, int] = {}
    for idx, c in enumerate(clause_list):
        for l in c:
            v = abs(l)
            if v in var_to_clause:
                union(var_to_clause[v], idx)
            else:
                var_to_clause[v] = idx

    groups: Dict[int, List[Clause]] = {}
    for idx, c in enumerate(clause_list):
        groups.setdefault(find(idx), []).append(c)
    return [frozenset(g) for g in groups.values()]


def minfill_order(cnf: CNF) -> List[int]:
    """A static branching order from min-fill elimination on the primal graph.

    Min-fill repeatedly eliminates the variable whose neighborhood needs the
    fewest fill-in edges to become a clique — a standard treewidth heuristic.
    Variables eliminated *last* sit in the densest, most central part of the
    interaction graph, so branching on them *first* tends to disconnect the
    residual clause set quickly, which is exactly what produces small
    decomposable circuits.  Returns the reversed elimination order.
    """
    adj: Dict[int, set] = {v: set() for v in range(1, cnf.num_vars + 1)}
    for clause in cnf.clauses:
        cvars = [abs(l) for l in clause]
        for i in range(len(cvars)):
            for j in range(i + 1, len(cvars)):
                if cvars[i] != cvars[j]:
                    adj[cvars[i]].add(cvars[j])
                    adj[cvars[j]].add(cvars[i])
    remaining = set(adj)
    elim: List[int] = []

    def fill_cost(v: int) -> int:
        nbrs = list(adj[v])
        cost = 0
        for i in range(len(nbrs)):
            for j in range(i + 1, len(nbrs)):
                if nbrs[j] not in adj[nbrs[i]]:
                    cost += 1
        return cost

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


def _pick_var(clauses: ClauseSet, order: Optional[Sequence[int]]) -> int:
    if order is not None:
        present = {abs(l) for c in clauses for l in c}
        for v in order:
            if v in present:
                return v
    counts: Counter = Counter(abs(l) for c in clauses for l in c)
    # Most frequent variable; deterministic tie-break on index.
    return min(counts, key=lambda v: (-counts[v], v))


def compile_cnf(
    cnf: CNF,
    var_order: Optional[Sequence[int]] = None,
    smooth: bool = False,
    heuristic: str = "dynamic",
) -> Circuit:
    """Compile a CNF into a decision-DNNF circuit.

    Parameters
    ----------
    cnf:
        The input formula.
    var_order:
        Optional static branching order (list of variable indices).
        Variables listed here are branched before any others (earliest
        first); components containing none of them fall back to the
        heuristic.  The order strongly influences circuit size.
    smooth:
        If True, the result is also smoothed (required for model counting
        and weighted model counting via semiring evaluation).
    heuristic:
        ``"dynamic"`` — most-occurrences scoring per component (default);
        ``"minfill"`` — a static order from min-fill elimination on the
        primal graph (see :func:`minfill_order`), usually much better on
        structured instances.  Ignored when ``var_order`` is given.
    """
    if var_order is None and heuristic == "minfill":
        var_order = minfill_order(cnf)
    elif var_order is None and heuristic != "dynamic":
        raise ValueError(f"unknown heuristic {heuristic!r}")
    builder = CircuitBuilder(cnf.num_vars)
    pre = _preprocess(cnf)
    if pre is None:
        circuit = builder.finish(builder.false())
        return circuit
    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(old_limit, 10000 + 50 * cnf.num_vars))
    try:
        memo: Dict[ClauseSet, int] = {}

        def solve(clauses: ClauseSet) -> int:
            cached = memo.get(clauses)
            if cached is not None:
                return cached
            implied, residual = _bcp(clauses)
            if implied is None:
                node = builder.false()
            else:
                parts = [builder.literal(l) for l in sorted(implied)]
                for comp in _components(residual):
                    comp_node = memo.get(comp)
                    if comp_node is None:
                        v = _pick_var(comp, var_order)
                        pos = builder.and_(
                            [builder.literal(v), solve(_assign(comp, v))]
                        )
                        neg = builder.and_(
                            [builder.literal(-v), solve(_assign(comp, -v))]
                        )
                        comp_node = builder.or_([pos, neg])
                        memo[comp] = comp_node
                    parts.append(comp_node)
                node = builder.and_(parts)
            memo[clauses] = node
            return node

        root = solve(frozenset(pre))
    finally:
        sys.setrecursionlimit(old_limit)
    circuit = builder.finish(root)
    if smooth:
        circuit = circuit.smooth()
    return circuit
