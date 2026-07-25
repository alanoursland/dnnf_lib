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
weighted model counting after :meth:`modenexus.circuit.Circuit.smooth`.
"""

from __future__ import annotations

import sys
from collections import Counter
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

from .circuit import FALSE, OR, Circuit, CircuitBuilder
from .cnf import CNF
from .compile_control import CompileControl

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


def _compile_binary_forest(
    clauses: Sequence[Clause],
    num_vars: int,
    builder: CircuitBuilder,
    *,
    smooth: bool,
    session=None,
) -> Optional[Tuple[Circuit, int]]:
    """Compile an acyclic unary/binary CNF by linear-time tree DP.

    General DPLL remains the fallback.  On a primal forest, however, each
    subtree has only two boundary contexts (the parent variable's values),
    so repeatedly rescanning and hashing the residual suffix is unnecessary.
    The resulting circuit is smooth by construction.  Singleton OR wrappers
    keep forced subtrees from being flattened and recopied at every ancestor.
    """
    if any(len(clause) > 2 for clause in clauses):
        return None

    unary_allowed: Dict[int, List[bool]] = {}
    edge_clauses: Dict[Tuple[int, int], List[Clause]] = {}
    adjacency: Dict[int, set] = {}
    relevant: set = set()
    edges: set = set()

    for clause in clauses:
        if len(clause) == 1:
            lit = clause[0]
            var = abs(lit)
            relevant.add(var)
            allowed = unary_allowed.setdefault(var, [True, True])
            allowed[0 if lit > 0 else 1] = False
            adjacency.setdefault(var, set())
            continue
        if len(clause) == 2:
            left, right = sorted((abs(clause[0]), abs(clause[1])))
            if left == right:
                return None
            pair = (left, right)
            relevant.update(pair)
            adjacency.setdefault(left, set()).add(right)
            adjacency.setdefault(right, set()).add(left)
            edge_clauses.setdefault(pair, []).append(clause)
            edges.add(pair)

    # A repeated clause on one edge is fine; only distinct primal edges
    # participate in the cycle check.
    parent = {var: var for var in relevant}

    def find(var: int) -> int:
        while parent[var] != var:
            parent[var] = parent[parent[var]]
            var = parent[var]
        return var

    for left, right in edges:
        root_left, root_right = find(left), find(right)
        if root_left == root_right:
            return None
        parent[root_right] = root_left

    edge_allowed: Dict[
        Tuple[int, int], Tuple[Tuple[bool, bool], Tuple[bool, bool]]
    ] = {}
    for pair, pair_clauses in edge_clauses.items():
        left, right = pair
        rows = []
        for left_value in (False, True):
            row = []
            for right_value in (False, True):
                values = {left: left_value, right: right_value}
                row.append(
                    all(
                        any(
                            values[abs(lit)] == (lit > 0)
                            for lit in clause
                        )
                        for clause in pair_clauses
                    )
                )
            rows.append(tuple(row))
        edge_allowed[pair] = tuple(rows)  # type: ignore[assignment]

    memo: Dict[Tuple[int, Optional[int], Optional[bool]], int] = {}

    def allowed_with_parent(
        var: int,
        value: bool,
        parent_var: Optional[int],
        parent_value: Optional[bool],
    ) -> bool:
        allowed = unary_allowed.get(var, (True, True))
        if not allowed[1 if value else 0]:
            return False
        if parent_var is None:
            return True
        pair = tuple(sorted((var, parent_var)))
        matrix = edge_allowed[pair]
        if var == pair[0]:
            return matrix[1 if value else 0][
                1 if parent_value else 0
            ]
        return matrix[1 if parent_value else 0][1 if value else 0]

    def build(
        var: int,
        parent_var: Optional[int],
        parent_value: Optional[bool],
    ) -> int:
        key = (var, parent_var, parent_value)
        cached = memo.get(key)
        if cached is not None:
            if session is not None:
                session.cache_hits += 1
            return cached
        if session is not None:
            session.check(len(builder.kinds), len(memo))
            session.decisions += 1
        children = sorted(adjacency.get(var, ()) - {parent_var})
        branches = []
        for value in (False, True):
            if not allowed_with_parent(
                var, value, parent_var, parent_value
            ):
                continue
            child_nodes = [
                build(child, var, value) for child in children
            ]
            branch = builder.and_(
                [
                    builder.literal(var if value else -var),
                    *child_nodes,
                ]
            )
            if builder.kinds[branch] != FALSE:
                branches.append(branch)
        if not branches:
            node = builder.false()
        elif len(branches) == 1:
            # CircuitBuilder.or_ intentionally collapses singleton ORs.
            # Keeping this wrapper prevents a forced descendant AND from
            # being flattened and recopied at every ancestor.
            node = builder._emit(OR, 0, (branches[0],))
        else:
            node = builder.or_(branches)
        memo[key] = node
        if session is not None:
            session.check(len(builder.kinds), len(memo))
        return node

    component_roots = []
    seen: set = set()
    for root in sorted(relevant):
        if root in seen:
            continue
        stack = [root]
        seen.add(root)
        while stack:
            current = stack.pop()
            for neighbor in adjacency.get(current, ()):
                if neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        component_roots.append(root)
    if session is not None:
        session.components += len(component_roots)

    root_nodes = [build(root, None, None) for root in component_roots]
    if smooth:
        for var in range(1, num_vars + 1):
            if var not in relevant:
                root_nodes.append(
                    builder.or_(
                        [builder.literal(var), builder.literal(-var)]
                    )
                )
    root = builder.and_(root_nodes)
    return builder.finish(root), len(memo)


def compile_cnf(
    cnf: CNF,
    var_order: Optional[Sequence[int]] = None,
    smooth: bool = False,
    heuristic: str = "dynamic",
    control: Optional[CompileControl] = None,
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
        Under the dynamic default, acyclic unary/binary instances use an
        exact tree-DP fast path whose work and circuit size are linear in
        the forest.
    control:
        Optional timeout, cancellation callback, node/cache budgets, and
        progress callback.  Interrupted exceptions carry partial statistics.
    """
    if var_order is None and heuristic == "minfill":
        var_order = minfill_order(cnf)
    elif var_order is None and heuristic != "dynamic":
        raise ValueError(f"unknown heuristic {heuristic!r}")
    builder = CircuitBuilder(cnf.num_vars)
    session = control._start() if control is not None else None
    if session is not None:
        session.check(len(builder.kinds), 0, force=True)
    pre = _preprocess(cnf)
    if pre is None:
        circuit = builder.finish(builder.false())
        if session is not None:
            session.check(len(circuit), 0)
            session.finish(len(circuit), 0)
        return circuit
    if var_order is None:
        forest_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(
            max(forest_limit, 10000 + 50 * cnf.num_vars)
        )
        try:
            forest_result = _compile_binary_forest(
                pre,
                cnf.num_vars,
                builder,
                smooth=smooth,
                session=session,
            )
        finally:
            sys.setrecursionlimit(forest_limit)
        if forest_result is not None:
            forest, forest_cache_entries = forest_result
            if session is not None:
                session.check(len(forest), forest_cache_entries)
                session.finish(len(forest), forest_cache_entries)
            return forest
    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(old_limit, 10000 + 50 * cnf.num_vars))
    try:
        memo: Dict[ClauseSet, int] = {}

        def solve(clauses: ClauseSet) -> int:
            if session is not None:
                session.check(len(builder.kinds), len(memo))
            cached = memo.get(clauses)
            if cached is not None:
                if session is not None:
                    session.cache_hits += 1
                return cached
            implied, residual = _bcp(clauses)
            if implied is None:
                node = builder.false()
            else:
                parts = [builder.literal(l) for l in sorted(implied)]
                comps = _components(residual)
                if session is not None:
                    session.components += len(comps)
                for comp in comps:
                    comp_node = memo.get(comp)
                    if comp_node is None:
                        v = _pick_var(comp, var_order)
                        if session is not None:
                            session.decisions += 1
                        pos = builder.and_(
                            [builder.literal(v), solve(_assign(comp, v))]
                        )
                        neg = builder.and_(
                            [builder.literal(-v), solve(_assign(comp, -v))]
                        )
                        comp_node = builder.or_([pos, neg])
                        memo[comp] = comp_node
                    elif session is not None:
                        session.cache_hits += 1
                    parts.append(comp_node)
                node = builder.and_(parts)
            memo[clauses] = node
            if session is not None:
                session.check(len(builder.kinds), len(memo))
            return node

        root = solve(frozenset(pre))
    finally:
        sys.setrecursionlimit(old_limit)
    circuit = builder.finish(root)
    if smooth:
        circuit = circuit.smooth()
    if session is not None:
        session.check(len(circuit), len(memo))
        session.finish(len(circuit), len(memo))
    return circuit
