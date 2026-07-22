"""Negation Normal Form circuits and the DNNF family.

An NNF circuit is a rooted DAG whose leaves are literals (or the constants
true/false) and whose internal nodes are AND / OR gates.  The properties that
make these circuits useful for tractable reasoning are:

* **Decomposability** (the "D" in DNNF): the children of every AND node
  mention pairwise-disjoint sets of variables.  This makes satisfiability,
  minimum-cost model extraction, and model enumeration linear-time in the
  circuit size.
* **Determinism** (d-DNNF): the children of every OR node are pairwise
  logically inconsistent.  Together with decomposability this makes (weighted)
  model counting linear-time.
* **Smoothness**: the children of every OR node mention the same set of
  variables.  Required for counting-style semiring evaluations to be correct;
  it can always be enforced with only a modest size increase.

Nodes are stored in flat parallel arrays, in topological order (children
always precede parents).  This representation is convenient both for the
pure-Python evaluators and for exporting the circuit as a layered tensor
program for GPU evaluation.

Literals use the DIMACS convention: variable ``v`` (1-based) appears as the
integers ``+v`` and ``-v``.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Node kinds
FALSE = 0
TRUE = 1
LIT = 2
AND = 3
OR = 4

_KIND_NAMES = {FALSE: "F", TRUE: "T", LIT: "L", AND: "A", OR: "O"}


def lit_index(lit: int) -> int:
    """Map a DIMACS literal to a dense index in ``[0, 2 * num_vars)``.

    ``+v -> 2*(v-1)`` and ``-v -> 2*(v-1) + 1``.
    """
    if lit > 0:
        return 2 * (lit - 1)
    return 2 * (-lit - 1) + 1


class Circuit:
    """An immutable NNF circuit in topological array form."""

    def __init__(
        self,
        num_vars: int,
        kinds: List[int],
        lits: List[int],
        children: List[Tuple[int, ...]],
        root: int,
    ):
        self.num_vars = num_vars
        self.kinds = kinds
        self.lits = lits  # 0 for non-literal nodes
        self.children = children  # () for leaves
        self.root = root

    # ------------------------------------------------------------------
    # Basic introspection
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.kinds)

    @property
    def num_edges(self) -> int:
        return sum(len(c) for c in self.children)

    def var_sets(self) -> List[frozenset]:
        """Per-node sets of mentioned variables, bottom-up."""
        out: List[frozenset] = []
        for i, kind in enumerate(self.kinds):
            if kind == LIT:
                out.append(frozenset((abs(self.lits[i]),)))
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

    # ------------------------------------------------------------------
    # Property checks
    # ------------------------------------------------------------------
    def is_decomposable(self) -> bool:
        vs = self.var_sets()
        for i, kind in enumerate(self.kinds):
            if kind != AND:
                continue
            seen: set = set()
            for c in self.children[i]:
                cv = vs[c]
                if seen & cv:
                    return False
                seen |= cv
        return True

    def is_smooth(self) -> bool:
        vs = self.var_sets()
        for i, kind in enumerate(self.kinds):
            if kind != OR:
                continue
            ch = self.children[i]
            if not ch:
                continue
            first = vs[ch[0]]
            if any(vs[c] != first for c in ch[1:]):
                return False
        return True

    def asserted_literals(self) -> List[frozenset]:
        """Per-node sets of literals every model of the node must satisfy
        (syntactic under-approximation): the literal itself for a leaf, the
        union over children for AND, empty for OR/constants."""
        asserted: List[frozenset] = []
        for i, kind in enumerate(self.kinds):
            if kind == LIT:
                asserted.append(frozenset((self.lits[i],)))
            elif kind == AND:
                s: frozenset = frozenset()
                for c in self.children[i]:
                    s = s | asserted[c]
                asserted.append(s)
            else:
                asserted.append(frozenset())
        return asserted

    def is_deterministic(self) -> bool:
        """Syntactic determinism check (sufficient, not necessary).

        Two OR-children are considered provably inconsistent when one asserts
        a literal whose negation the other asserts (see
        :meth:`asserted_literals`).  This covers decision nodes produced by
        the compiler and smoothing gadgets ``(v OR ~v)``.
        """
        asserted = self.asserted_literals()
        for i, kind in enumerate(self.kinds):
            if kind != OR:
                continue
            ch = self.children[i]
            for a in range(len(ch)):
                for b in range(a + 1, len(ch)):
                    sa, sb = asserted[ch[a]], asserted[ch[b]]
                    if not any(-l in sb for l in sa):
                        return False
        return True

    # ------------------------------------------------------------------
    # Transformations (each returns a new Circuit)
    # ------------------------------------------------------------------
    def condition(self, assignment: Dict[int, bool]) -> "Circuit":
        """Structurally condition on a partial assignment ``{var: value}``.

        Literals consistent with the assignment become TRUE, contradicted
        literals become FALSE, and the circuit is re-simplified.  The result
        mentions none of the assigned variables.
        """
        b = CircuitBuilder(self.num_vars)
        new_id: List[int] = []
        for i, kind in enumerate(self.kinds):
            if kind == FALSE:
                new_id.append(b.false())
            elif kind == TRUE:
                new_id.append(b.true())
            elif kind == LIT:
                lit = self.lits[i]
                v = abs(lit)
                if v in assignment:
                    holds = assignment[v] == (lit > 0)
                    new_id.append(b.true() if holds else b.false())
                else:
                    new_id.append(b.literal(lit))
            elif kind == AND:
                new_id.append(b.and_([new_id[c] for c in self.children[i]]))
            else:
                new_id.append(b.or_([new_id[c] for c in self.children[i]]))
        return b.finish(new_id[self.root])

    def smooth(self) -> "Circuit":
        """Return an equivalent smooth circuit mentioning all ``num_vars``
        variables at the root.

        Missing variables are filled in with deterministic gadgets
        ``(v OR ~v)``, so determinism and decomposability are preserved.
        A FALSE root is returned unchanged.
        """
        if self.kinds[self.root] == FALSE:
            return Circuit(self.num_vars, [FALSE], [0], [()], 0)
        vs = self.var_sets()
        b = CircuitBuilder(self.num_vars)
        gadgets: Dict[int, int] = {}

        def gadget(v: int) -> int:
            if v not in gadgets:
                gadgets[v] = b.or_([b.literal(v), b.literal(-v)])
            return gadgets[v]

        def pad(node_id: int, missing: Iterable[int]) -> int:
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
                new_id.append(b.literal(self.lits[i]))
            elif kind == AND:
                new_id.append(b.and_([new_id[c] for c in self.children[i]]))
            else:  # OR: pad children up to the union of mentioned vars
                union: frozenset = frozenset()
                for c in self.children[i]:
                    union = union | vs[c]
                padded = [
                    pad(new_id[c], union - vs[c]) for c in self.children[i]
                ]
                new_id.append(b.or_(padded))
        root = new_id[self.root]
        all_vars = frozenset(range(1, self.num_vars + 1))
        root = pad(root, all_vars - vs[self.root])
        return b.finish(root)

    # ------------------------------------------------------------------
    def stats(self) -> Dict[str, int]:
        from collections import Counter

        c = Counter(self.kinds)
        return {
            "nodes": len(self),
            "edges": self.num_edges,
            "and": c[AND],
            "or": c[OR],
            "lit": c[LIT],
            "vars": self.num_vars,
        }

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        s = self.stats()
        return (
            f"Circuit(nodes={s['nodes']}, edges={s['edges']}, "
            f"and={s['and']}, or={s['or']}, lit={s['lit']}, vars={s['vars']})"
        )


class CircuitBuilder:
    """Constructs circuits bottom-up with hash-consing and on-the-fly
    algebraic simplification.

    Simplifications applied:

    * AND: flattens nested ANDs, drops TRUE children, collapses to FALSE if
      any child is FALSE, deduplicates children, collapses singletons.
    * OR: drops FALSE children, collapses to TRUE if any child is TRUE,
      deduplicates children, collapses singletons.  Nested ORs are *not*
      flattened, since that could destroy determinism guarantees the caller
      is relying on.

    Nodes are emitted in topological order by construction, so a builder's
    output can be evaluated with a single forward sweep.
    """

    def __init__(self, num_vars: int):
        self.num_vars = num_vars
        self.kinds: List[int] = []
        self.lits: List[int] = []
        self.children: List[Tuple[int, ...]] = []
        self._unique: Dict[Tuple, int] = {}
        self._false: Optional[int] = None
        self._true: Optional[int] = None

    def _emit(self, kind: int, lit: int, children: Tuple[int, ...]) -> int:
        key = (kind, lit, children)
        idx = self._unique.get(key)
        if idx is not None:
            return idx
        idx = len(self.kinds)
        self.kinds.append(kind)
        self.lits.append(lit)
        self.children.append(children)
        self._unique[key] = idx
        return idx

    def false(self) -> int:
        if self._false is None:
            self._false = self._emit(FALSE, 0, ())
        return self._false

    def true(self) -> int:
        if self._true is None:
            self._true = self._emit(TRUE, 0, ())
        return self._true

    def literal(self, lit: int) -> int:
        if lit == 0 or abs(lit) > self.num_vars:
            raise ValueError(f"literal {lit} out of range for {self.num_vars} vars")
        return self._emit(LIT, lit, ())

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

    def finish(self, root: int) -> Circuit:
        """Produce a Circuit containing only nodes reachable from ``root``."""
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
            if not reachable[i]:
                continue
            remap[i] = len(kinds)
            kinds.append(self.kinds[i])
            lits.append(self.lits[i])
            children.append(tuple(remap[c] for c in self.children[i]))
        return Circuit(self.num_vars, kinds, lits, children, remap[root])
