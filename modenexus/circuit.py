"""Boolean NNF circuit representation, inspection, and transformations.

Circuits use flat topological arrays and DIMACS literals. See
``CONTRACTS.md`` for DNNF property definitions and evaluator requirements.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .invariants import ModeNexusInvariantError


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
        if self.kinds[self.root] == FALSE:
            return True
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
        return vs[self.root] == frozenset(
            range(1, self.num_vars + 1)
        )

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
        """Run the circuit's conservative syntactic determinism check."""
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
        """Return a simplified circuit conditioned by ``{var: bool}``."""
        for var, value in assignment.items():
            if not isinstance(var, int) or not 1 <= var <= self.num_vars:
                raise ValueError(
                    f"condition variable {var!r} is outside "
                    f"1..{self.num_vars}"
                )
            if not isinstance(value, bool):
                raise ValueError(
                    f"condition value for variable {var} must be boolean"
                )
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
        asserted = [
            b.literal(var if value else -var)
            for var, value in sorted(assignment.items())
        ]
        conditioned = b.finish(b.and_([new_id[self.root], *asserted]))
        if conditioned.kinds[conditioned.root] != FALSE:
            root_assertions = conditioned.asserted_literals()[
                conditioned.root
            ]
            expected = {
                var if value else -var
                for var, value in assignment.items()
            }
            if not expected <= root_assertions:
                raise ModeNexusInvariantError(
                    "conditioned circuit did not retain all evidence literals"
                )
        return conditioned

    def smooth(self) -> "Circuit":
        """Return the smoothed form used by counting evaluators."""
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
    """Build hash-consed circuits with basic AND/OR simplification."""

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
