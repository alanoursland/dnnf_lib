"""A small propositional formula language with Tseitin CNF encoding.

Formulas are built from :class:`Prop` atoms (propositional variables) with
the connectives ``&``, ``|``, ``~``, ``>>`` (implies), plus the helpers
:func:`iff`, :func:`xor`, :func:`exactly_one`, :func:`at_most_one`.

``encode(formulas, cnf)`` converts a list of constraints to clauses inside
a :class:`modenexus.cnf.CNF`, introducing fresh auxiliary (Tseitin) variables
for internal connectives.  The encoding uses full biconditional Tseitin
clauses, so auxiliary variables are *functionally determined* by the
original variables: model counts over the original variables are preserved,
and giving auxiliary literals neutral weight leaves WMC/MPE untouched.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

from .cnf import CNF


class Formula:
    def __and__(self, other: "Formula") -> "Formula":
        return And((self, other))

    def __or__(self, other: "Formula") -> "Formula":
        return Or((self, other))

    def __invert__(self) -> "Formula":
        return Not(self)

    def __rshift__(self, other: "Formula") -> "Formula":
        return Or((Not(self), other))


class Prop(Formula):
    """A propositional variable atom bound to DIMACS variable ``var``."""

    def __init__(self, var: int, name: str = ""):
        self.var = var
        self.name = name or f"x{var}"

    def __repr__(self) -> str:  # pragma: no cover
        return self.name


class Not(Formula):
    def __init__(self, child: Formula):
        self.child = child


class And(Formula):
    def __init__(self, children: Iterable[Formula]):
        self.children = tuple(children)


class Or(Formula):
    def __init__(self, children: Iterable[Formula]):
        self.children = tuple(children)


def iff(a: Formula, b: Formula) -> Formula:
    return And(((a >> b), (b >> a)))


def xor(a: Formula, b: Formula) -> Formula:
    return Not(iff(a, b))


def at_most_one(props: Sequence[Formula]) -> Formula:
    parts = []
    for i in range(len(props)):
        for j in range(i + 1, len(props)):
            parts.append(Or((Not(props[i]), Not(props[j]))))
    return And(parts) if parts else And(())


def exactly_one(props: Sequence[Formula]) -> Formula:
    return And((Or(props), at_most_one(props)))


# ----------------------------------------------------------------------
# Tseitin encoding
# ----------------------------------------------------------------------
def _as_literal_clause(f: Formula) -> List[int] | None:
    """If ``f`` is a disjunction of literals (or a single literal), return
    the clause; else None."""

    def lit_of(g: Formula) -> int | None:
        if isinstance(g, Prop):
            return g.var
        if isinstance(g, Not) and isinstance(g.child, Prop):
            return g.child.var
        return None

    if isinstance(g := f, (Prop, Not)):
        l = lit_of(g)
        if l is not None:
            return [l if isinstance(g, Prop) else -l]
        return None
    if isinstance(f, Or):
        clause = []
        for c in f.children:
            l = lit_of(c)
            if l is None:
                return None
            clause.append(l if isinstance(c, Prop) else -l)
        return clause
    return None


def encode(formulas: Iterable[Formula], cnf: CNF) -> None:
    """Encode constraints into ``cnf`` (mutated in place).

    Top-level conjunctions are split, and disjunctions of plain literals
    become single clauses without auxiliaries; everything else goes through
    biconditional Tseitin with hash-consed auxiliary variables.
    """
    cache: Dict[Tuple, int] = {}

    def lit(f: Formula) -> int:
        if isinstance(f, Prop):
            return f.var
        if isinstance(f, Not):
            return -lit(f.child)
        if isinstance(f, (And, Or)):
            child_lits = tuple(sorted(lit(c) for c in f.children))
            key = (type(f).__name__, child_lits)
            if key in cache:
                return cache[key]
            aux = cnf.add_var()
            if isinstance(f, And):
                for cl in child_lits:
                    cnf.add_clause((-aux, cl))
                cnf.add_clause((aux,) + tuple(-cl for cl in child_lits))
            else:
                cnf.add_clause((-aux,) + child_lits)
                for cl in child_lits:
                    cnf.add_clause((aux, -cl))
            cache[key] = aux
            return aux
        raise TypeError(f"unknown formula node: {f!r}")

    def top(f: Formula) -> None:
        if isinstance(f, And):
            for c in f.children:
                top(c)
            return
        clause = _as_literal_clause(f)
        if clause is not None:
            cnf.add_clause(clause)
            return
        cnf.add_clause((lit(f),))

    for f in formulas:
        top(f)
