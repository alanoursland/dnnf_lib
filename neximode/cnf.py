"""CNF formulas with DIMACS I/O.

Clauses are tuples of DIMACS literals (non-zero ints; ``+v`` / ``-v``).
"""

from __future__ import annotations

import io
from itertools import product
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple


class CNF:
    def __init__(self, num_vars: int = 0, clauses: Iterable[Sequence[int]] = ()):
        self.num_vars = num_vars
        self.clauses: List[Tuple[int, ...]] = []
        for c in clauses:
            self.add_clause(c)

    def add_clause(self, lits: Sequence[int]) -> None:
        clause = tuple(lits)
        for l in clause:
            if l == 0:
                raise ValueError("0 is not a valid literal")
            if abs(l) > self.num_vars:
                self.num_vars = abs(l)
        self.clauses.append(clause)

    def add_var(self) -> int:
        """Allocate a fresh variable and return its (1-based) index."""
        self.num_vars += 1
        return self.num_vars

    # ------------------------------------------------------------------
    # DIMACS
    # ------------------------------------------------------------------
    @classmethod
    def from_dimacs(cls, source) -> "CNF":
        """Parse DIMACS CNF from a path, file object, or string."""
        if isinstance(source, str) and "\n" not in source and "\0" not in source:
            with open(source, "r") as f:
                return cls._parse(f)
        if isinstance(source, str):
            return cls._parse(io.StringIO(source))
        return cls._parse(source)

    @classmethod
    def _parse(cls, f) -> "CNF":
        cnf = cls()
        declared_vars = 0
        pending: List[int] = []
        for line in f:
            line = line.strip()
            if not line or line.startswith(("c", "%")):
                continue
            if line.startswith("p"):
                parts = line.split()
                if len(parts) >= 4 and parts[1] == "cnf":
                    declared_vars = int(parts[2])
                continue
            for tok in line.split():
                lit = int(tok)
                if lit == 0:
                    cnf.add_clause(pending)
                    pending = []
                else:
                    pending.append(lit)
        if pending:
            cnf.add_clause(pending)
        cnf.num_vars = max(cnf.num_vars, declared_vars)
        return cnf

    def to_dimacs(self) -> str:
        lines = [f"p cnf {self.num_vars} {len(self.clauses)}"]
        for c in self.clauses:
            lines.append(" ".join(str(l) for l in c) + " 0")
        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------------
    # Reference semantics (exponential; for testing and tiny problems)
    # ------------------------------------------------------------------
    def satisfied_by(self, assignment: Dict[int, bool]) -> bool:
        for clause in self.clauses:
            if not any(assignment.get(abs(l), None) == (l > 0) for l in clause):
                return False
        return True

    def models(self) -> Iterator[Dict[int, bool]]:
        """Enumerate all models over variables ``1..num_vars`` (brute force)."""
        variables = list(range(1, self.num_vars + 1))
        for values in product((False, True), repeat=len(variables)):
            assignment = dict(zip(variables, values))
            if self.satisfied_by(assignment):
                yield assignment

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"CNF(num_vars={self.num_vars}, clauses={len(self.clauses)})"
