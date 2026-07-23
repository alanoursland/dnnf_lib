"""Brute-force reference implementations for cross-validation."""

from __future__ import annotations

import math
import random
from itertools import product
from typing import Dict, List, Sequence, Tuple

from modenexus.circuit import lit_index
from modenexus.cnf import CNF


def random_cnf(rng: random.Random, num_vars: int, num_clauses: int) -> CNF:
    cnf = CNF(num_vars=num_vars)
    for _ in range(num_clauses):
        width = rng.choice((1, 2, 2, 3, 3, 3))
        vs = rng.sample(range(1, num_vars + 1), min(width, num_vars))
        cnf.add_clause([v if rng.random() < 0.5 else -v for v in vs])
    return cnf


def random_weights(rng: random.Random, num_vars: int) -> List[float]:
    return [rng.uniform(0.1, 2.0) for _ in range(2 * num_vars)]


def assignment_weight(
    assignment: Dict[int, bool], weights: Sequence[float]
) -> float:
    w = 1.0
    for var, val in assignment.items():
        w *= weights[lit_index(var if val else -var)]
    return w


def assignment_cost(
    assignment: Dict[int, bool], costs: Sequence[float]
) -> float:
    return sum(
        costs[lit_index(var if val else -var)]
        for var, val in assignment.items()
    )


def brute_wmc(cnf: CNF, weights: Sequence[float]) -> float:
    return sum(assignment_weight(m, weights) for m in cnf.models())


def brute_mpe(cnf: CNF, costs: Sequence[float]) -> float:
    best = math.inf
    for m in cnf.models():
        best = min(best, assignment_cost(m, costs))
    return best


def brute_sorted_costs(cnf: CNF, costs: Sequence[float]) -> List[float]:
    return sorted(assignment_cost(m, costs) for m in cnf.models())
