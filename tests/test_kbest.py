import math
import random

import pytest

from neximode import CNF, compile_cnf, condition_weights, enumerate_models
from helpers import (
    assignment_cost,
    brute_sorted_costs,
    random_cnf,
    random_weights,
)


@pytest.mark.parametrize("seed", range(20))
def test_kbest_matches_brute_force_order(seed):
    rng = random.Random(seed)
    num_vars = rng.randint(3, 6)
    cnf = random_cnf(rng, num_vars, rng.randint(2, 2 * num_vars))
    circuit = compile_cnf(cnf, smooth=True)
    costs = random_weights(rng, num_vars)
    expected = brute_sorted_costs(cnf, costs)

    got = list(enumerate_models(circuit, costs))
    # Same number of models, identical sorted cost sequence.
    assert len(got) == len(expected)
    for (cost, assignment), exp_cost in zip(got, expected):
        assert cost == pytest.approx(exp_cost, rel=1e-9)
        assert cnf.satisfied_by(assignment)
        assert assignment_cost(assignment, costs) == pytest.approx(
            cost, rel=1e-9
        )
    # Nondecreasing and distinct assignments.
    costs_only = [c for c, _ in got]
    assert costs_only == sorted(costs_only)
    keys = {tuple(sorted(a.items())) for _, a in got}
    assert len(keys) == len(got)


def test_k_limits_output():
    cnf = CNF(num_vars=4, clauses=[(1, 2, 3, 4)])
    circuit = compile_cnf(cnf, smooth=True)
    costs = [0.5] * 8
    got = list(enumerate_models(circuit, costs, k=3))
    assert len(got) == 3


def test_evidence_suppresses_models():
    cnf = CNF(num_vars=2, clauses=[(1, 2)])
    circuit = compile_cnf(cnf, smooth=True)
    costs = condition_weights([0.1] * 4, {1: False}, annihilator=math.inf)
    got = list(enumerate_models(circuit, costs))
    assert len(got) == 1
    assert got[0][1] == {1: False, 2: True}


def test_unsat_yields_nothing():
    cnf = CNF(num_vars=1, clauses=[(1,), (-1,)])
    circuit = compile_cnf(cnf, smooth=True)
    assert list(enumerate_models(circuit, [0.1] * 2)) == []
