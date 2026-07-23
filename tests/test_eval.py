import math
import random

import pytest

from modenexus import (
    CNF,
    compile_cnf,
    condition_weights,
    log_wmc,
    mpe,
    wmc,
)
from helpers import (
    assignment_cost,
    brute_mpe,
    brute_wmc,
    random_cnf,
    random_weights,
)


@pytest.mark.parametrize("seed", range(20))
def test_wmc_matches_brute_force(seed):
    rng = random.Random(seed)
    num_vars = rng.randint(3, 7)
    cnf = random_cnf(rng, num_vars, rng.randint(2, 2 * num_vars))
    circuit = compile_cnf(cnf, smooth=True)
    weights = random_weights(rng, num_vars)
    expected = brute_wmc(cnf, weights)
    assert wmc(circuit, weights) == pytest.approx(expected, rel=1e-9)
    log_w = [math.log(x) for x in weights]
    if expected > 0:
        assert log_wmc(circuit, log_w) == pytest.approx(
            math.log(expected), rel=1e-9
        )
    else:
        assert log_wmc(circuit, log_w) == -math.inf


@pytest.mark.parametrize("seed", range(20))
def test_mpe_matches_brute_force(seed):
    rng = random.Random(1000 + seed)
    num_vars = rng.randint(3, 7)
    cnf = random_cnf(rng, num_vars, rng.randint(2, 2 * num_vars))
    circuit = compile_cnf(cnf, smooth=True)
    costs = random_weights(rng, num_vars)
    expected = brute_mpe(cnf, costs)
    cost, assignment = mpe(circuit, costs)
    if expected == math.inf:
        assert cost == math.inf and assignment is None
    else:
        assert cost == pytest.approx(expected, rel=1e-9)
        # decoded assignment is a real model achieving the cost
        assert cnf.satisfied_by(assignment)
        assert set(assignment) == set(range(1, num_vars + 1))
        assert assignment_cost(assignment, costs) == pytest.approx(
            cost, rel=1e-9
        )


def test_conditioning_via_weights():
    # (x1 or x2), condition on x1=False -> only models with x2=True
    cnf = CNF(num_vars=2, clauses=[(1, 2)])
    circuit = compile_cnf(cnf, smooth=True)
    weights = [1.0] * 4
    conditioned = condition_weights(weights, {1: False}, annihilator=0.0)
    assert wmc(circuit, conditioned) == 1.0


def test_wmc_requires_smooth():
    cnf = CNF(num_vars=3, clauses=[(1, 2)])
    circuit = compile_cnf(cnf, smooth=False)
    with pytest.raises(ValueError):
        wmc(circuit, [1.0] * 6)
