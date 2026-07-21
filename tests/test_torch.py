import math
import random

import pytest

torch = pytest.importorskip("torch")

from dnnf import CNF, compile_cnf, log_wmc, mpe
from dnnf.circuit import lit_index
from dnnf.torch_backend import TorchCircuit
from helpers import brute_wmc, random_cnf, random_weights


@pytest.mark.parametrize("seed", range(10))
def test_torch_log_wmc_matches_cpu(seed):
    rng = random.Random(seed)
    num_vars = rng.randint(3, 7)
    cnf = random_cnf(rng, num_vars, rng.randint(2, 2 * num_vars))
    circuit = compile_cnf(cnf, smooth=True)
    weights = random_weights(rng, num_vars)
    log_w = [math.log(x) for x in weights]
    expected = log_wmc(circuit, log_w)

    tc = TorchCircuit(circuit, semiring="logprob")
    got = tc(torch.tensor(log_w, dtype=torch.float64)).item()
    if expected == -math.inf:
        assert got == -math.inf
    else:
        assert got == pytest.approx(expected, rel=1e-9)


def test_torch_batched_rows_independent():
    rng = random.Random(42)
    cnf = random_cnf(rng, 5, 8)
    circuit = compile_cnf(cnf, smooth=True)
    tc = TorchCircuit(circuit, semiring="logprob")
    rows = [random_weights(rng, 5) for _ in range(4)]
    batch = torch.tensor(
        [[math.log(x) for x in row] for row in rows], dtype=torch.float64
    )
    out = tc(batch)
    for i, row in enumerate(rows):
        expected = brute_wmc(cnf, row)
        if expected == 0:
            assert out[i].item() == -math.inf
        else:
            assert out[i].item() == pytest.approx(math.log(expected), rel=1e-9)


@pytest.mark.parametrize("seed", range(5))
def test_marginals_match_brute_force(seed):
    rng = random.Random(seed)
    num_vars = rng.randint(3, 5)
    cnf = random_cnf(rng, num_vars, rng.randint(2, num_vars + 2))
    circuit = compile_cnf(cnf, smooth=True)
    weights = random_weights(rng, num_vars)
    z = brute_wmc(cnf, weights)
    if z == 0:
        pytest.skip("unsatisfiable instance")

    tc = TorchCircuit(circuit, semiring="logprob")
    w = torch.tensor([math.log(x) for x in weights], dtype=torch.float64)
    marg = tc.marginals(w)[0]

    from helpers import assignment_weight

    for var in range(1, num_vars + 1):
        for value in (True, False):
            mass = sum(
                assignment_weight(m, weights)
                for m in cnf.models()
                if m[var] == value
            )
            idx = lit_index(var if value else -var)
            assert marg[idx].item() == pytest.approx(mass / z, rel=1e-8)


@pytest.mark.parametrize("seed", range(10))
def test_torch_mpe_matches_cpu(seed):
    rng = random.Random(500 + seed)
    num_vars = rng.randint(3, 7)
    cnf = random_cnf(rng, num_vars, rng.randint(2, 2 * num_vars))
    circuit = compile_cnf(cnf, smooth=True)
    costs = random_weights(rng, num_vars)
    expected_cost, _ = mpe(circuit, costs)

    tc = TorchCircuit(circuit, semiring="neglog")
    got_costs, assignments = tc.mpe(torch.tensor(costs, dtype=torch.float64))
    if expected_cost == math.inf:
        assert not torch.isfinite(got_costs[0])
        assert assignments[0] is None
    else:
        assert got_costs[0].item() == pytest.approx(expected_cost, rel=1e-9)
        assert cnf.satisfied_by(assignments[0])


def test_evidence_conditioning():
    cnf = CNF(num_vars=2, clauses=[(1, 2)])
    circuit = compile_cnf(cnf, smooth=True)
    tc = TorchCircuit(circuit, semiring="logprob")
    w = torch.zeros(1, 4, dtype=torch.float64)  # weight 1 per literal
    cond = tc.condition(w, {1: False})
    # models: {~1, 2} only
    assert tc(cond)[0].item() == pytest.approx(0.0)
