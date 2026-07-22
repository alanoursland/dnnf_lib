"""Finite-domain core cross-validated against brute force."""

import math
import random
from collections import defaultdict

import pytest

from neximode import fd


def random_fd_cnf(rng, num_vars, num_clauses):
    cnf = fd.FDCnf()
    sizes = [rng.randint(2, 4) for _ in range(num_vars)]
    for s in sizes:
        cnf.spec.add_var(s)
    for _ in range(num_clauses):
        width = rng.choice((1, 2, 2, 3))
        cvars = rng.sample(range(num_vars), min(width, num_vars))
        lits = []
        for v in cvars:
            k = rng.randint(1, sizes[v] - 1)
            lits.append((v, rng.sample(range(sizes[v]), k)))
        cnf.add_clause(lits)
    return cnf


def random_fd_weights(rng, spec):
    return [rng.uniform(0.1, 2.0) for _ in range(spec.total)]


def model_weight(assignment, spec, weights):
    w = 1.0
    for var, val in enumerate(assignment):
        w *= weights[spec.mvlit(var, val)]
    return w


@pytest.mark.parametrize("seed", range(25))
def test_fd_compile_count_wmc_mpe(seed):
    rng = random.Random(seed)
    cnf = random_fd_cnf(rng, rng.randint(3, 6), rng.randint(2, 10))
    circuit = fd.compile_fd(cnf, smooth=True)
    assert circuit.is_decomposable()
    assert circuit.is_deterministic()
    assert circuit.is_smooth()

    models = list(cnf.models())
    assert fd.model_count(circuit) == len(models)

    weights = random_fd_weights(rng, cnf.spec)
    expected_wmc = sum(model_weight(m, cnf.spec, weights) for m in models)
    assert fd.wmc(circuit, weights) == pytest.approx(expected_wmc, rel=1e-9)
    log_w = [math.log(x) for x in weights]
    if expected_wmc > 0:
        assert fd.log_wmc(circuit, log_w) == pytest.approx(
            math.log(expected_wmc), rel=1e-9
        )

    costs = random_fd_weights(rng, cnf.spec)
    if models:
        expected_best = min(
            sum(costs[cnf.spec.mvlit(v, val)] for v, val in enumerate(m))
            for m in models
        )
        got_cost, got_assign = fd.mpe(circuit, costs)
        assert got_cost == pytest.approx(expected_best, rel=1e-9)
        assert cnf.satisfied_by([got_assign[v] for v in range(cnf.spec.num_vars)])
    else:
        assert fd.mpe(circuit, costs) == (math.inf, None)


@pytest.mark.parametrize("seed", range(15))
def test_fd_enumerate_models_ordered(seed):
    rng = random.Random(100 + seed)
    cnf = random_fd_cnf(rng, rng.randint(3, 5), rng.randint(2, 8))
    circuit = fd.compile_fd(cnf, smooth=True)
    costs = random_fd_weights(rng, cnf.spec)
    expected = sorted(
        sum(costs[cnf.spec.mvlit(v, val)] for v, val in enumerate(m))
        for m in cnf.models()
    )
    got = list(fd.enumerate_models(circuit, costs))
    assert len(got) == len(expected)
    for (cost, assignment), exp in zip(got, expected):
        assert cost == pytest.approx(exp, rel=1e-9)
        assert cnf.satisfied_by(
            [assignment[v] for v in range(cnf.spec.num_vars)]
        )
    keys = {tuple(sorted(a.items())) for _, a in got}
    assert len(keys) == len(got)


@pytest.mark.parametrize("seed", range(15))
def test_fd_enumerate_map_matches_brute_force(seed):
    rng = random.Random(200 + seed)
    cnf = random_fd_cnf(rng, rng.randint(3, 6), rng.randint(2, 8))
    map_vars = [0, 1]
    circuit = fd.compile_fd(cnf, var_order=map_vars, smooth=True)
    weights = random_fd_weights(rng, cnf.spec)
    log_w = [math.log(x) for x in weights]

    mass = defaultdict(float)
    for m in cnf.models():
        mass[(m[0], m[1])] += model_weight(m, cnf.spec, weights)
    expected = sorted((-math.log(v), k) for k, v in mass.items() if v > 0)

    got = list(fd.enumerate_map(circuit, log_w, map_vars))
    assert len(got) == len(expected)
    for (cost, assignment), (exp_cost, exp_key) in zip(got, expected):
        assert cost == pytest.approx(exp_cost, rel=1e-9)
        assert (assignment[0], assignment[1]) == exp_key


def test_fd_encode_formula():
    """iff/xor over FD atoms round-trip through Tseitin correctly."""
    from neximode.formula import iff

    cnf = fd.FDCnf()
    x = cnf.spec.add_var(3)  # domain {0,1,2}
    b = cnf.spec.add_var(2)
    atom_x2 = fd.FDAtom(x, frozenset((2,)))
    atom_b = fd.FDAtom(b, frozenset((1,)))
    fd.encode([iff(atom_b, atom_x2)], cnf)
    circuit = fd.compile_fd(cnf, smooth=True)
    # Models over (x, b, aux...): b=1 iff x=2 -> 3 combinations of (x,b),
    # aux vars determined; count projected to x,b should be 3.
    count = fd.model_count(circuit)
    assert count == 3


def test_fd_unsat_and_trivial():
    cnf = fd.FDCnf()
    v = cnf.spec.add_var(3)
    cnf.add_clause([(v, {0})])
    cnf.add_clause([(v, {1, 2})])
    circuit = fd.compile_fd(cnf, smooth=True)
    assert not fd.is_satisfiable(circuit)
    assert fd.model_count(circuit) == 0

    cnf2 = fd.FDCnf()
    cnf2.spec.add_var(4)
    circuit2 = fd.compile_fd(cnf2, smooth=True)
    assert fd.model_count(circuit2) == 4


def test_fd_set_literal_thresholds():
    """A 'quantized' style constraint: level in {0,1} (below threshold)."""
    cnf = fd.FDCnf()
    level = cnf.spec.add_var(5)
    cnf.add_clause([(level, {0, 1})])
    circuit = fd.compile_fd(cnf, smooth=True)
    assert fd.model_count(circuit) == 2
