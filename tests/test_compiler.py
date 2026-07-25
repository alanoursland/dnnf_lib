import random

import pytest

from modenexus import (
    CNF,
    CircuitBuilder,
    compile_cnf,
    is_satisfiable,
    model_count,
)
from helpers import random_cnf


def test_trivial_true():
    circuit = compile_cnf(CNF(num_vars=2), smooth=True)
    assert is_satisfiable(circuit)
    assert model_count(circuit) == 4


def test_trivial_false():
    cnf = CNF(num_vars=1, clauses=[(1,), (-1,)])
    circuit = compile_cnf(cnf, smooth=True)
    assert not is_satisfiable(circuit)
    assert model_count(circuit) == 0


def test_single_clause():
    cnf = CNF(num_vars=3, clauses=[(1, -2)])
    circuit = compile_cnf(cnf, smooth=True)
    assert model_count(circuit) == 6  # 8 - 2 falsifying


def test_tautological_clause_dropped():
    cnf = CNF(num_vars=2, clauses=[(1, -1), (2,)])
    circuit = compile_cnf(cnf, smooth=True)
    assert model_count(circuit) == 2


@pytest.mark.parametrize("seed", range(30))
def test_random_cnf_model_count_matches_brute_force(seed):
    rng = random.Random(seed)
    num_vars = rng.randint(3, 8)
    cnf = random_cnf(rng, num_vars, rng.randint(2, 3 * num_vars))
    circuit = compile_cnf(cnf, smooth=True)
    assert circuit.is_decomposable()
    assert circuit.is_deterministic()
    assert circuit.is_smooth()
    expected = sum(1 for _ in cnf.models())
    assert model_count(circuit) == expected


@pytest.mark.parametrize("seed", range(5))
def test_static_var_order(seed):
    rng = random.Random(100 + seed)
    cnf = random_cnf(rng, 6, 10)
    order = list(range(1, 7))
    rng.shuffle(order)
    circuit = compile_cnf(cnf, var_order=order, smooth=True)
    expected = sum(1 for _ in cnf.models())
    assert model_count(circuit) == expected


def test_condition():
    cnf = CNF(num_vars=3, clauses=[(1, 2), (-1, 3)])
    circuit = compile_cnf(cnf, smooth=True)
    conditioned = circuit.condition({1: True})
    assert 1 in conditioned.mentioned_vars()
    assert 1 in conditioned.asserted_literals()[conditioned.root]
    # Models with x1=True need x3; x2 remains free.
    assert model_count(conditioned.smooth()) == 2


def test_is_smooth_requires_all_declared_variables():
    builder = CircuitBuilder(3)
    partial = builder.finish(builder.literal(1))
    assert not partial.is_smooth()
    assert partial.smooth().is_smooth()


def test_binary_forest_compiler_scales_with_linear_output():
    sizes = []
    for num_vars in (50, 100, 200):
        cnf = CNF(
            num_vars=num_vars,
            clauses=[
                (-var, var + 1)
                for var in range(1, num_vars)
            ],
        )
        circuit = compile_cnf(cnf, smooth=True)
        assert model_count(circuit) == num_vars + 1
        assert circuit.is_decomposable()
        assert circuit.is_deterministic()
        assert circuit.is_smooth()
        sizes.append(len(circuit))
    assert sizes[1] <= 2 * sizes[0] + 10
    assert sizes[2] <= 2 * sizes[1] + 10


def test_cache_shares_subcircuits():
    # A chain formula where both branches reconverge quickly should produce
    # a DAG smaller than the full decision tree.
    cnf = CNF(num_vars=12)
    for v in range(1, 12):
        cnf.add_clause((-v, v + 1))
    circuit = compile_cnf(cnf)
    assert len(circuit) < 2**12
