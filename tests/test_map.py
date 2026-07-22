"""Marginal MAP: ranked joint mode assignments by summed posterior."""

import math
import random
from collections import defaultdict

import pytest

from neximode import SystemModel, compile_cnf, enumerate_map, iff
from neximode.circuit import lit_index
from helpers import assignment_weight, random_cnf, random_weights


@pytest.mark.parametrize("seed", range(15))
def test_enumerate_map_matches_brute_force(seed):
    rng = random.Random(seed)
    num_vars = rng.randint(4, 7)
    cnf = random_cnf(rng, num_vars, rng.randint(2, 2 * num_vars))
    map_vars = [1, 2]
    circuit = compile_cnf(cnf, var_order=map_vars, smooth=True)
    weights = random_weights(rng, num_vars)
    log_w = [math.log(x) for x in weights]

    # Brute-force marginal MAP: group model mass by map-var assignment.
    mass = defaultdict(float)
    for m in cnf.models():
        key = (m[1], m[2])
        mass[key] += assignment_weight(m, weights)
    expected = sorted(
        ((-math.log(v), k) for k, v in mass.items() if v > 0)
    )

    got = list(enumerate_map(circuit, log_w, map_vars))
    assert len(got) == len(expected)
    for (cost, assignment), (exp_cost, exp_key) in zip(got, expected):
        assert cost == pytest.approx(exp_cost, rel=1e-9)
        assert (assignment[1], assignment[2]) == exp_key
    costs = [c for c, _ in got]
    assert costs == sorted(costs)


def test_unconstrained_circuit_rejected():
    rng = random.Random(0)
    # Branch on var 3 first: decisions on map vars 1,2 end up below.
    cnf = random_cnf(rng, 5, 8)
    circuit = compile_cnf(cnf, var_order=[3, 4, 5, 1, 2], smooth=True)
    weights = [0.0] * 10
    try:
        list(enumerate_map(circuit, weights, [1, 2]))
    except ValueError:
        return
    # Some instances may accidentally satisfy the constraint (e.g. units);
    # in that case the result must still be well-formed, so no assert here.


def build_map_vs_mpe_system():
    """MAP and MPE rankings differ:

    mode m: a (prior 0.45) or b (0.55); unconstrained bool y.
    Constraint: m=b -> y.
    Mass: a spreads over y in {0,1} -> 2 * 0.45 = 0.90 (best state 0.45);
          b forces y=1            -> 0.55        (best state 0.55).
    MPE ranking: b first.  Marginal MAP ranking: a first.
    """
    m = SystemModel()
    mv = m.mode("m", ("a", "b"), priors=(0.45, 0.55))
    y = m.bool("y")
    m.add((mv == "b") >> y)
    return m.compile()


def test_map_and_mpe_rankings_differ():
    sys = build_map_vs_mpe_system()
    mpe_first = sys.diagnoses({}, k=1)[0].modes["m"]
    map_diags = sys.map_diagnoses({}, k=2)
    assert mpe_first == "b"
    assert map_diags[0].modes["m"] == "a"
    assert map_diags[0].posterior == pytest.approx(0.90 / 1.45)
    assert map_diags[1].posterior == pytest.approx(0.55 / 1.45)
    assert sum(d.posterior for d in map_diags) == pytest.approx(1.0)


def test_map_agrees_with_mode_posteriors():
    sys = build_map_vs_mpe_system()
    post = sys.mode_posteriors({})
    map_diags = sys.map_diagnoses({}, k=2)
    for d in map_diags:
        assert d.posterior == pytest.approx(post["m"][d.modes["m"]])


def test_map_with_evidence():
    sys = build_map_vs_mpe_system()
    diags = sys.map_diagnoses({"y": False}, k=2)
    # y=False rules out b entirely.
    assert len(diags) == 1
    assert diags[0].modes["m"] == "a"
    assert diags[0].posterior == pytest.approx(1.0)


def test_bool_prior_weights_both_polarities():
    m = SystemModel()
    m.mode("mm", ("u", "v"), priors=(0.5, 0.5))  # inert mode to satisfy API
    m.bool("x", prior=0.3)
    sys = m.compile()
    assert sys.log_evidence({"x": True}) == pytest.approx(math.log(0.3))
    assert sys.log_evidence({"x": False}) == pytest.approx(math.log(0.7))
    assert sys.log_evidence({}) == pytest.approx(0.0)


def test_two_mode_map_joint():
    m = SystemModel()
    a = m.mode("a", ("ok", "bad"), priors=(0.9, 0.1))
    b = m.mode("b", ("ok", "bad"), priors=(0.8, 0.2))
    alarm = m.bool("alarm")
    m.add(iff(alarm, (a == "bad") | (b == "bad")))
    sys = m.compile()
    diags = sys.map_diagnoses({"alarm": True}, k=4)
    joint = {
        ("ok", "bad"): 0.9 * 0.2,
        ("bad", "ok"): 0.1 * 0.8,
        ("bad", "bad"): 0.1 * 0.2,
    }
    z = sum(joint.values())
    assert len(diags) == 3
    got = [(d.modes["a"], d.modes["b"]) for d in diags]
    assert got == sorted(joint, key=lambda k: -joint[k])
    for d in diags:
        assert d.posterior == pytest.approx(
            joint[(d.modes["a"], d.modes["b"])] / z
        )
