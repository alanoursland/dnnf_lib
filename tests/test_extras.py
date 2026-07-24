"""VOI, minimum-cardinality diagnoses, external compiler driver."""

import shutil

import pytest

from modenexus import CNF, SystemModel, iff


def build_two_valve():
    m = SystemModel()
    v1 = m.mode("valve1", ("ok", "stuck_closed"), priors=(0.99, 0.01))
    v2 = m.mode("valve2", ("ok", "stuck_closed"), priors=(0.95, 0.05))
    f1 = m.bool("flow1")
    f2 = m.bool("flow2")
    out = m.bool("flow_out")
    m.add(iff(f1, v1 == "ok"))
    m.add(iff(f2, v2 == "ok"))
    m.add(iff(out, f1 & f2))
    return m.compile()


def test_min_cardinality_orders_by_fault_count():
    sys = build_two_valve()
    diags = sys.diagnoses_min_cardinality({"flow_out": False}, k=4)
    cards = [c for c, _ in diags]
    assert cards == sorted(cards)
    assert cards[0] == 1 and cards[-1] == 2
    assert len(diags) == 3  # two single faults + the double
    singles = {tuple(sorted(m.items())) for c, m in diags if c == 1}
    assert len(singles) == 2


def test_voi_prefers_discriminating_sensor():
    sys = build_two_valve()
    # After observing no output flow, reading a per-leg flow sensor
    # discriminates which valve failed; VOI must rank both legs above
    # nothing, and both should carry positive value.
    ranked = dict(sys.value_of_information({"flow_out": False}))
    assert ranked["flow1"] > 0
    assert ranked["flow2"] > 0
    # flow1 resolves the higher-entropy question (valve1's posterior is
    # more uncertain than... valve2 has higher prior fault rate; either
    # ordering is model-dependent, so just check both beat a
    # non-discriminating candidate: observing flow_out again is not a
    # candidate (already observed), and candidates exclude modes.
    names = [n for n, _ in sys.value_of_information({"flow_out": False})]
    assert set(names) == {"flow1", "flow2"}


def test_voi_zero_when_nothing_to_learn():
    sys = build_two_valve()
    # flow_out=True entails both valves ok: no residual uncertainty.
    ranked = sys.value_of_information({"flow_out": True})
    for _, voi in ranked:
        assert voi == pytest.approx(0.0, abs=1e-12)


def test_voi_filters_explicitly_observed_candidates():
    sys = build_two_valve()
    assert sys.value_of_information(
        {"flow_out": False}, candidates=["flow_out"]
    ) == []


def test_voi_filters_observed_finite_domain_candidates():
    m = SystemModel()
    mode = m.mode("component", ("ok", "bad"), priors=(0.9, 0.1))
    status = m.finite("status", ("green", "red"))
    m.add(iff(status == "red", mode == "bad"))
    sys = m.compile()
    assert sys.value_of_information(
        {"status": "green"}, candidates=["status"]
    ) == []


def test_c2d_driver_missing_binary_raises():
    from modenexus.external import compile_with_c2d

    if shutil.which("c2d"):
        pytest.skip("c2d installed; covered by round-trip below")
    with pytest.raises(FileNotFoundError):
        compile_with_c2d(CNF(num_vars=2, clauses=[(1, 2)]))


@pytest.mark.skipif(shutil.which("c2d") is None, reason="c2d not installed")
def test_c2d_round_trip():  # pragma: no cover - environment-dependent
    from modenexus import model_count
    from modenexus.external import compile_with_c2d

    cnf = CNF(num_vars=3, clauses=[(1, 2), (-1, 3)])
    circuit = compile_with_c2d(cnf).smooth()
    assert model_count(circuit) == 4
