"""Serialization round-trip and MEXEC-style reconfiguration planning."""

import math

import pytest

from neximode import SystemModel, iff
from neximode.diagnosis import CompiledSystem
from neximode.planning import Planner


def test_save_load_round_trip(tmp_path):
    m = SystemModel()
    v = m.mode("valve", ("ok", "stuck"), priors=(0.95, 0.05))
    level = m.quantized("level", (0.0, 10.0, 50.0), priors=(0.4, 0.6))
    m.sensor("alarm", (v == "stuck") | level.below(10.0),
             false_positive=0.05)
    sys1 = m.compile()
    path = str(tmp_path / "system.json")
    sys1.save(path)
    sys2 = CompiledSystem.load(path)

    # Header states all resource requirements up front (single-
    # allocation loading).
    import json

    header = json.load(open(path))["header"]
    assert header["num_nodes"] == len(sys1.circuit)
    assert header["num_edges"] == sys1.circuit.num_edges
    assert header["num_weight_slots"] == sys1.circuit.spec.total
    assert header["max_children"] >= 2

    for ev in ({}, {"alarm": True}, {"alarm": False, "level": 30.0}):
        assert sys2.log_evidence(ev) == pytest.approx(
            sys1.log_evidence(ev), rel=1e-12
        )
    p1 = sys1.mode_posteriors({"alarm": True})
    p2 = sys2.mode_posteriors({"alarm": True})
    for name in p1:
        for value, prob in p1[name].items():
            assert p2[name][value] == pytest.approx(prob, rel=1e-12)
    d1 = sys1.map_diagnoses({"alarm": True}, k=2)
    d2 = sys2.map_diagnoses({"alarm": True}, k=2)
    assert [d.modes for d in d1] == [d.modes for d in d2]


def siderostat_planner():
    """The MEXEC paper's running example."""
    p = Planner()
    p.mode("sw", ("Tracking", "Idling"))
    p.command("cmd", ("idle", "track", "none"))
    p.transition("sw", "Tracking", "Idling", command=("cmd", "idle"))
    p.transition("sw", "Idling", "Tracking", command=("cmd", "track"))
    return p


def test_one_step_reconfiguration():
    cp = siderostat_planner().compile(horizon=1)
    cost, steps = cp.plan({"sw": "Tracking"}, {"sw": "Idling"})
    assert steps == [{"cmd": "idle"}]
    cost2, steps2 = cp.plan({"sw": "Idling"}, {"sw": "Tracking"})
    assert steps2 == [{"cmd": "track"}]
    # Already at target: noop plan, zero cost.
    cost3, steps3 = cp.plan({"sw": "Idling"}, {"sw": "Idling"})
    assert cost3 == pytest.approx(0.0)


def test_multi_step_chain_and_unreachability():
    p = Planner()
    p.mode("m", ("A", "B", "C"))
    p.command("go", ("ab", "bc", "none"))
    p.transition("m", "A", "B", command=("go", "ab"))
    p.transition("m", "B", "C", command=("go", "bc"))

    assert p.compile(horizon=1).plan({"m": "A"}, {"m": "C"}) is None
    cost, steps = p.compile(horizon=2).plan({"m": "A"}, {"m": "C"})
    assert [s["go"] for s in steps] == ["ab", "bc"]


def test_costs_pick_the_likelier_route():
    p = Planner()
    p.mode("m", ("A", "B"))
    p.command("c", ("risky", "safe", "none"))
    p.transition("m", "A", "B", command=("c", "risky"), cost=2.0)
    p.transition("m", "A", "B", command=("c", "safe"), cost=0.5)
    cost, steps = p.compile(horizon=1).plan({"m": "A"}, {"m": "B"})
    assert steps == [{"c": "safe"}]
    assert cost == pytest.approx(0.5, abs=1e-9)
