"""One compiled circuit: estimation, planning, planning-from-sensors."""

import math

import pytest

from modenexus.formula import iff
from modenexus.planning import Planner


def siderostat(horizon=2):
    """The MEXEC paper's example, now with its observable: valid=True
    iff the siderostat is Tracking; priors favor Idling slightly less
    than the paper's costs but preserve the ordering (Tracking cost 20,
    Idling cost 5 -> Idling more likely a priori)."""
    p = Planner()
    p.mode("sw", ("Tracking", "Idling"),
           priors=(math.exp(-20), math.exp(-5)))
    p.command("cmd", ("idle", "track", "none"))
    p.observable("valid")
    p.behavior(lambda v: iff(v["valid"] == True, v["sw"] == "Tracking"))  # noqa: E712
    p.transition("sw", "Tracking", "Idling", command=("cmd", "idle"))
    p.transition("sw", "Idling", "Tracking", command=("cmd", "track"))
    return p.compile(horizon)


def test_estimation_from_observations():
    cp = siderostat()
    # The paper's Figure 4 scenario: valid sensed true twice with no
    # commanded change -> Tracking both steps despite Idling's better
    # prior (the observation overrides the prior, cost 20).
    cost, traj = cp.estimate(
        [{"valid": True}, {"valid": True}],
        commands=[{"cmd": "none"}],
    )
    assert [s["sw"] for s in traj] == ["Tracking", "Tracking"]
    # Priors are normalized, so the trajectory cost is the normalized
    # initial neg-log: 20 - (-log(e^-20 + e^-5)) = 15 up to ~3e-7.
    assert cost == pytest.approx(15.0, abs=1e-5)
    # No observations at all: prior wins.
    _, traj2 = cp.estimate([{}, {}])
    assert traj2[0]["sw"] == "Idling"


def test_estimation_detects_transition():
    cp = siderostat()
    cost, traj = cp.estimate(
        [{"valid": True}, {"valid": False}], commands=[{"cmd": "idle"}]
    )
    assert [s["sw"] for s in traj] == ["Tracking", "Idling"]


def test_estimation_inconsistent_returns_none():
    cp = siderostat()
    # valid flips with no command issued and no transition possible.
    assert cp.estimate(
        [{"valid": True}, {"valid": False}], commands=[{"cmd": "none"}]
    ) is None


def test_plan_from_sensors_without_stating_current_mode():
    cp = siderostat()
    # We never say what mode the siderostat is in -- only what the
    # sensor reads at step 0.  valid=False => Idling => plan must track.
    cost, steps = cp.plan(
        target={"sw": "Tracking"},
        observations=[{"valid": False}],
    )
    assert steps[0]["cmd"] == "track"


def test_plan_and_estimate_share_one_circuit():
    cp = siderostat()
    circuit_id = id(cp.system.circuit)
    cp.estimate([{"valid": True}])
    cp.plan(current={"sw": "Tracking"}, target={"sw": "Idling"})
    assert id(cp.system.circuit) == circuit_id  # same artifact, no recompiles


def test_observation_bounds_checked():
    cp = siderostat(horizon=1)
    with pytest.raises(ValueError):
        cp.estimate([{}, {}, {}])
