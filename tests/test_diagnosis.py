import math

import pytest

from modenexus import SystemModel, iff


def build_two_valve_system():
    """Two valves feeding a line; flow observed at the outlet.

    Each valve is 'ok' (passes flow when commanded open) or 'stuck_closed'.
    Both valves are commanded open; outlet flow requires both to pass.
    """
    m = SystemModel()
    v1 = m.mode("valve1", ("ok", "stuck_closed"), priors=(0.99, 0.01))
    v2 = m.mode("valve2", ("ok", "stuck_closed"), priors=(0.95, 0.05))
    flow = m.bool("flow")
    m.add(iff(flow, (v1 == "ok") & (v2 == "ok")))
    return m.compile()


def test_nominal_observation():
    sys = build_two_valve_system()
    diags = sys.diagnoses({"flow": True}, k=3)
    assert diags[0].modes == {"valve1": "ok", "valve2": "ok"}
    assert diags[0].posterior == pytest.approx(1.0)
    assert len(diags) == 1  # flow=True entails both ok


def test_fault_observation_ranking():
    sys = build_two_valve_system()
    diags = sys.diagnoses({"flow": False}, k=4)
    modes = [d.modes for d in diags]
    # Most probable single fault first: valve2 (prior 0.05) over valve1 (0.01),
    # double fault last.
    assert modes[0] == {"valve1": "ok", "valve2": "stuck_closed"}
    assert modes[1] == {"valve1": "stuck_closed", "valve2": "ok"}
    assert modes[2] == {"valve1": "stuck_closed", "valve2": "stuck_closed"}
    assert len(modes) == 3
    # Posteriors normalize over the three consistent diagnoses.
    total_p = 0.99 * 0.05 + 0.01 * 0.95 + 0.01 * 0.05
    assert diags[0].posterior == pytest.approx(0.99 * 0.05 / total_p)
    assert diags[1].posterior == pytest.approx(0.01 * 0.95 / total_p)
    assert diags[2].posterior == pytest.approx(0.01 * 0.05 / total_p)
    # Ranked most-to-least probable.
    posts = [d.posterior for d in diags]
    assert posts == sorted(posts, reverse=True)


def test_mode_posteriors():
    sys = build_two_valve_system()
    post = sys.mode_posteriors({"flow": False})
    total_p = 0.99 * 0.05 + 0.01 * 0.95 + 0.01 * 0.05
    expected_v1_ok = 0.99 * 0.05 / total_p
    assert post["valve1"]["ok"] == pytest.approx(expected_v1_ok)
    assert post["valve1"]["stuck_closed"] == pytest.approx(1 - expected_v1_ok)
    for name in ("valve1", "valve2"):
        assert sum(post[name].values()) == pytest.approx(1.0)


def test_observed_variable_posterior_is_a_point_mass_and_voi_is_bounded():
    model = SystemModel()
    mode = model.mode("mode", ("ok", "bad"), priors=(0.9, 0.1))
    model.sensor(
        "alarm",
        mode == "bad",
        false_positive=0.05,
        false_negative=0.05,
    )
    system = model.compile()

    for observed in ("ok", "bad"):
        posterior = system.mode_posteriors({"mode": observed})["mode"]
        assert posterior == pytest.approx(
            {
                "ok": 1.0 if observed == "ok" else 0.0,
                "bad": 1.0 if observed == "bad" else 0.0,
            }
        )
        for _, value in system.value_of_information(
            {"mode": observed}
        ):
            assert value == pytest.approx(0.0, abs=1e-12)


def test_log_evidence():
    sys = build_two_valve_system()
    p_flow = 0.99 * 0.95
    assert sys.log_evidence({"flow": True}) == pytest.approx(math.log(p_flow))
    assert sys.log_evidence({"flow": False}) == pytest.approx(
        math.log(1 - p_flow)
    )


def test_inconsistent_evidence():
    sys = build_two_valve_system()
    diags = sys.diagnoses({"flow": True, "valve1": "stuck_closed"}, k=3)
    assert diags == []


def test_diagnosis_state_includes_observables():
    sys = build_two_valve_system()
    diags = sys.diagnoses({"flow": False}, k=1)
    assert diags[0].state["flow"] is False


@pytest.mark.parametrize(
    "priors",
    [
        (1.0, -0.1),
        (0.0, 0.0),
        (1.0, math.inf),
        (1.0, math.nan),
    ],
)
def test_mode_priors_reject_invalid_relative_weights(priors):
    m = SystemModel()
    with pytest.raises(ValueError, match="component.*priors"):
        m.mode("component", ("ok", "bad"), priors)


def test_mode_priors_are_documented_relative_weights():
    m = SystemModel()
    m.mode("component", ("ok", "bad"), (0.9, 0.9))
    sys = m.compile()
    assert sys.mode_posteriors({})["component"] == pytest.approx(
        {"ok": 0.5, "bad": 0.5}
    )


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("false_positive", -0.1),
        ("false_positive", 1.1),
        ("false_positive", math.inf),
        ("false_negative", -0.1),
        ("false_negative", 1.1),
        ("false_negative", math.nan),
    ],
)
def test_sensor_rates_reject_invalid_probabilities(parameter, value):
    m = SystemModel()
    condition = m.bool("condition")
    kwargs = {parameter: value}
    with pytest.raises(ValueError, match=f"alarm {parameter}"):
        m.sensor("alarm", condition, **kwargs)


@pytest.mark.parametrize("rate", [0.0, 1.0])
def test_sensor_rate_boundaries_are_valid(rate):
    m = SystemModel()
    condition = m.bool("condition")
    m.sensor(
        "alarm",
        condition,
        false_positive=rate,
        false_negative=rate,
    )
