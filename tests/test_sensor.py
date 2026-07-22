"""Noisy sensor primitive: P(s|expr) = 1-fn, P(s|~expr) = fp."""

import math

import pytest

from neximode import SystemModel

P_BAD = 0.01
FP = 0.1
FN = 0.2


def build():
    m = SystemModel()
    mv = m.mode("m", ("ok", "bad"), priors=(1 - P_BAD, P_BAD))
    m.sensor("alarm", mv == "bad", false_positive=FP, false_negative=FN)
    return m.compile()


def test_evidence_probability():
    sys = build()
    p_alarm = (1 - P_BAD) * FP + P_BAD * (1 - FN)
    assert sys.log_evidence({"alarm": True}) == pytest.approx(
        math.log(p_alarm)
    )
    assert sys.log_evidence({"alarm": False}) == pytest.approx(
        math.log(1 - p_alarm)
    )


def test_posterior_with_noise():
    sys = build()
    p_alarm = (1 - P_BAD) * FP + P_BAD * (1 - FN)
    post = sys.mode_posteriors({"alarm": True})
    assert post["m"]["bad"] == pytest.approx(P_BAD * (1 - FN) / p_alarm)


def test_hidden_noise_vars_not_in_state():
    sys = build()
    d = sys.map_diagnoses({"alarm": True}, k=1)[0]
    assert all(not k.startswith("_") for k in d.state)


def test_noiseless_sensor_is_exact():
    m = SystemModel()
    mv = m.mode("m", ("ok", "bad"), priors=(0.9, 0.1))
    m.sensor("alarm", mv == "bad")
    sys = m.compile()
    post = sys.mode_posteriors({"alarm": True})
    assert post["m"]["bad"] == pytest.approx(1.0)


def test_fn_only_and_fp_only():
    for fp, fn in ((0.0, 0.2), (0.1, 0.0)):
        m = SystemModel()
        mv = m.mode("m", ("ok", "bad"), priors=(0.5, 0.5))
        m.sensor("alarm", mv == "bad", false_positive=fp, false_negative=fn)
        sys = m.compile()
        p_alarm = 0.5 * fp + 0.5 * (1 - fn)
        assert sys.log_evidence({"alarm": True}) == pytest.approx(
            math.log(p_alarm)
        )
