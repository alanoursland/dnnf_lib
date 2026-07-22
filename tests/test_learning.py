"""Sampling and EM prior learning from partially observed telemetry."""

import math
import random
from collections import Counter

import pytest

from neximode import SystemModel, fd


def build_valve(prior_bad=0.3, fp=0.1, fn=0.1):
    m = SystemModel()
    v = m.mode("v", ("ok", "bad"), priors=(1 - prior_bad, prior_bad))
    m.sensor("alarm", v == "bad", false_positive=fp, false_negative=fn)
    return m.compile()


# ----------------------------------------------------------------------
# Sampling
# ----------------------------------------------------------------------
def test_sample_matches_distribution():
    sys = build_valve(prior_bad=0.3)
    rng = random.Random(0)
    n = 20000
    counts = Counter()
    for _ in range(n):
        state = sys.sample_state(rng)
        counts[(state["v"], state["alarm"])] += 1
    # P(alarm=True) = 0.3*0.9 + 0.7*0.1 = 0.34
    p_alarm = sum(c for (v, a), c in counts.items() if a) / n
    assert p_alarm == pytest.approx(0.34, abs=0.02)
    p_bad = sum(c for (v, a), c in counts.items() if v == "bad") / n
    assert p_bad == pytest.approx(0.3, abs=0.02)


def test_sample_respects_constraints():
    m = SystemModel()
    a = m.mode("a", ("x", "y"), priors=(0.5, 0.5))
    b = m.mode("b", ("x", "y"), priors=(0.5, 0.5))
    m.add((a == "x") >> (b == "x"))
    sys = m.compile()
    rng = random.Random(1)
    for _ in range(200):
        s = sys.sample_state(rng)
        assert not (s["a"] == "x" and s["b"] == "y")


def test_fd_sample_zero_mass():
    cnf = fd.FDCnf()
    v = cnf.spec.add_var(2)
    cnf.add_clause([(v, {0})])
    cnf.add_clause([(v, {1})])
    circuit = fd.compile_fd(cnf, smooth=True)
    assert fd.sample(circuit, [0.0, 0.0], random.Random(0)) is None


# ----------------------------------------------------------------------
# EM prior learning
# ----------------------------------------------------------------------
def test_em_recovers_prior_from_alarms_only():
    """Generate telemetry from a true failure rate, observe ONLY the
    noisy alarm, and recover the rate by EM.  Identifiable because
    P(alarm) = p*(1-fn) + (1-p)*fp is invertible in p."""
    true = build_valve(prior_bad=0.3)
    rng = random.Random(42)
    observations = [
        {"alarm": true.sample_state(rng)["alarm"]} for _ in range(4000)
    ]

    learner = build_valve(prior_bad=0.5)  # start from the wrong prior
    history = learner.fit_priors(observations, names=["v"])

    # Log-likelihood is non-decreasing (EM guarantee).
    for a, b in zip(history, history[1:]):
        assert b >= a - 1e-9

    fitted_p_bad = learner._weights[
        learner.circuit.spec.mvlit(learner.vars["v"].fd_var, 1)
    ]
    # Compare against the analytic MLE from the sampled alarm frequency,
    # not the true 0.3, to keep the test exact under sampling noise.
    freq = sum(1 for o in observations if o["alarm"]) / len(observations)
    mle = (freq - 0.1) / 0.8
    assert fitted_p_bad == pytest.approx(mle, abs=1e-3)
    assert fitted_p_bad == pytest.approx(0.3, abs=0.05)


def test_em_two_component_recovery():
    def build(p1, p2):
        m = SystemModel()
        v1 = m.mode("v1", ("ok", "bad"), priors=(1 - p1, p1))
        v2 = m.mode("v2", ("ok", "bad"), priors=(1 - p2, p2))
        m.sensor("a1", v1 == "bad", false_positive=0.05)
        m.sensor("a2", v2 == "bad", false_positive=0.05)
        return m.compile()

    rng = random.Random(7)
    true = build(0.2, 0.05)
    obs = []
    for _ in range(3000):
        s = true.sample_state(rng)
        obs.append({"a1": s["a1"], "a2": s["a2"]})

    learner = build(0.5, 0.5)
    learner.fit_priors(obs)
    spec = learner.circuit.spec
    p1 = learner._weights[spec.mvlit(learner.vars["v1"].fd_var, 1)]
    p2 = learner._weights[spec.mvlit(learner.vars["v2"].fd_var, 1)]
    assert p1 == pytest.approx(0.2, abs=0.04)
    assert p2 == pytest.approx(0.05, abs=0.03)


def test_posteriors_of_hidden_vars():
    sys = build_valve(prior_bad=0.3, fp=0.1, fn=0.1)
    post = sys.posteriors({"alarm": True}, names=["v", "_alarm_fp"])
    # P(v=bad | alarm) = 0.27 / 0.34
    assert post["v"]["bad"] == pytest.approx(0.27 / 0.34)
    # The fp glitch fires with alarm when either it caused the alarm
    # (v=ok: 0.7*0.1) or it fired irrelevantly alongside a real fault
    # (v=bad, no fn glitch: 0.3*0.9*0.1).
    assert post["_alarm_fp"][True] == pytest.approx(
        (0.7 * 0.1 + 0.3 * 0.9 * 0.1) / 0.34
    )


# ----------------------------------------------------------------------
# Command-conditioned transitions
# ----------------------------------------------------------------------
def test_step_transition_override():
    from neximode import ModeTracker
    from neximode.formula import iff

    m = SystemModel()
    v = m.mode("v", ("ok", "bad"), priors=(1.0, 0.0))
    m.sensor("alarm", v == "bad")
    sys = m.compile()

    idle = {"v": {"ok": {"ok": 1.0, "bad": 0.0}, "bad": {"bad": 1.0, "ok": 0.0}}}
    stress = {"v": {"ok": {"ok": 0.5, "bad": 0.5}, "bad": {"bad": 1.0, "ok": 0.0}}}

    tracker = ModeTracker(sys, idle, beam=2)
    tracker.step({})  # idle step: cannot fail
    assert tracker.marginals()["v"]["bad"] == pytest.approx(0.0)
    tracker.step({}, transitions=stress)  # commanded step: may fail
    assert tracker.marginals()["v"]["bad"] == pytest.approx(0.5)
    tracker.step({})  # default transitions restored automatically
    assert tracker.marginals()["v"]["bad"] == pytest.approx(0.5)
