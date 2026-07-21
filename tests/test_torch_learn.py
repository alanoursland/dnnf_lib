"""Gradient prior learning and correlated transitions."""

import math
import random

import pytest

torch = pytest.importorskip("torch")

from dnnf import ModeTracker, SystemModel


def build_valve(prior_bad=0.3, fp=0.1, fn=0.1):
    m = SystemModel()
    v = m.mode("v", ("ok", "bad"), priors=(1 - prior_bad, prior_bad))
    m.sensor("alarm", v == "bad", false_positive=fp, false_negative=fn)
    return m.compile()


def test_sgd_recovers_prior_and_matches_em():
    true = build_valve(prior_bad=0.3)
    rng = random.Random(11)
    observations = [
        {"alarm": true.sample_state(rng)["alarm"]} for _ in range(3000)
    ]
    freq = sum(1 for o in observations if o["alarm"]) / len(observations)
    mle = (freq - 0.1) / 0.8

    sgd = build_valve(prior_bad=0.5)
    history = sgd.fit_priors_torch(observations, epochs=400, lr=0.1)
    assert history[-1] > history[0]  # likelihood improved

    from dnnf.torch_learn import PriorLearner

    spec = sgd.circuit.spec
    p_bad = sgd._weights[spec.mvlit(sgd.vars["v"].fd_var, 1)]
    assert p_bad == pytest.approx(mle, abs=0.01)

    em = build_valve(prior_bad=0.5)
    em.fit_priors(observations, names=["v"])
    p_bad_em = em._weights[spec.mvlit(em.vars["v"].fd_var, 1)]
    assert p_bad == pytest.approx(p_bad_em, abs=0.01)


def test_written_back_priors_affect_queries():
    true = build_valve(prior_bad=0.25)
    rng = random.Random(3)
    obs = [{"alarm": true.sample_state(rng)["alarm"]} for _ in range(2500)]
    learner = build_valve(prior_bad=0.5)
    learner.fit_priors_torch(obs, epochs=300, lr=0.1)
    # Post-training posterior should be close to the true model's.
    p_true = true.posteriors({"alarm": True})["v"]["bad"]
    p_fit = learner.posteriors({"alarm": True})["v"]["bad"]
    assert p_fit == pytest.approx(p_true, abs=0.05)


def test_learner_probs_normalized():
    from dnnf.torch_learn import PriorLearner

    sys = build_valve()
    learner = PriorLearner(sys, names=["v"])
    dist = learner.probs()["v"]
    assert sum(dist.values()) == pytest.approx(1.0)
    assert dist["bad"] == pytest.approx(0.3)  # initialized from current


# ----------------------------------------------------------------------
# Correlated (previous-state-dependent) transitions
# ----------------------------------------------------------------------
def exact_two_component_forward(steps, base_fail, coupled_fail):
    """Exact filter over joint (A, B), both absorbing; B's failure rate
    depends on whether A was failed at the previous step."""
    belief = {("ok", "ok"): 1.0, ("ok", "bad"): 0.0,
              ("bad", "ok"): 0.0, ("bad", "bad"): 0.0}
    for _ in range(steps):
        new = {k: 0.0 for k in belief}
        for (a, b), p in belief.items():
            if p == 0.0:
                continue
            pa_fail = 0.0 if a == "bad" else base_fail
            pb_fail = 0.0 if b == "bad" else (
                coupled_fail if a == "bad" else base_fail
            )
            for a2, pa in (("bad", pa_fail if a == "ok" else 1.0),
                           ("ok", 1 - pa_fail if a == "ok" else 0.0)):
                for b2, pb in (("bad", pb_fail if b == "ok" else 1.0),
                               ("ok", 1 - pb_fail if b == "ok" else 0.0)):
                    new[(a2, b2)] += p * pa * pb
        belief = new
    return belief


def test_transition_fn_correlated_failure():
    BASE, COUPLED = 0.1, 0.4

    m = SystemModel()
    m.mode("A", ("ok", "bad"), priors=(1.0, 0.0))
    m.mode("B", ("ok", "bad"), priors=(1.0, 0.0))
    sys = m.compile()

    def dynamics(prev):
        pa = 0.0 if prev["A"] == "bad" else BASE
        pb = 0.0 if prev["B"] == "bad" else (
            COUPLED if prev["A"] == "bad" else BASE
        )
        return {
            "A": {"ok": (0.0 if prev["A"] == "bad" else 1 - pa),
                  "bad": (1.0 if prev["A"] == "bad" else pa)},
            "B": {"ok": (0.0 if prev["B"] == "bad" else 1 - pb),
                  "bad": (1.0 if prev["B"] == "bad" else pb)},
        }

    tracker = ModeTracker(sys, transition_fn=dynamics, beam=4)
    steps = 3
    for _ in range(steps):
        tracker.step({})
    expected = exact_two_component_forward(steps, BASE, COUPLED)
    got = {tuple(modes[k] for k in ("A", "B")): p
           for modes, p in tracker.belief()}
    for key, p in expected.items():
        assert got.get(key, 0.0) == pytest.approx(p, rel=1e-9, abs=1e-12)
