"""ModeTracker vs. an exact hand-rolled HMM forward recursion.

System: one component m in {ok, bad}; a glitchy alarm sensor:
alarm is true iff (m == bad) XOR glitch, with glitch ~ Bernoulli(0.1)
resampled independently each step.  So the observation model is
P(alarm=True | ok) = 0.1, P(alarm=True | bad) = 0.9.

With beam covering the full mode space (2 states), the tracker must
reproduce the exact forward algorithm.
"""

import math
from types import SimpleNamespace

import pytest

from modenexus import ModeTracker, SystemModel, iff, xor

P_GLITCH = 0.1
PRIOR_BAD = 0.01
P_FAIL = 0.05  # ok -> bad per step; bad is absorbing

TRANS = {
    "m": {
        "ok": {"ok": 1 - P_FAIL, "bad": P_FAIL},
        "bad": {"bad": 1.0, "ok": 0.0},
    }
}


def build_system():
    m = SystemModel()
    mv = m.mode("m", ("ok", "bad"), priors=(1 - PRIOR_BAD, PRIOR_BAD))
    glitch = m.bool("glitch", prior=P_GLITCH)
    alarm = m.bool("alarm")
    m.add(iff(alarm, xor(mv == "bad", glitch)))
    return m.compile()


def obs_likelihood(mode: str, alarm: bool) -> float:
    p_alarm = (1 - P_GLITCH) if mode == "bad" else P_GLITCH
    return p_alarm if alarm else 1 - p_alarm


def exact_forward(observations):
    """Standard HMM filter over {ok, bad}."""
    belief = {"ok": 1 - PRIOR_BAD, "bad": PRIOR_BAD}
    for alarm in observations:
        predicted = {
            to: sum(
                belief[fr] * TRANS["m"][fr].get(to, 0.0)
                for fr in ("ok", "bad")
            )
            for to in ("ok", "bad")
        }
        updated = {
            s: predicted[s] * obs_likelihood(s, alarm) for s in ("ok", "bad")
        }
        z = sum(updated.values())
        belief = {s: v / z for s, v in updated.items()}
    return belief


@pytest.mark.parametrize(
    "observations",
    [
        [False, False, False],
        [True],
        [True, True],
        [False, True, True, True],
        [True, False, False, False, False],
    ],
)
def test_tracker_matches_exact_hmm_filter(observations):
    tracker = ModeTracker(build_system(), TRANS, beam=2)
    for alarm in observations:
        tracker.step({"alarm": alarm})
    expected = exact_forward(observations)
    got = tracker.marginals()["m"]
    for state in ("ok", "bad"):
        assert got[state] == pytest.approx(expected[state], rel=1e-9)


def test_initial_belief_is_prior():
    tracker = ModeTracker(build_system(), TRANS, beam=2)
    belief = dict(
        (modes["m"], p) for modes, p in tracker.belief()
    )
    assert belief["ok"] == pytest.approx(1 - PRIOR_BAD)
    assert belief["bad"] == pytest.approx(PRIOR_BAD)


def test_persistent_alarms_convict_the_component():
    tracker = ModeTracker(build_system(), TRANS, beam=2)
    for _ in range(6):
        tracker.step({"alarm": True})
    modes, prob = tracker.most_probable()
    assert modes["m"] == "bad"
    assert prob > 0.99


def test_absorbing_fault_stays_faulty():
    tracker = ModeTracker(build_system(), TRANS, beam=2)
    for _ in range(4):
        tracker.step({"alarm": True})
    p_bad_after_alarms = tracker.marginals()["m"]["bad"]
    tracker.step({"alarm": False})  # one glitchy quiet step
    p_bad = tracker.marginals()["m"]["bad"]
    # bad is absorbing: one clean reading shouldn't exonerate much
    # (P(alarm=False | bad) = 0.1, and ok requires never having failed).
    assert p_bad_after_alarms > 0.99
    assert p_bad > 0.9


def test_untracked_mode_var_resampled_from_prior():
    # Two components, transitions only for the first: the second is
    # memoryless and its posterior should reset toward prior each step.
    m = SystemModel()
    a = m.mode("a", ("ok", "bad"), priors=(0.99, 0.01))
    b = m.mode("b", ("ok", "bad"), priors=(0.9, 0.1))
    alarm = m.bool("alarm")
    m.add(iff(alarm, (a == "bad") | (b == "bad")))
    sys = m.compile()
    trans = {"a": {"ok": {"ok": 1.0, "bad": 0.0}, "bad": {"bad": 1.0, "ok": 0.0}}}
    tracker = ModeTracker(sys, trans, beam=4)
    tracker.step({"alarm": False})
    tracker.step({"alarm": False})
    marg = tracker.marginals()
    # With a pinned ok (no transitions to bad) and quiet alarms, b's
    # posterior after a quiet step equals P(b=bad | alarm=False) with
    # fresh prior: 0 mass from a=bad, so alarm=False => b=ok exactly.
    assert marg["b"]["ok"] == pytest.approx(1.0)


def test_belief_collapse_raises():
    m = SystemModel()
    mv = m.mode("m", ("ok", "bad"), priors=(0.5, 0.5))
    alarm = m.bool("alarm")
    m.add(iff(alarm, mv == "bad"))
    sys = m.compile()
    # bad is unreachable from ok and vice versa; observing alarm=True with
    # a beam trapped in ok must collapse.
    trans = {"m": {"ok": {"ok": 1.0, "bad": 0.0}, "bad": {"bad": 1.0, "ok": 0.0}}}
    tracker = ModeTracker(sys, trans, beam=1)  # beam=1 keeps only ok
    tracker.step({"alarm": False})
    with pytest.raises(ValueError):
        tracker.step({"alarm": True})


def test_exact_mode_sizes_beam_from_joint_state_space():
    m = SystemModel()
    m.mode("a", ("x", "y", "z"), priors=(1, 1, 1))
    m.mode("b", ("u", "v"), priors=(1, 1))
    tracker = ModeTracker(m.compile(), exact=True)
    assert tracker.joint_state_count == 6
    assert tracker.beam == 6
    assert tracker.expand == 6
    assert tracker.is_exact
    tracker.step({})
    assert tracker.last_step_info.exact
    assert tracker.last_step_info.retained_probability_mass == pytest.approx(1)


def test_exact_mode_has_configurable_state_guard():
    m = SystemModel()
    m.mode("a", ("x", "y", "z"), priors=(1, 1, 1))
    m.mode("b", ("u", "v"), priors=(1, 1))
    with pytest.raises(ValueError, match="6 joint states"):
        ModeTracker(m.compile(), exact=True, max_exact_states=5)


def test_step_info_reports_approximation():
    m = SystemModel()
    m.mode("m", ("a", "b", "c"), priors=(0.6, 0.3, 0.1))
    tracker = ModeTracker(m.compile(), beam=1, expand=1)
    initial = tracker.belief()
    assert not initial.exact
    assert initial.retained_probability_mass == pytest.approx(0.6)
    tracker.step({})
    info = tracker.last_step_info
    assert info.expansion_truncated
    assert info.retained_probability_mass is None
    assert not info.exact
    belief = tracker.belief()
    assert not belief.exact
    assert belief.retained_probability_mass is None


def test_refine_replays_retained_history_without_mutating_source():
    tracker = ModeTracker(
        build_system(),
        beam=1,
        expand=1,
        retain_history=True,
    )
    first = {"alarm": False}
    tracker.step(first, transitions=TRANS)
    first["alarm"] = True
    tracker.step({"alarm": True}, transitions=TRANS)

    refined = tracker.refine(exact=True)
    expected = exact_forward([False, True])

    assert tracker.beam == 1
    assert not tracker.belief().exact
    assert tracker.history[0].evidence == {"alarm": False}
    assert refined.beam == 2
    assert refined.belief().exact
    assert refined.last_replay_info.steps_replayed == 2
    assert refined.last_replay_info.generated_candidates > 0
    assert refined.last_replay_info.replay_seconds >= 0
    assert refined.last_replay_info.source_beam == 1
    assert refined.last_replay_info.target_beam == 2
    for state, probability in refined.marginals()["m"].items():
        assert probability == pytest.approx(expected[state], rel=1e-9)


def test_refine_requires_history_after_filtering():
    tracker = ModeTracker(build_system(), TRANS, beam=1, expand=1)
    tracker.step({"alarm": False})
    with pytest.raises(RuntimeError, match="retain_history=True"):
        tracker.refine(exact=True)


def test_refine_until_reports_resource_work_and_certificate_scope():
    tracker = ModeTracker(
        build_system(),
        TRANS,
        beam=1,
        expand=1,
        retain_history=True,
    )
    tracker.step({"alarm": True})

    def evaluate(belief):
        return SimpleNamespace(
            root_action_certified=belief.exact,
            certificate_scope=(
                "exact-tracker-belief"
                if belief.exact
                else "tracked-belief-mass-unknown"
            ),
            maximum_regret=0.0 if belief.exact else 1.0,
        )

    result = tracker.refine_until(
        evaluate,
        lambda decision: decision.root_action_certified,
    )

    assert result.accepted
    assert result.tracker.belief().exact
    assert len(result.attempts) == 2
    approximate, exact = result.attempts
    assert approximate.steps_replayed == 0
    assert approximate.certificate_scope == "tracked-belief-mass-unknown"
    assert not approximate.accepted
    assert exact.steps_replayed == 1
    assert exact.generated_candidates > 0
    assert exact.replay_seconds >= 0
    assert exact.evaluation_seconds >= 0
    assert exact.certificate_scope == "exact-tracker-belief"
    assert exact.maximum_regret == 0.0
    assert exact.accepted
    assert result.total_steps_replayed == 1
    assert result.total_replay_seconds >= 0
    assert result.total_evaluation_seconds >= 0
