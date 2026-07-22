"""Compiled joint transition relations via prev-slice variables."""

import pytest

from neximode import ModeTracker, SystemModel

FAIL = 0.2
ABSORBING = {
    "ok": {"ok": 1 - FAIL, "bad": FAIL},
    "bad": {"bad": 1.0, "ok": 0.0},
}


def build(joint_constraint: bool):
    m = SystemModel()
    a = m.mode("a", ("ok", "bad"), priors=(1.0, 0.0))
    b = m.mode("b", ("ok", "bad"), priors=(1.0, 0.0))
    if joint_constraint:
        # No common-cause pair failure: both cannot fail on the same
        # tick starting from a healthy pair.
        m.add(~((m.prev("a") == "ok") & (m.prev("b") == "ok")
                & (a == "bad") & (b == "bad")))
    return m.compile()


def test_joint_constraint_prunes_and_renormalizes():
    trans = {"a": ABSORBING, "b": ABSORBING}
    free = ModeTracker(build(False), trans, beam=4)
    constrained = ModeTracker(build(True), trans, beam=4)
    free.step({})
    constrained.step({})

    got_free = {tuple(m[k] for k in ("a", "b")): p
                for m, p in free.belief()}
    got_con = {tuple(m[k] for k in ("a", "b")): p
               for m, p in constrained.belief()}

    # Unconstrained: independent product.
    assert got_free[("bad", "bad")] == pytest.approx(FAIL * FAIL)
    # Constrained: (bad,bad) pruned, rest renormalized over 1 - 0.04.
    z = 1 - FAIL * FAIL
    assert ("bad", "bad") not in got_con
    assert got_con[("ok", "ok")] == pytest.approx((1 - FAIL) ** 2 / z)
    assert got_con[("ok", "bad")] == pytest.approx((1 - FAIL) * FAIL / z)
    assert got_con[("bad", "ok")] == pytest.approx((1 - FAIL) * FAIL / z)


def test_second_step_reaches_double_fault_sequentially():
    # (bad,bad) is unreachable in one tick but reachable via a single
    # fault followed by the other.
    tracker = ModeTracker(build(True), {"a": ABSORBING, "b": ABSORBING},
                          beam=4)
    tracker.step({})
    tracker.step({})
    belief = {tuple(m[k] for k in ("a", "b")): p
              for m, p in tracker.belief()}
    # Hand recursion: after step 1 (z = 1 - 0.04):
    #   (ok,ok)=0.64/z, (ok,bad)=(bad,ok)=0.16/z.
    # Step 2: from (ok,ok) same conditional row as step 1; from a
    # single-fault state the constraint is inactive (prev pair not
    # both ok), so the healthy one fails freely at 0.2.
    # Global renormalization at step 2: the healthy particle loses its
    # pruned (bad,bad) mass; single-fault particles lose nothing.
    z = 1 - FAIL * FAIL
    p_oo1, p_ob1 = 0.64 / z, 0.16 / z
    total = p_oo1 * z + 2 * p_ob1
    expected_bb = 2 * p_ob1 * FAIL / total
    expected_oo = p_oo1 * 0.64 / total
    assert belief[("bad", "bad")] == pytest.approx(expected_bb)
    assert belief[("ok", "ok")] == pytest.approx(expected_oo)


def test_prev_requires_mode_var():
    m = SystemModel()
    m.bool("x")
    with pytest.raises(KeyError):
        m.prev("x")


def test_prev_vars_hidden_from_diagnoses():
    sys = build(True)
    diags = sys.map_diagnoses({}, k=2)
    assert diags, "initial belief should exist with prev unconstrained"
    for d in diags:
        assert all("@prev" not in k for k in d.modes)
