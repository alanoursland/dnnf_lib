"""QSIM-style qualitative envisionment of the classic cascaded tanks.

System: constant inflow -> tank A -> tank B -> drain, with qualitative
flow laws outflowX = M+(levelX) (monotonic, corresponding at zero).

One-time-slice qualitative state: for each tank a magnitude over the
quantity space {zero, between, full} and a direction of change over
{dec, std, inc}.  Constraints:

* correspondence: outflowX = zero  iff  levelX = zero
* sign of derivative: d(levelA) = sign(inflow - outflowA),
                      d(levelB) = sign(outflowA - outflowB),
  where sign(plus - plus) is ambiguous (unconstrained) — QSIM's
  irreducible ambiguity.
* landmark consistency: at zero a level cannot fall; at full it cannot
  rise.

The expected envisionment sizes below are derived by hand-enumeration of
these rules (the standard textbook analysis):

* inflow = plus:  33 consistent qualitative states
* inflow = zero:  15 states, exactly one fully quiescent
  (both tanks empty, both std) — the global equilibrium
* inflow free:    48 states
* inflow = plus and tank B rising (dB = inc): 10 states

The point of the test: the compiled circuit's model set IS the
envisionment — counting and filtering it are single sweeps.
"""

import math

import pytest

from modenexus import SystemModel, fd, iff

LEVELS = ("zero", "between", "full")
DIRS = ("dec", "std", "inc")


def sign_constraints(m, p, q, d):
    """d = sign(p - q) over {zero, plus} magnitudes; plus-plus is free."""
    m.add(((p == "zero") & (q == "zero")) >> (d == "std"))
    m.add(((p == "zero") & (q == "plus")) >> (d == "dec"))
    m.add(((p == "plus") & (q == "zero")) >> (d == "inc"))


def build(inflow_value=None):
    m = SystemModel()
    la = m.finite("levelA", LEVELS)
    da = m.finite("dA", DIRS)
    lb = m.finite("levelB", LEVELS)
    db = m.finite("dB", DIRS)
    inflow = m.finite("inflow", ("zero", "plus"))
    oa = m.finite("outflowA", ("zero", "plus"))
    ob = m.finite("outflowB", ("zero", "plus"))

    # M+ with corresponding value at zero.
    m.add(iff(oa == "zero", la == "zero"))
    m.add(iff(ob == "zero", lb == "zero"))

    sign_constraints(m, inflow, oa, da)
    sign_constraints(m, oa, ob, db)

    # Landmark consistency.
    m.add((la == "zero") >> (da != "dec"))
    m.add((la == "full") >> (da != "inc"))
    m.add((lb == "zero") >> (db != "dec"))
    m.add((lb == "full") >> (db != "inc"))

    if inflow_value is not None:
        m.add(inflow == inflow_value)
    return m.compile()


def count(system, evidence=None):
    log_c = fd.log_wmc(
        system.circuit, system.log_weights_for(evidence or {})
    )
    return 0 if log_c == -math.inf else round(math.exp(log_c))


def test_envisionment_size_inflow_on():
    assert count(build("plus")) == 33


def test_envisionment_size_inflow_off():
    assert count(build("zero")) == 15


def test_envisionment_size_inflow_free():
    assert count(build()) == 48


def test_filtering_tank_b_rising():
    assert count(build("plus"), {"dB": "inc"}) == 10


def test_unique_quiescent_state_is_empty_tanks():
    system = build("zero")
    assert count(system, {"dA": "std", "dB": "std"}) == 1
    # Decode the single quiescent state: both tanks must be empty.
    lw = system.log_weights_for({"dA": "std", "dB": "std"})
    costs = [math.inf if x == -math.inf else 0.0 for x in lw]
    states = list(fd.enumerate_models(system.circuit, costs, k=2))
    assert len(states) == 1
    _, assignment = states[0]
    decoded = system._decode_state(assignment)
    assert decoded["levelA"] == "zero"
    assert decoded["levelB"] == "zero"


def test_no_quiescence_while_inflow_runs():
    # With inflow on, a fully quiescent state requires inflow to balance
    # outflowA exactly (plus-plus ambiguity) with tank A non-empty --
    # possible for A, but B then receives flow and must also balance.
    # The states dA=std & dB=std require both tanks non-empty: verify
    # none of them has an empty tank.
    system = build("plus")
    n = count(system, {"dA": "std", "dB": "std"})
    assert n == 4  # levelA in {between, full} x levelB in {between, full}
    assert count(
        system, {"dA": "std", "dB": "std", "levelA": "zero"}
    ) == 0
    assert count(
        system, {"dA": "std", "dB": "std", "levelB": "zero"}
    ) == 0


def test_overflow_risk_query():
    """The design-stage question: can tank B be full and still rising?
    Landmark consistency must forbid it."""
    assert count(build("plus"), {"levelB": "full", "dB": "inc"}) == 0
