"""Chapter 6 exercise solutions."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from modenexus import SystemModel, iff


def build():
    m = SystemModel()
    pump = m.mode("pump", ("ok", "dead"), priors=(0.97, 0.03))
    drip = m.bool("drip")
    m.add(iff(drip, pump == "ok"))
    m.sensor("moist", drip, false_positive=0.03, false_negative=0.08)
    return m.compile()


s = build()
# Ex 1: P(dry|dead) uses the FALSE-POSITIVE rate (a dead pump's dry
# soil reads moist only if the sensor invents moisture): 1-fp = 0.97.
bayes = 0.03 * 0.97 / (0.03 * 0.97 + 0.97 * 0.08)
got = s.posteriors({"moist": False})["pump"]["dead"]
assert abs(got - bayes) < 1e-12
# The tempting-but-wrong version applies the miss rate (fn) there:
wrong = 0.03 * 0.92 / (0.03 * 0.92 + 0.97 * 0.08)
assert abs(wrong - got) > 0.005  # close, and wrong

# Ex 2: (1, 0) soft evidence == hard evidence moist=False.
soft_extreme = s.posteriors({"moist": (1.0, 0.0)})["pump"]["dead"]
hard = s.posteriors({"moist": False})["pump"]["dead"]
assert abs(soft_extreme - hard) < 1e-12
print("ch06 solutions OK:", round(got, 4))
