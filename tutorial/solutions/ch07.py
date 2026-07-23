"""Chapter 7 exercise solutions (ex. 3: the right kind of failure)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from modenexus import ModeTracker, SystemModel


def build():
    m = SystemModel()
    pump = m.mode("pump", ("ok", "weak", "dead"), priors=(1.0, 0.0, 0.0))
    m.sensor("flow_ok", pump == "ok", false_positive=0.05,
             false_negative=0.05)
    m.sensor("any_flow", pump != "dead", false_positive=0.05,
             false_negative=0.05)
    return m.compile()


NEVER_FAILS = {"pump": {
    "ok": {"ok": 1.0, "weak": 0.0, "dead": 0.0},
    "weak": {"weak": 1.0, "ok": 0.0, "dead": 0.0},
    "dead": {"dead": 1.0, "ok": 0.0, "weak": 0.0},
}}

tracker = ModeTracker(build(), NEVER_FAILS, beam=3)
for _ in range(3):
    tracker.step({"flow_ok": True, "any_flow": True})
# With "never fails", weak/dead have zero mass forever; contradictory
# evidence can only be explained by sensor noise -- belief stays ok,
# it does NOT collapse (noise keeps every day consistent):
for _ in range(3):
    tracker.step({"flow_ok": False, "any_flow": True})
assert tracker.marginals()["pump"]["ok"] == 1.0
# The fix belongs in the MATRIX (the model was wrong about reality),
# not in the beam (search width was never the problem).
print("ch07 solutions OK")
