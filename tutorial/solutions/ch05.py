"""Chapter 5 exercise solutions."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from dnnf import SystemModel, iff


def build(with_valve=False):
    m = SystemModel()
    pump = m.mode("pump", ("ok", "weak", "dead"), priors=(0.45, 0.52, 0.03))
    drip = m.bool("drip")
    moist = m.bool("moist")
    m.add((pump == "weak") >> ~drip)
    m.add((pump == "dead") >> ~drip)
    if with_valve:
        valve = m.mode("valve", ("ok", "stuck"), priors=(0.95, 0.05))
        m.add((valve == "stuck") >> ~drip)
    m.add(iff(moist, drip))
    return m.compile()


sys5 = build()
# Ex 1: Z = .45+.45+.52+.03 = 1.45; moist=True keeps only (ok, drip):
post = sys5.posteriors({"moist": True})["pump"]
assert abs(post["ok"] - 1.0) < 1e-9  # only ok worlds allow moisture
# Ex 2: with a valve the disagreement survives (ok still spreads over
# timer AND valve worlds; weak still owns the heaviest single state):
sys5v = build(with_valve=True)
mpe_top = sys5v.diagnoses({}, k=1)[0].modes["pump"]
map_top = sys5v.map_diagnoses({}, k=1)[0].modes["pump"]
assert (mpe_top, map_top) == ("weak", "ok"), (mpe_top, map_top)
print("ch05 solutions OK")
