"""Chapter 8 exercise solutions (ex. 1: the loop, one turn)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from modenexus import SystemModel, iff


def build():
    m = SystemModel()
    pump = m.mode("pump", ("ok", "dead"), priors=(0.97, 0.03))
    valve = m.mode("valve", ("ok", "stuck"), priors=(0.95, 0.05))
    drip = m.bool("drip")
    m.add(iff(drip, (pump == "ok") & (valve == "ok")))
    m.sensor("moist", drip, false_negative=0.05)
    m.sensor("pump_hum", pump == "ok", false_positive=0.02,
             false_negative=0.02)
    return m.compile()


s = build()
ev = {"moist": False, "pump_hum": True}   # heard the hum
top = s.map_diagnoses(ev, k=1)[0]
assert top.modes["pump"] == "ok"          # hum exonerates the pump
voi = s.value_of_information(ev)
# Remaining uncertainty is the valve; only 'drip' can still inform:
assert voi[0][0] == "drip"
# Stopping rule (one sentence): stop when max VOI is below the cost
# of taking the observation (here: when it drops near zero).
print("ch08 solutions OK:", top.modes, [(n, round(v, 3)) for n, v in voi])
