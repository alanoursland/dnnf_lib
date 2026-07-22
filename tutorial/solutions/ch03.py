"""Chapter 3 exercise solutions."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from dnnf import SystemModel, iff, fd
sys.path.insert(0, os.path.dirname(__file__))
from ch01 import survivors, DOMAINS, BASE


def build(extra_light=True, broken=False):
    m = SystemModel()
    pump = m.mode("pump", ("ok", "weak", "dead"), priors=(0.90, 0.07, 0.03))
    drip = m.bool("drip")
    moist = m.bool("moist")
    m.add(iff(drip, pump != "dead"))
    m.add(moist >> drip)
    if extra_light:
        bulb = m.mode("bulb", ("ok", "burnt_out"), priors=(0.95, 0.05))
        light = m.bool("light")
        m.add(iff(light, bulb == "ok"))
    if broken:
        m.add(drip & ~drip)
    return m.compile()


# Exercise 1: independent subsystem -> nodes grow by ~a constant chunk,
# not multiplicatively (separate component in the circuit DAG).
small = build(extra_light=False)
big = build(extra_light=True)
assert len(big.circuit) - len(small.circuit) < len(small.circuit), (
    len(small.circuit), len(big.circuit))

# Exercise 2: unsatisfiable systems compile to the FALSE circuit.
dead = build(broken=True)
assert not fd.is_satisfiable(dead.circuit)
assert len(dead.circuit) == 1

# Exercise 3: automated cross-check, the repo discipline.
assert fd.model_count(big.circuit) == len(survivors(BASE, DOMAINS)) == 10
print("ch03 solutions OK:", len(small.circuit), "->", len(big.circuit))
