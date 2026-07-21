"""The classic 'car won't start' as compiled diagnosis.

Shows finite-domain observables (headlights are off/dim/bright, not a
boolean) and mixed-domain modes.  Run: python examples/car_wont_start.py
"""

from dnnf import SystemModel, iff

m = SystemModel()

battery = m.mode("battery", ("ok", "weak", "dead"), priors=(0.90, 0.06, 0.04))
starter = m.mode("starter", ("ok", "dead"), priors=(0.97, 0.03))
ignition = m.mode("ignition", ("ok", "fault"), priors=(0.97, 0.03))
fuel = m.mode("fuel", ("has_fuel", "empty"), priors=(0.95, 0.05))

lights = m.finite("lights", ("off", "dim", "bright"))
cranks = m.bool("cranks")
starts = m.bool("starts")

m.add(iff(lights == "off", battery == "dead"))
m.add(iff(lights == "dim", battery == "weak"))
m.add(iff(cranks, (starter == "ok") & (battery == "ok")))
m.add(iff(starts, cranks & (fuel == "has_fuel") & (ignition == "ok")))

system = m.compile()
print(f"compiled: {system.circuit!r}\n")

for label, ev in [
    ("won't start, lights bright, cranks fine",
     {"starts": False, "lights": "bright", "cranks": True}),
    ("won't start, everything silent, lights off",
     {"starts": False, "lights": "off", "cranks": False}),
    ("won't start, lights bright, no crank",
     {"starts": False, "lights": "bright", "cranks": False}),
]:
    print(f"--- {label}")
    for d in system.map_diagnoses(ev, k=3):
        print(f"  p={d.posterior:.3f}  " +
              ", ".join(f"{k}={v}" for k, v in sorted(d.modes.items())))
    print()
