"""A realistic modeling experiment: home solar + battery system.

This is the "would a domain engineer tolerate the DSL?" test.  The system:

    PV array --> charge controller --> battery --> inverter --> house loads
                                          |
                                       grid tie (backup / export)

Components and failure modes:
  * pv:        ok | degraded | dead          (panel string failures, shading)
  * ctrl:      ok | stuck_off                (charge controller)
  * battery:   ok | weak | dead              (capacity fade, cell failure)
  * inverter:  ok | fault                    (trips offline)
  * grid:      up | down                     (external, not a fault mode)

Observables (booleans, as a monitoring system would sample them):
  * sun            — daylight, from clock/irradiance (an input, observed)
  * pv_current     — PV producing current
  * charging       — battery charge current present
  * soc_rising     — state of charge trending up
  * house_powered  — loads energized
  * on_grid        — meter shows grid import/export

Expectations (the "physics"):
  * PV produces current iff there is sun and the array isn't dead.
    (degraded still produces *some* current on this coarse abstraction)
  * Battery charges iff PV produces, controller works, battery accepts
    charge (not dead).
  * SOC rises iff charging and battery holds charge (not weak/dead).
  * House is powered iff inverter ok and (battery not dead or grid up).
  * on_grid iff grid is up.

Run:  python examples/home_battery.py
"""

from neximode import ModeTracker, SystemModel, iff

m = SystemModel()

pv = m.mode("pv", ("ok", "degraded", "dead"), priors=(0.95, 0.04, 0.01))
ctrl = m.mode("ctrl", ("ok", "stuck_off"), priors=(0.99, 0.01))
battery = m.mode("battery", ("ok", "weak", "dead"), priors=(0.94, 0.05, 0.01))
inverter = m.mode("inverter", ("ok", "fault"), priors=(0.99, 0.01))
grid = m.mode("grid", ("up", "down"), priors=(0.98, 0.02))

sun = m.bool("sun")
pv_current = m.bool("pv_current")
charging = m.bool("charging")
house_powered = m.bool("house_powered")
on_grid = m.bool("on_grid")

m.add(iff(pv_current, sun & (pv != "dead")))
m.add(iff(charging, pv_current & (ctrl == "ok") & (battery != "dead")))
# SOC estimation is noisy: it can read flat on a healthy day (5%) or
# drift up on a weak battery (2%).  The noise keeps one bad reading from
# convicting the battery outright.
soc_rising = m.sensor(
    "soc_rising", charging & (battery == "ok"),
    false_positive=0.02, false_negative=0.05,
)
m.add(iff(house_powered, (inverter == "ok") & ((battery != "dead") | (grid == "up"))))
m.add(iff(on_grid, grid == "up"))

system = m.compile()
print(f"compiled: {system.circuit!r}\n")

SCENARIOS = [
    ("nominal sunny day",
     dict(sun=True, pv_current=True, charging=True, soc_rising=True,
          house_powered=True, on_grid=True)),
    ("charging but SOC flat (weak battery?)",
     dict(sun=True, pv_current=True, charging=True, soc_rising=False,
          house_powered=True, on_grid=True)),
    ("sun but no PV current",
     dict(sun=True, pv_current=False, house_powered=True, on_grid=True)),
    ("PV fine, not charging (controller vs dead battery)",
     dict(sun=True, pv_current=True, charging=False, house_powered=True,
          on_grid=True)),
    ("house dark during outage",
     dict(sun=False, house_powered=False, on_grid=False)),
]

for label, evidence in SCENARIOS:
    print(f"--- {label}: {evidence}")
    for d in system.map_diagnoses(evidence, k=3):
        modes = ", ".join(f"{k}={v}" for k, v in sorted(d.modes.items()))
        print(f"  p={d.posterior:.4f}  {modes}")
    print()

# ----------------------------------------------------------------------
# Temporal tracking: a battery degrading over a week of daily snapshots.
# ----------------------------------------------------------------------
print("=== tracking: daily snapshots, battery fading ===")
TRANS = {
    "battery": {
        "ok": {"ok": 0.995, "weak": 0.004, "dead": 0.001},
        "weak": {"weak": 0.98, "dead": 0.02, "ok": 0.0},
        "dead": {"dead": 1.0, "ok": 0.0, "weak": 0.0},
    },
    "pv": {
        "ok": {"ok": 0.999, "degraded": 0.0009, "dead": 0.0001},
        "degraded": {"degraded": 0.999, "dead": 0.001, "ok": 0.0},
        "dead": {"dead": 1.0, "ok": 0.0, "degraded": 0.0},
    },
    # ctrl / inverter / grid: memoryless (resampled from priors) is fine
    # for this demo; grid outages genuinely are.
}

tracker = ModeTracker(system, TRANS, beam=8)
week = (
    [dict(sun=True, pv_current=True, charging=True, soc_rising=True,
          house_powered=True, on_grid=True)] * 3          # healthy
    + [dict(sun=True, pv_current=True, charging=True, soc_rising=False,
            house_powered=True, on_grid=True)] * 4        # SOC stops rising
)
for day, evidence in enumerate(week, 1):
    tracker.step(evidence)
    p_batt = tracker.marginals()["battery"]
    top, p = tracker.most_probable()
    print(f"day {day}: P(battery)= " +
          " ".join(f"{k}:{v:.3f}" for k, v in p_batt.items()) +
          f"   best={'{' + ', '.join(f'{k}={v}' for k, v in sorted(top.items())) + '}'} ({p:.3f})")
