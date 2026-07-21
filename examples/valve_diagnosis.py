"""Model-based diagnosis of a small propulsion feed system.

A tank feeds two parallel legs; each leg has a valve, and each valve can be
ok, stuck_open, or stuck_closed.  A flow sensor sits on each leg and one on
the combined outlet.  Valves are commanded open.  We observe sensors and ask
for the most probable mode assignments, ordered — the DNNF-based diagnosis
loop: compile once offline, then answer observation queries online.

Run:  python examples/valve_diagnosis.py
"""

from dnnf import SystemModel, iff

m = SystemModel()

v1 = m.mode("valve1", ("ok", "stuck_open", "stuck_closed"), priors=(0.98, 0.01, 0.01))
v2 = m.mode("valve2", ("ok", "stuck_open", "stuck_closed"), priors=(0.98, 0.01, 0.01))

flow1 = m.bool("flow1")
flow2 = m.bool("flow2")
flow_out = m.bool("flow_out")

# Valves are commanded open: a leg flows unless its valve is stuck closed.
m.add(iff(flow1, v1 != "stuck_closed"))
m.add(iff(flow2, v2 != "stuck_closed"))
m.add(iff(flow_out, flow1 | flow2))

system = m.compile()
print(f"compiled circuit: {system.circuit!r}")

for evidence in (
    {"flow1": True, "flow2": True, "flow_out": True},   # nominal
    {"flow1": False, "flow2": True, "flow_out": True},  # leg-1 anomaly
    {"flow_out": False},                                # only outlet sensed
):
    print(f"\nevidence: {evidence}")
    for d in system.diagnoses(evidence, k=4):
        print(f"  {d}")
    posteriors = system.mode_posteriors(evidence)
    p1 = posteriors["valve1"]
    print("  P(valve1 | evidence): "
          + ", ".join(f"{k}={v:.4f}" for k, v in p1.items()))
