"""Chapter 9 exercise solutions (ex. 1 and 2)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from dnnf import Planner

# Ex 1: Safe reachable only from Idling.
p = Planner()
p.mode("sw", ("Tracking", "Idling", "Safe"))
p.command("cmd", ("idle", "track", "safe", "none"))
p.transition("sw", "Tracking", "Idling", command=("cmd", "idle"))
p.transition("sw", "Idling", "Tracking", command=("cmd", "track"))
p.transition("sw", "Idling", "Safe", command=("cmd", "safe"))
assert p.compile(1).plan({"sw": "Tracking"}, {"sw": "Safe"}) is None
cost, steps = p.compile(2).plan({"sw": "Tracking"}, {"sw": "Safe"})
assert [s["cmd"] for s in steps] == ["idle", "safe"]

# Ex 2: costed routes -- direct (0.7) vs two cheap hops (0.2 + 0.2).
q = Planner()
q.mode("m", ("A", "B", "W"))
q.command("c", ("direct", "warm", "go", "none"))
q.transition("m", "A", "B", command=("c", "direct"), cost=0.7)
q.transition("m", "A", "W", command=("c", "warm"), cost=0.2)
q.transition("m", "W", "B", command=("c", "go"), cost=0.2)
cost2, steps2 = q.compile(2).plan({"m": "A"}, {"m": "B"})
assert [s["c"] for s in steps2] == ["warm", "go"]
assert abs(cost2 - 0.4) < 1e-9
print("ch09 solutions OK")
