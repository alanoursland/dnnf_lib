"""Chapter 4 exercise solutions."""
import math, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from neximode import SystemModel, iff, fd


def build():
    m = SystemModel()
    pump = m.mode("pump", ("ok", "weak", "dead"), priors=(0.90, 0.07, 0.03))
    drip = m.bool("drip")
    moist = m.bool("moist")
    m.add(iff(drip, pump != "dead"))
    m.add(moist >> drip)
    return m.compile()


system = build()

# Exercise 2: direct product underflows; log-sum does not.
# (Gotcha: a naive `while p > 0: p *= 0.9` loop never terminates --
# at the smallest denormal, 5e-324 * 0.9 rounds back to 5e-324.
# Use pow, which underflows honestly.)
n = next(k for k in range(1, 20000) if 0.9 ** k == 0.0)
assert n == 7073, n
logs = 1000 * math.log(0.9)  # fine: -105.36

# Exercise 3: second-best world by cost surgery on the best one's
# distinguishing leaf.
costs = system._conditioned_costs({})
best_cost, best = fd.mpe(system.circuit, costs)
state = system._decode_state(best)
# Forbid the best world's moist value; the tie-partner appears.
var = system.vars["moist"]
idx = var.values.index(state["moist"])
costs2 = list(costs)
costs2[system.circuit.spec.mvlit(var.fd_var, idx)] = math.inf
second_cost, second = fd.mpe(system.circuit, costs2)
assert second_cost >= best_cost
print("ch04 solutions OK:", n, round(best_cost, 4), round(second_cost, 4))
