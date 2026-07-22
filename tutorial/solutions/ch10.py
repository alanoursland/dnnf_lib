"""Chapter 10 exercise solutions (ex. 1: scatter vs N)."""
import random, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from neximode import SystemModel


def build(pb):
    m = SystemModel()
    v = m.mode("valve", ("ok", "leaky"), priors=(1 - pb, pb))
    m.sensor("puddle", v == "leaky", false_positive=0.05,
             false_negative=0.10)
    return m.compile()


def fit(obs, start):
    s = build(start)
    s.fit_priors(obs, names=["valve"], iterations=200, tol=1e-12)
    spec = s.circuit.spec
    return s._weights[spec.mvlit(s.vars["valve"].fd_var, 1)]


rng = random.Random(3)
true = build(0.20)
big = [{"puddle": true.sample_state(rng)["puddle"]} for _ in range(3000)]

for n, tol in ((50, 0.25), (3000, 0.05)):
    fits = [fit(big[:n], s) for s in (0.1, 0.5, 0.9)]
    spread = max(fits) - min(fits)
    # identifiable: all starts land together (tight tol matters --
    # EM's default early stop is on LIKELIHOOD change, and equal
    # likelihoods tolerate slightly different parameters):
    assert spread < 1e-4, spread
    assert abs(fits[0] - 0.20) < tol   # ...accuracy is limited by N only
print("ch10 solutions OK")
