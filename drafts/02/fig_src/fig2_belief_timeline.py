"""Figure 2: tracked month from the case study (seeded; regenerates
the §5 timeline). Reuses the case-study model and evidence process."""
import os, random, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from case_study import build, TRANS  # noqa: E402
from dnnf import ModeTracker

OUT = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"font.size": 11, "figure.dpi": 200})
rng = random.Random(2026)
def flip(truth, fp, fn):
    return (rng.random() > fn) if truth else (rng.random() < fp)

tracker = ModeTracker(build(), TRANS, beam=8)
hist = []
for day in range(1, 31):
    tb = "ok" if day < 13 else "weak"
    soc = tb == "ok"
    tracker.step({"sun": True,
                  "pv_current": flip(True, 0.02, 0.02),
                  "charging": flip(True, 0.02, 0.02),
                  "soc_rising": flip(soc, 0.02, 0.05),
                  "house_powered": flip(True, 0.01, 0.01)})
    hist.append(dict(tracker.marginals()["battery"]))

fig, ax = plt.subplots(figsize=(6.6, 2.9))
days = range(1, 31)
ax.stackplot(days, [h["ok"] for h in hist], [h["weak"] for h in hist],
             [h["dead"] for h in hist], labels=["ok", "weak", "dead"],
             colors=["#9fd19f", "#f2d091", "#e69a9a"])
ax.axvline(12.5, color="#444", ls=":", lw=1)
ax.annotate("fault onset (day 13)", (12.7, 0.5), fontsize=8)
ax.annotate("detected day 14", (14.2, 0.15), fontsize=8)
ax.set_xlabel("day"); ax.set_ylabel("P(battery mode)")
ax.set_xlim(1, 30); ax.set_ylim(0, 1)
ax.legend(loc="center right", fontsize=8)
fig.savefig(os.path.join(OUT, "fig2_belief_timeline.png"),
            bbox_inches="tight")
print("wrote fig2_belief_timeline.png")
