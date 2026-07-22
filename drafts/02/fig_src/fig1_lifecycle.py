"""Figure 1: the lifecycle on one compiled artifact (paper restyle)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"font.size": 11, "figure.dpi": 200})

fig, ax = plt.subplots(figsize=(6.6, 3.4))
steps = [("MODEL", "modes, sensors,\nconstraints"),
         ("COMPILE", "once, offline\n(§2)"),
         ("QUERY", "diagnose · sense ·\ntrack · plan (§3–4)"),
         ("LEARN", "priors from\ntelemetry (§4)")]
centers = [(0.14, 0.72), (0.86, 0.72), (0.86, 0.24), (0.14, 0.24)]
for (x, y), (head, sub) in zip(centers, steps):
    ax.add_patch(FancyBboxPatch((x - 0.13, y - 0.13), 0.26, 0.26,
                                boxstyle="round,pad=0.012",
                                fc="#f2f6fc", ec="#2c4a6e", lw=1.1))
    ax.text(x, y + 0.045, head, ha="center", weight="bold", fontsize=10)
    ax.text(x, y - 0.05, sub, ha="center", fontsize=8, color="#333")
for a, b, rad in ((0, 1, 0.0), (1, 2, 0.25), (2, 3, 0.0), (3, 0, 0.25)):
    (x0, y0), (x1, y1) = centers[a], centers[b]
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="->",
                                 mutation_scale=14, color="#2c4a6e",
                                 shrinkA=30, shrinkB=30,
                                 connectionstyle=f"arc3,rad={rad}"))
ax.text(0.5, 0.48, "one compiled circuit\n(5.2 KiB in §5)", ha="center",
        va="center", fontsize=10, style="italic", color="#2c4a6e")
ax.set_xlim(-0.04, 1.04); ax.set_ylim(0, 1); ax.axis("off")
fig.savefig(os.path.join(OUT, "fig1_lifecycle.png"), bbox_inches="tight")
print("wrote fig1_lifecycle.png")
