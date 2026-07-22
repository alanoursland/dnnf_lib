"""Figure 3: calibration on real hydraulic-rig data (§6).
Requires the dataset at /tmp/hydraulic (see real_data_study.py)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

OUT = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"font.size": 11, "figure.dpi": 200})

# Re-run the study to get per-bin numbers (fast: ~2k queries on tiny
# circuits). Import machinery from the study script.
import importlib.util
spec = importlib.util.spec_from_file_location(
    "rds", os.path.join(os.path.dirname(__file__), "..",
                        "real_data_study.py"))
# real_data_study.py runs at import; capture its bin data by re-running
# the evaluation loop here instead.
D = "/tmp/hydraulic"
assert os.path.isdir(D), "download dataset first (see real_data_study.py)"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
exec(open(os.path.join(os.path.dirname(__file__), "..",
                       "real_data_study.py")).read().split(
    'print("target')[0])  # reuse defs up to the evaluation loop

fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.9), sharey=True)
for ax, target in zip(axes, TARGETS):
    labels, classes, feats = TARGETS[target]
    sysm, _ = build(target)
    bins = [[0, 0] for _ in range(5)]
    for i in test:
        ev = {}
        for f in feats:
            feat, bf = FEATS[f]
            one = [EPS] * K
            one[bf(feat[i])] = 1.0
            ev[f] = tuple(one)
        post = sysm.posteriors(ev, names=[target])[target]
        pred = max(post, key=post.get)
        b = min(int(post[pred] * 5), 4)
        bins[b][0] += pred == str(labels[i])
        bins[b][1] += 1
    xs, accs, ns, ideal = [], [], [], []
    for b, (h, t) in enumerate(bins):
        if t:
            xs.append(0.2 * b + 0.1)
            accs.append(h / t)
            ns.append(t)
            ideal.append(0.2 * b + 0.1)
    ax.bar(xs, accs, width=0.16, color="#3b6ea5", label="observed accuracy")
    ax.plot([0, 1], [0, 1], "--", color="#888", lw=1, label="perfect calibration")
    for x, a, n in zip(xs, accs, ns):
        ax.annotate(f"n={n}", (x, a + 0.03), ha="center", fontsize=7)
    ax.set_title(target, fontsize=10)
    ax.set_xlabel("reported confidence")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.12)
axes[0].set_ylabel("observed accuracy")
axes[1].legend(fontsize=7, loc="upper left")
fig.savefig(os.path.join(OUT, "fig3_calibration.png"), bbox_inches="tight")
print("wrote fig3_calibration.png")
