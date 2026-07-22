"""Figures + verified numbers for tutorial chapters 5-8.
Run: python tutorial/images/generate_ch5_8.py  (prints the numbers the
chapters quote; regenerates their figures)."""

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dnnf import ModeTracker, SystemModel, iff

OUT = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"font.size": 12, "figure.dpi": 130})


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- ch5
def greenhouse_ch5():
    m = SystemModel()
    pump = m.mode("pump", ("ok", "weak", "dead"), priors=(0.45, 0.52, 0.03))
    drip = m.bool("drip")
    moist = m.bool("moist")
    # weak or dead pumps can't push water; an OK pump drips whenever
    # the (unmodeled) irrigation timer opens -- so under ok, drip is
    # genuinely open: TWO ok-worlds survive, one per timer state.
    m.add((pump == "weak") >> ~drip)
    m.add((pump == "dead") >> ~drip)
    m.add(iff(moist, drip))
    return m.compile()


def ch5_numbers():
    sys5 = greenhouse_ch5()
    print("\n-- ch5: MPE vs MAP (no evidence) --")
    print("MPE:", [(d.modes, round(d.posterior, 4))
                   for d in sys5.diagnoses({}, k=2)])
    print("MAP:", [(d.modes, round(d.posterior, 4))
                   for d in sys5.map_diagnoses({}, k=2)])
    print("-- ch5: evidence moist=False --")
    post = sys5.posteriors({"moist": False})
    print("post:", {k: {v: round(p, 4) for v, p in d.items()}
                    for k, d in post.items()})
    return sys5, post


def img_05a():
    fig, ax = plt.subplots(figsize=(8, 3.6))
    boxes = [
        (0.10, 0.4, "PUMP\nok / weak / dead", "#eef4ff"),
        (0.44, 0.4, "drip line", "#f7f7f7"),
        (0.78, 0.4, "soil", "#f7f7f7"),
    ]
    for x, y, label, c in boxes:
        ax.add_patch(plt.Rectangle((x, y), 0.2, 0.28, fc=c, ec="#345"))
        ax.text(x + 0.1, y + 0.14, label, ha="center", va="center",
                fontsize=10)
    for x0, x1 in ((0.30, 0.44), (0.64, 0.78)):
        ax.annotate("", (x1, 0.54), (x0, 0.54),
                    arrowprops=dict(arrowstyle="->"))
    ax.text(0.62, 0.2, "moist sensor\n(reads the soil)", ha="center",
            fontsize=9, style="italic")
    ax.annotate("", (0.86, 0.38), (0.68, 0.26),
                arrowprops=dict(arrowstyle="->", ls=":"))
    ax.set_xlim(0, 1.1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Greenhouse, chapter 5")
    save(fig, "05a_greenhouse.png")


def img_05b(sys5, post):
    prior = {"ok": 0.45, "weak": 0.52, "dead": 0.03}
    fig, ax = plt.subplots(figsize=(7, 3.8))
    modes = list(prior)
    x = range(len(modes))
    ax.bar([i - 0.18 for i in x], [prior[m] for m in modes], 0.36,
           label="prior", color="#b3c7e6")
    ax.bar([i + 0.18 for i in x], [post["pump"][m] for m in modes], 0.36,
           label="posterior | soil dry", color="#e6a4a4")
    ax.set_xticks(list(x), modes)
    ax.set_ylabel("P(pump = ·)")
    ax.legend()
    ax.set_title("One dry reading moves belief — exactly, not by rule")
    save(fig, "05b_posterior_bars.png")


def img_05c(sys5):
    mpe = sys5.diagnoses({}, k=4)
    mp = sys5.map_diagnoses({}, k=3)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    states = [
        ("ok\ntimer on\n(drip)", 0.45), ("ok\ntimer off\n(dry)", 0.45),
        ("weak\n(dry)", 0.52), ("dead\n(dry)", 0.03),
    ]
    axes[0].bar([s for s, _ in states],
                [v for _, v in states], color="#b3c7e6")
    axes[0].set_title("MPE view: individual states\n(best single bar wins)")
    modes = [("ok", 0.90), ("weak", 0.52), ("dead", 0.03)]
    axes[1].bar([m for m, _ in modes], [v for _, v in modes],
                color="#e6c8a4")
    axes[1].set_title("MAP view: states summed per mode\n(biggest area wins)")
    for ax in axes:
        ax.set_ylabel("probability mass")
    save(fig, "05c_mpe_vs_map.png")


# ---------------------------------------------------------------- ch6
def build_ch6(noisy):
    m = SystemModel()
    pump = m.mode("pump", ("ok", "dead"), priors=(0.97, 0.03))
    drip = m.bool("drip")
    m.add(iff(drip, pump == "ok"))
    if noisy:
        m.sensor("moist", drip, false_positive=0.03, false_negative=0.08)
    else:
        m.add(iff(m.bool("moist"), drip))
    return m.compile()


def img_06a():
    readings = [True, True, True, False, True, False, False, False]
    fig, ax = plt.subplots(figsize=(8, 4))
    for noisy, label, color in ((False, "exact sensor", "#c44"),
                                (True, "noisy sensor (fp 3%, fn 8%)", "#268")):
        sys6 = build_ch6(noisy)
        tracker = ModeTracker(
            sys6,
            {"pump": {"ok": {"ok": 0.99, "dead": 0.01},
                      "dead": {"dead": 1.0, "ok": 0.0}}},
            beam=2,
        )
        ps = []
        for r in readings:
            try:
                tracker.step({"moist": r})
                ps.append(tracker.marginals()["pump"]["dead"])
            except ValueError:  # exact sensor: belief collapse
                ps.append(1.0)
        ax.plot(range(1, len(ps) + 1), ps, "o-", label=label, color=color)
    ax.set_xlabel("day")
    ax.set_ylabel("P(pump = dead)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.set_title("A single dry day convicts the exact-sensor model;\n"
                 "the noisy-sensor model waits for the pattern")
    save(fig, "06a_snap_vs_gradual.png")


def img_06b():
    fig, axes = plt.subplots(1, 2, figsize=(9, 2.8))
    for ax, title, vals in (
        (axes[0], 'hard evidence: moist=False', [1.0, 0.0]),
        (axes[1], 'soft evidence: (0.8, 0.2) — "probably dry"', [0.8, 0.2]),
    ):
        ax.bar(["w(moist=F)", "w(moist=T)"], vals,
               color=["#268bd2", "#93c5e8"])
        ax.set_ylim(0, 1.1)
        ax.set_title(title, fontsize=10)
    fig.suptitle("Evidence is just weight surgery on two leaves")
    save(fig, "06b_evidence_vectors.png")


# ---------------------------------------------------------------- ch7
def build_ch7():
    m = SystemModel()
    pump = m.mode("pump", ("ok", "weak", "dead"), priors=(1.0, 0.0, 0.0))
    m.sensor("flow_ok", pump == "ok", false_positive=0.05,
             false_negative=0.05)
    m.sensor("any_flow", pump != "dead", false_positive=0.05,
             false_negative=0.05)
    return m.compile()


TRANS7 = {"pump": {
    "ok": {"ok": 0.97, "weak": 0.025, "dead": 0.005},
    "weak": {"weak": 0.95, "dead": 0.05, "ok": 0.0},
    "dead": {"dead": 1.0, "ok": 0.0, "weak": 0.0},
}}


def ch7_run():
    sys7 = build_ch7()
    tracker = ModeTracker(sys7, TRANS7, beam=3)
    days = ([{"flow_ok": True, "any_flow": True}] * 10
            + [{"flow_ok": False, "any_flow": True}] * 10
            + [{"flow_ok": False, "any_flow": False}] * 10)
    history = []
    for ev in days:
        tracker.step(ev)
        history.append(dict(tracker.marginals()["pump"]))
    return history


def img_07b(history):
    fig, ax = plt.subplots(figsize=(8.5, 4))
    days = range(1, len(history) + 1)
    ok = [h["ok"] for h in history]
    weak = [h["weak"] for h in history]
    dead = [h["dead"] for h in history]
    ax.stackplot(days, ok, weak, dead,
                 labels=["ok", "weak", "dead"],
                 colors=["#9fd19f", "#f2d091", "#e69a9a"])
    ax.axvline(10.5, color="#666", ls=":")
    ax.axvline(20.5, color="#666", ls=":")
    ax.text(5, 1.04, "flow_ok ✓", ha="center", fontsize=9)
    ax.text(15, 1.04, "flow_ok ✗, any_flow ✓", ha="center", fontsize=9)
    ax.text(25, 1.04, "both ✗", ha="center", fontsize=9)
    ax.set_xlabel("day")
    ax.set_ylabel("belief")
    ax.set_ylim(0, 1)
    ax.legend(loc="center left")
    ax.set_title("A pump dying in three acts, tracked", pad=22)
    save(fig, "07b_belief_timeline.png")


# ---------------------------------------------------------------- ch8
def build_ch8():
    m = SystemModel()
    pump = m.mode("pump", ("ok", "dead"), priors=(0.97, 0.03))
    valve = m.mode("valve", ("ok", "stuck"), priors=(0.95, 0.05))
    drip = m.bool("drip")
    m.add(iff(drip, (pump == "ok") & (valve == "ok")))
    m.sensor("moist", drip, false_negative=0.05)
    m.sensor("pump_hum", pump == "ok", false_positive=0.02,
             false_negative=0.02)
    return m.compile()


def ch8_numbers():
    sys8 = build_ch8()
    print("\n-- ch8: after moist=False --")
    print("mincard:", sys8.diagnoses_min_cardinality({"moist": False}, k=4))
    voi = sys8.value_of_information({"moist": False})
    print("voi:", [(n, round(v, 4)) for n, v in voi])
    return sys8, voi


def img_08b(sys8, voi):
    import math as _m

    def entropy(ev):
        h = 0.0
        for name in sys8.mode_vars:
            for p in sys8.posteriors(ev, names=[name])[name].values():
                if p > 0:
                    h -= p * _m.log(p)
        return h

    h0 = entropy({"moist": False})
    fig, ax = plt.subplots(figsize=(7, 3.6))
    names = ["(now)"] + [n for n, _ in voi]
    values = [h0] + [h0 - v for _, v in voi]
    ax.bar(names, values, color=["#888"] + ["#268bd2"] * len(voi))
    ax.set_ylabel("expected remaining entropy (nats)")
    ax.set_title("Which sensor to read next?\n"
                 "(lower expected entropy = more informative)")
    save(fig, "08b_voi.png")


if __name__ == "__main__":
    sys5, post5 = ch5_numbers()
    img_05a()
    img_05b(sys5, post5)
    img_05c(sys5)
    img_06a()
    img_06b()
    print("\n-- ch6 numbers --")
    n = build_ch6(True)
    print("noisy posterior after 1 dry day:",
          round(n.posteriors({"moist": False})["pump"]["dead"], 4))
    e = build_ch6(False)
    print("exact posterior after 1 dry day:",
          round(e.posteriors({"moist": False})["pump"]["dead"], 4))
    hist = ch7_run()
    print("\n-- ch7: belief at days 10/20/30 --")
    for d in (9, 19, 29):
        print(f"day {d+1}:", {k: round(v, 3) for k, v in hist[d].items()})
    img_07b(hist)
    sys8, voi = ch8_numbers()
    img_08b(sys8, voi)
    print("done")
