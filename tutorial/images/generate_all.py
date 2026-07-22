"""Generate all tutorial images. Run: python tutorial/images/generate_all.py

Every figure is produced from code (mostly from the library itself) so
the images regenerate when the library changes. Matplotlib only — no
graphviz binary required (layouts come from neximode.viz.layered_layout).
"""

import math
import os
import sys
from itertools import product

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, Rectangle

from neximode import SystemModel, fd, iff
from neximode.circuit import AND, LIT, OR
from neximode.viz import layered_layout, node_label

OUT = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"font.size": 12, "figure.dpi": 130})

AND_C, OR_C, LIT_C = "#fff4e6", "#eef4ff", "#f0f0f0"


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# ----------------------------------------------------------------------
def greenhouse_fragment():
    """Ch.1-4 fragment: pump mode, moisture bool, drip bool."""
    m = SystemModel()
    pump = m.mode("pump", ("ok", "weak", "dead"), priors=(0.90, 0.07, 0.03))
    drip = m.bool("drip")
    moist = m.bool("moist")
    m.add(iff(drip, pump != "dead"))
    m.add(moist >> drip)  # soil can only be moist if water reaches it
    return m


def draw_circuit(ax, circuit, names, values, node_vals=None, fontsize=9):
    pos = layered_layout(circuit)
    for i in range(len(circuit)):
        for c in circuit.children[i]:
            x0, y0 = pos[c]
            x1, y1 = pos[i]
            ax.add_patch(FancyArrowPatch((x0, y0 + 0.12), (x1, y1 - 0.12),
                                         arrowstyle="-", color="#999999",
                                         lw=0.8, zorder=1))
    for i in range(len(circuit)):
        x, y = pos[i]
        kind = circuit.kinds[i]
        label = node_label(circuit, i, names, values)
        if node_vals is not None:
            v = node_vals[i]
            label += f"\n{v:.3g}" if isinstance(v, float) else f"\n{v}"
        color = AND_C if kind == AND else OR_C if kind == OR else LIT_C
        if kind == LIT:
            ax.add_patch(Rectangle((x - 0.45, y - 0.14), 0.9, 0.28,
                                   fc=color, ec="#666666", zorder=2))
        else:
            ax.add_patch(Ellipse((x, y), 0.9, 0.3, fc=color, ec="#666666",
                                 zorder=2))
        ax.text(x, y, label, ha="center", va="center", fontsize=fontsize,
                zorder=3)
    ax.set_xlim(-0.2, max(x for x, _ in pos.values()) + 0.8)
    ax.set_ylim(-0.5, max(y for _, y in pos.values()) + 0.5)
    ax.axis("off")


def sys_names(system):
    c = system.circuit
    names = [f"x{v}" for v in range(c.spec.num_vars)]
    values = [list(range(s)) for s in c.spec.sizes]
    for var in system.vars.values():
        names[var.fd_var] = var.name
        values[var.fd_var] = [
            {True: "T", False: "F"}.get(v, v) for v in var.values
        ]
    return names, values


# ---- 01a: world table -------------------------------------------------
def img_01a():
    rows = []
    for pump, drip, moist in product(("ok", "weak", "dead"),
                                     (False, True), (False, True)):
        legal = (drip == (pump != "dead")) and (not moist or drip)
        rows.append((pump, drip, moist, legal))
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    ax.axis("off")
    headers = ["pump", "drip", "moist", "world survives?"]
    for j, h in enumerate(headers):
        ax.text(j + 0.5, len(rows) + 0.5, h, ha="center", weight="bold")
    for r, (p, d, mo, ok) in enumerate(rows):
        y = len(rows) - r - 0.5
        bg = "#e8f5e9" if ok else "#ffebee"
        ax.add_patch(Rectangle((0, y - 0.5), 4, 1, fc=bg, ec="white"))
        for j, cell in enumerate((p, d, mo, "yes" if ok else "no")):
            ax.text(j + 0.5, y, str(cell), ha="center", va="center")
    ax.set_xlim(0, 4)
    ax.set_ylim(0, len(rows) + 1)
    kept = sum(1 for *_, ok in rows if ok)
    ax.set_title("All 12 worlds of the greenhouse fragment\n"
                 f"(constraints keep {kept})")
    save(fig, "01a_world_table.png")


# ---- 01b: explosion ---------------------------------------------------
def img_01b():
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ns = list(range(1, 266, 4))
    ax.semilogy(ns, [2.0 ** n for n in ns], lw=2)
    for n, label in [(33, "all humans, counting one\nworld per second, one year"),
                     (77, "atoms in the Milky Way... roughly"),
                     (266, "atoms in the visible universe")]:
        ax.axvline(n, color="#cccccc", lw=0.8)
        ax.text(n, 2.0 ** 20, label, rotation=90, fontsize=8,
                ha="right", va="bottom")
    ax.set_xlabel("boolean variables n")
    ax.set_ylabel("number of worlds (log scale)")
    ax.set_title("Brute force dies fast: $2^n$ worlds")
    save(fig, "01b_explosion.png")


# ---- 02b: the two properties -----------------------------------------
def img_02b():
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, gate, kids, note, color in (
        (axes[0], "AND", ["counts 3\nover {pump}", "counts 2\nover {drip, moist}"],
         "children mention DISJOINT variables\n⇒ multiply:  3 × 2 = 6", AND_C),
        (axes[1], "OR", ["counts 4\nworlds where pump=ok",
                         "counts 2\nworlds where pump=weak"],
         "children are MUTUALLY EXCLUSIVE\n⇒ add:  4 + 2 = 6", OR_C),
    ):
        ax.add_patch(Ellipse((0.5, 0.8), 0.24, 0.14, fc=color, ec="#666"))
        ax.text(0.5, 0.8, gate, ha="center", va="center", weight="bold")
        for x, kid in zip((0.25, 0.75), kids):
            ax.add_patch(Rectangle((x - 0.16, 0.28), 0.32, 0.2,
                                   fc="#fafafa", ec="#666"))
            ax.text(x, 0.38, kid, ha="center", va="center", fontsize=9)
            ax.add_patch(FancyArrowPatch((x, 0.5), (0.5, 0.72),
                                         arrowstyle="-", color="#999"))
        ax.text(0.5, 0.05, note, ha="center", fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
    axes[0].set_title("decomposability")
    axes[1].set_title("determinism")
    save(fig, "02b_two_properties.png")


# ---- 02c: a real compiled circuit ------------------------------------
def img_02c():
    system = greenhouse_fragment().compile()
    names, values = sys_names(system)
    fig, ax = plt.subplots(figsize=(9, 5))
    draw_circuit(ax, system.circuit, names, values)
    ax.set_title("The greenhouse fragment, compiled "
                 f"({len(system.circuit)} nodes)")
    save(fig, "02c_first_circuit.png")
    return system


# ---- 04a: three sweeps ------------------------------------------------
def img_04a(system):
    c = system.circuit
    names, values = sys_names(system)
    # counting sweep
    ones = [1.0] * c.spec.total
    count_vals = []
    from neximode.eval import _forward
    count_vals = _forward(c, lambda ml: 1, lambda a, b: a + b,
                          lambda a, b: a * b, 0, 1)
    # probability sweep (priors; observables uniform 0.5 for readability)
    w = list(system._weights)
    for var in system.vars.values():
        if var.name in ("drip", "moist"):
            for i in range(len(var.values)):
                w[c.spec.mvlit(var.fd_var, i)] = 0.5
    prob_vals = _forward(c, lambda ml: w[ml], lambda a, b: a + b,
                         lambda a, b: a * b, 0.0, 1.0)
    cost_vals = _forward(c, lambda ml: (math.inf if w[ml] <= 0
                                        else -math.log(w[ml])),
                         min, lambda a, b: a + b, math.inf, 0.0)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, vals, title in (
        (axes[0], count_vals, "count semiring  (+, ×)\nroot = #worlds"),
        (axes[1], prob_vals, "probability semiring  (+, ×)\nroot = P(theory)"),
        (axes[2], cost_vals, "tropical semiring  (min, +) on −log\nroot = cost of likeliest world"),
    ):
        draw_circuit(ax, c, names, values, node_vals=vals, fontsize=7)
        ax.set_title(title, fontsize=11)
    fig.suptitle("One circuit, three questions — only the two operators change",
                 fontsize=13)
    save(fig, "04a_three_sweeps.png")


# ---- 04b: neglog number line ------------------------------------------
def img_04b():
    fig, ax = plt.subplots(figsize=(7, 2.6))
    ps = [1.0, 0.9, 0.5, 0.1, 0.01, 0.001]
    xs = [-math.log(p) for p in ps]
    ax.hlines(0, -0.3, 7.5, color="#333")
    for p, x in zip(ps, xs):
        ax.plot([x], [0], "o", color="#1f77b4")
        ax.annotate(f"p={p}", (x, 0.02), ha="center", fontsize=9)
        ax.annotate(f"cost={x:.2f}", (x, -0.045), ha="center", fontsize=8,
                    color="#666")
    ax.annotate("p → 0,  cost → ∞", (7.4, 0.0), fontsize=10, va="center")
    ax.set_ylim(-0.1, 0.1)
    ax.axis("off")
    ax.set_title("cost = −log p:  likely is cheap, impossible is infinite\n"
                 "(probabilities multiply ⇔ costs add)")
    save(fig, "04b_neglog_line.png")


# ---- 00a: lifecycle ----------------------------------------------------
def img_00a():
    fig, ax = plt.subplots(figsize=(7.5, 5))
    steps = ["MODEL\nmodes, sensors,\nconstraints",
             "COMPILE\nonce, offline",
             "QUERY\ncount · diagnose ·\ntrack · plan",
             "LEARN\npriors from\ntelemetry"]
    centers = [(0.2, 0.75), (0.8, 0.75), (0.8, 0.25), (0.2, 0.25)]
    for (x, y), s in zip(centers, steps):
        ax.add_patch(Rectangle((x - 0.16, y - 0.13), 0.32, 0.26,
                               fc="#eef4ff", ec="#336"))
        ax.text(x, y, s, ha="center", va="center", fontsize=10)
    arrows = [(0, 1), (1, 2), (2, 3), (3, 0)]
    for a, b in arrows:
        (x0, y0), (x1, y1) = centers[a], centers[b]
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="->",
                                     mutation_scale=18, color="#333",
                                     shrinkA=32, shrinkB=32,
                                     connectionstyle="arc3,rad=0.12"))
    ax.text(0.5, 0.5, "one compiled\ncircuit", ha="center", va="center",
            fontsize=12, style="italic")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    save(fig, "00a_lifecycle.png")


if __name__ == "__main__":
    img_00a()
    img_01a()
    img_01b()
    img_02b()
    system = img_02c()
    img_04a(system)
    img_04b()
    print("done")
