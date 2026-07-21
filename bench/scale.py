"""Scale sweep: how far does the pure-Python compiler go on realistic
system-model families?

Two parameterized generators:

* adder(k): k-bit ripple-carry adder, 5 gates per bit, each gate a mode
  variable {ok, stuck0, stuck1} — the classic gate-level model-based
  diagnosis benchmark (polybox's big sibling).
* process(n): n-stage serial process line, each stage a pump
  {ok, weak, dead} + valve {ok, stuck_open, stuck_closed} + noisy flow
  sensor — the shape of industrial monitoring models.

For each size: compile time and circuit size with modes_first on
(marginal MAP available) and off (unconstrained heuristic), plus a
ranked-diagnosis query time on the modes_first circuit.

Usage: python bench/scale.py [--budget 60]
"""

from __future__ import annotations

import argparse
import sys
import time

sys.path.insert(0, ".")

from dnnf import SystemModel, iff, xor
from dnnf.formula import Or


def gate(m, name, out, expr, priors=(0.98, 0.01, 0.01)):
    g = m.mode(name, ("ok", "stuck0", "stuck1"), priors=priors)
    m.add((g == "ok") >> iff(out, expr))
    m.add((g == "stuck0") >> ~out)
    m.add((g == "stuck1") >> out)
    return g


def adder(k: int) -> SystemModel:
    m = SystemModel()
    carry = m.bool("c0")
    m.add(~carry)  # carry-in 0
    for i in range(k):
        a = m.bool(f"a{i}")
        b = m.bool(f"b{i}")
        x1 = m.bool(f"x1_{i}")
        a1 = m.bool(f"a1_{i}")
        a2 = m.bool(f"a2_{i}")
        s = m.bool(f"sum{i}")
        cout = m.bool(f"c{i + 1}")
        gate(m, f"gx1_{i}", x1, xor(a, b))
        gate(m, f"gs_{i}", s, xor(x1, carry))
        gate(m, f"ga1_{i}", a1, a & b)
        gate(m, f"ga2_{i}", a2, x1 & carry)
        gate(m, f"go_{i}", cout, Or((a1, a2)))
        carry = cout
    return m


def adder_evidence(k: int):
    """a = all ones, b = 1: expect sum = 0, carry out 1; observe sum0
    stuck wrong (reads 1) -> single-fault diagnoses around bit 0."""
    ev = {}
    for i in range(k):
        ev[f"a{i}"] = True
        ev[f"b{i}"] = i == 0
    for i in range(k):
        ev[f"sum{i}"] = i == 0  # sum0 wrongly reads 1
    ev[f"c{k}"] = True
    return ev


def process(n: int) -> SystemModel:
    m = SystemModel()
    flow = m.bool("flow0")
    m.add(flow)
    for i in range(n):
        pump = m.mode(f"pump{i}", ("ok", "weak", "dead"),
                      priors=(0.96, 0.02, 0.02))
        valve = m.mode(f"valve{i}", ("ok", "stuck_open", "stuck_closed"),
                       priors=(0.96, 0.02, 0.02))
        out = m.bool(f"flow{i + 1}")
        m.add(iff(out, flow & (pump != "dead") & (valve != "stuck_closed")))
        m.sensor(f"fs{i}", out, false_positive=0.02, false_negative=0.02)
        flow = out
    return m


def process_evidence(n: int):
    ev = {f"fs{i}": True for i in range(n)}
    ev[f"fs{n // 2}"] = False  # mid-line sensor reads no-flow
    return ev


def run(family, sizes, make, make_ev, budget: float):
    """Free compilation (MPE diagnoses) is the scaling path; modes_first
    (exact marginal MAP) is tracked until it exceeds the budget once,
    then dropped — its constrained order is known to cost circuit size."""
    print(f"\n== {family} ==")
    hdr = (f"{'size':>5} {'fdvars':>7} | {'free nodes':>10} {'t':>7} "
           f"{'mpe query':>9} | {'modes1st nodes':>14} {'t':>7}")
    print(hdr)
    print("-" * len(hdr))
    mf_ok = True
    for size in sizes:
        t0 = time.time()
        sys_free = make(size).compile(modes_first=False)
        t_free = time.time() - t0
        if t_free > budget:
            print(f"{size:>5}  free compile exceeded budget "
                  f"({t_free:.1f}s); stopping family")
            break
        t0 = time.time()
        diags = sys_free.diagnoses(make_ev(size), k=3)
        t_q = time.time() - t0

        mf_cell = "     (skipped)"
        t_mf_cell = "      "
        if mf_ok:
            t0 = time.time()
            sys_mf = make(size).compile(modes_first=True)
            t_mf = time.time() - t0
            mf_cell = f"{len(sys_mf.circuit):>14}"
            t_mf_cell = f"{t_mf:>6.2f}s"
            # The constrained order can blow up exponentially where the
            # free order stays linear; stop tracking it as soon as it is
            # both slow and far off the free compile's pace.
            if t_mf > 0.5 and t_mf > 5 * max(t_free, 0.05):
                mf_ok = False
        nv = sys_free.circuit.spec.num_vars
        print(f"{size:>5} {nv:>7} | {len(sys_free.circuit):>10} "
              f"{t_free:>6.2f}s {t_q:>8.3f}s | {mf_cell} {t_mf_cell}")
        if diags:
            top = ", ".join(
                f"{k}={v}" for k, v in sorted(diags[0].modes.items())
                if not v.startswith("ok")
            )
            print(f"       top diagnosis: {top or '(all ok)'} "
                  f"p={diags[0].posterior:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=60.0)
    args = ap.parse_args()
    run("ripple-carry adder (5 gates/bit, stuck-at faults)",
        (1, 2, 4, 8, 12, 16, 24, 32), adder, adder_evidence, args.budget)
    run("process line (pump+valve+noisy sensor per stage)",
        (5, 10, 20, 40, 80, 120), process, process_evidence, args.budget)


if __name__ == "__main__":
    main()
