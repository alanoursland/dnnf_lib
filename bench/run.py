"""Benchmark harness: compare compilation heuristics across instance families.

Usage:
    python bench/run.py [--families random,chain,grid,php,diag] [--timeout 60]

Reports circuit size (nodes/edges), compile time, and a query time per
heuristic.  Circuit size is the number that matters: every online query is
linear in it.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from typing import Callable, Dict, List, Optional, Tuple

sys.path.insert(0, ".")

from neximode import CNF, compile_cnf, model_count
from neximode.compiler import minfill_order


# ----------------------------------------------------------------------
# Instance families
# ----------------------------------------------------------------------
def random_3cnf(n: int, ratio: float, seed: int) -> CNF:
    rng = random.Random(seed)
    cnf = CNF(num_vars=n)
    for _ in range(int(n * ratio)):
        vs = rng.sample(range(1, n + 1), 3)
        cnf.add_clause([v if rng.random() < 0.5 else -v for v in vs])
    return cnf


def chain(n: int) -> CNF:
    cnf = CNF(num_vars=n)
    for v in range(1, n):
        cnf.add_clause((-v, v + 1))
    return cnf


def grid(rows: int, cols: int, seed: int) -> CNF:
    """Grid-structured implications: bounded treewidth, where min-fill
    should shine over the dynamic heuristic."""
    rng = random.Random(seed)
    cnf = CNF(num_vars=rows * cols)

    def var(r: int, c: int) -> int:
        return r * cols + c + 1

    for r in range(rows):
        for c in range(cols):
            v = var(r, c)
            if c + 1 < cols:
                cnf.add_clause((-v if rng.random() < 0.5 else v, var(r, c + 1)))
            if r + 1 < rows:
                cnf.add_clause((-v if rng.random() < 0.5 else v, var(r + 1, c)))
    return cnf


def pigeonhole(pigeons: int, holes: int) -> CNF:
    """PHP(p, h): unsat when p > h; classically hard for resolution-style
    reasoning and a stress test for the cache."""
    cnf = CNF(num_vars=pigeons * holes)

    def var(p: int, h: int) -> int:
        return p * holes + h + 1

    for p in range(pigeons):
        cnf.add_clause([var(p, h) for h in range(holes)])
    for h in range(holes):
        for p1 in range(pigeons):
            for p2 in range(p1 + 1, pigeons):
                cnf.add_clause((-var(p1, h), -var(p2, h)))
    return cnf


def diagnosis_chain(num_components: int):
    """A serial system: component i (3 modes) passes its input to i+1
    unless stuck_closed; per-stage flow sensors.  Structurally similar to
    compiled system models: local interactions, long chains.  Returns an
    FDCnf (native multi-valued)."""
    from neximode import fd
    from neximode.diagnosis import SystemModel
    from neximode.formula import iff

    m = SystemModel()
    flows = [m.bool("flow0")]
    m.add(flows[0])
    for i in range(num_components):
        v = m.mode(
            f"c{i}", ("ok", "stuck_open", "stuck_closed"),
            priors=(0.98, 0.01, 0.01),
        )
        out = m.bool(f"flow{i + 1}")
        m.add(iff(out, flows[-1] & (v != "stuck_closed")))
        flows.append(out)
    fd.encode(m._constraints, m.cnf)
    return m.cnf


FAMILIES: Dict[str, List[Tuple[str, Callable[[], CNF]]]] = {
    "random": [
        (f"random3cnf n=40 r=2.0 s={s}", lambda s=s: random_3cnf(40, 2.0, s))
        for s in (1, 2, 3)
    ],
    "chain": [("chain n=300", lambda: chain(300))],
    "grid": [
        ("grid 6x6 s=1", lambda: grid(6, 6, 1)),
        ("grid 8x8 s=1", lambda: grid(8, 8, 1)),
        ("grid 10x10 s=1", lambda: grid(10, 10, 1)),
    ],
    "php": [("php 6->5", lambda: pigeonhole(6, 5))],
    "diag": [
        ("diag-chain 10", lambda: diagnosis_chain(10)),
        ("diag-chain 25", lambda: diagnosis_chain(25)),
    ],
}

HEURISTICS = ("dynamic", "minfill")


def run_one(name: str, cnf, heuristic: str) -> Optional[Dict[str, object]]:
    from neximode import fd

    if isinstance(cnf, fd.FDCnf):
        t0 = time.time()
        circuit = fd.compile_fd(cnf, smooth=True, heuristic=heuristic)
        compile_s = time.time() - t0
        t0 = time.time()
        count = fd.model_count(circuit)
        query_s = time.time() - t0
        return {
            "instance": name,
            "heuristic": heuristic,
            "nodes": len(circuit),
            "edges": circuit.num_edges,
            "compile_s": compile_s,
            "query_s": query_s,
            "count": count,
        }
    t0 = time.time()
    circuit = compile_cnf(cnf, smooth=True, heuristic=heuristic)
    compile_s = time.time() - t0
    t0 = time.time()
    count = model_count(circuit)
    query_s = time.time() - t0
    return {
        "instance": name,
        "heuristic": heuristic,
        "nodes": len(circuit),
        "edges": circuit.num_edges,
        "compile_s": compile_s,
        "query_s": query_s,
        "count": count,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default=",".join(FAMILIES))
    args = ap.parse_args()

    rows: List[Dict[str, object]] = []
    for fam in args.families.split(","):
        for name, make in FAMILIES[fam.strip()]:
            cnf = make()
            counts = set()
            for h in HEURISTICS:
                r = run_one(name, cnf, h)
                if r is None:
                    continue
                rows.append(r)
                counts.add(r["count"])
            assert len(counts) == 1, f"count mismatch on {name}: {counts}"

    hdr = f"{'instance':28} {'heur':8} {'nodes':>8} {'edges':>9} {'compile':>8} {'query':>7}  count"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['instance']:28} {r['heuristic']:8} {r['nodes']:>8} "
            f"{r['edges']:>9} {r['compile_s']:>7.2f}s {r['query_s']:>6.3f}s  {r['count']}"
        )


if __name__ == "__main__":
    main()
