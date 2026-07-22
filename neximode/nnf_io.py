"""Read/write circuits in the c2d ``.nnf`` text format.

This is the de-facto interchange format emitted by c2d and dsharp, letting
this library evaluate circuits produced by industrial-strength external
compilers::

    nnf <num_nodes> <num_edges> <num_vars>
    L <lit>                        (literal leaf)
    A <count> <child...>           (AND; "A 0" is TRUE)
    O <decision_var> <count> <child...>   (OR; "O 0 0" is FALSE)

Node ids are 0-based line order; children refer to earlier lines.
"""

from __future__ import annotations

import io
from typing import List, Tuple

from .circuit import AND, FALSE, LIT, OR, TRUE, Circuit


def loads(text: str) -> Circuit:
    return load(io.StringIO(text))


def load(source) -> Circuit:
    if isinstance(source, str):
        with open(source, "r") as f:
            return load(f)
    header = None
    kinds: List[int] = []
    lits: List[int] = []
    children: List[Tuple[int, ...]] = []
    num_vars = 0
    for line in source:
        line = line.strip()
        if not line or line.startswith("c"):
            continue
        parts = line.split()
        if parts[0] == "nnf":
            header = (int(parts[1]), int(parts[2]), int(parts[3]))
            num_vars = header[2]
            continue
        if parts[0] == "L":
            lit = int(parts[1])
            kinds.append(LIT)
            lits.append(lit)
            children.append(())
            num_vars = max(num_vars, abs(lit))
        elif parts[0] == "A":
            count = int(parts[1])
            kids = tuple(int(x) for x in parts[2 : 2 + count])
            if count == 0:
                kinds.append(TRUE)
                lits.append(0)
                children.append(())
            else:
                kinds.append(AND)
                lits.append(0)
                children.append(kids)
        elif parts[0] == "O":
            count = int(parts[2])
            kids = tuple(int(x) for x in parts[3 : 3 + count])
            if count == 0:
                kinds.append(FALSE)
                lits.append(0)
                children.append(())
            else:
                kinds.append(OR)
                lits.append(0)
                children.append(kids)
        else:
            raise ValueError(f"unrecognized .nnf line: {line!r}")
    if not kinds:
        raise ValueError("empty .nnf input")
    return Circuit(num_vars, kinds, lits, children, root=len(kinds) - 1)


def dumps(circuit: Circuit) -> str:
    """Serialize; the root must be the last node (true for compiler output)."""
    lines = [
        f"nnf {len(circuit)} {circuit.num_edges} {circuit.num_vars}"
    ]
    order = list(range(len(circuit)))
    if circuit.root != len(circuit) - 1:
        # Re-topologize so the root is last.
        order = _topo_with_root_last(circuit)
    remap = {old: new for new, old in enumerate(order)}
    for old in order:
        kind = circuit.kinds[old]
        if kind == LIT:
            lines.append(f"L {circuit.lits[old]}")
        elif kind == TRUE:
            lines.append("A 0")
        elif kind == FALSE:
            lines.append("O 0 0")
        elif kind == AND:
            kids = " ".join(str(remap[c]) for c in circuit.children[old])
            lines.append(f"A {len(circuit.children[old])} {kids}")
        else:
            kids = " ".join(str(remap[c]) for c in circuit.children[old])
            lines.append(f"O 0 {len(circuit.children[old])} {kids}")
    return "\n".join(lines) + "\n"


def dump(circuit: Circuit, path: str) -> None:
    with open(path, "w") as f:
        f.write(dumps(circuit))


def _topo_with_root_last(circuit: Circuit) -> List[int]:
    order: List[int] = []
    visited = [False] * len(circuit)

    def visit(i: int) -> None:
        stack = [(i, False)]
        while stack:
            node, done = stack.pop()
            if done:
                order.append(node)
                continue
            if visited[node]:
                continue
            visited[node] = True
            stack.append((node, True))
            for c in circuit.children[node]:
                if not visited[c]:
                    stack.append((c, False))

    for i in range(len(circuit)):
        if i != circuit.root:
            visit(i)
    visit(circuit.root)
    # Nodes not ancestors of root may appear after; keep root truly last by
    # dropping unreachable nodes.
    reachable = set()
    stack = [circuit.root]
    reachable.add(circuit.root)
    while stack:
        n = stack.pop()
        for c in circuit.children[n]:
            if c not in reachable:
                reachable.add(c)
                stack.append(c)
    return [n for n in order if n in reachable]
