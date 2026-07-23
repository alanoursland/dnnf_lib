"""Circuit visualization helpers (no hard dependencies).

Two utilities used by the tutorial's generated figures and handy for
debugging small circuits:

* :func:`circuit_to_dot` — emit Graphviz DOT text for an
  :class:`modenexus.fd.FDCircuit` (render with any dot tool, or just read it).
* :func:`layered_layout` — pure-Python (x, y) positions, one row per
  depth layer, for plotting with matplotlib without graphviz installed.

Both are meant for *small* circuits (tens of nodes) — the ones you can
learn from by looking.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from .circuit import AND, FALSE, LIT, OR, TRUE


def leaf_label(
    circuit,
    node: int,
    var_names: Optional[Sequence[str]] = None,
    value_names: Optional[Sequence[Sequence]] = None,
) -> str:
    var, val = circuit.spec.decode(circuit.lits[node])
    name = var_names[var] if var_names else f"x{var}"
    value = value_names[var][val] if value_names else val
    return f"{name}={value}"


def node_label(circuit, node, var_names=None, value_names=None) -> str:
    kind = circuit.kinds[node]
    if kind == LIT:
        return leaf_label(circuit, node, var_names, value_names)
    return {AND: "AND", OR: "OR", TRUE: "true", FALSE: "false"}[kind]


def circuit_to_dot(
    circuit,
    var_names: Optional[Sequence[str]] = None,
    value_names: Optional[Sequence[Sequence]] = None,
    node_values: Optional[Sequence] = None,
) -> str:
    """Graphviz DOT for an FD circuit; ``node_values`` (e.g. a semiring
    sweep from :func:`modenexus.fd.log_values`) annotates every node."""
    lines = [
        "digraph circuit {",
        "  rankdir=BT;",
        '  node [fontname="Helvetica"];',
    ]
    for i in range(len(circuit)):
        kind = circuit.kinds[i]
        label = node_label(circuit, i, var_names, value_names)
        if node_values is not None:
            v = node_values[i]
            label += f"\\n{v:.3g}" if isinstance(v, float) else f"\\n{v}"
        shape = "box" if kind == LIT else "ellipse"
        style = ' style=filled fillcolor="#eef4ff"' if kind == OR else (
            ' style=filled fillcolor="#fff4e6"' if kind == AND else ""
        )
        lines.append(f'  n{i} [label="{label}" shape={shape}{style}];')
    for i in range(len(circuit)):
        for c in circuit.children[i]:
            lines.append(f"  n{c} -> n{i};")
    lines.append("}")
    return "\n".join(lines)


def layered_layout(circuit) -> Dict[int, Tuple[float, float]]:
    """(x, y) positions: y = depth (leaves at 0), x spreads each layer."""
    n = len(circuit)
    depth = [0] * n
    for i in range(n):
        ch = circuit.children[i]
        depth[i] = 1 + max((depth[c] for c in ch), default=-1)
    layers: Dict[int, List[int]] = {}
    for i in range(n):
        layers.setdefault(depth[i], []).append(i)
    pos: Dict[int, Tuple[float, float]] = {}
    width = max(len(nodes) for nodes in layers.values())
    for d, nodes in layers.items():
        step = width / (len(nodes) + 1)
        for k, i in enumerate(nodes):
            pos[i] = ((k + 1) * step, float(d))
    return pos
