from modenexus import SystemModel, iff
from modenexus.viz import circuit_to_dot, layered_layout, node_label


def small_system():
    m = SystemModel()
    v = m.mode("valve", ("ok", "stuck"), priors=(0.9, 0.1))
    m.add(iff(m.bool("flow"), v == "ok"))
    return m.compile()


def test_dot_contains_all_nodes_and_edges():
    sys = small_system()
    c = sys.circuit
    names = [None] * c.spec.num_vars
    values = [None] * c.spec.num_vars
    for var in sys.vars.values():
        names[var.fd_var] = var.name
        values[var.fd_var] = var.values
    dot = circuit_to_dot(c, names, values)
    assert dot.count("->") == c.num_edges
    assert "valve=stuck" in dot and "flow=True" in dot
    assert dot.strip().startswith("digraph") and dot.strip().endswith("}")


def test_dot_with_node_values():
    sys = small_system()
    from modenexus import fd

    vals = fd.log_values(sys.circuit, sys.log_weights_for({}))
    dot = circuit_to_dot(sys.circuit, node_values=vals)
    assert "\\n" in dot  # annotations present


def test_layout_covers_all_nodes():
    c = small_system().circuit
    pos = layered_layout(c)
    assert set(pos) == set(range(len(c)))
    # Leaves at y=0, root above everything it depends on.
    for i, kind in enumerate(c.kinds):
        if not c.children[i]:
            assert pos[i][1] == 0.0
    for i in range(len(c)):
        for ch in c.children[i]:
            assert pos[i][1] > pos[ch][1]


def test_node_label_kinds():
    c = small_system().circuit
    labels = {node_label(c, i) for i in range(len(c))}
    assert "AND" in labels and "OR" in labels
