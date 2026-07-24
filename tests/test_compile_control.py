import pytest

from modenexus import (
    CNF,
    CompilationBudgetExceeded,
    CompilationCancelled,
    CompileControl,
    Planner,
    SystemModel,
    compile_cnf,
    fd,
)


def test_boolean_compile_reports_final_stats():
    snapshots = []
    circuit = compile_cnf(
        CNF(3, [(1, 2), (-1, 3)]),
        smooth=True,
        control=CompileControl(
            progress=snapshots.append, progress_interval=0.0
        ),
    )
    final = snapshots[-1]
    assert final.complete
    assert final.nodes == len(circuit)
    assert final.decisions > 0
    assert final.cache_entries > 0


def test_compile_can_be_cancelled_cooperatively():
    with pytest.raises(CompilationCancelled) as exc:
        compile_cnf(
            CNF(2, [(1, 2)]),
            control=CompileControl(cancel=lambda: True),
        )
    assert not exc.value.stats.complete
    assert exc.value.stats.nodes == 0


def test_fd_compile_honors_node_budget():
    cnf = fd.FDCnf()
    for _ in range(5):
        cnf.spec.add_var(3)
    cnf.add_clause((v, {0, 1}) for v in range(5))
    with pytest.raises(CompilationBudgetExceeded) as exc:
        fd.compile_fd(cnf, control=CompileControl(max_nodes=2))
    assert exc.value.stats.nodes > 2


def test_high_level_compilers_forward_controls():
    m = SystemModel()
    m.mode("mode", ("ok", "bad"), (0.9, 0.1))
    with pytest.raises(CompilationCancelled):
        m.compile(control=CompileControl(cancel=lambda: True))

    planner = Planner()
    planner.mode("mode", ("ok", "bad"))
    planner.transition("mode", "ok", "bad")
    with pytest.raises(CompilationCancelled):
        planner.compile(1, control=CompileControl(cancel=lambda: True))
