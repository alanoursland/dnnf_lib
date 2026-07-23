"""The FD compiler must not depend on Python recursion depth."""

import sys

from modenexus import fd


def test_deep_chain_compiles_under_tiny_recursion_limit():
    n = 800
    cnf = fd.FDCnf()
    for _ in range(n):
        cnf.spec.add_var(2)
    for i in range(n - 1):
        cnf.add_clause([(i, {0}), (i + 1, {1})])  # v_i=1 -> v_{i+1}=1

    old = sys.getrecursionlimit()
    sys.setrecursionlimit(100)
    try:
        circuit = fd.compile_fd(cnf, smooth=True)
    finally:
        sys.setrecursionlimit(old)
    # Monotone chain: models are 0^k 1^(n-k) for k = 0..n.
    assert fd.model_count(circuit) == n + 1
