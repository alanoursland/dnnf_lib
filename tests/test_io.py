import random

from dnnf import CNF, compile_cnf, model_count, nnf_io
from helpers import random_cnf


def test_dimacs_round_trip():
    cnf = CNF(num_vars=4, clauses=[(1, -2), (3, 4), (-1,)])
    text = cnf.to_dimacs()
    back = CNF.from_dimacs(text)
    assert back.num_vars == 4
    assert back.clauses == cnf.clauses


def test_dimacs_parse_with_comments():
    text = "c a comment\np cnf 3 2\n1 -2 0\n2 3 0\n"
    cnf = CNF.from_dimacs(text)
    assert cnf.num_vars == 3
    assert cnf.clauses == [(1, -2), (2, 3)]


def test_nnf_round_trip_preserves_semantics():
    rng = random.Random(7)
    cnf = random_cnf(rng, 6, 10)
    circuit = compile_cnf(cnf, smooth=True)
    text = nnf_io.dumps(circuit)
    back = nnf_io.loads(text)
    assert back.num_vars == circuit.num_vars
    assert model_count(back) == model_count(circuit)


def test_nnf_constants():
    unsat = compile_cnf(CNF(num_vars=1, clauses=[(1,), (-1,)]))
    text = nnf_io.dumps(unsat)
    back = nnf_io.loads(text)
    assert back.kinds[back.root] == 0  # FALSE
