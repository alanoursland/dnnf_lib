"""dnnf: compilation of propositional theories to Decomposable Negation
Normal Form, tractable weighted reasoning, ordered model enumeration, and
model-based diagnosis — with an optional PyTorch backend for batched
GPU evaluation.

Typical offline/online split::

    import dnnf

    cnf = dnnf.CNF.from_dimacs("theory.cnf")
    circuit = dnnf.compile_cnf(cnf, smooth=True)   # offline, once

    dnnf.model_count(circuit)                      # online queries
    dnnf.wmc(circuit, weights)
    cost, best = dnnf.mpe(circuit, costs)
    for cost, model in dnnf.enumerate_models(circuit, costs, k=10):
        ...

GPU evaluation (requires ``pip install dnnf-lib[torch]``)::

    from dnnf.torch_backend import TorchCircuit
    tc = TorchCircuit(circuit, semiring="logprob", device="cuda")
    log_z = tc(weights)          # (B,) batched log-WMC
    marg = tc.marginals(weights) # (B, 2n) posterior literal marginals
"""

from .circuit import Circuit, CircuitBuilder, lit_index
from .cnf import CNF
from .compiler import compile_cnf, minfill_order
from .diagnosis import CompiledSystem, Diagnosis, SystemModel
from .eval import (
    condition_weights,
    costs_from_probs,
    is_satisfiable,
    log_wmc,
    model_count,
    mpe,
    uniform_weights,
    weights_from_probs,
    wmc,
)
from .formula import And, Formula, Not, Or, Prop, at_most_one, exactly_one, iff, xor
from .kbest import enumerate_map, enumerate_models
from .tracking import ModeTracker
from . import nnf_io

__version__ = "0.1.0"

__all__ = [
    "CNF",
    "Circuit",
    "CircuitBuilder",
    "CompiledSystem",
    "Diagnosis",
    "SystemModel",
    "And",
    "Formula",
    "Not",
    "Or",
    "Prop",
    "at_most_one",
    "exactly_one",
    "iff",
    "xor",
    "compile_cnf",
    "condition_weights",
    "costs_from_probs",
    "enumerate_map",
    "enumerate_models",
    "minfill_order",
    "ModeTracker",
    "is_satisfiable",
    "lit_index",
    "log_wmc",
    "model_count",
    "mpe",
    "nnf_io",
    "uniform_weights",
    "weights_from_probs",
    "wmc",
]
