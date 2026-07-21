"""Drivers for external d-DNNF compilers (boolean CNF path).

For instances beyond the pure-Python compilers, shell out to an
installed industrial compiler and load its output through
:mod:`dnnf.nnf_io`.  Currently supported: **c2d** (Darwiche), whose
output is the `.nnf` format we already parse.  A D4 driver requires a
parser for its arc-based output format and is left as future work.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

from . import nnf_io
from .circuit import Circuit
from .cnf import CNF


def compile_with_c2d(
    cnf: CNF, binary: str = "c2d", timeout: float = 600.0
) -> Circuit:
    """Compile via an installed c2d binary; returns the parsed circuit.
    Raises FileNotFoundError if the binary is absent."""
    if shutil.which(binary) is None:
        raise FileNotFoundError(
            f"{binary!r} not found on PATH; install c2d or use compile_fd"
        )
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "input.cnf")
        with open(path, "w") as f:
            f.write(cnf.to_dimacs())
        subprocess.run(
            [binary, "-in", path], check=True, timeout=timeout,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        out = path + ".nnf"
        circuit = nnf_io.load(out)
    if circuit.num_vars < cnf.num_vars:
        circuit = Circuit(
            cnf.num_vars, circuit.kinds, circuit.lits,
            circuit.children, circuit.root,
        )
    return circuit
