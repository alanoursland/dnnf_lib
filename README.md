# dnnf_lib

Compilation of propositional theories to **Decomposable Negation Normal
Form (DNNF)**, tractable weighted reasoning on the compiled circuits, and
model-based diagnosis — with an optional **PyTorch backend** for batched
GPU evaluation and autograd-powered marginals.

The architecture follows the knowledge-compilation approach used in
DNNF-based diagnosis at NASA JPL: encode a system model (component modes
with priors, observables, expectations) into logic, compile **once**
offline into a small circuit, then answer observation queries **online** in
time linear in circuit size — including enumerating complete system states
ordered from most to least probable, with leaf weights read as negative
log probabilities. See [docs/DESIGN.md](docs/DESIGN.md) for the full
technical treatment and [docs/FUTURE_WORK.md](docs/FUTURE_WORK.md) for the
roadmap from next steps to blue sky.

## Install

```bash
pip install -e .            # core: pure Python, no dependencies
pip install -e .[torch]     # + PyTorch backend
pip install -e .[dev]       # + pytest
```

## Quick start: compile and query

```python
import dnnf

cnf = dnnf.CNF(num_vars=3, clauses=[(1, 2), (-1, 3)])
circuit = dnnf.compile_cnf(cnf, smooth=True)   # decision-DNNF: decomposable,
                                               # deterministic, smooth
dnnf.model_count(circuit)                      # 4
dnnf.wmc(circuit, dnnf.weights_from_probs(3, {1: 0.9, 2: 0.5, 3: 0.5}))

# MPE / most probable model under neg-log costs
costs = dnnf.costs_from_probs(3, {1: 0.9, 2: 0.5, 3: 0.5})
cost, best = dnnf.mpe(circuit, costs)

# Models ordered most-probable-first (lazy k-best)
for cost, model in dnnf.enumerate_models(circuit, costs, k=5):
    print(cost, model)
```

DIMACS CNF (`dnnf.CNF.from_dimacs`) and the c2d `.nnf` circuit format
(`dnnf.nnf_io`) are supported, so circuits from external compilers
(c2d, dsharp, D4) plug into the same evaluators.

## Diagnosis: modes, priors, ranked explanations

```python
from dnnf import SystemModel, iff

m = SystemModel()
v1 = m.mode("valve1", ("ok", "stuck_open", "stuck_closed"), priors=(0.98, 0.01, 0.01))
v2 = m.mode("valve2", ("ok", "stuck_open", "stuck_closed"), priors=(0.98, 0.01, 0.01))
flow1, flow2 = m.bool("flow1"), m.bool("flow2")
m.add(iff(flow1, v1 != "stuck_closed"))
m.add(iff(flow2, v2 != "stuck_closed"))

system = m.compile()                                   # offline
for d in system.diagnoses({"flow1": False, "flow2": True}, k=3):  # online
    print(d)                # ranked by best supporting state (MPE)
for d in system.map_diagnoses({"flow1": False}, k=3):
    print(d)                # ranked by exact summed posterior (marginal MAP)
system.mode_posteriors({"flow1": False})   # exact P(mode=value | evidence)
```

`map_diagnoses` is exact marginal MAP — mode variables are branched first
during compilation (`modes_first=True`, the default), which constrains the
circuit so that max-over-modes / sum-over-everything-else is a single
sweep plus lazy k-best enumeration.

The diagnosis stack runs on a **native finite-domain core** (`dnnf.fd`):
variables carry their domains directly, circuit leaves are atomic
assignments like `valve1=stuck_closed`, and decision nodes branch d-ways —
no one-hot encoding, no exactly-one clauses, ~40% smaller circuits on
mode-heavy models than the boolean lowering. Continuous quantities can be
**quantized into bounded ranges** with threshold atoms and automatic
bucketing of numeric evidence:

```python
level = m.quantized("level", (0.0, 10.0, 50.0, 100.0), priors=(0.2, 0.5, 0.3))
m.sensor("low_alarm", level.below(10.0))
m.add((leak == "large") >> level.below(10.0))
sys.log_evidence({"level": 37.2})     # numeric evidence, bucketed for you
```

Compiled systems are also **generative and learnable**:

```python
state = sys.sample_state(rng)            # exact simulation from the model
telemetry = [{"alarm": sys.sample_state(rng)["alarm"]} for _ in range(4000)]
sys.fit_priors(telemetry)                # EM: learn failure rates from
                                         # partially observed telemetry
sys.posteriors(evidence, names=[...])    # exact marginals for any variable
```

`fit_priors` is exact expectation-maximization on the circuit (E-step:
WMC-ratio posteriors; M-step: average), with a guaranteed non-decreasing
likelihood — failure priors estimated from fleet data instead of
engineering guesses, with the logical model as a hard constraint.

Sensors can be noisy (`m.sensor("alarm", expr, false_positive=0.1,
false_negative=0.2)`), and `dnnf.ModeTracker` runs the monitoring loop:
per-mode transition matrices, observations each timestep, and a
beam-filtered belief over joint mode assignments (exact HMM filtering when
the beam covers the mode space — see `examples/home_battery.py` for a
degrading-battery week of telemetry, and `docs/MODELING_NOTES.md` for
ergonomics findings from that experiment).

## GPU / batched evaluation (PyTorch)

```python
import torch
from dnnf.torch_backend import TorchCircuit

tc = TorchCircuit(circuit, semiring="logprob", device="cuda")
w = tc.weights_from_probs({1: 0.9}, batch=1024)   # (B, 2n) log-weights
log_z = tc(tc.condition(w, {2: True}))            # batched log-WMC
marginals = tc.marginals(w)                       # (B, 2n) posteriors:
                                                  # one backward pass computes
                                                  # every P(lit | evidence)

mp = TorchCircuit(circuit, semiring="neglog")     # min-sum semiring
costs, states = mp.mpe(cost_tensor)               # batched MPE
```

Because evaluation is ordinary autograd-friendly tensor code, the circuit
composes with other PyTorch models — e.g. neural observation models
producing leaf weights, trained end-to-end through exact inference.

## Layout

| Module | Contents |
|---|---|
| `dnnf.cnf` | CNF container, DIMACS I/O |
| `dnnf.formula` | propositional AST, Tseitin encoding |
| `dnnf.compiler` | CNF → decision-DNNF (DPLL + components + caching) |
| `dnnf.circuit` | circuit arrays, property checks, smooth/condition |
| `dnnf.eval` | semiring sweeps: SAT, counting, WMC, log-WMC, MPE |
| `dnnf.kbest` | lazy ordered model enumeration |
| `dnnf.torch_backend` | layered batched tensor evaluation, marginals |
| `dnnf.fd` | native finite-domain core: FD-CNF, compiler, queries, MAP |
| `dnnf.diagnosis` | `SystemModel` / ranked diagnoses / posteriors / quantized vars |
| `dnnf.tracking` | `ModeTracker` temporal filtering |
| `dnnf.nnf_io` | c2d `.nnf` interop |

## Tests

```bash
python -m pytest
```

All evaluators, the compiler, enumeration, the torch backend, and the
diagnosis layer are cross-validated against brute-force model enumeration
on randomized instances.
