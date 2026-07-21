# Design: a DNNF compilation and reasoning library

## 1. Motivation

This library reconstructs — and extends toward GPUs — the reasoning
architecture used in DNNF-based model-based diagnosis at NASA JPL (see NTRS
20060043135): a system description (components with operational modes,
sensors for observables, expectations between them) is encoded into
propositional logic, compiled **offline** into a *small* Decomposable
Negation Normal Form circuit, and then queried **online** with guaranteed
polytime operations. The online engine returns complete leaf
interpretations (system states) ordered from most to least probable, where
per-leaf weights are interpreted as **negative log probabilities** — an
observation that turns the diagnosis engine into exact probabilistic
inference over a tractable circuit.

Two facts make the architecture attractive:

1. **All the hard work happens once.** Compilation is #P-hard in general,
   but it is done offline, and the compiler uses search + heuristics to find
   a *small* (not minimal — minimality is intractable) circuit.
2. **Online queries are linear in circuit size** and are simple feed-forward
   sweeps — exactly the shape of computation GPUs are good at.

## 2. Background

### 2.1 The NNF property lattice

An NNF circuit is a rooted DAG with literal leaves and AND/OR internal
nodes. Tractability comes from structural properties:

| Property | Definition | Buys you |
|---|---|---|
| **Decomposability** (DNNF) | AND-children mention disjoint variable sets | SAT, min-cost model, projection, enumeration in linear time |
| **Determinism** (d-DNNF) | OR-children pairwise logically inconsistent | model counting, weighted model counting |
| **Smoothness** | OR-children mention identical variable sets | makes counting semiring sweeps correct; enforceable in near-linear time |
| **Decision** | OR nodes have form `(v ∧ α) ∨ (¬v ∧ β)` | what DPLL-trace compilers emit; implies determinism |

Key references: Darwiche, *Decomposable Negation Normal Form* (JACM 2001);
Darwiche & Marquis, *A Knowledge Compilation Map* (JAIR 2002); Darwiche,
*A Differential Approach to Inference in Bayesian Networks* (JACM 2003);
Muise et al., *dsharp* (2012); Lagniez & Marquis, *D4* (IJCAI 2017).

### 2.2 Why every query is a semiring sweep

A smooth DNNF evaluated bottom-up with OR ↦ ⊕ and AND ↦ ⊗ over a
commutative semiring computes ⊕ over models of ⊗ over literal weights
(determinism required when ⊕ is not idempotent):

| Semiring | ⊕, ⊗ | Query |
|---|---|---|
| Boolean | or, and | consistency (SAT) |
| Counting | +, × | model counting |
| Real | +, × | weighted model counting (WMC) |
| Log | logsumexp, + | log-space WMC (numerically stable) |
| Tropical / min-sum | min, + | MPE over neg-log costs |

The neg-log-probability reading: give literal `l` weight `-log P(l)`. Then
min-sum evaluation yields the cost of the most probable model; the arg-min
traversal decodes it; and k-best enumeration yields models most-probable
first. This is precisely the "leaf weights are neglog probabilities"
interpretation of the JPL engine, made formal.

## 3. Architecture

```
formula.py     propositional AST  ──Tseitin──►  cnf.py (DIMACS)
                                                    │
                                             compiler.py  (offline)
                                                    │  exhaustive DPLL +
                                                    │  components + caching
                                                    ▼
                                  circuit.py  decision-DNNF (arrays, DAG)
                                       │  smooth() / condition()
        ┌──────────────┬───────────────┼─────────────────┬──────────────┐
        ▼              ▼               ▼                 ▼              ▼
    eval.py        kbest.py     torch_backend.py     nnf_io.py    diagnosis.py
  semiring sweeps  ordered      layered batched      c2d .nnf     SystemModel /
  (SAT/#/WMC/MPE)  models       GPU eval + autograd  interop      ranked diagnoses
```

### 3.1 Circuit representation (`circuit.py`)

Nodes live in flat parallel arrays (`kinds`, `lits`, `children`) in
topological order — children strictly precede parents. Construction goes
through `CircuitBuilder`, which hash-conses nodes (structural sharing is
what makes compiled circuits DAGs rather than trees) and applies algebraic
simplifications (unit AND/OR collapse, constant absorption, AND
flattening). ORs are never flattened, since merging OR nodes can silently
destroy determinism.

The array form serves both backends: the pure-Python evaluators do a single
forward loop, and the torch backend lowers the same arrays to a layered
tensor program.

### 3.2 Compilation (`compiler.py`)

Exhaustive DPLL with the three classic ingredients of c2d/dsharp/D4-style
model counters, each mapping to a circuit construct:

* **Unit propagation** → implied literals become AND conjuncts.
* **Connected-component decomposition** of the residual clause set →
  decomposable AND nodes (children over disjoint variables by
  construction).
* **Branching** `(v ∧ Δ|v) ∨ (¬v ∧ Δ|¬v)` → deterministic decision-OR
  nodes.
* **Component caching** keyed on the residual clause set (literal-level
  canonical form, which is sound: identical residuals compile identically)
  → sub-circuit reuse across branches. This cache is why the output is a
  compact DAG.

Circuit size is governed by the branching heuristic — this is the "search
and heuristics to find a small form" knob. v1 ships a dynamic
most-occurrences heuristic plus support for user-supplied static orders;
better orders (min-fill over the primal graph, hypergraph partitioning /
dtree-driven as in c2d) are the highest-leverage future work.

For problems beyond the pure-Python compiler, `nnf_io.py` reads the c2d
`.nnf` format, so circuits produced by c2d/dsharp/D4 (d4 can emit
compatible output) drop into the same evaluators.

### 3.3 Queries (`eval.py`, `kbest.py`)

`eval.py` implements the semiring sweeps with a guard that counting-style
queries only run on verified smooth d-DNNF (smoothness + syntactic
determinism + full variable coverage at the root, cached per circuit).
`mpe()` adds the standard arg-min back-traversal to decode a minimizing
model.

`kbest.py` enumerates models in nondecreasing cost order using the lazy
k-best hypergraph scheme (Huang & Chiang 2005): OR nodes lazily heap-merge
children's ranked streams; AND nodes lazily explore the monotone product
frontier. Decomposability makes AND-combination consistent; determinism
makes the stream duplicate-free (a root-level dedup guards arbitrary
inputs). Cost `+inf` (annihilated by evidence) terminates the stream, so
conditioning composes with enumeration for free.

### 3.4 GPU evaluation (`torch_backend.py`)

A smooth d-DNNF *is* an arithmetic circuit, so evaluation lowers to a
layered tensor program:

1. Nodes renumbered by depth (longest path from leaves); each depth is one
   layer, ANDs before ORs within a layer.
2. Per layer: `index_select` gathers child values from the computed prefix;
   `scatter_add` reduces AND segments; OR segments use `scatter_reduce
   (amin)` for min-sum or a max-shifted segment log-sum-exp for log-space
   WMC (the max is detached, so autograd sees the exact softmax gradient).
3. A batch dimension carries B weight vectors — B evidence sets, B
   parameterizations, B sensor frames — through one pass.

**Marginals for free:** on the `logprob` semiring,
`∂ log WMC / ∂ log w(l) = P(l | evidence)` (Darwiche's differential
semantics). `TorchCircuit.marginals()` is literally one forward + one
`backward()`: every posterior literal marginal, batched, on GPU. This is
the main payoff of the PyTorch choice beyond raw speed — the circuit
composes with anything differentiable (neural observation models producing
the leaf weights, learned priors, end-to-end training through exact
inference, integration with other torch-based libraries).

`neglog` forward + per-row decode gives batched MPE.

### 3.5 Diagnosis (`diagnosis.py`, `formula.py`)

The JPL-style workflow, made a first-class API:

* `SystemModel` declares **mode variables** (finite domains, one-hot
  encoded with exactly-one constraints, priors attached to positive
  literals — the standard literal-weighted WMC encoding of discrete
  distributions), **observables**, and propositional constraints built
  with a small formula AST.
* Constraints are Tseitin-encoded with *biconditional* clauses, so
  auxiliary variables are functionally determined by the originals: model
  counts project correctly and neutral-weight auxiliaries never perturb
  WMC or MPE.
* `compile()` runs the DNNF compiler once; the result answers:
  - `diagnoses(evidence, k)` — k-best enumeration under neg-log costs with
    evidence annihilation, projected to distinct mode assignments,
    normalized by `log P(evidence)`: ranked leaf interpretations, most
    probable first;
  - `mode_posteriors(evidence)` — exact `P(mode=value | evidence)` by WMC
    ratios;
  - `log_evidence(evidence)`.

Semantic note, documented in the API: `diagnoses()` ranks mode assignments
by their best supporting complete state (MPE semantics), not by the summed
posterior of the mode assignment (marginal MAP, which is harder — it
requires a circuit whose OR-decisions on mode variables dominate the
non-mode variables, i.e. constrained vtrees). `mode_posteriors()` provides
the exact summed marginals as a complement.

## 4. Correctness strategy

Every nontrivial path is cross-validated against brute-force model
enumeration on randomized small instances (seeded):

* compiled circuits: decomposable + deterministic + smooth, model counts
  equal brute-force counts (30 random CNFs, plus static-order variants);
* WMC / log-WMC / MPE values and decoded models vs. exhaustive weighted
  enumeration;
* k-best streams: exact cost-sequence equality with the sorted brute-force
  model list, validity and distinctness of every model;
* torch backend vs. CPU evaluators and vs. brute force, including batched
  rows, evidence conditioning, gradient marginals vs. enumerated
  posteriors;
* diagnosis: closed-form posterior checks on a two-valve system.

## 5. Roadmap

Ordered by leverage:

1. **Better branching heuristics / dtrees** — min-fill and hypergraph
   partitioning static orders; VSADS-style dynamic scoring. This is the
   dominant factor in circuit size.
2. **Scale the compiler** — iterative (non-recursive) search, watched
   literals, clause-index component keys, sat-solver-backed failed-literal
   pruning; or lean on external compilers via `.nnf` interop (already
   supported) and treat the Python compiler as the reference
   implementation.
3. **Marginal MAP over modes** — constrained decision order putting mode
   variables above the rest, enabling summed-posterior diagnosis ranking.
4. **Torch-native k-best** — batched beam/top-k semiring (each node carries
   its k best costs as a tensor slice) for GPU-ranked diagnoses.
5. **Learning** — leaf weights as `nn.Parameter`; train priors/observation
   models end-to-end through exact WMC (the circuit is already
   differentiable).
6. **Incremental evidence** — reuse of layer prefixes across evidence
   updates that touch few literals.
7. **Projected compilation / existential quantification** on DNNF for
   observable-only views.
