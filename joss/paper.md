---
title: 'neximode: compiled DNNF reasoning, model-based diagnosis, and learning on one circuit'
tags:
  - Python
  - knowledge compilation
  - model counting
  - model-based diagnosis
  - probabilistic circuits
  - PyTorch
authors:
  - name: Alan Oursland
    affiliation: 1
affiliations:
  - name: Independent researcher, USA
    index: 1
date: 21 July 2026
bibliography: paper.bib
---

# Summary

`neximode` compiles finite-domain constraint theories into smooth
deterministic decomposable negation normal form (d-DNNF) circuits
[@darwiche2001; @darwiche2002map] and answers probabilistic queries on
the compiled artifact in time linear in its size. Queries are semiring
evaluations [@kimmig2017; @chavira2008]: satisfiability, model
counting, weighted model counting, most-probable explanations, exact
marginal MAP (via constrained compilation), lazily ordered model
enumeration, and exact sampling. Leaves are atomic assignments
(`valve=stuck_closed`); leaf weights read as negative log
probabilities, so best-first enumeration returns explanations ordered
from most to least probable.

On the circuit core sits a model-based diagnosis layer in the tradition
of compiled spacecraft health management [@barrett2005]: component
modes with priors, noisy sensors, quantized continuous observables,
and soft (virtual) evidence; temporal mode tracking with
command-conditioned, correlated, and compiled joint transition
constraints; and learning — exact EM and gradient training of priors,
plus neural observation models trained through the circuit with no
labels for the observables. A batched PyTorch evaluator makes the
circuit differentiable; by the differential semantics of d-DNNF
[@darwiche2003], one backward pass yields every posterior marginal.

# Statement of need

Open-source tooling covers fragments of this stack: d-DNNF compilers
without query or learning layers [@muise2012; @lagniez2017],
decision-diagram packages without diagnosis semantics, and
probabilistic-circuit learners without logical modeling front ends
[@peharz2020; @choi2020]. Neurosymbolic systems that train networks
through weighted model counting [@xu2018; @manhaeve2018] target logic
programming rather than system-model diagnosis. `neximode` integrates
the full lifecycle — model, compile, simulate, monitor, diagnose,
learn — behind one representation, in pure Python with an optional
torch extra, and cross-validates every inference path against
brute-force enumeration or closed forms (280+ tests). Output from the
external compiler c2d is loadable through the same query stack.

The library also serves research use directly: it is an exactly
solvable instance family for studying the expectation-maximization
structure of gradient-based training [@oursland2025], with a
machine-checked experiment suite (`tests/test_implicit_em.py`)
verifying that circuit gradients equal posterior responsibilities to
machine precision.

# Functionality overview

- `neximode.fd`: finite-domain CNF, native d-way decision-DNNF compiler
  (unit propagation, component decomposition, caching; iterative core),
  semiring queries, enumeration, marginal MAP, sampling, smoothing.
- `neximode.diagnosis`: `SystemModel` DSL (modes, sensors with
  false-positive/negative rates, quantized ranges with threshold
  atoms), ranked and minimum-cardinality diagnoses, exact posteriors,
  value-of-information sensor ranking, EM prior learning.
- `neximode.tracking`: beam-filtered belief over joint mode assignments,
  exact HMM filtering when the beam covers the space; transition
  matrices, per-step overrides, previous-state-dependent rates, and
  compiled joint transition constraints.
- `neximode.torch_backend` / `neximode.torch_learn`: layered batched
  evaluation (log-probability and tropical semirings), autograd
  marginals, batched MPE, gradient prior learning, neural observation
  model training.
- `neximode.cnf` / `neximode.nnf_io` / `neximode.external`: DIMACS and c2d `.nnf`
  interop, external compiler driver.

Benchmarks (`bench/`) include a scale study: circuit size grows
linearly on structured models (a 32-bit ripple-carry adder with 160
stuck-at fault modes compiles to 5,777 nodes; a 200-stage process line
to 9,201), and the constrained marginal-MAP order is measured to be
free on chain-structured models and exponential on gate-level
networks.

# Acknowledgements

The architecture follows the DNNF diagnosis lineage of Adnan Darwiche
and the NASA/JPL compiled-diagnosis systems. Development was assisted
by Claude (Anthropic); the design direction, the underlying
probabilistic interpretation, and the verification standards are the
author's.

# References
