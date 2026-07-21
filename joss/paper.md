---
title: 'dnnf_lib: compiled DNNF reasoning, model-based diagnosis, and learning on one circuit'
tags: [Python, knowledge compilation, model counting, diagnosis, probabilistic circuits, PyTorch]
authors:
  - name: Alan Oursland
    affiliation: 1
affiliations:
  - name: Independent researcher
    index: 1
date: 21 July 2026
bibliography: paper.bib
---

# Summary

dnnf_lib compiles finite-domain constraint theories into smooth
deterministic decomposable negation normal form (d-DNNF) circuits and
answers probabilistic queries on the compiled artifact in time linear
in its size: weighted model counting, most-probable explanations,
exact marginal MAP, ordered model enumeration, and exact sampling. On
top of the circuit core it provides a model-based diagnosis layer
(component modes with priors, noisy sensors, quantized continuous
observables, soft/virtual evidence), temporal mode tracking with
command-conditioned, correlated, and compiled joint transition
constraints, and learning: exact EM and gradient training of priors,
and neural observation models trained through the circuit. A batched
PyTorch evaluator makes the circuit differentiable; one backward pass
yields all posterior marginals.

# Statement of need

Existing open tools cover fragments of this stack: d-DNNF compilers
without query/learning layers (c2d, dsharp, D4), decision-diagram
packages without diagnosis semantics, and probabilistic-circuit
learners without logical modeling front ends. dnnf_lib integrates the
lifecycle behind one representation, in pure Python with an optional
torch extra, and validates every inference path against brute-force
enumeration or closed forms (275+ tests). It follows the compiled-
diagnosis architecture used in early-2000s spacecraft health
management, extended with exact marginal MAP and differentiable
learning; it also serves as an exactly solvable testbed for research
on the EM-structure of gradient descent.

# Acknowledgements

The architecture follows the DNNF diagnosis lineage of Darwiche and
the NASA/JPL compiled-diagnosis systems.
