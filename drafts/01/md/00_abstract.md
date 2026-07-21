# Abstract

*Gradient Descent as Implicit EM* [Oursland2025] argues that objectives
with log-sum-exp structure over distances train by implicit
expectation-maximization: the gradient with respect to each distance is
the negative posterior responsibility of the corresponding component.
Applied to neural networks, the claim is interpretive — the
distance-based reading of network computation must be argued. We exhibit
a model class in which the structure is explicit and every quantity is
exactly computable: smooth deterministic decomposable negation normal
form (d-DNNF) circuits, whose log-space evaluation is a DAG of
log-sum-exp nodes over negative-log-weight distances. In this setting we
verify the thesis against ground truth. Backpropagation through a
compiled diagnosis circuit reproduces exact posterior responsibilities
at every leaf to machine precision (max error 2.2e-16), at arbitrary
composition depth; exact EM and softmax-parameterized gradient ascent
converge to the same fixed point, matching the analytic maximum
likelihood estimate to six decimals; and deleting each of the theorem's
structural conditions produces its predicted degeneracy — an unbounded
gauge direction without normalization, posterior collapse without a
clamped observation. Both degeneracies were encountered during
independent system development before the theorem was known to the
developers, and were cured by restoring exactly the named conditions.
As a corollary, Darwiche's differential semantics of d-DNNF is Fisher's
identity applied recursively, connecting the implicit-EM thesis to the
knowledge compilation literature. All experiments are machine-checked
in the open-source library dnnf_lib.
