# 2. Background: circuits, distances, and derivatives

**d-DNNF.** A negation normal form circuit is a rooted DAG whose leaves
are literals and whose internal nodes are AND and OR. Three structural
properties make it tractable [Darwiche2001; DarwicheMarquis2002]:
*decomposability* (AND children mention disjoint variables),
*determinism* (OR children are pairwise logically inconsistent), and
*smoothness* (OR children mention the same variables). Compilers in the
exhaustive-DPLL family produce such circuits from constraint theories
[Muise2012; LagniezMarquis2017]; our experiments use dnnf_lib's native
finite-domain compiler, whose leaves are atomic assignments
$X{=}x$ and whose decision nodes branch over a variable's domain.

**Weighted model counting.** Assign each leaf $X{=}x$ a weight
$w_{X=x} \geq 0$. Evaluating the circuit with OR as $+$ and AND as
$\times$ yields the weighted model count
$\mathrm{WMC} = \sum_{\text{models } m} \prod_{X=x \in m} w_{X=x}$,
which is exact on smooth d-DNNF [Darwiche2003; ChaviraDarwiche2008];
with suitable weights this computes any discrete-factor marginal
likelihood. Evidence enters by annihilating the weights of ruled-out
values, and *soft* (virtual) evidence [Pearl1988] by scaling them.

**Log space: distances and LSE.** Write $d_{X=x} = -\log w_{X=x}$.
In log space AND nodes sum their children and every OR node computes

$$\log \sum_{c \in \mathrm{ch}} \exp(-d_c),$$

a log-sum-exp over the (accumulated) distances of its children. A
smooth d-DNNF is therefore a DAG of LSE nodes over distances — a
hierarchical mixture model in which OR nodes mix mutually exclusive
alternatives and AND nodes multiply independent factors. This is the
companion paper's setting [Oursland2025], with two additions: exactness
(determinism makes the mixture components disjoint, so the "posterior
responsibility" has an unambiguous meaning as a conditional
probability) and depth (LSE nodes compose through the DAG).

**Differential semantics.** Darwiche showed that partial derivatives
of the circuit polynomial carry exact inferential meaning:
$\partial \mathrm{WMC} / \partial w_{X=x}$ recovers the posterior
marginal of $X{=}x$ given the evidence [Darwiche2003]. In log space:

$$\frac{\partial \log \mathrm{WMC}}{\partial \log w_{X=x}}
  = P(X{=}x \mid \text{evidence}).$$

One backward pass computes every marginal. Section 3 observes that this
celebrated result is precisely Fisher's identity applied recursively,
and Section 4 verifies the equality numerically against marginals
computed without differentiation.

**Learning.** With priors on a variable parameterized as
$\pi = \mathrm{softmax}(\theta)$, maximizing
$\log P(\text{evidence})$ over partially observed data admits both an
exact EM procedure (E-step: posterior responsibilities by WMC ratios;
M-step: set $\pi$ to the average responsibility) and direct gradient
ascent through the differentiable evaluation. The relation between the
two is classical [Dempster1977; NealHinton1998] and is exercised
quantitatively in Section 4. Analogous expectation computations appear
as "circuit flows" in the probabilistic-circuits literature
[ChoiVergariVdB2020; PeharzEtAl2020].
