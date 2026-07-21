# 1. Introduction

The companion paper [Oursland2025] makes a simple algebraic observation
and a large claim. The observation: for an objective
$L = \log \sum_j \exp(-d_j)$ over distances $d_j$,

$$\frac{\partial L}{\partial d_i} = -r_i, \qquad
r_i = \frac{\exp(-d_i)}{\sum_k \exp(-d_k)},$$

so the gradient with respect to each distance *is* the negative
posterior responsibility of the corresponding component — Fisher's
identity for mixtures [Fisher1925; Dempster1977]. The claim: because
standard neural objectives instantiate this structure, ordinary
gradient descent implicitly performs expectation-maximization, and
mixture learning, attention, and classification are three regimes of one
mechanism.

The natural objection is that this is a just-so interpretation: an
identity true of any log-sum-exp, projected onto systems whose internal
quantities need not be distances or responsibilities in any committed
sense. Interpretations of this kind explain everything and forbid
nothing. The way out is to find a setting where the structure is not
interpretive but explicit, where responsibilities have an independent
definition against which gradients can be *checked*, and where the
theorem's preconditions can be deleted to see whether the predicted
failures occur.

Compiled logical-probabilistic circuits are that setting. A smooth
deterministic decomposable negation normal form (d-DNNF) circuit
[Darwiche2001; DarwicheMarquis2002] evaluated in log space is a DAG in
which every OR node is a log-sum-exp and every leaf weight is a
negative log probability — a distance in precisely the companion
paper's sense. Ground-truth responsibilities are available by
construction: they are exact posterior marginals, computable
independently by weighted model counting [ChaviraDarwiche2008]. The
setting also exercises what single-node expositions of the identity do
not: *composition*, since responsibilities must chain correctly through
arbitrarily deep alternations of mixtures (OR) and independent factors
(AND).

We report three results, all machine-checked in the open-source library
dnnf_lib and reproducible with one command (Section 4):

1. **Gradients are responsibilities, exactly and at depth.** The
   backward pass of the log weighted model count through a multi-level
   diagnosis circuit equals independently computed exact posteriors at
   every leaf, with maximum error 2.2e-16 (double-precision machine
   epsilon). As a corollary, Darwiche's differential semantics of
   d-DNNF [Darwiche2003] — that partial derivatives of the circuit
   polynomial are posterior marginals — is Fisher's identity applied
   recursively through the DAG.
2. **EM and gradient ascent share one fixed point.** Under softmax
   parameterization of priors, $\nabla_\theta \log P(\text{evidence}) =
   \bar r - \pi$; the EM update jumps to the stationary point
   $\pi = \bar r$ while gradient ascent walks to it [NealHinton1998].
   From 2{,}500 partially observed samples, exact EM, Adam, the average
   responsibility at convergence, and the analytic maximum likelihood
   estimate coincide to six decimal places.
3. **The structural conditions are load-bearing.** Deleting
   normalization exposes an unbounded pure-gauge direction that raises
   the objective while changing no posterior; deleting the clamped
   observation makes an input-ignoring detector optimal (posterior
   collapse). Both degeneracies were encountered *during independent
   development of the library, before the companion paper was read*,
   and were repaired by measures — per-block log-softmax, a grounded
   evidence channel — that turn out to be exactly the theorem's named
   conditions. They are observations, not constructed illustrations.

We close by mapping the companion paper's three regimes onto literal
operations of the system (Section 5) and by stating plainly what this
testbed does and does not establish about neural networks (Section 6).
