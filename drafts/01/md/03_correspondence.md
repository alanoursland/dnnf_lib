# 3. The correspondence

**The identity at one node.** For an OR node with children of
accumulated distance $d_1,\dots,d_K$, its log-space value is
$L = \log \sum_j \exp(-d_j)$ and

$$\frac{\partial L}{\partial d_i}
 = -\frac{\exp(-d_i)}{\sum_k \exp(-d_k)} = -r_i,$$

the negative responsibility of child $i$ — Theorem 1 of
[Oursland2025]. By determinism the children are mutually exclusive
events, so $r_i$ is literally the conditional probability that child
$i$'s branch explains the evidence reaching this node: the E-step
quantity, not an analogy to it.

**Composition.** The root value $\log \mathrm{WMC}$ is a composition of
LSE (OR) and sum (AND) nodes. The chain rule multiplies local
responsibilities along paths and sums over paths; the quantity arriving
at leaf $X{=}x$ is the total posterior probability of the branches
selecting it, i.e. $P(X{=}x \mid e)$. This is exactly Darwiche's
differential semantics [Darwiche2003]; read through the companion
paper's lens, **differential semantics is Fisher's identity applied
recursively through a DAG of mixtures**. Neither literature, to our
knowledge, states the connection: knowledge compilation derives the
result from the circuit polynomial, EM theory from the incomplete-data
likelihood, and they are the same computation. Backpropagation through
the circuit *is* a hierarchical E-step.

**Zero temperature.** Replacing LSE with max (the tropical semiring
used for most-probable-explanation queries) replaces the softmax with
an argmax indicator: Viterbi responsibilities. The semiring family of
circuit evaluations is the temperature family of the identity.

**Learning and the shared fixed point.** Let one variable carry
trainable prior $\pi = \mathrm{softmax}(\theta)$ and let $\bar r$ be
the average posterior responsibility of its values over a dataset of
(possibly partial) observations. Differentiating through the identity,

$$\nabla_\theta \tfrac{1}{N}\textstyle\sum_n \log P(e_n; \theta)
 = \bar r - \pi.$$

Gradient ascent therefore moves the prior toward the average
responsibility; the EM M-step *sets* $\pi := \bar r$, solving the same
stationarity condition $\bar r = \pi$ in closed form. The two learners
are one fixed-point equation iterated softly versus hard
[NealHinton1998]. Section 4 confirms the shared fixed point
numerically, including the stationarity condition itself.

**The structural conditions.** The identity's inferential reading
requires (i) exponentiated distances, (ii) normalization across
alternatives, (iii) gradient (or M-step) updates [Oursland2025].
Conditions (i) and (iii) are built into the setting; condition (ii) and
the constrained regime's clamping term can be *deleted* — by feeding
unnormalized virtual evidence, and by removing all hard observations —
which yields the falsifiable predictions tested in Section 4.3.
