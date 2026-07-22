# 3. The query family on one circuit

Every online capability is a semiring evaluation of the compiled
circuit, differing only in the (resolve, combine) operator pair and in
which leaves evidence has annihilated or reweighted:

| query | operators | requires | returns |
|---|---|---|---|
| consistency | (or, and) | DNNF | SAT under evidence |
| weighted model count / posteriors | (+, ×) | smooth d-DNNF | exact P(evidence), P(var=value \| e) by ratio |
| ranked diagnoses (MPE) | (min, +) on −log, lazy k-best | DNNF | complete states, most probable first |
| exact marginal MAP | max over modes, + / logsumexp below | constrained order | mode assignments by *summed* posterior |
| minimum cardinality | (min, +), unit fault costs | DNNF | fewest-faults-first [DarwicheMarquis2004] |
| value of information | posterior sweeps per candidate | smooth d-DNNF | sensors ranked by expected entropy reduction |
| sampling | top-down, children ∝ mass | smooth d-DNNF | exact simulation |

Two of these deserve elaboration.

**Marginal MAP, and its measured price.** MPE ranks *complete states*;
when a mode's probability is smeared across latent detail (an
unmodeled timer, free observables), the ranking of best states can
invert the true posterior over modes (our library's tests construct
such cases). The principled diagnosis query sums over non-mode
variables — marginal MAP, NP^PP-hard in general. It becomes a single
sweep if mode variables are *decided above all others* during
compilation; the compiler accepts that constraint, verifies the
resulting structure, and the query machinery enumerates mode
assignments best-first by exact summed posterior.

The constraint's cost is strongly structure-dependent, and we measured
it. On chain-structured monitoring models (process lines to 120
stages) the modes-first circuit is *identical in size* to the
unconstrained one at every tested size: exact marginal MAP is free
exactly where diagnosis lives. On gate-network-shaped models it is
catastrophic: a ripple-carry adder with per-gate fault modes compiles
to 147 nodes unconstrained at 1 bit versus 368 constrained, 317 versus
3,589 at 2 bits, and is intractable by 4 bits — an empirical face for
the complexity separation, with operational guidance attached: use the
modes-first default for monitoring-shaped models, fall back to MPE
ranking (different semantics, stated openly) for densely coupled ones.

**Active sensing.** Because posteriors are exact and cheap,
"which sensor should be read next" is itself a query: for each
unobserved variable, the expected reduction in summed mode-marginal
entropy, averaging over that variable's posterior predictive. This
turns ranked diagnosis into a troubleshooting *loop* (Section 5 shows
one turn resolving an honest ambiguity). The ranking is one-step
greedy — a stated limitation (Section 8).

All evaluations are also available batched on GPU through a layered
tensor lowering of the same circuit; by the differential semantics of
d-DNNF [Darwiche2003], one backward pass yields every posterior
marginal simultaneously, a property Section 4 uses for learning.
