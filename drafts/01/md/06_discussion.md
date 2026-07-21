# 6. Discussion

**What the testbed establishes.** Within its scope the verification is
unusually strong: the identity holds to machine precision at
composition depth; the EM/SGD fixed-point equivalence is quantitative;
and the structural conditions demonstrably forbid something — their
deletion produces specific, reproducible degeneracies. A framework open
to the charge of just-so interpretation earns its keep exactly here:
it made prohibitions, and independent engineering ran into them. The
development history strengthens this beyond a constructed
demonstration: both degeneracies appeared as bugs in a system built
without knowledge of the theorem, and the independently derived fixes
(per-block log-softmax; a grounded evidence channel) coincide with the
theorem's conditions.

**A design checklist that transfers.** Read as engineering guidance,
the conditions become a checklist for any architecture with soft
evidence or learned likelihood-producing components: (i) normalize at
every choice point — an unnormalized potential hands the optimizer a
gauge direction; (ii) ground every soft channel — some observation must
clamp responsibility somewhere, or ignoring the input is optimal;
(iii) when both hold, the backward pass can be trusted as an E-step,
which licenses uses beyond training (our marginal computations and
sampling both consume the same responsibilities). These lessons were
learned here at toy scale in minutes; degenerate optima require only
the right objective, not scale, to manifest.

**Relation to probabilistic circuits.** The expected-statistics
computations known as circuit flows in the probabilistic-circuits
literature [ChoiVergariVdB2020; PeharzEtAl2020] are the same
responsibilities obtained here by differentiation, consistent with the
identity: EM for circuits and backpropagation through circuits compute
one quantity. The companion paper supplies the unifying reading;
knowledge compilation supplies an exact, discrete-structured instance
family on which future claims about implicit EM (temperature/annealing
behavior, responsibility dynamics under partial observability) can be
tested with ground truth.

**Scope fence.** These results verify the mechanism where the
LSE-over-distances structure is *explicit*. They do not, by themselves,
establish the companion paper's extension to general neural networks,
where the distance reading of internal computation is an interpretive
step argued separately [Oursland2025; Oursland2024]. Nor does the
testbed exhibit representation learning: circuit structure is compiled
from a specification, not discovered. What the testbed contributes to
the neural claim is indirect but real: the mechanism is exactly right
somewhere, its failure modes are portable (both have well-known neural
counterparts: unnormalized-potential pathologies and posterior
collapse [BowmanEtAl2016]), and its conditions have operational
content. A reader who doubts the neural extension can now localize the
doubt precisely: it lives entirely in the distance interpretation, not
in the EM mechanics.

**Limitations.** Experiment 4.2 is small and its model well-specified;
it demonstrates the fixed-point structure, not statistical robustness
(the companion library documents misspecification risk separately).
The gauge and collapse experiments use analytically transparent
instances by design — their value is that the *same* degeneracies were
first met unstaged, in ordinary development.
