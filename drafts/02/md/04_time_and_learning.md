# 4. Time and learning

**Tracking.** Modes evolve; the library maintains a belief over joint
mode assignments by beam filtering: each tracked assignment is pushed
through per-variable transition rows (zeros encode structure: absorbing
faults, no self-repair), the same compiled circuit is conditioned on
the step's evidence, and candidates merge by summed mass. When the
beam covers the joint mode space the recursion *is* exact HMM
filtering — verified in the test suite against an independently coded
forward algorithm — and otherwise it is the best-first approximation
standard in tracking diagnosis engines, with dropped-mass
renormalization and explicit failure (an exception, not silent mass
invention) when evidence contradicts every tracked trajectory.

Three dynamic refinements compose, each one sentence here and one test
in the repository: per-step transition overrides (command-conditioned
risk: a valve only risks sticking when actuated), transition functions
of the entire previous joint assignment (correlated wear), and hard
joint constraints between consecutive slices compiled into the circuit
itself via previous-timestep mode copies (e.g., forbidding
common-cause pair failures), which prune illegal transitions during
filtering with renormalization semantics.

**Learning priors.** The compiled artifact is a likelihood, so its
parameters can be fit. For value priors under partial observation the
library provides exact EM — E-step posteriors are weighted-model-count
ratios on the circuit; M-step averages them — with the guaranteed
non-decreasing likelihood trace, and an SGD path that trains the same
parameters through the differentiable evaluator with softmax
reparameterization. The two provably share the stationarity condition
(prior = average responsibility) and empirically land together: on an
identifiable single-component benchmark both recover the analytic
maximum-likelihood estimate to ~10⁻⁴ from telemetry that never
observes the modes. Identifiability is the operative caveat and is
documented with multi-start and simulate-then-recover disciplines
(exercised in Section 5's fleet experiment).

**Neural observation front-ends.** Real telemetry is rarely boolean.
A network mapping raw sensor input to per-value log-likelihoods (soft
evidence in Pearl's virtual-evidence sense) can be trained end-to-end
through the circuit by maximizing log-WMC — with **no labels for the
observables**, supervision coming from hard evidence elsewhere in the
constraint structure. Two degeneracies gate this construction, and we
report them as confirmations rather than novelties: without per-
observable normalization the objective is unbounded along a pure gauge
direction, and without any clamped observation an input-ignoring
detector is optimal (posterior collapse). Both occurred during
development before their common explanation was recognized, and both
are predicted exactly by the EM reading of gradient descent developed
in the companion paper [Oursland2025], whose machine-checked
verification suite lives in this same repository; the fixes
(log-softmax per observable; a grounded evidence channel) are the
theory's named conditions. We cite rather than re-derive.
