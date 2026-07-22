# Abstract

The compiled model-based diagnosis architecture deployed in
early-2000s spacecraft executives — device models compiled offline to
decomposable circuits, evaluated online in linear time over neg-log
costs — ranked diagnoses but never measured probabilities. We carry
the probability reading of those costs, established contemporaneously
in the knowledge-compilation literature, fully onto the architecture:
the compiled artifact becomes an exact, differentiable likelihood. On
one circuit per system model we obtain normalized posteriors, exact
marginal MAP over modes via constrained compilation (measured free on
monitoring-shaped models, exponentially costly on gate networks),
information-guided sensor selection, beam-filtered mode tracking that
is provably exact HMM filtering when the beam covers the mode space,
exact EM and gradient prior-learning that share a fixed point, and
neural observation front-ends trained without observable labels. An
end-to-end case study compiles a residential solar+battery model to a
5 KiB artifact in 4 ms that diagnoses in 2 ms, detects an injected
fault with one day's latency over a tracked month, and recovers fleet
failure priors from unlabeled telemetry. On 2,205 real hydraulic-rig
cycles, confident posteriors are perfectly calibrated while mid-range
ones are overconfident 2–3× — misspecification measured, not
hypothesized. The D4 compiler independently matches our model counts
on all shared instances while compiling 4–6× faster, fixing our
compiler's role as reference implementation with industrial interop.
