# 1. Introduction

Twenty years ago, model-based diagnosis reached flight software by
way of a compromise with certification: do the intractable reasoning
**offline**. Systems in the Livingstone lineage [WilliamsNayak1996],
and compiled executives like MEXEC [Barrett2005] most sharply,
compiled a declarative device model into a decomposable circuit on
the ground, then flew an online evaluator simple enough to certify —
every query a linear-time sweep, mode estimation and reconfiguration
planning the same structure evaluated two ways. Numerically these
systems ran on *costs*: integer per-mode penalties understood as
negative log-likelihoods, minimized by a tropical sweep to rank
diagnoses.

The theory underneath was moving past them even then. Darwiche and
Marquis showed weighted compiled bases support min-cost diagnosis with
the log transform stated explicitly [DarwicheMarquis2004]; Darwiche's
differential semantics [Darwiche2003] and weighted model counting
[ChaviraDarwiche2008] established that the *same* circuit, evaluated
under (+, ×) instead of (min, +), is an exact probability
distribution — partition functions, posteriors, marginals from a
derivative. The deployed architecture never absorbed this: its
circuits ranked, but did not measure.

This paper carries the probability reading the rest of the way onto
the deployed architecture, and then past it to what the reading makes
possible now: the compiled diagnosis artifact as a **differentiable
likelihood**. Concretely, on one compiled circuit per system model, we
provide and evaluate:

1. **The full exact query family** (§3): normalized posteriors, ranked
   diagnoses, minimum-cardinality, information-guided sensor selection
   — and exact *marginal MAP* over modes via constrained compilation,
   with a measured structure-sensitivity result: the constraint is
   free on monitoring-shaped models and exponentially costly on
   gate-network-shaped ones.
2. **Time and learning** (§4): beam-filtered mode tracking (exact when
   the beam covers the space, verified against HMM filtering) with
   command-conditioned, correlated, and compiled joint transition
   constraints; exact EM and gradient learning of priors that provably
   share a fixed point; and neural observation front-ends trained
   through the circuit without observable labels, whose two failure
   modes are predicted by the EM reading of gradient descent developed
   in a companion paper [Oursland2025].
3. **An end-to-end case study** (§5): a residential solar+battery
   model whose 5 KiB compiled artifact serves millisecond diagnosis,
   sensor selection, a tracked month with one-day fault-detection
   latency, and fleet-scale prior recovery from unlabeled telemetry.
4. **Real data and an external baseline** (§6): calibration measured
   on 2,205 real hydraulic-rig cycles — confident posteriors perfectly
   calibrated, mid-range ones overconfident 2–3×, misspecification
   made visible rather than hypothesized — and validation against the
   D4 compiler, which matches our model counts on every shared
   instance while compiling 4–6× faster, fixing the reference-
   implementation-plus-interop role of our compiler.

Everything is released as an open-source library in which every
inference path is cross-validated against brute-force enumeration or
closed forms, and every number in this paper is produced by a script
in its repository.

A remark on lineage: the observation that diagnosis costs are
negative log-probabilities belongs to the record cited above, not to
this paper; what the deployed systems lacked was not the observation
but its consequences. The architecture's defining split — compile
offline, evaluate simply online — turns out to be the compile/execute
split of modern differentiable computing, which is why a design
optimized for 2005 flight processors extends, without alteration, to
GPU-batched evaluation and gradient-based learning in 2026.
