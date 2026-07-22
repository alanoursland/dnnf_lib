# 8. Limitations and conclusion

Five limitations bound the claims made here (an extended list ships
with the library).

**Structure-conditional tractability.** Circuit size is measured
linear on monitoring-shaped models but the worst case is genuinely
exponential; the architecture's promise is conditional on model
locality, and densely coupled models can defeat it.

**Marginal-MAP fragility.** The modes-first order that makes summed-
posterior diagnosis exact is free on monitoring shapes and
exponentially costly on gate-network shapes (§3); the fallback (MPE
ranking) answers a different question, and the practitioner must know
which they are asking.

**Beam-approximate, forward-only tracking.** Exact only when the beam
covers the joint mode space; no smoothing or ranked trajectories yet.

**The real-data gap, narrowed but open.** §6 measures calibration
under misspecification on real hardware snapshots; real *sequences*
(tracking on hardware telemetry) and learning of transition rates
from real data remain unevaluated. Sensor noise is modeled i.i.d.
per step; learned parameters are only as identifiable as the data
allows — both are places exactness sharpens, rather than hides, the
consequences of a wrong model.

**Single-designer ergonomics.** The modeling DSL has been exercised
by its authors; external-user evidence is pending.

**Conclusion.** The compiled-diagnosis architecture of the early
2000s was defined by one discipline — do the hard reasoning offline,
once, and make every online operation a linear sweep — and one
reading of its numbers, costs as negative log-probabilities, that its
deployed systems never carried past ranking. Carried all the way,
that reading makes the compiled artifact a probability distribution:
posteriors, marginal MAP, information-guided sensing, filtered
tracking, exact simulation, and maximum-likelihood learning all
become evaluations of one small file (§5: 5 KiB, 4 ms to build, one
day to detect an injected fault, fleet priors recovered without
labels). The lifecycle closes — model, compile, diagnose, track,
learn, better model — on a single artifact, and the twenty-year-old
offline/online split lands exactly on the compile/evaluate split of
modern differentiable computing.
