# Honest limitations

*(Draft section for the DX paper. Each item states the limitation, its
evidence, and what would retire it.)*

**1. Compilation scale is engineering-bound, and worst-case is real.**
The pure-Python compiler handles ~200 components (measured: 9.2k-node
circuit from 400 mode variables in 31 s; `docs/SCALE.md`), with
superlinear constants beyond. Nothing prevents a C implementation —
circuit growth is measured *linear* on structured models — but we have
not built one. And the worst case is not an implementation artifact:
densely coupled models can have exponentially large circuits,
fundamentally. The architecture's promise is conditional on structure.

**2. Exact marginal MAP is structure-fragile.** The modes-first
compilation that makes summed-posterior diagnosis exact is measured
*free* on chain/monitoring-shaped models and *exponentially
catastrophic* on gate-network-shaped ones (intractable by 4 bits of a
ripple-carry adder). This mirrors the complexity separation (marginal
MAP is NP^PP-hard) but means the headline query silently depends on
model shape; the fallback (MPE ranking) has different semantics.

**3. Tracking is beam-approximate and forward-only.** Belief
propagation keeps the k best joint mode assignments: exact when k
covers the space (verified against HMM filtering), otherwise mass
outside the beam is dropped and renormalized; a never-believed fault
combination cannot re-enter except through a transition. There is no
smoothing/retrodiction and no ranked *trajectory* enumeration yet.

**4. All validation is synthetic or self-generated.** The case study's
telemetry is sampled with the *modeled* error rates — misspecification
is structurally absent. This is the largest gap between the present
paper and a deployment claim: exact inference on a wrong model is
confidently wrong, learning cannot invent unmodeled failure modes, and
sensor-noise rates are assumed known during tracking. Real-hardware
telemetry with labeled incidents, and a calibration analysis
(posterior vs. empirical frequency), is required and outstanding.
**Update:** partially retired — `real_data_study.md` evaluates 2,205 real rig cycles: confident predictions perfectly calibrated, mid-range posteriors overconfident 2-3x (misspecification measured, not hypothesized). Real *sequences* and rate learning on real data remain open.

**5. Learning covers priors, not dynamics or structure.** EM/SGD learn
value priors of declared variables (identifiability caveats
documented and tested); transition rates are not yet learnable
in-library (grid search over tracker likelihood is the workaround),
and structure learning is out of scope by design.

**6. The sensor model is i.i.d. per step.** Glitch variables are
independent across time and sensors: no drift, no correlated noise,
no common-mode failures of the sensing layer itself. Persistent
sensor faults must be modeled as explicit modes (possible, but
manual).

**7. Quantization is static.** Continuous quantities need boundaries
chosen up front; there is no landmark discovery or adaptive
refinement, and threshold atoms must align with the chosen boundaries.

**8. VOI is myopic.** Sensor ranking is one-step greedy over marginal
mode entropies; it can under-rank observations whose value lies in
combination with a second reading, and it ignores joint (vs marginal)
uncertainty by construction.

**9. Planner expressiveness is v1.** Transition preconditions are
single command values (not general formulas over state); costs are
static; commands after target attainment are unconstrained filler; no
replanning loop or execution monitor is provided.

**10. No external-compiler baseline yet.** The c2d driver exists and
round-trips, but a systematic size/time comparison against an
industrial compiler (D4) on the paper's own benchmark families has not
been run **Update:** retired — `d4_comparison.md`: counts match on all shared instances (independent validation); d4 is 4-6x faster with 1.6-4.3x smaller circuits, so production-scale use should route through external compilers via the interop path.

**11. Single-designer ergonomics.** The modeling DSL has been
exercised by its authors only; `docs/MODELING_NOTES.md` records the
self-assessment and its circularity. External-user evidence (e.g.
JOSS review, a student cohort on the tutorial) is pending.
