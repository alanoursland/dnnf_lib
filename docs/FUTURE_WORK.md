# Future work: from next steps to blue sky

Organized in horizons. Each item notes why it matters and what it depends
on. The single most important fact shaping this list: **circuit size is
everything.** Every online query is linear in circuit size, so effort spent
making circuits smaller (or evaluating them faster) multiplies across every
query forever after.

---

## Horizon 0 — Housekeeping and hardening (days)

Low-risk work that makes everything after it easier.

- **CI**: GitHub Actions running the test matrix (with and without torch),
  plus a lint pass. The brute-force cross-validation suite is the safety
  net for all compiler work below; it should run on every commit.
- **Benchmark harness**: ✅ *Started:* `bench/run.py` compares heuristics
  on random k-CNF, chains, grids, pigeonhole, and generated diagnosis
  models, with a built-in count cross-check. Measured result so far:
  min-fill helps on structured instances (grids, diagnosis chains) and
  hurts on random CNF, so `dynamic` remains the default. Still to add:
  standard public instances (ISCAS85 diagnosis circuits such as c432/c499;
  DQMR and grid networks for counting) and per-commit tracking.
- **Property-based fuzzing**: `hypothesis` strategies generating CNFs and
  evidence sets, asserting the invariants we already test (count parity,
  k-best order, torch/CPU agreement). Random seeds catch bugs; shrinking
  finds minimal reproducers.
- **Iterative compiler core**: replace recursion with an explicit stack so
  deep instances can't hit Python's recursion limit. Mechanical but
  unlocks larger inputs.

## Horizon 1 — Compiler quality: the big lever (weeks)

The gap between this compiler and c2d/dsharp/D4 is almost entirely
decomposition-heuristic quality and constant factors. Ordered by expected
leverage:

1. **Static decomposition orders.** ✅ *Partially done:* min-fill order is
   implemented (`compile_cnf(..., heuristic="minfill")`); benchmarks show
   it helps on structured instances and hurts on random ones. The real
   prize remains: build a **dtree** by recursive hypergraph partitioning
   of the clause set (KaHyPar or a simple FM heuristic) and branch to cut
   components apart, the way c2d does. System models — mostly-local
   component interconnections — are exactly the structured instances where
   this wins big. This is the highest-value item in the whole document.
2. **Dynamic scoring.** VSADS-style hybrid of static structure and
   activity, favoring variables that split components.
3. **Sound preprocessing.** Subsumption, self-subsuming resolution,
   vivification, equivalent-literal substitution (with a reconstruction
   map so counts and models are recoverable). Note: *pure-literal
   elimination is unsound for counting* — the preprocessing menu is
   smaller than a SAT solver's, and each rule needs a count-preservation
   argument in its docstring.
4. **Cheaper mechanics.** Watched-literal BCP; components via clause-index
   union-find on a preallocated arena instead of frozensets of literal
   tuples; cache keys packing clause indices + deleted-literal masks into
   bytes (dsharp's trick) rather than hashing full residual clause sets;
   memory-bounded cache with eviction.
5. **External compiler drivers.** We already read c2d `.nnf`. Add
   subprocess adapters that shell out to installed `c2d`/`dsharp`/`d4`
   binaries and parse their output (including D4's decision-DNNF format),
   so the Python compiler becomes the readable reference implementation
   and industrial instances get industrial compilers, with all queries
   downstream unchanged.
6. **Failed-literal probing / clause learning.** Conflict-driven learning
   in a compilation setting (as in sharpSAT lineage) prunes dead branches
   before they generate circuit nodes.

## Horizon 2 — Query layer completeness (weeks, parallel to H1)

- **Marginal MAP over modes.** ✅ *Done:* `SystemModel.compile()` now
  branches mode variables first by default, the constrained structure is
  verified, and `CompiledSystem.map_diagnoses()` / `dnnf.enumerate_map()`
  return joint mode assignments ranked by exact summed posterior (lazy
  k-best over the mode region with log-sum-exp values as terminal costs).
  Remaining refinement: constrained *dtree* construction so the modes-first
  restriction costs less circuit size on large models.
- **Minimum-cardinality diagnoses.** Darwiche's classic: min-sum with
  unit fault costs gives minimum fault cardinality; enumerate within a
  cardinality bound. Cheap to add on the existing tropical machinery and
  matches how diagnosis engineers often think ("all double faults").
- **Projection / existential quantification.** Forgetting variables is
  polytime on DNNF (drop-and-simplify). Enables observable-only views,
  smaller circuits for fixed query patterns, and is a prerequisite for
  some marginal-MAP variants.
- **Incremental evidence.** Consecutive sensor frames differ in a few
  literals. Cache node values and recompute only the cone of influence of
  changed leaves (upward closure). For a monitoring loop this is the
  difference between re-evaluating 1% and 100% of the circuit per tick.
- **Entailment and implicant queries.** Clausal entailment, prime
  implicants over mode variables ("what must be broken"), consequence
  enumeration — all polytime on DNNF and useful for explanation UIs.
- **Anytime bounds.** Partial k-best enumeration already yields a lower
  bound on evidence probability; add the complementary upper bound from
  the min-sum relaxation so callers can stop early with certified
  intervals.

## Horizon 3 — Tensor backend performance (weeks)

- **Kill the autograd-path `cat`.** The no-grad path already writes into a
  preallocated buffer; do the same for the gradient path with a custom
  `torch.autograd.Function` per layer (saving only layer inputs), or
  segment the circuit into chunks with checkpointing. Target: marginals at
  the same throughput as forward.
- **`torch.compile` / CUDA graphs.** The forward is a fixed sequence of
  gathers and segment-reductions — ideal fusion fodder. Measure first;
  the win may already be large on GPU.
- **Precision strategy.** float32 (or bfloat16) with a float64 fallback
  for near-cancellation cases in log-space; min-sum is precision-robust
  and can run fully in float32.
- **Batched k-best on GPU.** A top-k semiring: each node carries its k
  smallest costs as a fixed-width vector; OR = top-k of concatenation,
  AND = top-k of the pairwise-sum matrix (k² → k). One forward pass
  yields the k best costs *batched over evidence sets*; a second pass
  decodes assignments. This is "ranked diagnoses for a whole fleet of
  telemetry frames in one kernel launch" — the natural GPU version of the
  original mission use case.
- **Multi-circuit batching.** Pad/pack several circuits into one layered
  program (block-diagonal segments) to amortize kernel launches when
  monitoring many subsystems.
- **Deployment export.** The forward pass is index arithmetic + reductions
  with no control flow: export to ONNX/ExecuTorch/C for embedded targets.
  There is a pleasing echo of the original architecture here — the online
  engine is simple enough to certify for flight software, and that
  simplicity survives the port to tensors.

## Horizon 4 — Learning (months)

This is where the neglog-probability observation pays out beyond faster
diagnosis: the compiled circuit is an exact, differentiable likelihood.

- **Learned priors.** ✅ *CPU EM done:* `CompiledSystem.fit_priors()`
  learns value priors from partially observed telemetry by exact EM
  (posterior WMC ratios as the E-step), verified to recover known failure
  rates from alarm-only observations; `sample_state()` provides exact
  model sampling for simulation and synthetic data. ✅ *Gradient path
  done too:* `dnnf.torch_learn.PriorLearner` /
  `CompiledSystem.fit_priors_torch` train the same priors by Adam on the
  differentiable backend (batched masked log-WMC over deduplicated
  evidence patterns), verified to match EM and the analytic MLE.
  Result: failure priors estimated from fleet data instead of engineering
  guesses, with the logical model as a hard constraint.
- **Neural observation models.** Raw sensor streams rarely arrive as clean
  booleans. Put a small network in front: `sensor trace → P(observable
  literal)` → leaf weights → exact inference in the circuit. Train
  end-to-end through log-WMC. This is the semantic-loss / neurosymbolic
  pattern with a system model, and it is the strongest argument for the
  PyTorch substrate: the circuit becomes a differentiable layer other
  models can sit on top of.
- **Calibration and validation loop.** Compare learned posteriors against
  held-out labeled incidents; the diagnosis layer gives exact posteriors,
  so miscalibration cleanly indicts the model or the priors, not the
  inference.

## Horizon 5 — Blue sky (research-grade)

- **Temporal diagnosis / mode tracking.** ✅ *First version done:*
  `dnnf.ModeTracker` maintains a beam-filtered belief over joint mode
  assignments with per-variable transition matrices, exact HMM filtering
  when the beam covers the mode space (verified against a hand-rolled
  forward recursion). Zero-probability transitions, per-step
  (command-conditioned) overrides, and `transition_fn` — each variable's
  next-value distribution as a function of the entire previous joint
  assignment, for correlated dynamics — are supported and verified
  against an exact joint forward filter. Remaining: compiled transition
  *relations* (hard joint constraints between consecutive slices),
  fixed-lag smoothing via k-step unrolling, ranked fault *trajectories*
  (k-best over paths, not just states), and running the per-step sweeps
  on the GPU backend.
- **Decision-making on top.** With P(mode | evidence) exact and cheap,
  add value-of-information queries: which next sensor reading or active
  test most reduces diagnosis entropy? (Greedy VOI is just a few
  conditioned WMC sweeps.) That closes the loop from diagnosis to
  recovery planning — the other half of what Remote-Agent-class systems
  did.
- **Probabilistic-circuit convergence.** d-DNNF, SDDs, PSDDs, and
  sum-product networks are one family. Import PySDD/PyJuice circuits into
  the same layered evaluator; export our circuits to their formats. The
  library then stops being "a DNNF tool" and becomes a bridge between the
  logic side (hard constraints, diagnosis) and the ML side (learned
  densities) of the same mathematical object.
- **First-order / relational models.** System descriptions are naturally
  relational ("N identical thrusters"). A modeling layer that grounds
  parameterized components into the propositional DSL — with compilation
  caching across isomorphic components — would let one component library
  serve many vehicle configurations. (True lifted compilation is a
  research problem; grounding with structure sharing is not.)
- **LLM-authored models with verified inference.** The DSL is small and
  declarative; language models are good at drafting such models from
  prose specs and P&ID-style descriptions. The circuit then provides
  something LLMs cannot: exact, auditable posteriors. "LLM writes the
  model, knowledge compilation checks and runs it" is a genuinely
  current research direction, and this library has both halves of the
  interface.
- **Approximate compilation with guarantees.** When exact compilation
  blows up, compile a relaxation (drop clauses → upper bound on WMC) and
  a strengthening (add clauses → lower bound), and report certified
  intervals. Diagnosis rankings are often stable under bounded error;
  quantifying that would extend the approach to models an exact compiler
  cannot touch.

---

## Suggested sequencing

If effort is scarce, the dependency-aware shortlist:

1. CI + benchmark harness (H0) — everything else is measured against it.
2. dtree/min-fill decomposition (H1.1) — the leverage item.
3. Marginal MAP over modes (H2) — makes the diagnosis semantics exact.
4. Autograd-path buffer + `torch.compile` (H3) — makes the GPU story real
   at scale.
5. Learned priors + neural observation front-end (H4) — the payoff that
   none of the 2005-era systems had.
