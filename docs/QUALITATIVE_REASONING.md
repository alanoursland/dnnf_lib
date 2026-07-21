# Qualitative reasoning and compiled circuits

Thinking notes — not an implementation plan. The question: does DNNF
compilation have something to offer qualitative reasoning (QR), the
QSIM/de-Kleer-Brown lineage of reasoning about physical systems without
numbers? Short answer: yes, and the fit is unusually good, because the
library's native finite-domain machinery already speaks QR's language.

## Why the fit is natural

Qualitative physics discretizes continuous state exactly the way our
quantized variables do:

- A **quantity space** is an ordered set of landmarks and the intervals
  between them — `level ∈ {0, (0,full), full}`. That *is*
  `m.quantized("level", boundaries)`: an ordered finite domain whose
  atoms are threshold sets.
- A **qualitative state** assigns each quantity a magnitude (interval or
  landmark) and a direction of change (`dec | std | inc`). Direction is
  just a second finite-domain variable per quantity.
- **Qualitative constraints** — monotonic influences (`M+`, `M-`),
  correspondences, sums (`ADD(a,b,c)` at the sign level), derivative
  links — are relations over small finite domains. Every one of them is
  expressible as FD clauses; a monotonic function constraint, for
  instance, is a short list of forbidden value pairs.

So a QR model *compiles*. And once compiled, the queries QR systems
historically computed by generate-and-test search come almost for free:

## What compilation buys QR

1. **Envisionment as a circuit.** The set of all consistent qualitative
   states (the "total envisionment") is exactly the model set of the
   compiled theory. QSIM-era systems enumerated it explicitly and choked
   on state-space explosion; a DNNF circuit *is* that state set,
   factored. Counting consistent states is one sweep; enumerating them
   lazily (best-first, if weighted) is exactly `enumerate_models`.
2. **Conditioning is filtering.** "Which qualitative states are possible
   given that the outlet is dry and pressure is rising?" — mask value
   weights, sweep. This is the QR analog of our diagnosis conditioning,
   and it is linear-time online, no re-search.
3. **Probabilities over qualitative behavior.** Classic QR is purely
   possibilistic — every consistent behavior is equally "possible," which
   was one of its practical weaknesses (ambiguity explosion: QSIM
   branches on every unresolvable comparison). Leaf weights fix this:
   put priors or learned frequencies on qualitative values and the
   ambiguity explosion becomes a *ranked* list — most plausible
   qualitative states first, exactly the move the diagnosis layer makes.
   "Qualitative simulation with a posterior" is, to my knowledge, still
   an underexplored combination.
4. **Ambiguity itself becomes a query.** Marginals (`mode_posteriors`, or
   the torch backend's gradient marginals) directly measure how
   underdetermined each quantity is given observations — a principled
   version of QSIM's branching factor, usable to decide *which sensor to
   read next* (value-of-information over qualitative states).
5. **Transitions are the tracking layer.** QSIM's continuity rules — a
   magnitude moves to an adjacent interval, passing through landmarks;
   direction changes pass through `std` — are constraints between
   consecutive states. Our `ModeTracker` already advances a belief
   through per-variable transition structure; QSIM continuity is a
   *joint* transition relation, which is precisely the "compiled
   transition relations" item on the tracking roadmap. With it, the
   attainable envisionment (states reachable from an initial state) is
   beam search over compiled steps, and "most plausible qualitative
   behavior" is a ranked trajectory query.

## Applications worth taking seriously

- **Early design-stage analysis.** Before parameters exist, engineers
  reason qualitatively ("if the pump weakens, can the tank still
  overflow?"). A compiled envisionment answers reachability-of-a-bad-
  qualitative-region questions interactively over the whole design space,
  with counts quantifying how much of the space is safe.
- **Diagnosis without good numeric models.** Most industrial equipment
  lacks a calibrated quantitative model but has a perfectly stateable
  qualitative one (flows, levels, temperatures, monotonic dependencies).
  Qualitative system model + noisy threshold sensors + priors is exactly
  the stack this library now implements end-to-end — the QR framing
  argues that the *modeling bar* for useful diagnosis is lower than it
  looks: you never need the differential equations.
- **Guardrails for learned models.** A qualitative envisionment is a
  certificate of which state combinations are physically possible. Used
  as a semantic-loss layer (the torch backend), it constrains neural
  state estimators to physically consistent outputs — a lightweight,
  auditable physics prior that doesn't require simulation.
- **Explanation.** Qualitative states are human-readable by construction
  ("level low and falling, valve commanded open, no flow downstream").
  Ranked qualitative diagnoses are close to the explanation a human
  operator would give, which matters in any setting where the consumer
  of the diagnosis is a person, not a controller.

## What stays hard (honesty section)

- **Continuity/temporal structure is where the real QR work lives.**
  Pure DNNF handles one time-slice perfectly; behaviors need the
  transition-relation machinery, and long horizons need either unrolling
  (circuit growth linear in horizon) or beam filtering (approximation).
  QSIM's infinite-time behaviors (limit cycles, quiescence) don't map
  directly onto finite unrolling.
- **Landmark discovery.** QSIM invents new landmarks during simulation
  (where curves cross). A static quantization can't; boundaries must be
  chosen up front. Adaptive re-quantization is possible but is genuinely
  new work.
- **Sign algebra ambiguity is real ambiguity.** Compilation doesn't
  reduce it — it just represents it compactly and lets weights rank it.
  That's an honest improvement, not magic.

## The experiment — done, and it worked

The classic QR textbook system — two cascaded tanks with monotonic flow
laws (`outflow = M+(level)`), quantity spaces `{zero, between, full}`,
direction-of-change variables `{dec, std, inc}`, sign-algebra derivative
constraints, and landmark consistency rules — is modeled in
`examples/cascaded_tanks.py` and verified in `tests/test_qsim_tanks.py`.

Results (all checked against hand-enumeration of the qualitative rules):

- The one-slice envisionment compiles to a **62-node circuit** whose
  model set is exactly the expected state space: **33** consistent
  states with inflow on, **15** with inflow off, **48** with inflow
  free. Counting each is one sweep.
- The unique quiescent state under no inflow (both tanks empty, both
  directions `std`) falls out as the single model consistent with
  `dA=std, dB=std` — the compiled circuit "knows" the global
  equilibrium.
- Filtering works as filtering should: "tank B rising" narrows 33
  states to 10; the design-safety query "tank B full *and* still
  rising" is provably impossible (`log P = -inf`) by landmark
  consistency alone.
- With mild plausibility priors on qualitative values, QSIM's ambiguity
  explosion becomes a **ranked posterior over qualitative states**
  (`map_diagnoses`): under "faucet on, B rising," the four plausible
  mid-range states come back ordered with exact probabilities, ambiguity
  quantified instead of enumerated.

So the thesis holds at textbook scale: *QSIM's state graph as a
queryable, weightable database*, with the ambiguity ranked rather than
exploding. The open frontier is unchanged: behaviors over time need the
compiled transition-relation machinery (QSIM continuity rules as joint
constraints between consecutive slices), which is also what the tracking
layer needs next.
