# Scale study: where the compiler stands

`bench/scale.py`, run on the two parameterized families plus the small
examples. Numbers from a single-core container; trends matter more than
absolutes.

## Families

- **Ripple-carry adder, k bits** — 5 gates/bit, each gate a mode
  `{ok, stuck0, stuck1}`: the classic gate-level model-based-diagnosis
  benchmark. At k=32 that is 160 fault modes and 1313 FD variables.
- **Process line, n stages** — pump `{ok,weak,dead}` + valve
  `{ok,stuck_open,stuck_closed}` + noisy flow sensor per stage. At n=120
  that is 240 mode variables and 1801 FD variables.

## Results (free variable order, MPE diagnoses)

| family | size | FD vars | nodes | compile | ranked-dx query |
|---|---|---|---|---|---|
| adder | 8 | 329 | 1409 | 0.4s | 0.04s |
| adder | 32 | 1313 | 5777 | 3.5s | 0.45s |
| process | 40 | 601 | 1841 | 1.3s | 0.23s |
| process | 80 | 1201 | 3681 | 6.4s | 1.5s |
| process | 120 | 1801 | 5521 | 39.6s | 4.6s |

Diagnosis quality checks out at every size: the injected adder fault
(sum bit 0 reads wrong) is pinned to `gs_0=stuck1` from 1 bit to 32
bits; the process line correctly prefers "one sensor glitched" over any
component fault because downstream sensors exonerate the components —
nontrivial diagnostic reasoning falling out of exact inference.

## Findings

1. **Circuit size scales linearly on structured models.** ~180
   nodes/bit on adders, ~46 nodes/stage on process lines, at every size
   tested. Component caching does exactly what it is supposed to on
   chain-decomposable structure. The compiled artifacts are *small*;
   there is no representational wall in sight for models of this shape.
2. **Compile time is the wall, and it is mechanics, not math.** Process
   lines: 1.3s at 40 stages, 6.4s at 80, 39.6s at 120 — superlinear
   time against perfectly linear circuit growth. The blowup is the
   pure-Python engine (frozenset clause-set hashing at every search
   node, no watched literals), not the algorithm. This confirms the
   "cheaper mechanics" roadmap item as the top scaling priority, and
   puts a number on today's practical envelope: **~100 components /
   ~2000 FD variables per compile at interactive-offline patience.**
   (Compile is offline and once-per-model; queries stay sub-second up
   to ~5k nodes.)
3. **The modes-first (marginal MAP) constraint is free on
   chain-structured models and catastrophic on gate networks.** On
   process lines the constrained circuit is *identical in size* to the
   free one at every tested size — modes are decided locally anyway. On
   adders it blows up exponentially (368 vs 147 nodes at 1 bit, 3589 vs
   317 at 2 bits, intractable by 4) because forcing all 160 gate modes
   above the wire variables destroys the carry-chain decomposition.
   Practical guidance now encoded in the sweep: monitoring-shaped
   models get exact marginal MAP for free; densely coupled fault
   networks should use MPE diagnoses (`diagnoses()`) or an
   interleaved-order compilation, and this is a fundamental structure
   sensitivity (marginal MAP is NP^PP-hard in general), not an
   implementation artifact.

## Consequences for the roadmap

- Promote compiler mechanics (clause-index cache keys, watched
  literals, iterative core) — the ~40s/120-stage compile should be
  seconds, since the output circuit is only 5.5k nodes.
- Marginal MAP guidance: default `modes_first=True` stays correct for
  the monitoring use case the library targets; document the gate-network
  caveat prominently (done — this file).
- The adder family is the right regression benchmark for future
  heuristic work: linear today, and any decomposition regression will
  show up immediately.
