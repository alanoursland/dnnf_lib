# Appendices

## A — API cheat sheet

**Modeling** (`neximode.SystemModel`)

| call | what | ch. |
|---|---|---|
| `m.bool(name, prior=None)` | boolean var; returns the "is true" atom | 3 |
| `m.finite(name, values)` | finite-domain var; `v == "x"`, `v != "x"`, `v.in_((...))` atoms | 3 |
| `m.mode(name, values, priors)` | finite var marked diagnosable, with fault priors | 5 |
| `m.quantized(name, boundaries)` | interval var; `below/at_least/between` atoms; numeric evidence auto-buckets | 6 |
| `m.sensor(name, expr, false_positive=, false_negative=)` | noisy observable (desugars to glitch vars) | 6 |
| `m.add(formula)` | constraint; operators `& \| ~ >>`, helpers `iff`, `xor` | 3 |
| `m.prev(mode_name)` | previous-timestep copy for joint transition constraints | 7 |
| `m.compile(modes_first=True)` | → `CompiledSystem` | 3 |

**Queries** (`CompiledSystem`)

| call | returns | ch. |
|---|---|---|
| `log_evidence(ev)` | log unnormalized mass of the evidence | 4 |
| `posteriors(ev, names=None)` | exact P(var=value \| ev) per variable | 5 |
| `diagnoses(ev, k)` | k best complete states (MPE ranking) | 5 |
| `map_diagnoses(ev, k)` | k best mode assignments by summed posterior | 5 |
| `diagnoses_min_cardinality(ev, k)` | fewest-faults-first | 8 |
| `value_of_information(ev)` | sensors ranked by expected entropy drop | 8 |
| `sample_state(rng)` | one exact draw from the model | 10 |
| `fit_priors(observations, names=)` / `fit_priors_torch(...)` | EM / SGD prior learning, in place | 10 |
| `save(path)` / `CompiledSystem.load(path)` | JSON persistence, single-allocation header | 12 |

Evidence values: hard (`True`, `"stuck"`, `37.2`) or soft likelihood
vectors (`(0.8, 0.2)` / `{False: 0.8, True: 0.2}`) — ch. 6.

**Time** (`neximode.ModeTracker`) — ch. 7: `ModeTracker(system, transitions,
beam=)`, `.step(ev, transitions=None)`, `.marginals()`, `.belief()`;
`transition_fn=` for correlated dynamics.

**Planning** (`neximode.Planner`) — ch. 9: `mode/command/observable/
behavior/transition`, `.compile(horizon)`, then `.estimate(obs,
commands)` / `.plan(current, target, observations=)`.

**Circuit level** (`neximode.fd`) — chs. 2–4: `compile_fd`, `model_count`,
`wmc`, `log_wmc`, `mpe`, `enumerate_models`, `enumerate_map`, `sample`;
`neximode.viz.circuit_to_dot` to look at small ones. GPU: 
`neximode.torch_backend.TorchCircuit` (batched log-WMC, marginals via one
backward pass).

## B — Glossary

- **world / model**: one complete assignment to every variable (ch. 1).
- **model counting**: how many worlds satisfy the constraints (ch. 1).
- **NNF circuit**: DAG of AND/OR over `var=value` leaves (ch. 2).
- **decomposability**: AND children share no variables ⇒ multiply (ch. 2).
- **determinism**: OR children mutually exclusive ⇒ add (ch. 2).
- **smoothness**: OR branches mention the same variables (padding
  gadgets make counting come out right) (ch. 2).
- **semiring**: the (resolve, combine) operator pair of a sweep (ch. 4).
- **WMC**: weighted model count — sum over worlds of products of leaf
  weights (ch. 4).
- **neg-log cost**: −log p; probabilities multiply ⇔ costs add;
  impossibility = ∞ (ch. 4).
- **MPE**: most probable *complete state* (ch. 5).
- **marginal MAP**: most probable *mode assignment*, summing over
  everything else (ch. 5).
- **prior / posterior**: belief before / after conditioning (ch. 5).
- **virtual (soft) evidence**: a likelihood vector multiplied into a
  variable's leaves; only ratios matter (ch. 6).
- **identifiability**: whether data can distinguish parameter values
  at all (ch. 10).
- **beam**: the k mode assignments a tracker keeps; exact when it
  covers the space (ch. 7).
- **horizon**: how many steps a planning circuit unrolls (ch. 9).

## C — Further reading (one line on why)

- Darwiche, *Decomposable Negation Normal Form* (JACM 2001) — the
  properties of chapter 2, from the source.
- Darwiche & Marquis, *A Knowledge Compilation Map* (JAIR 2002) — the
  atlas: which circuit families support which queries.
- Darwiche & Marquis, *Compiling Propositional Weighted Bases* (AIJ
  2004) — where "neg-log weights on compiled circuits" enters the
  record.
- Darwiche, *A Differential Approach to Inference in Bayesian
  Networks* (JACM 2003) — derivatives of the circuit are posteriors;
  the bridge to chapter 10's GPU footnote.
- Chavira & Darwiche, *On Probabilistic Inference by Weighted Model
  Counting* (AIJ 2008) — chapter 4, made general and rigorous.
- Barrett, *Model Compilation for Real-Time Planning and Diagnosis
  with Feedback* (IJCAI 2005) — chapter 9's spacecraft ancestor.
- Kimmig, Van den Broeck & De Raedt, *Algebraic Model Counting* (JAL
  2017) — the full semiring generalization of the operator swap.
- Choi, Vergari & Van den Broeck, *Probabilistic Circuits* (2020
  survey) — the same mathematics arrived at from machine learning.
