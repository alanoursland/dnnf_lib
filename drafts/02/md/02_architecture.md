# 2. Architecture

The system keeps the classic two-phase shape [Barrett2005]: an
**offline compiler** turns a declarative system description into a
circuit; an **online evaluator** answers every operational question by
sweeping that circuit. What is inside each phase is where this work
departs from its ancestors.

**Representation.** Models are natively finite-domain: a component's
modes form a domain (`valve ∈ {ok, stuck_open, stuck_closed}`),
circuit leaves are atomic assignments `var=value`, and decision nodes
branch d-ways over a domain. Constraints are written in a small
propositional DSL over mode atoms, booleans, and quantized continuous
quantities (interval domains with threshold atoms). Compilation is
exhaustive search with memoization — unit propagation, connected-
component decomposition (whence decomposable AND nodes), domain
branching (whence deterministic OR nodes), and component caching
(whence a DAG rather than a tree) — producing a smooth deterministic
decomposable circuit [Darwiche2001; DarwicheMarquis2002].

**Probability lives on the leaves.** Each `(variable, value)` leaf
carries a weight: mode priors are their values' weights, evidence
annihilates the weights of ruled-out values, *soft* evidence rescales
them (virtual evidence [Pearl1988]), and sensor noise is not a runtime
feature but desugaring — a declared false-positive/negative rate
becomes hidden glitch variables and a constraint, i.e., more model.
The weighted circuit computes weighted model counts exactly
[ChaviraDarwiche2008], and every quantity reported to a user is a
ratio of such counts: always normalized, always exact.

**One artifact.** The compiled circuit, its weights, and the variable
registry serialize to a single small file whose header states all
resource bounds up front, permitting single-allocation loading in an
embedded setting. Every capability in the remainder of the paper —
the query family (§3), tracking and learning (§4) — consumes this one
artifact; nothing recompiles online. A layered lowering of the same
circuit to batched tensor operations makes it differentiable, which
§4 exploits for gradient learning; full engineering detail is in the
open-source library's documentation rather than here.
