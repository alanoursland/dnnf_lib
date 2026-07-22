# Chapter 11 — Under the hood (optional, recommended)

## Learning goals

- Trace, by hand, what the compiler does: propagate, split, branch,
  cache.
- Know what makes circuits large, and the measured limits of this
  library.
- Debug a blowup with `stats()` and the small-scale discipline.

## The compiler in four moves

Compilation is exhaustive search *that remembers*. On the residual
constraint set at each step:

1. **Propagate** — anything forced (a clause with one surviving
   option) is applied immediately. Forced facts become leaves glued on
   with AND.
2. **Split** — if the remaining constraints fall into groups sharing
   no variables, each group becomes an independent subproblem. This is
   where **decomposable AND nodes** come from: the split *is* the
   proof of disjointness. (Chapter 3, exercise 1: your light subsystem
   added a constant, not a multiple, because it split.)
3. **Branch** — pick a variable, make one child per domain value.
   That's an OR node, **deterministic by construction**: each child
   asserts a different value of the same variable.
4. **Cache** — before solving any subproblem, check whether an
   identical residual was solved before; if so, *point* at the old
   answer. The cache turns the search tree into a DAG, and it is the
   entire difference between exponential and small. In the chapter-2
   figure, every shared node is a cache hit.

Everything else — semirings, diagnosis, tracking, planning, learning —
never looks at the constraints again. Only these four moves do.

## What makes circuits big

The cache only hits when residual subproblems *repeat*, and they
repeat when the model has locality: chains, trees, loose couplings.
Measured on this repo's benchmark families (`docs/SCALE.md`):

| model | variables | circuit | compile |
|---|---|---|---|
| 32-bit adder, per-gate stuck-at faults | 1,313 | 5,777 nodes | 0.8 s |
| 200-stage process line | ~2,800 | 9,201 nodes | 31 s |

Circuit size grows **linearly** in both families — thousands of
components are fine when structure is local. Two things genuinely
hurt:

- **Dense coupling**: when everything constrains everything, residuals
  never repeat and never split. (Worst case is real: circuits *can* be
  exponential; the compiler finds small ones when they exist, not
  always.)
- **`modes_first=True` on the wrong shape.** Diagnosis compiles modes
  above everything else so `map_diagnoses` is exact. On
  monitoring-shaped models this is measured to be *free* (identical
  size). On densely coupled networks it's catastrophic — the adder
  blows up exponentially by 4 bits under it. If your model is
  gate-network-shaped, pass `modes_first=False` and use `diagnoses`
  (MPE) instead of `map_diagnoses`; that trade is fundamental
  (marginal MAP is genuinely harder), not a library quirk.

Branching *order* also matters (that's move 3's freedom):
`fd.compile_fd(..., heuristic=...)` offers `"dynamic"` (default),
`"minfill"`, `"dtree"` — measured honestly in `docs/SCALE.md`, none
dominates, dynamic is the sane default.

## The debugging discipline

When compilation is slow or the circuit huge:

1. **Shrink first.** Halve the model until it's fast; `stats()` at
   each size. Linear growth → you're fine, keep scaling. Doubling per
   step → find the coupling you added at the size where the curve
   bent.
2. **Suspect global constraints.** One innocent-looking "no two
   components may fail simultaneously" couples *every* mode pair and
   kills the split move. Localize or drop it, remeasure.
3. **Check `modes_first`** (above) before blaming the model.
4. **Never debug probability on a big model.** Correctness bugs
   reproduce at toy size, where brute force can referee (chapter 3's
   assert). Size bugs need big models; *truth* bugs never do.

## Exercise

Build a ring: n components, each constrained with its successor
(mod n). Compile at n = 4, 8, 16, 32 with each heuristic and tabulate
nodes. Rings are the minimal structure where variable order visibly
matters — explain why breaking the ring anywhere leaves a chain.

## Checkpoint

*Colleague: "compilation is exponential, so this whole approach can't
scale." Give the two-sentence honest answer.* (Worst case yes;
structured case measured linear to thousands of components — and the
compile is offline, once, while every online query is linear in the
result.)

Next: [Chapter 12 — Your project](12_your_project.md)
