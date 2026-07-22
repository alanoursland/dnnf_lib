# 7. Related work

**Knowledge compilation.** DNNF and its tractability map are
Darwiche's [Darwiche2001; DarwicheMarquis2002]; weighted min-cost
diagnosis on compiled circuits is Darwiche & Marquis
[DarwicheMarquis2004]; probabilistic inference as weighted model
counting, with the differential semantics we rely on for marginals and
gradients, is [Darwiche2003; ChaviraDarwiche2008]; the semiring
generalization is algebraic model counting [KimmigEtAl2017]. Our
contribution is not to this theory but to carrying it, integrated,
onto the deployed-diagnosis architecture.

**Compilers.** c2d, dsharp, and D4 [Darwiche2004c2d; MuiseEtAl2012;
LagniezMarquis2017] are the industrial line; §6 uses D4 as baseline
and validator, and the library interoperates rather than competes.

**Model-based diagnosis and mode estimation.** Livingstone's
conflict-directed mode estimation [WilliamsNayak1996] and the compiled
executives that followed it — notably MEXEC [Barrett2005], which
shares our offline/online split, native multi-valued circuits, and
min-sum evaluation, and to which this work adds true posteriors,
marginal MAP, tracking with learned dynamics hooks, and learning.

**Probabilistic circuits.** Sum-product networks and their
descendants [ChoiVergariVdB2020; PeharzEtAl2020] are the same
mathematical object grown from the learning side; their "circuit
flows" for EM are the responsibilities our backward pass computes.
The convergence of the two traditions on one structure is part of
this paper's premise.

**Neurosymbolic learning.** Training networks through WMC-based
losses is the semantic-loss / DeepProbLog line [XuEtAl2018;
ManhaeveEtAl2018]; our neural observation front-end (§4) is that
pattern applied to system-model diagnosis, with the training
degeneracies analyzed through the EM reading of gradient descent in
the companion paper [Oursland2025].
