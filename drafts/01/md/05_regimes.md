# 5. The three regimes as system operations

The companion paper distinguishes three regimes of the one mechanism by
what is observed versus latent [Oursland2025]. In the circuit system
each regime is a distinct, pre-existing API operation — none was built
with the regimes in mind.

| Regime [Oursland2025] | Circuit operation | Responsibilities weight |
|---|---|---|
| Unsupervised (mixture learning): components compete freely | Prior learning over mode values (`fit_priors`, `fit_priors_torch`): no value is clamped; $\nabla_\theta = \bar r - \pi$ | which mode value explains each observation; the M-step is the simplex projection |
| Conditional (attention): internal LSE, responsibility-weighted value gradients | Internal (non-root) OR nodes: gradients through them are responsibility-weighted by the chain rule; their children are the "values," their accumulated distances the "scores" | which branch of a sub-mixture explains the evidence reaching it |
| Constrained (classification): the label clamps responsibility to 1 | Hard evidence: $-\infty$ annihilation of competing values forces responsibility 1 on the observed value; the training objective becomes clamped-vs-free log-partition, the structure of cross-entropy | the deficit between clamped and free responsibilities |

Two remarks. First, the constrained regime's clamping is not an added
loss term here but an *operation on the measure* — evidence
conditioning — which makes its role in shaping gradients unusually
visible: Section 4.3(b) is what training looks like when it is removed.
Second, the conditional regime appears without any attention-like
architecture: internal responsibility-weighting is a property of
composed LSE, not of a particular network design, which supports the
companion paper's claim that the three regimes are one mechanism.
