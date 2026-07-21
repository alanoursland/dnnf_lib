"""Executable companion to "Gradient Descent as Implicit EM" (Oursland,
https://github.com/alanoursland/gradient_descent_as_implicit_em).

The paper's Theorem 1: for log-sum-exp objectives over distances,
``dL/dd_i = -r_i`` — the gradient of an LSE with respect to each distance
is the negative posterior responsibility of the corresponding component
(Fisher's identity).  A smooth d-DNNF evaluated in log space is a *DAG of
LSE nodes over neg-log-weight distances*, so this library is an explicit,
exactly-solvable instance of the paper's setting, with composition depth.

Three verifications:

1. **Gradients are responsibilities, at depth.**  The backward pass of
   log-WMC through a compiled multi-level diagnosis circuit equals the
   exact posterior marginals computed independently by WMC ratios — the
   identity survives arbitrary composition through the circuit DAG
   (Darwiche's differential semantics is Theorem 1 applied recursively).
2. **EM and SGD share one fixed point.**  With priors pi = softmax(theta),
   d log P(evidence)/d theta = mean-responsibility - pi.  EM jumps to the
   stationary point (M-step: pi := mean responsibility); SGD walks to it.
   Both learners land on the same parameters, and at the fixed point the
   average posterior responsibility equals the prior.
3. **Violating the structural conditions degenerates the objective,
   exactly as the theorem predicts.**  (i) Without normalization the
   "responsibility" reading of the gradient breaks: a pure scale (gauge)
   direction increases the objective unboundedly while changing no
   posterior.  (ii) Without an observation clamping responsibility
   anywhere, ignoring the input is optimal (posterior collapse): a
   constant detector reporting the a-priori likely value beats a
   truthful one.
"""

import math
import random

import pytest

from dnnf import SystemModel, iff


def build_deep_system():
    """Two-level system so the circuit has genuinely nested OR (LSE)
    nodes: modes -> intermediate flows -> observed alarm."""
    m = SystemModel()
    v1 = m.mode("v1", ("ok", "bad"), priors=(0.9, 0.1))
    v2 = m.mode("v2", ("ok", "weak", "bad"), priors=(0.8, 0.15, 0.05))
    f1 = m.bool("f1")
    f2 = m.bool("f2")
    m.add(iff(f1, v1 == "ok"))
    m.add(iff(f2, f1 & (v2 != "bad")))
    m.sensor("alarm", ~f2, false_positive=0.05, false_negative=0.05)
    return m.compile()


def test_1_gradients_are_responsibilities_at_depth():
    torch = pytest.importorskip("torch")
    from dnnf.torch_backend import TorchCircuit

    sys = build_deep_system()
    tc = TorchCircuit(sys.circuit, semiring="logprob")
    evidence = {"alarm": True}

    # Backward pass: d logWMC / d log w  (one E-step for every leaf).
    w = torch.tensor(sys.log_weights_for(evidence), dtype=torch.float64)
    grads = tc.marginals(w)[0]

    # Independent E-step: posterior of every variable by WMC ratios.
    spec = sys.circuit.spec
    for name, var in sys.vars.items():
        if name in evidence:
            # Observed variable: all responsibility on the observed value.
            i = var.values.index(evidence[name])
            assert grads[spec.mvlit(var.fd_var, i)].item() == \
                pytest.approx(1.0, abs=1e-9)
            continue
        post = sys.posteriors(evidence, names=[name])[name]
        for i, value in enumerate(var.values):
            r = grads[spec.mvlit(var.fd_var, i)].item()
            assert r == pytest.approx(post[value], abs=1e-9), (
                f"gradient != responsibility for {name}={value}"
            )
    # Responsibilities are normalized per variable (they lie on the
    # simplex): the structural reason the objective is well-posed.
    for name, var in sys.vars.items():
        total = sum(
            grads[spec.mvlit(var.fd_var, i)].item()
            for i in range(len(var.values))
        )
        assert total == pytest.approx(1.0, abs=1e-9)


def test_2_em_and_sgd_share_the_fixed_point():
    pytest.importorskip("torch")

    def build(p):
        m = SystemModel()
        v = m.mode("v", ("ok", "bad"), priors=(1 - p, p))
        m.sensor("alarm", v == "bad", false_positive=0.1, false_negative=0.1)
        return m.compile()

    rng = random.Random(5)
    true = build(0.3)
    obs = [{"alarm": true.sample_state(rng)["alarm"]} for _ in range(2500)]

    em, sgd = build(0.5), build(0.5)
    em.fit_priors(obs, names=["v"], iterations=100, tol=1e-13)
    sgd.fit_priors_torch(obs, names=["v"], epochs=500, lr=0.1)

    spec = em.circuit.spec
    slot = spec.mvlit(em.vars["v"].fd_var, 1)
    assert em._weights[slot] == pytest.approx(sgd._weights[slot], abs=0.01)

    # Fixed-point property: at convergence the prior equals the average
    # posterior responsibility over the telemetry (gradient = r_bar - pi = 0).
    r_bar = sum(
        em.posteriors(o, names=["v"])["v"]["bad"] for o in obs
    ) / len(obs)
    assert em._weights[slot] == pytest.approx(r_bar, abs=1e-5)


def test_3a_no_normalization_gauge_explosion():
    sys = build_deep_system()
    # Unnormalized virtual evidence scaled by c: the objective (log
    # evidence) grows by log c without bound...
    base = sys.log_evidence({"f2": (1.0, 1.0)})
    scaled = sys.log_evidence({"f2": (10.0, 10.0)})
    assert scaled == pytest.approx(base + math.log(10.0), abs=1e-9)
    # ...while every posterior is unchanged: the runaway direction is
    # pure gauge, which is why normalization (log-softmax / the M-step's
    # simplex projection) removes it exactly rather than approximately.
    p1 = sys.posteriors({"f2": (1.0, 1.0)})
    p2 = sys.posteriors({"f2": (10.0, 10.0)})
    for name in p1:
        for value in p1[name]:
            assert p1[name][value] == pytest.approx(
                p2[name][value], abs=1e-9
            )


def test_3b_no_clamping_posterior_collapse():
    torch = pytest.importorskip("torch")
    from dnnf.torch_learn import ObservationTrainer

    m = SystemModel()
    v = m.mode("v", ("ok", "bad"), priors=(0.7, 0.3))
    flow = m.bool("flow")
    m.add(iff(flow, v == "ok"))
    sys = m.compile()
    trainer = ObservationTrainer(sys, names=["flow"])

    B = 8
    # A "truthful" detector: confident, correct half the time each way.
    truthful = torch.tensor(
        [[0.0, 4.0], [4.0, 0.0]] * (B // 2), dtype=torch.float64
    )
    # A collapsed detector: always reports the a-priori likely value.
    collapsed = torch.tensor([[0.0, 4.0]] * B, dtype=torch.float64)

    # With no hard evidence anywhere (masks=None), collapse wins: the
    # objective is satisfiable without using the input.
    ll_truthful = trainer.log_likelihood(truthful).mean().item()
    ll_collapsed = trainer.log_likelihood(collapsed).mean().item()
    assert ll_collapsed > ll_truthful

    # Ground the structure with a hard observation channel and the
    # ordering reverses for the rows where the evidence disagrees with
    # the collapsed report: clamping is what makes truth pay.
    m2 = SystemModel()
    v2 = m2.mode("v", ("ok", "bad"), priors=(0.7, 0.3))
    flow2 = m2.bool("flow")
    m2.add(iff(flow2, v2 == "ok"))
    m2.add(iff(m2.bool("alarm"), v2 == "bad"))
    sys2 = m2.compile()
    tr2 = ObservationTrainer(sys2, names=["flow"])
    hard = [{"alarm": i % 2 == 1} for i in range(B)]  # alternating truth
    masks = tr2.masks_for(hard)
    ll_truthful2 = tr2.log_likelihood(truthful, masks).mean().item()
    ll_collapsed2 = tr2.log_likelihood(collapsed, masks).mean().item()
    assert ll_truthful2 > ll_collapsed2
