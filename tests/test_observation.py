"""Soft (virtual) evidence and neural observation-model training."""

import math
import random

import pytest

from neximode import SystemModel, iff


def build():
    m = SystemModel()
    v = m.mode("v", ("ok", "bad"), priors=(0.7, 0.3))
    flow = m.bool("flow")
    m.add(iff(flow, v == "ok"))
    return m.compile()


def test_soft_evidence_posterior():
    sys = build()
    # Virtual evidence on flow: L(reading|flow=F)=0.8, L(|flow=T)=0.2.
    # flow=T iff v=ok, so P(bad|reading) = .3*.8 / (.3*.8 + .7*.2).
    post = sys.posteriors({"flow": (0.8, 0.2)})  # value order (False, True)
    assert post["v"]["bad"] == pytest.approx(0.24 / 0.38)
    # Dict form, and hard evidence still works.
    post2 = sys.posteriors({"flow": {False: 0.8, True: 0.2}})
    assert post2["v"]["bad"] == pytest.approx(0.24 / 0.38)
    assert sys.posteriors({"flow": False})["v"]["bad"] == pytest.approx(1.0)


def test_soft_evidence_in_mpe_costs():
    sys = build()
    costs = sys._conditioned_costs({"flow": (0.9, 0.1)})
    import neximode.fd as fd

    cost, best = fd.mpe(sys.circuit, costs)
    # ok path: cost -log(.7) - log(.1); bad path: -log(.3) - log(.9)
    assert cost == pytest.approx(-math.log(0.3) - math.log(0.9))
    decoded = sys._decode_state(best)
    assert decoded["v"] == "bad"


def test_observation_trainer_learns_without_labels():
    torch = pytest.importorskip("torch")
    from neximode.torch_learn import ObservationTrainer

    # Grounded setup: alarm (hard-observed, iff v=bad) anchors the
    # hidden state; the net reads raw x for flow (iff v=ok).  Flow is
    # never labeled — the structure links it to the alarm telemetry.
    m = SystemModel()
    v = m.mode("v", ("ok", "bad"), priors=(0.7, 0.3))
    flow = m.bool("flow")
    m.add(iff(flow, v == "ok"))
    m.add(iff(m.bool("alarm"), v == "bad"))
    sys = m.compile()

    rng = random.Random(0)
    xs, hard = [], []
    for _ in range(600):
        bad = rng.random() < 0.3
        xs.append([rng.gauss(-1.0 if bad else 1.0, 0.7)])
        hard.append({"alarm": bad})
    inputs = torch.tensor(xs, dtype=torch.float64)

    torch.manual_seed(0)
    net = torch.nn.Sequential(
        torch.nn.Linear(1, 8), torch.nn.Tanh(), torch.nn.Linear(8, 2)
    ).double()
    trainer = ObservationTrainer(sys, names=["flow"])
    masks = trainer.masks_for(hard)
    history = trainer.fit(net, inputs, masks, epochs=150, lr=0.05)
    assert history[-1] > history[0]  # likelihood improved

    # The detector must have learned the direction: strong positive x
    # should favor flow=True (slot order False, True).
    with torch.no_grad():
        lo = net(torch.tensor([[-2.0]], dtype=torch.float64))[0]
        hi = net(torch.tensor([[2.0]], dtype=torch.float64))[0]
    assert (hi[1] - hi[0]) > (lo[1] - lo[0])
    assert hi[1] > hi[0]      # x=+2 -> flow likely
    assert lo[0] > lo[1]      # x=-2 -> no-flow likely

    # And its soft outputs plug straight into exact diagnosis.
    lik = torch.softmax(lo, dim=0).tolist()
    post = sys.posteriors({"flow": (lik[0], lik[1])})
    assert post["v"]["bad"] > 0.5
