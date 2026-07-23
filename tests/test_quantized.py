"""Quantized continuous variables: threshold atoms and numeric evidence."""

import math

import pytest

from modenexus import SystemModel


def build_tank():
    """Tank level quantized into [0,10) [10,50) [50,100); a low-level
    alarm watches level < 10."""
    m = SystemModel()
    level = m.quantized(
        "level", (0.0, 10.0, 50.0, 100.0), priors=(0.2, 0.5, 0.3)
    )
    m.sensor("low_alarm", level.below(10.0))
    return m, level


def test_threshold_atom_probability():
    m, _ = build_tank()
    sys = m.compile()
    assert sys.log_evidence({"low_alarm": True}) == pytest.approx(
        math.log(0.2)
    )
    assert sys.log_evidence({"low_alarm": False}) == pytest.approx(
        math.log(0.8)
    )


def test_numeric_evidence_buckets():
    m, _ = build_tank()
    sys = m.compile()
    # 37.2 falls in [10,50): alarm must be off.
    assert sys.log_evidence({"level": 37.2, "low_alarm": False}) == \
        pytest.approx(math.log(0.5))
    assert sys.log_evidence({"level": 37.2, "low_alarm": True}) == -math.inf
    # 3.0 falls in [0,10).
    assert sys.log_evidence({"level": 3.0, "low_alarm": True}) == \
        pytest.approx(math.log(0.2))


def test_posterior_over_buckets():
    m, _ = build_tank()
    sys = m.compile()
    post = sys.mode_posteriors({"low_alarm": False})
    assert post["level"]["[0.0,10.0)"] == pytest.approx(0.0)
    assert post["level"]["[10.0,50.0)"] == pytest.approx(0.5 / 0.8)
    assert post["level"]["[50.0,100.0)"] == pytest.approx(0.3 / 0.8)


def test_between_and_at_least():
    m = SystemModel()
    t = m.quantized("temp", (0, 20, 40, 60, 80), priors=(0.1, 0.2, 0.3, 0.4))
    m.sensor("mid", t.between(20, 60))
    m.sensor("hot", t.at_least(60))
    sys = m.compile()
    assert sys.log_evidence({"mid": True}) == pytest.approx(math.log(0.5))
    assert sys.log_evidence({"hot": True}) == pytest.approx(math.log(0.4))
    assert sys.log_evidence({"mid": True, "hot": True}) == -math.inf


def test_boundary_validation():
    m = SystemModel()
    t = m.quantized("t", (0, 10, 20))
    with pytest.raises(ValueError):
        t.below(15)  # not a boundary
    with pytest.raises(ValueError):
        t.bucket_of(25)  # out of range
    with pytest.raises(ValueError):
        m.quantized("bad", (0, 10))  # only one bucket
    with pytest.raises(ValueError):
        m.quantized("bad2", (0, 10, 5))  # not increasing


def test_quantized_mode_in_diagnoses():
    """A quantized quantity can itself be a mode-like hidden state."""
    m = SystemModel()
    leak = m.mode("leak", ("none", "small", "large"), priors=(0.9, 0.08, 0.02))
    level = m.quantized("level", (0.0, 10.0, 50.0, 100.0))
    # Physics: large leak -> level below 10; small leak -> below 50;
    # no leak -> at least 50.
    m.add((leak == "large") >> level.below(10.0))
    m.add((leak == "small") >> level.between(10.0, 50.0))
    m.add((leak == "none") >> level.at_least(50.0))
    sys = m.compile()
    diags = sys.map_diagnoses({"level": 30.0}, k=3)
    assert diags[0].modes["leak"] == "small"
    assert diags[0].posterior == pytest.approx(1.0)
    diags = sys.map_diagnoses({"level": 75.0}, k=3)
    assert diags[0].modes["leak"] == "none"
