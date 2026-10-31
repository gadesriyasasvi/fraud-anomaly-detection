import pytest
from fraud_engine.risk import combine_scores, decision


def test_fusion():
    assert combine_scores(.8, .9) == pytest.approx(83.0)


def test_invalid_probability_rejected():
    with pytest.raises(ValueError):
        combine_scores(1.2, .4)


def test_decision():
    assert decision(10, 70) == "ALLOW"
    assert decision(45, 70) == "REVIEW"
    assert decision(75, 70) == "FLAG"
