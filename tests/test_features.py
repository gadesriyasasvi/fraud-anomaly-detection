import pytest
from fraud_engine.features import validate_prediction_payload


def payload():
    return {"transaction_id": "TX1", "amount": 1200, "hour": 12,
            "recent_transaction_count": 2, "amount_to_average_ratio": 1.4}


def test_valid_payload():
    assert validate_prediction_payload(payload())["amount"] == 1200.0


def test_negative_amount_rejected():
    item = payload()
    item["amount"] = -1
    with pytest.raises(ValueError):
        validate_prediction_payload(item)


def test_invalid_hour_rejected():
    item = payload()
    item["hour"] = 24
    with pytest.raises(ValueError):
        validate_prediction_payload(item)
