FEATURES = ["amount", "hour", "recent_transaction_count", "amount_to_average_ratio"]
TARGET = "is_fraud"


def validate_prediction_payload(payload):
    required = ["transaction_id", *FEATURES]
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")
    result = {"transaction_id": str(payload["transaction_id"]).strip()}
    if not result["transaction_id"]:
        raise ValueError("transaction_id cannot be blank")
    for key in FEATURES:
        try:
            result[key] = float(payload[key])
        except (TypeError, ValueError):
            raise ValueError(f"{key} must be numeric") from None
    if result["amount"] < 0:
        raise ValueError("amount cannot be negative")
    if not result["hour"].is_integer() or not 0 <= result["hour"] <= 23:
        raise ValueError("hour must be an integer between 0 and 23")
    if result["recent_transaction_count"] < 0 or not result["recent_transaction_count"].is_integer():
        raise ValueError("recent_transaction_count must be a non-negative integer")
    if result["amount_to_average_ratio"] < 0:
        raise ValueError("amount_to_average_ratio cannot be negative")
    return result
