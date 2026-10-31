def explain(transaction, anomaly_score):
    reasons = []
    if transaction["amount"] >= 20000:
        reasons.append("Transaction amount is high relative to the demo rule threshold")
    if transaction["hour"] <= 4 or transaction["hour"] >= 23:
        reasons.append("Transaction occurred during late-night hours")
    if transaction["recent_transaction_count"] >= 5:
        reasons.append("Recent transaction count is elevated")
    if transaction["amount_to_average_ratio"] >= 5:
        reasons.append("Amount is much higher than the account reference average")
    if anomaly_score >= .90:
        reasons.append("Transaction is highly unusual relative to the anomaly reference data")
    return reasons or ["No strong rule-based warning signal was triggered"]
