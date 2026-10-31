# Fraud & Anomaly Detection Engine

A complete starter project combining supervised classification, unsupervised anomaly detection, class-imbalance handling, risk fusion, explanations, evaluation, and a FastAPI prediction service.

**Data note:** The default workflow generates synthetic transaction data so you can run everything immediately without a Kaggle account. Its metrics verify that the pipeline runs; they do not demonstrate real-world fraud performance.

## Requirements
- Windows 10/11
- Python 3.11
- VS Code recommended

## Setup (Windows / VS Code)

1. Extract `fraud-anomaly-detection.zip`.
2. Open the extracted `fraud-anomaly-detection` folder in VS Code.
3. Select **Terminal → New Terminal** and run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal and activate again. If `py -3.11` is not found, install Python 3.11 from https://www.python.org/downloads/ and enable **Add Python to PATH**.

## Run the project

### 1. Generate demo data
```powershell
python -m fraud_engine.cli generate-data
```

### 2. Train both models
```powershell
python -m fraud_engine.cli train
```

### 3. Evaluate on the held-out test set
```powershell
python -m fraud_engine.cli evaluate
```

Training creates `artifacts/` files; evaluation writes `reports/test_evaluation.json`.

### 4. Start the API
```powershell
uvicorn fraud_engine.api:app --reload
```

Open http://127.0.0.1:8000/docs. Expand `POST /predict`, click **Try it out**, and submit:

```json
{
  "transaction_id": "TX-DEMO-001",
  "amount": 35000,
  "hour": 2,
  "recent_transaction_count": 7,
  "amount_to_average_ratio": 18.5
}
```

You can also open `/health` and `/model-info`. Stop the server with `Ctrl+C`.

### 5. Run tests
In a second terminal, activate the environment if needed, then:
```powershell
python -m pytest -v
```

## Project flow
1. Generate/load and validate labeled transactions.
2. Split into train (60%), validation (20%), and test (20%).
3. Train a class-weighted Logistic Regression fraud classifier.
4. Train Isolation Forest on legitimate training rows.
5. Normalize anomaly scores against the training reference distribution.
6. Tune a hybrid risk threshold on validation data.
7. Evaluate on the untouched test set.
8. Return risk score, decision, and transparent rule-based reasons via FastAPI.

## Important interpretation
- Precision, recall, F1, PR-AUC, ROC-AUC, and confusion matrix are reported.
- The risk score is a prototype index, **not a calibrated fraud probability**.
- Reasons are transparent input-based rules, not causal proof or SHAP values.
- The demo schema is synthetic and includes amount, hour, recent transaction count, amount-to-average ratio, and fraud label.
- For real data, adapt `features.py` and the data pipeline to the real schema. Do not invent unavailable fields. For timestamped transactions, consider chronological splitting and leakage-safe historical features.
- Never oversample before splitting. Thresholds and fusion choices must be selected using validation data, not the test set.
