import json
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fraud_engine.explain import explain
from fraud_engine.features import FEATURES, validate_prediction_payload
from fraud_engine.models import raw_anomaly_scores, normalize_scores
from fraud_engine.risk import combine_scores, decision
from fraud_engine.train import ARTIFACTS

app = FastAPI(title="Fraud & Anomaly Detection Engine", version="1.0.0",
              description="Educational prototype; a risk score does not prove fraud.")


class Transaction(BaseModel):
    transaction_id: str = Field(min_length=1)
    amount: float = Field(ge=0)
    hour: int = Field(ge=0, le=23)
    recent_transaction_count: int = Field(ge=0)
    amount_to_average_ratio: float = Field(ge=0)


class Batch(BaseModel):
    transactions: list[Transaction] = Field(min_length=1, max_length=500)


def load_models():
    paths = [ARTIFACTS / "classifier.joblib", ARTIFACTS / "anomaly_detector.joblib",
             ARTIFACTS / "reference_anomaly_scores.npy", ARTIFACTS / "metadata.json"]
    if not all(p.exists() for p in paths):
        raise RuntimeError("Models missing. Run `python -m fraud_engine.cli train` first.")
    return (joblib.load(paths[0]), joblib.load(paths[1]), np.load(paths[2]),
            json.loads(paths[3].read_text(encoding="utf-8")))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    try:
        _, _, _, meta = load_models()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"model_version": meta["model_version"], "features": meta["features"], "data_note": meta["data_note"]}


def score_payload(payload):
    try:
        row = validate_prediction_payload(payload)
        classifier, detector, reference, meta = load_models()
    except (ValueError, RuntimeError) as exc:
        code = 400 if isinstance(exc, ValueError) else 503
        raise HTTPException(status_code=code, detail=str(exc)) from exc
    x = pd.DataFrame([[row[k] for k in FEATURES]], columns=FEATURES)
    probability = float(classifier.predict_proba(x)[0, 1])
    anomaly = float(normalize_scores(reference, raw_anomaly_scores(detector, x.to_numpy()))[0])
    risk = float(combine_scores(probability, anomaly))
    threshold = float(meta["hybrid_threshold"])
    return {
        "transaction_id": row["transaction_id"],
        "fraud_probability": round(probability, 4),
        "anomaly_score": round(anomaly, 4),
        "risk_score": round(risk, 2),
        "decision": decision(risk, threshold),
        "decision_threshold": round(threshold, 2),
        "reasons": explain(row, anomaly),
        "model_version": meta["model_version"],
        "disclaimer": "Risk score is not a calibrated fraud probability. Review flagged transactions before action."
    }


@app.post("/predict")
def predict(transaction: Transaction):
    return score_payload(transaction.model_dump())


@app.post("/predict/batch")
def predict_batch(batch: Batch):
    return {"results": [score_payload(item.model_dump()) for item in batch.transactions]}
