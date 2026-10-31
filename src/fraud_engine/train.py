import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from fraud_engine.data import make_splits
from fraud_engine.features import FEATURES, TARGET
from fraud_engine.models import build_classifier, build_detector, raw_anomaly_scores, normalize_scores
from fraud_engine.risk import combine_scores, select_threshold

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"


def metric_set(y, pred, score):
    result = {
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y, score)),
    }
    try:
        result["roc_auc"] = float(roc_auc_score(y, score))
    except ValueError:
        result["roc_auc"] = None
    return result


def train():
    train_df, val_df, _ = make_splits()
    xtr, ytr = train_df[FEATURES], train_df[TARGET].to_numpy()
    xv, yv = val_df[FEATURES], val_df[TARGET].to_numpy()
    classifier = build_classifier().fit(xtr, ytr)
    probability = classifier.predict_proba(xv)[:, 1]

    normal_x = xtr.loc[ytr == 0]
    detector = build_detector().fit(normal_x)
    reference = raw_anomaly_scores(detector, normal_x.to_numpy())
    anomaly = normalize_scores(reference, raw_anomaly_scores(detector, xv.to_numpy()))
    hybrid = combine_scores(probability, anomaly)
    hybrid_t, hybrid_f1 = select_threshold(yv, hybrid)
    supervised_scores = 100 * probability
    supervised_t, supervised_f1 = select_threshold(yv, supervised_scores)

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, ARTIFACTS / "classifier.joblib")
    joblib.dump(detector, ARTIFACTS / "anomaly_detector.joblib")
    np.save(ARTIFACTS / "reference_anomaly_scores.npy", reference)
    metadata = {
        "model_version": "1.0.0",
        "features": FEATURES,
        "supervised_weight": .70,
        "hybrid_threshold": hybrid_t,
        "supervised_threshold": supervised_t,
        "validation_hybrid_f1": hybrid_f1,
        "validation_supervised_f1": supervised_f1,
        "training_rows": len(train_df),
        "validation_rows": len(val_df),
        "fraud_rate_train": float(ytr.mean()),
        "data_note": "Synthetic demonstration data unless user replaces input CSV",
    }
    (ARTIFACTS / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (REPORTS / "training_summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
