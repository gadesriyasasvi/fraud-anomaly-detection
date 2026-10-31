import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from fraud_engine.features import FEATURES, TARGET
from fraud_engine.models import raw_anomaly_scores, normalize_scores
from fraud_engine.risk import combine_scores
from fraud_engine.train import ARTIFACTS, REPORTS, metric_set


def evaluate():
    root = Path(__file__).resolve().parents[2]
    test_path = root / "data" / "processed" / "test.csv"
    if not test_path.exists() or not (ARTIFACTS / "metadata.json").exists():
        raise FileNotFoundError("Run `python -m fraud_engine.cli train` first.")
    df = pd.read_csv(test_path)
    x, y = df[FEATURES], df[TARGET].to_numpy()
    classifier = joblib.load(ARTIFACTS / "classifier.joblib")
    detector = joblib.load(ARTIFACTS / "anomaly_detector.joblib")
    reference = np.load(ARTIFACTS / "reference_anomaly_scores.npy")
    meta = json.loads((ARTIFACTS / "metadata.json").read_text(encoding="utf-8"))
    prob = classifier.predict_proba(x)[:, 1]
    anomaly = normalize_scores(reference, raw_anomaly_scores(detector, x.to_numpy()))
    hybrid = combine_scores(prob, anomaly)
    hybrid_pred = (hybrid >= meta["hybrid_threshold"]).astype(int)
    supervised_score = 100 * prob
    supervised_pred = (supervised_score >= meta["supervised_threshold"]).astype(int)
    report = {
        "test_rows": int(len(df)),
        "fraud_rows": int(y.sum()),
        "supervised_only": metric_set(y, supervised_pred, prob),
        "hybrid": metric_set(y, hybrid_pred, hybrid),
        "hybrid_confusion_matrix_tn_fp_fn_tp": confusion_matrix(y, hybrid_pred, labels=[0, 1]).ravel().tolist(),
        "note": "Held-out test set; synthetic data results are only for pipeline verification.",
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "test_evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
