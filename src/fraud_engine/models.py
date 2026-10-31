import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_classifier():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(class_weight="balanced", max_iter=1500, random_state=42)),
    ])


def build_detector():
    return IsolationForest(n_estimators=250, contamination="auto", random_state=42, n_jobs=-1)


def raw_anomaly_scores(model, x):
    return -model.decision_function(x)


def normalize_scores(reference, scores):
    reference = np.sort(np.asarray(reference, dtype=float))
    if not len(reference):
        raise ValueError("Reference anomaly scores cannot be empty")
    return np.clip(np.searchsorted(reference, scores, side="right") / len(reference), 0, 1)
