from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from fraud_engine.features import FEATURES, TARGET

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "transactions.csv"
PROCESSED = ROOT / "data" / "processed"


def generate_data(n=12000, seed=42):
    rng = np.random.default_rng(seed)
    amount = np.clip(rng.lognormal(5.7, 1.0, n), 5, 150000)
    hour = rng.integers(0, 24, n)
    recent = rng.poisson(1.7, n)
    ratio = np.clip(rng.lognormal(0, 0.7, n), 0.05, 80)
    night = ((hour <= 4) | (hour >= 23)).astype(float)
    amount_signal = np.clip(np.log1p(amount) - 7, 0, 4)
    velocity_signal = np.clip(recent - 2, 0, 8) / 4
    ratio_signal = np.clip(np.log1p(ratio) - 1, 0, 4)
    logits = -5 + .75 * night + .60 * amount_signal + .55 * velocity_signal + .75 * ratio_signal
    p = 1 / (1 + np.exp(-logits))
    labels = rng.binomial(1, p)
    if labels.sum() < 30:
        labels[np.argsort(p)[-max(30, n // 100):]] = 1
    return pd.DataFrame({
        "transaction_id": [f"TX{i:07d}" for i in range(n)],
        "amount": amount.round(2), "hour": hour,
        "recent_transaction_count": recent,
        "amount_to_average_ratio": ratio.round(3), TARGET: labels.astype(int)
    })


def ensure_data():
    RAW.parent.mkdir(parents=True, exist_ok=True)
    if not RAW.exists():
        generate_data().to_csv(RAW, index=False)
    return RAW


def make_splits():
    frame = pd.read_csv(ensure_data())
    required = ["transaction_id", *FEATURES, TARGET]
    missing = [c for c in required if c not in frame]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    if frame[required].isna().any().any():
        raise ValueError("Training CSV contains missing values")
    if frame.transaction_id.duplicated().any():
        raise ValueError("transaction_id values must be unique")
    if not set(frame[TARGET].unique()).issubset({0, 1}):
        raise ValueError("is_fraud must contain only 0 and 1")
    if frame[TARGET].nunique() != 2:
        raise ValueError("Training data must contain both classes")
    train_val, test = train_test_split(frame, test_size=.2, random_state=42, stratify=frame[TARGET])
    train, val = train_test_split(train_val, test_size=.25, random_state=42, stratify=train_val[TARGET])
    PROCESSED.mkdir(parents=True, exist_ok=True)
    train.to_csv(PROCESSED / "train.csv", index=False)
    val.to_csv(PROCESSED / "validation.csv", index=False)
    test.to_csv(PROCESSED / "test.csv", index=False)
    return train, val, test
