import numpy as np
from sklearn.metrics import f1_score

WEIGHT = 0.70


def combine_scores(probability, anomaly, weight=WEIGHT):
    p, a = np.asarray(probability, dtype=float), np.asarray(anomaly, dtype=float)
    if not 0 <= weight <= 1:
        raise ValueError("weight must be between 0 and 1")
    if np.any((p < 0) | (p > 1)) or np.any((a < 0) | (a > 1)):
        raise ValueError("Scores must be between 0 and 1")
    return 100 * (weight * p + (1 - weight) * a)


def select_threshold(y, scores):
    candidates = np.unique(np.quantile(scores, np.linspace(.05, .95, 91)))
    best_t, best_f1 = float(np.median(scores)), -1
    for threshold in candidates:
        f1 = f1_score(y, scores >= threshold, zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = float(threshold), float(f1)
    return best_t, best_f1


def decision(score, threshold):
    if score >= threshold:
        return "FLAG"
    if score >= threshold * .55:
        return "REVIEW"
    return "ALLOW"
