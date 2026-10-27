"""Prediction and ranking metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def rmse(y, p) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))


def mae(y, p) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def logloss(y, p, eps=1e-7) -> float:
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def auc(y, p) -> float:
    y = np.asarray(y)
    if y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, p))


def ece(y, p, n_bins: int = 10) -> tuple[float, list[dict]]:
    """Expected calibration error with equal-width bins; also returns the reliability table."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    table, err = [], 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            conf, acc = p[m].mean(), y[m].mean()
            err += m.mean() * abs(conf - acc)
            table.append({"bin": b, "n": int(m.sum()), "mean_pred": float(conf), "frac_pos": float(acc)})
    return float(err), table


def regression_calibration(y, p, n_bins: int = 10) -> list[dict]:
    """Mean target vs mean prediction in prediction-quantile bins (for watch-ratio regression)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    qs = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    bins = np.clip(np.searchsorted(qs, p, side="right") - 1, 0, n_bins - 1)
    return [{"bin": b, "n": int((bins == b).sum()), "mean_pred": float(p[bins == b].mean()),
             "mean_true": float(y[bins == b].mean())} for b in range(n_bins) if (bins == b).any()]


def gauc(user, y, p) -> float:
    """Impression-weighted mean of per-user AUC over users with both classes."""
    user, y, p = np.asarray(user), np.asarray(y), np.asarray(p)
    order = np.argsort(user, kind="stable")
    user, y, p = user[order], y[order], p[order]
    starts = np.r_[0, np.flatnonzero(user[1:] != user[:-1]) + 1, len(user)]
    tot, wsum = 0.0, 0
    for s, e in zip(starts[:-1], starts[1:]):
        yy = y[s:e]
        if 0 < yy.sum() < len(yy):
            tot += roc_auc_score(yy, p[s:e]) * (e - s)
            wsum += e - s
    return float(tot / max(wsum, 1))


def ndcg_at_k(true_gain: np.ndarray, scores: np.ndarray, k: int) -> np.ndarray:
    """Row-wise NDCG@k with graded gains (true_gain >= 0). NaN gains are treated as 0 and never ranked first."""
    g = np.nan_to_num(true_gain, nan=0.0)
    s = np.where(np.isnan(true_gain), -np.inf, scores)
    top = np.argsort(-s, axis=1)[:, :k]
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = (np.take_along_axis(g, top, axis=1) * disc).sum(1)
    ideal = (-np.sort(-g, axis=1)[:, :k] * disc).sum(1)
    return dcg / np.maximum(ideal, 1e-12)


def spearman_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata
    out = np.empty(len(a))
    for i in range(len(a)):
        m = ~(np.isnan(a[i]) | np.isnan(b[i]))
        ra, rb = rankdata(a[i, m]), rankdata(b[i, m])
        out[i] = np.corrcoef(ra, rb)[0, 1]
    return out
