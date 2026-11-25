"""Pure numpy/pandas/sklearn helpers (no Polars) so they can be unit-tested."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


class DropEmptyColumns(BaseEstimator, TransformerMixin):
    """Drop columns that have NO observed value in the TRAINING data (e.g. a field FDIC did not report yet
    in the early quarters). Learned in fit(), so it is part of the leak-free pipeline."""

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.keep_ = ~np.all(np.isnan(X), axis=0)
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)[:, self.keep_]


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip each column to [lo, hi] percentiles learned on the TRAINING data only."""

    def __init__(self, lo: float = 0.5, hi: float = 99.5):
        self.lo, self.hi = lo, hi

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lo_ = np.nanpercentile(X, self.lo, axis=0)
        self.hi_ = np.nanpercentile(X, self.hi, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)


def purged_expanding_folds(qidx: np.ndarray, blocks, purge_q: int):
    """Yield (train_idx, val_idx) for expanding-window CV with a purge gap.

    Validation = quarters in [start, end] of the block. Training = quarters <= start - purge_q
    (label window is 4 quarters, +1 for the filing lag), so no training label can look into the
    validation period. Quarters are absolute indices (year*4+q-1).
    """
    qidx = np.asarray(qidx)
    for (s, e) in blocks:
        tr = np.where(qidx <= s - purge_q)[0]
        va = np.where((qidx >= s) & (qidx <= e))[0]
        if len(tr) and len(va):
            yield tr, va


def event_confusion(cert: np.ndarray, y: np.ndarray, alarm: np.ndarray):
    """Event-level confusion for early warning.

    A failure event (bank) is CAUGHT if it raised >= 1 alarm in its pre-failure window
    (rows with y == 1). FP = alarm-quarters of banks that do not fail in the window.
    """
    cert, y, alarm = np.asarray(cert), np.asarray(y).astype(bool), np.asarray(alarm).astype(bool)
    fail_banks = np.unique(cert[y])
    caught = np.unique(cert[y & alarm])
    fp = int((alarm & ~y).sum())
    return {"events": int(len(fail_banks)), "caught": int(len(caught)),
            "missed": int(len(fail_banks) - len(caught)), "false_alarms": fp,
            "alarms": int(alarm.sum())}


def expected_cost(conf: dict, c_fn: float, c_fp: float) -> float:
    return conf["missed"] * c_fn + conf["false_alarms"] * c_fp


def choose_threshold(cert, y, score, c_fn: float, c_fp: float, n_grid: int = 400):
    """Threshold minimising event-level expected cost on development (out-of-fold) scores."""
    score = np.asarray(score)
    grid = np.unique(np.quantile(score, np.linspace(0.5, 0.99999, n_grid)))
    best = (np.inf, None)
    curve = []
    for t in grid:
        c = expected_cost(event_confusion(cert, y, score >= t), c_fn, c_fp)
        curve.append((float(t), float(c)))
        if c < best[0]:
            best = (c, float(t))
    return best[1], best[0], curve


def metrics(y, score, cert=None, top_frac: float = 0.05) -> dict:
    y = np.asarray(y)
    out = {"n": int(len(y)), "positives": int(y.sum()), "base_rate": float(y.mean())}
    if y.sum() == 0 or y.sum() == len(y):
        return out
    out["pr_auc"] = float(average_precision_score(y, score))
    out["roc_auc"] = float(roc_auc_score(y, score))
    k = max(1, int(len(y) * top_frac))
    top = np.argsort(-np.asarray(score))[:k]
    out[f"recall_top{int(top_frac * 100)}pct_rows"] = float(y[top].sum() / y.sum())
    out["precision_top5pct_rows"] = float(y[top].mean())
    out["lift_top5pct"] = float(y[top].mean() / y.mean())
    try:
        out["brier"] = float(brier_score_loss(y, np.clip(score, 0, 1)))
    except ValueError:
        pass
    return out


def quarterly_recall_at_k(df: pd.DataFrame, score_col: str, k: float = 0.05) -> float:
    """Share of failing banks that are in the top-k% of the quarter's risk ranking at least once."""
    df = df.copy()
    df["rank_pct"] = df.groupby("qidx")[score_col].rank(ascending=False, pct=True)
    pos = df[df["y"] == 1]
    if pos.empty:
        return float("nan")
    return float(pos.groupby("CERT")["rank_pct"].min().le(k).mean())