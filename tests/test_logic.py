"""Unit tests for the pure-Python logic (no data download needed).  Run: python -m pytest tests -q
(or: python tests/test_logic.py)"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from modeling_utils import (Winsorizer, choose_threshold, event_confusion, expected_cost,  # noqa: E402
                            purged_expanding_folds)


def test_purged_folds_never_see_future_labels():
    q = np.repeat(np.arange(1984 * 4, 2010 * 4), 3)
    blocks = [(1990 * 4, 1992 * 4 + 3), (2000 * 4, 2003 * 4 + 3)]
    for tr, va in purged_expanding_folds(q, blocks, purge_q=5):
        # label window is 4 quarters: the last training label must end before validation starts
        assert q[tr].max() + 4 < q[va].min()
        assert q[tr].max() <= q[va].min() - 5


def test_event_confusion_counts_banks_not_rows():
    cert = np.array([1, 1, 1, 1, 2, 2, 3, 3])
    y = np.array([0, 0, 1, 1, 0, 0, 0, 0])
    alarm = np.array([0, 0, 0, 1, 1, 0, 0, 0])
    c = event_confusion(cert, y, alarm)
    assert c["events"] == 1 and c["caught"] == 1 and c["missed"] == 0   # bank 1 caught once is enough
    assert c["false_alarms"] == 1                                      # bank 2 alarm


def test_threshold_moves_with_cost_ratio():
    rng = np.random.RandomState(0)
    n = 20000
    cert = np.arange(n)
    y = (rng.rand(n) < 0.02).astype(int)
    s = np.clip(0.2 * y + rng.rand(n) * 0.8, 0, 1)
    t_cheap, _, _ = choose_threshold(cert, y, s, c_fn=100.0, c_fp=0.1)    # false alarms cheap -> low threshold
    t_dear, _, _ = choose_threshold(cert, y, s, c_fn=100.0, c_fp=20.0)    # false alarms dear -> high threshold
    assert t_cheap < t_dear


def test_winsorizer_uses_training_quantiles_only():
    tr = np.array([[0.0], [1.0], [2.0], [3.0], [1000.0]])
    w = Winsorizer(0, 80).fit(tr)
    out = w.transform(np.array([[5000.0], [-5.0]]))
    assert out[0, 0] == w.hi_[0] and out[1, 0] == w.lo_[0]


def test_label_definition_pandas_mirror():
    """Mirror of the Polars label: y = 1{FAILDATE in (REPDTE, REPDTE + 1y]}."""
    rep = pd.to_datetime(["2008-12-31", "2008-12-31", "2008-12-31", "2008-12-31"])
    fail = pd.to_datetime(["2009-01-15", "2009-12-31", "2010-01-01", "2008-12-31"])
    end = rep + pd.DateOffset(years=1)
    y = ((fail > rep) & (fail <= end)).astype(int)
    assert list(y) == [1, 1, 0, 0]       # fails on the report date itself -> not labelled (not a "future" event)


def test_ytd_decumulation_logic():
    """Q1=YTD, Q2=YTD2-YTD1, ... ; gap -> missing."""
    df = pd.DataFrame({"qidx": [8000, 8001, 8002, 8003, 8004, 8006],
                       "m": [3, 6, 9, 12, 3, 9], "ytd": [10, 25, 45, 70, 8, 30]})
    prev_q, prev_v = df.qidx.shift(1), df.ytd.shift(1)
    q = np.where(df.m == 3, df.ytd, np.where(df.qidx - prev_q == 1, df.ytd - prev_v, np.nan))
    assert list(q[:5]) == [10, 15, 20, 25, 8] and np.isnan(q[5])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
