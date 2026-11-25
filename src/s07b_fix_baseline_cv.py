"""Recompute the one-variable baselines (B1, B2) of the development CV on EXACTLY the same rows as the
fitted models (the union of the purged validation blocks), and patch results/model.json.
Uses development data only: the test period is NOT touched (no re-evaluation, the guard flag stays valid)."""
import numpy as np
import polars as pl

from common import load_result, save_result
from config import CV_BLOCKS, DEV_END, PROC, PURGE_Q, qidx
from modeling_utils import metrics, purged_expanding_folds

f = pl.read_parquet(PROC / "features.parquet").filter(~pl.col("label_incomplete")).select(
    "qidx", "y", "equity_ratio", "texas_ratio").to_pandas()
dev = f[f["qidx"] <= qidx(*DEV_END)].reset_index(drop=True)
folds = list(purged_expanding_folds(dev["qidx"].values, [(qidx(*a), qidx(*b)) for a, b in CV_BLOCKS], PURGE_Q))
idx = np.unique(np.concatenate([va for _, va in folds]))
res = load_result("model")
print("validation rows:", len(idx), "| model rows in model.json:", res["cv_results"]["M2 gradient boosting, tuned"]["pooled"]["n"])
for nm, col, sign in (("B1 rule: low equity ratio", "equity_ratio", -1), ("B2 rule: Texas ratio", "texas_ratio", +1)):
    v = sign * dev[col].values
    s = np.nan_to_num(v, nan=np.nanmedian(v))
    m = metrics(dev["y"].values[idx], s[idx])
    res["cv_results"][nm]["pooled"] = m
    print(nm, {k: round(x, 4) for k, x in m.items() if isinstance(x, float)})
save_result("model", res)
print("model.json patched")
