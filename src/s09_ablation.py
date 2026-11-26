"""PART 4 evidence - feature-group ablation on DEVELOPMENT data only (the test period is never loaded).

Question: is every feature group worth keeping, and do the history-based (lagged / rolling) features earn their
construction cost?  Method: same purged expanding-window CV folds, same pipeline and same tuned M2 hyper-parameters as
s07_model.py; refit with one group removed at a time and report the change in PR-AUC against the full feature set.
Because failures are few in some folds, a mean-fold change smaller than the fold-to-fold spread is not significant;
the per-fold deltas are stored so the report can show how many folds agree on the sign.

Outputs: results/ablation.json, figures/ablation.png
"""
from __future__ import annotations

import time

import numpy as np
import polars as pl

from common import PALETTE, load_result, save_result, savefig, style_axes
from config import CV_BLOCKS, DEV_END, PROC, PURGE_Q, qidx
from modeling_utils import metrics, purged_expanding_folds
from s06_features import FEATURES
from s07_model import cv_scores

GROUPS = {
    "size / age / history": ["log_asset", "age_years", "n_quarters_seen"],
    "capital": ["equity_ratio", "RBC1AAJ", "RBCRWAJ", "d_equity_ratio_4q"],
    "asset quality": ["npl_ratio", "past_due_30_89", "past_due_90", "nonaccrual_assets", "ore_ratio", "allowance_cov",
                      "texas_ratio", "d_npl_ratio_4q", "nco_rate", "provision_rate"],
    "earnings / efficiency": ["roa_q", "ROA", "ROE", "NIMY", "efficiency", "nonint_income_share", "roa_q_mean4",
                              "roa_q_std4", "d_roa_4q"],
    "growth / loan mix": ["asset_growth_4q", "loan_growth_4q", "deposit_growth_4q", "loans_assets", "loan_deposit",
                          "re_loan_share", "ci_loan_share", "con_loan_share"],
    "funding / liquidity": ["brokered_share", "liquid_assets", "securities_share", "d_brokered_4q"],
}
# features that need a bank's own history (lags / trailing windows) = the part built with window functions
HISTORY = ["d_equity_ratio_4q", "d_npl_ratio_4q", "d_roa_4q", "d_brokered_4q", "roa_q_mean4", "roa_q_std4",
           "asset_growth_4q", "loan_growth_4q", "deposit_growth_4q"]


def main():
    t0 = time.time()
    assert set(sum(GROUPS.values(), [])) == set(FEATURES), "groups must partition FEATURES"
    mdl = load_result("model")
    params = mdl["cv_results"]["M2 gradient boosting, tuned"]["params"]
    feats = pl.read_parquet(PROC / "features.parquet").filter(~pl.col("label_incomplete"))
    dev = feats.filter(pl.col("qidx") <= qidx(*DEV_END)).to_pandas().reset_index(drop=True)   # test rows never loaded
    assert dev["qidx"].max() <= qidx(*DEV_END)
    folds = list(purged_expanding_folds(dev["qidx"].values, [(qidx(*a), qidx(*b)) for a, b in CV_BLOCKS], PURGE_Q))

    variants = {"all features (reference)": FEATURES}
    for g, cols in GROUPS.items():
        variants[f"without {g}"] = [c for c in FEATURES if c not in cols]
    variants["without history-based features"] = [c for c in FEATURES if c not in HISTORY]

    res = {}
    for name, cols in variants.items():
        _, pf, pooled = cv_scores("abl", "hgb", cols, dev, folds, 0.25, lr=params["lr"], depth=params["depth"], l2=params["l2"])
        fold_pr = [m.get("pr_auc", float("nan")) for m in pf]
        res[name] = {"n_features": len(cols), "mean_fold_pr_auc": float(np.nanmean(fold_pr)),
                     "pooled_pr_auc": pooled.get("pr_auc"), "fold_pr_auc": fold_pr}
        print(f"  {name:42s} k={len(cols):2d} mean-fold PR-AUC {res[name]['mean_fold_pr_auc']:.4f}  pooled {pooled.get('pr_auc'):.4f}", flush=True)

    ref = res["all features (reference)"]
    for name, r in res.items():
        r["delta_mean_fold"] = r["mean_fold_pr_auc"] - ref["mean_fold_pr_auc"]
        r["delta_pooled"] = r["pooled_pr_auc"] - ref["pooled_pr_auc"]
        r["folds_worse_than_ref"] = int(sum(a < b for a, b in zip(r["fold_pr_auc"], ref["fold_pr_auc"])))
    out = {"params": params, "n_folds": len(folds), "neg_frac": 0.25, "variants": res,
           "note": "development period only; M2 hyper-parameters from s07_model; same purged folds", "runtime_s": round(time.time() - t0, 1)}
    save_result("ablation", out)

    plt = style_axes()
    names = [n for n in res if n != "all features (reference)"]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.barh(range(len(names)), [res[n]["delta_mean_fold"] for n in names],
            color=[PALETTE["red"] if res[n]["delta_mean_fold"] < 0 else PALETTE["teal"] for n in names])
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7); ax.axvline(0, color=PALETTE["grey"], lw=0.8)
    ax.set_xlabel("change in mean-fold PR-AUC vs all features (development CV)")
    ax.set_title("Feature-group ablation (negative = the group helps)")
    savefig(fig, "ablation")
    print("OK ablation ->", out["runtime_s"], "s")


if __name__ == "__main__":
    main()
