"""PART 5 - Modelling & validation on time-ordered, highly imbalanced data.

Protocol
  1. Development = quarters <= DEV_END, test = TEST_START..TEST_END (gap quarters dropped).
  2. Model selection with expanding-window, purged CV INSIDE the development period only.
  3. Threshold from event-level expected cost on out-of-fold dev scores (cost data from Failures.COST).
  4. Final model refit on all development data, evaluated ONCE on test (a guard file blocks re-runs).
Outputs: results/model.json, figures/model_*.png
"""
from __future__ import annotations

import argparse
import itertools
import time

import numpy as np
import pandas as pd
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import PALETTE, save_result, savefig, style_axes
from config import (CV_BLOCKS, DEV_END, FP_RATIO_BASE, FP_RATIO_GRID, GAP_Q, MIN_TEST_FAILURES, PROC, PURGE_Q, RES,
                    SEED, TEST_END, TEST_START, qidx, qlabel)
from modeling_utils import (DropEmptyColumns, Winsorizer, choose_threshold, event_confusion, expected_cost, metrics,
                            purged_expanding_folds, quarterly_recall_at_k)
from s06_features import FEATURES

BASELINE_FEATURES = ["equity_ratio", "npl_ratio", "roa_q_mean4", "texas_ratio"]   # CAMELS-style, 4 variables
GUARD = RES / "test_evaluated.flag"


def make_pipe(kind: str, cols: list[str], **kw) -> Pipeline:
    if kind == "logit":
        return Pipeline([("drop", DropEmptyColumns()), ("win", Winsorizer()),
                         ("imp", SimpleImputer(strategy="median", add_indicator=True)),
                         ("sc", StandardScaler()),
                         ("clf", LogisticRegression(C=kw.get("C", 0.1), max_iter=400, solver="lbfgs"))])
    return Pipeline([("drop", DropEmptyColumns()), ("win", Winsorizer()),
                     ("clf", HistGradientBoostingClassifier(
                         learning_rate=kw.get("lr", 0.08), max_depth=kw.get("depth", 4), max_iter=kw.get("iter", 200),
                         l2_regularization=kw.get("l2", 1.0), min_samples_leaf=200, random_state=SEED))])


def fit_predict(kind, cols, df_tr, df_va, neg_frac=1.0, **kw):
    """Fit on df_tr (negatives optionally subsampled and re-weighted) and score df_va."""
    rng = np.random.RandomState(SEED)
    tr = df_tr
    w = np.ones(len(tr))
    if neg_frac < 1.0:
        keep = (tr["y"].values == 1) | (rng.rand(len(tr)) < neg_frac)
        tr = tr[keep]
        w = np.where(tr["y"].values == 1, 1.0, 1.0 / neg_frac)
    pipe = make_pipe(kind, cols, **kw)
    pipe.fit(tr[cols].values.astype("float32"), tr["y"].values, clf__sample_weight=w)
    return pipe, pipe.predict_proba(df_va[cols].values.astype("float32"))[:, 1]


def cv_scores(name, kind, cols, dev, folds, neg_frac, **kw):
    oof = np.full(len(dev), np.nan)
    per_fold = []
    for k, (tr, va) in enumerate(folds):
        _, p = fit_predict(kind, cols, dev.iloc[tr], dev.iloc[va], neg_frac, **kw)
        oof[va] = p
        m = metrics(dev["y"].values[va], p)
        m["fold"] = k + 1
        per_fold.append(m)
    ok = ~np.isnan(oof)
    pooled = metrics(dev["y"].values[ok], oof[ok])
    return oof, per_fold, pooled


def main(fast: bool = False, allow_retest: bool = False):
    t_start = time.time()
    feats = pl.read_parquet(PROC / "features.parquet").filter(~pl.col("label_incomplete"))
    banks = pl.read_parquet(PROC / "banks.parquet")
    df = feats.to_pandas()
    q_dev_end, q_ts, q_te = qidx(*DEV_END), qidx(*TEST_START), qidx(*TEST_END)
    assert q_ts - q_dev_end - 1 >= GAP_Q, "gap between development and test must be >= 4 quarters"
    dev = df[df["qidx"] <= q_dev_end].reset_index(drop=True)
    test = df[(df["qidx"] >= q_ts) & (df["qidx"] <= q_te)].reset_index(drop=True)
    dev_fail, test_fail = dev.loc[dev.y == 1, "CERT"].nunique(), test.loc[test.y == 1, "CERT"].nunique()
    assert test_fail >= MIN_TEST_FAILURES, f"test period has only {test_fail} failures (< {MIN_TEST_FAILURES})"
    out = {"split": {"dev": f"{qlabel(int(dev.qidx.min()))}-{qlabel(q_dev_end)}", "gap": f"{qlabel(q_dev_end + 1)}-{qlabel(q_ts - 1)}",
                     "test": f"{qlabel(q_ts)}-{qlabel(q_te)}", "dev_rows": len(dev), "test_rows": len(test),
                     "dev_failures": int(dev_fail), "test_failures": int(test_fail),
                     "dev_positive_rows": int(dev.y.sum()), "test_positive_rows": int(test.y.sum()),
                     "dev_base_rate": float(dev.y.mean()), "test_base_rate": float(test.y.mean())}}
    print(out["split"])

    folds = list(purged_expanding_folds(dev["qidx"].values, [(qidx(*a), qidx(*b)) for a, b in CV_BLOCKS], PURGE_Q))
    out["cv"] = {"n_folds": len(folds), "blocks": [f"{qlabel(qidx(*a))}-{qlabel(qidx(*b))}" for a, b in CV_BLOCKS],
                 "purge_quarters": PURGE_Q,
                 "fold_sizes": [{"train": int(len(a)), "val": int(len(b)), "val_pos": int(dev.y.values[b].sum())} for a, b in folds]}
    neg_frac = 0.2 if fast else 0.25

    # ---------- candidate models (development CV only)
    results, oofs = {}, {}
    # B0: single-variable ranking rules (no fitting)
    val_idx = np.unique(np.concatenate([va for _, va in folds]))     # same rows as the fitted models' OOF scores
    for nm, col, sign in (("B1 rule: low equity ratio", "equity_ratio", -1), ("B2 rule: Texas ratio", "texas_ratio", +1)):
        s = np.nan_to_num(sign * dev[col].values, nan=np.nanmedian(sign * dev[col].values))
        results[nm] = {"pooled": metrics(dev.y.values[val_idx], s[val_idx]), "per_fold": []}
    # B3: logistic on 4 CAMELS-style variables
    oof, pf, pooled = cv_scores("B3", "logit", BASELINE_FEATURES, dev, folds, neg_frac)
    results["B3 logit, 4 CAMELS-style vars"] = {"pooled": pooled, "per_fold": pf}; oofs["B3"] = oof
    # M1: logistic on all features
    oof, pf, pooled = cv_scores("M1", "logit", FEATURES, dev, folds, neg_frac, C=0.1)
    results["M1 logit, all features"] = {"pooled": pooled, "per_fold": pf}; oofs["M1"] = oof
    # M2: gradient boosting, small grid tuned on dev CV (mean fold PR-AUC)
    grid = list(itertools.product([0.05, 0.1], [3, 5], [1.0, 10.0])) if not fast else [(0.1, 4, 1.0)]
    tuning = []
    best = (-1, None, None)
    for lr, depth, l2 in grid:
        oof, pf, pooled = cv_scores("M2", "hgb", FEATURES, dev, folds, neg_frac, lr=lr, depth=depth, l2=l2)
        score = float(np.nanmean([m.get("pr_auc", np.nan) for m in pf]))
        tuning.append({"lr": lr, "depth": depth, "l2": l2, "mean_fold_pr_auc": score, "pooled_pr_auc": pooled.get("pr_auc")})
        print("   HGB", lr, depth, l2, round(score, 4))
        if score > best[0]:
            best = (score, dict(lr=lr, depth=depth, l2=l2), (oof, pf, pooled))
    results["M2 gradient boosting, tuned"] = {"pooled": best[2][2], "per_fold": best[2][1], "params": best[1]}
    oofs["M2"] = best[2][0]
    out["tuning"] = tuning
    out["cv_results"] = results

    # ---------- choose final model on development only
    cand = {"B3": "B3 logit, 4 CAMELS-style vars", "M1": "M1 logit, all features", "M2": "M2 gradient boosting, tuned"}
    final_key = max(cand, key=lambda k: results[cand[k]]["pooled"].get("pr_auc", 0))
    out["selected_model"] = cand[final_key]
    oof = oofs[final_key]
    ok = ~np.isnan(oof)

    # ---------- cost model from the data (development failures only)
    dev_fail_certs = dev.loc[dev.y == 1, "CERT"].unique()
    cost = banks.filter(pl.col("CERT").is_in(dev_fail_certs.tolist()))["COST"].drop_nulls()
    c_fn = float(cost.mean())
    out["cost_model"] = {"c_fn_mean_cost_kUSD": c_fn, "c_fn_median_cost_kUSD": float(cost.median()), "n_cost_obs": len(cost),
                         "fp_ratio_base": FP_RATIO_BASE, "c_fp_kUSD": FP_RATIO_BASE * c_fn,
                         "break_even_precision": FP_RATIO_BASE / (1 + FP_RATIO_BASE),
                         "note": "c_FN data-derived (Failures.COST, dev failures); c_FP = ratio x c_FN is an assumption."}
    thr, c_best, curve = choose_threshold(dev["CERT"].values[ok], dev["y"].values[ok], oof[ok], c_fn, FP_RATIO_BASE * c_fn)
    sens = []
    for r in FP_RATIO_GRID:
        t_r, c_r, _ = choose_threshold(dev["CERT"].values[ok], dev["y"].values[ok], oof[ok], c_fn, r * c_fn)
        conf_r = event_confusion(dev["CERT"].values[ok], dev["y"].values[ok], oof[ok] >= t_r)
        sens.append({"fp_ratio": r, "threshold": t_r, **conf_r, "cost": c_r})
    conf_dev = event_confusion(dev["CERT"].values[ok], dev["y"].values[ok], oof[ok] >= thr)
    out["threshold"] = {"value": thr, "dev_oof_cost_kUSD": c_best, "dev_oof_confusion": conf_dev, "sensitivity": sens,
                        "cost_never_alarm": conf_dev["events"] * c_fn}

    # ---------- single evaluation on the test period
    if GUARD.exists() and not allow_retest:
        raise SystemExit(f"{GUARD} exists: the test period was already used once. Refusing to re-evaluate "
                         "(delete the flag only if you accept the loss of an honest test).")
    cols = {"B3": BASELINE_FEATURES, "M1": FEATURES, "M2": FEATURES}[final_key]
    kind = "hgb" if final_key == "M2" else "logit"
    kw = results[cand[final_key]].get("params", {})
    pipe, p_test = fit_predict(kind, cols, dev, test, neg_frac=neg_frac, **kw)
    # reference models on the same single test evaluation (all fitted on dev only)
    ref = {}
    for nm, (k_, c_, kw_) in {"B3 logit, 4 CAMELS-style vars": ("logit", BASELINE_FEATURES, {}),
                              "M1 logit, all features": ("logit", FEATURES, {"C": 0.1})}.items():
        _, pr = fit_predict(k_, c_, dev, test, neg_frac=neg_frac, **kw_)
        ref[nm] = metrics(test.y.values, pr)
        test[f"s_{k_}_{len(c_)}"] = pr
    for nm, col, sign in (("B1 rule: low equity ratio", "equity_ratio", -1), ("B2 rule: Texas ratio", "texas_ratio", +1)):
        v = sign * test[col].values
        ref[nm] = metrics(test.y.values, np.nan_to_num(v, nan=np.nanmedian(sign * dev[col].values)))
    test["score"] = p_test
    m_test = metrics(test.y.values, p_test)
    conf_test = event_confusion(test["CERT"].values, test["y"].values, p_test >= thr)
    test_cost = expected_cost(conf_test, c_fn, FP_RATIO_BASE * c_fn)
    out["test"] = {"model": cand[final_key], "metrics": m_test, "reference_models": ref,
                   "confusion_at_threshold": conf_test, "expected_cost_kUSD": test_cost,
                   "cost_never_alarm_kUSD": conf_test["events"] * c_fn,
                   "cost_alarm_everyone_kUSD": expected_cost({"missed": 0, "false_alarms": int((test.y == 0).sum())}, c_fn, FP_RATIO_BASE * c_fn),
                   "cross_sectional_recall_top5pct_of_banks": quarterly_recall_at_k(test, "score", 0.05),
                   "cross_sectional_recall_top10pct_of_banks": quarterly_recall_at_k(test, "score", 0.10),
                   "alarm_rate": float((p_test >= thr).mean())}
    by_year = test.assign(year=test.qidx // 4).groupby("year").apply(
        lambda g: pd.Series({"rows": len(g), "failures": g.loc[g.y == 1, "CERT"].nunique(),
                             "pr_auc": metrics(g.y.values, g.score.values).get("pr_auc", np.nan),
                             "roc_auc": metrics(g.y.values, g.score.values).get("roc_auc", np.nan)}), include_groups=False)
    out["test"]["by_year"] = by_year.reset_index().to_dict(orient="records")
    GUARD.write_text("test evaluated once\n")

    # interpretability: standardised logit coefficients (M1) or permutation-free gain proxy not available for HGB
    if kind == "logit":
        clf = pipe.named_steps["clf"]
        kept = [c for c, k in zip(cols, pipe.named_steps["drop"].keep_) if k]
        out["coefficients"] = sorted(zip(kept, clf.coef_[0][:len(kept)].tolist()), key=lambda t: -abs(t[1]))[:15]
    out["runtime_s"] = round(time.time() - t_start, 1)
    save_result("model", out)
    plots(out, oof[ok], dev.y.values[ok], test)
    print("OK. selected:", out["selected_model"], "| test PR-AUC:", round(m_test.get("pr_auc", float("nan")), 4))


def plots(out, oof, y_dev, test):
    from sklearn.metrics import precision_recall_curve
    plt = style_axes()
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    p, r, _ = precision_recall_curve(test.y.values, test.score.values)
    ax[0].plot(r, p, color=PALETTE["blue"], label=out["test"]["model"])
    ax[0].axhline(out["split"]["test_base_rate"], color=PALETTE["grey"], ls="--", label="base rate")
    ax[0].set_xlabel("recall"); ax[0].set_ylabel("precision"); ax[0].set_title("Precision-recall, test period"); ax[0].legend(fontsize=6)
    by = pd.DataFrame(out["test"]["by_year"])
    ax[1].bar(by["year"], by["failures"], color=PALETTE["red"], alpha=0.7)
    ax[1].set_title("Failures in the test period (by year of observation)")
    savefig(fig, "model_test")
    fig, ax = plt.subplots(figsize=(5, 3))
    sens = pd.DataFrame(out["threshold"]["sensitivity"])
    ax.plot(sens["fp_ratio"], sens["caught"] / sens["events"], marker="o", color=PALETTE["teal"], label="share of failures caught")
    ax.set_xscale("log"); ax.set_xlabel("c_FP / c_FN"); ax.set_title("Dev: failures caught vs assumed false-alarm cost")
    savefig(fig, "model_cost_sensitivity")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="smaller tuning grid (smoke test)")
    ap.add_argument("--allow-retest", action="store_true")
    a = ap.parse_args()
    main(a.fast, a.allow_retest)