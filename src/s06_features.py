"""PART 4 - Point-in-time feature table (Polars) + automated leakage test.

Every feature at (bank i, quarter q) uses ONLY rows of bank i dated <= q (backward-looking).
  * Year-to-date flow fields (NETINC, INTINC, ...) are converted to single-quarter flows.
  * Lags are gap-aware: lag_k(x) is null unless the row k quarters earlier really exists.
  * No imputation / scaling here: those are fitted on training data inside the sklearn Pipeline (s07).
Outputs: data_proc/features.parquet, results/features.json, results/leakage_test.json
"""
from __future__ import annotations

import random

import polars as pl

from common import save_result
from config import LAST_LABELLED_Q, PROC, SEED, YTD_FIELDS, qidx


def sdiv(num, den):
    """Safe ratio: null when denominator is null or <= 0 (no inf)."""
    return pl.when(den > 0).then(num / den).otherwise(None)


def lag(col: str, k: int) -> pl.Expr:
    """Value of `col` exactly k quarters earlier (null if that quarter is missing)."""
    ok = (pl.col("qidx") - pl.col("qidx").shift(k).over("CERT")) == k
    return pl.when(ok).then(pl.col(col).shift(k).over("CERT")).otherwise(None)


FEATURES = [
    # size, age, history
    "log_asset", "age_years", "n_quarters_seen",
    # capital
    "equity_ratio", "RBC1AAJ", "RBCRWAJ", "d_equity_ratio_4q",
    # asset quality
    "npl_ratio", "past_due_30_89", "past_due_90", "nonaccrual_assets", "ore_ratio",
    "allowance_cov", "texas_ratio", "d_npl_ratio_4q", "nco_rate", "provision_rate",
    # earnings / efficiency
    "roa_q", "ROA", "ROE", "NIMY", "efficiency", "nonint_income_share", "roa_q_mean4", "roa_q_std4", "d_roa_4q",
    # growth / balance-sheet structure
    "asset_growth_4q", "loan_growth_4q", "deposit_growth_4q", "loans_assets", "loan_deposit",
    "re_loan_share", "ci_loan_share", "con_loan_share",
    # funding / liquidity
    "brokered_share", "liquid_assets", "securities_share", "d_brokered_4q",
]


def build_features(panel: pl.DataFrame) -> pl.DataFrame:
    p = panel.sort(["CERT", "REPDTE"])
    # 1) de-cumulate year-to-date flows: Q1 = YTD, otherwise YTD_q - YTD_{q-1} (same year, consecutive)
    same_year_prev = (pl.col("qidx") - pl.col("qidx").shift(1).over("CERT") == 1) & (pl.col("REPDTE").dt.month() != 3)
    p = p.with_columns([
        pl.when(pl.col("REPDTE").dt.month() == 3).then(pl.col(c))
        .when(same_year_prev).then(pl.col(c) - pl.col(c).shift(1).over("CERT"))
        .otherwise(None).alias(f"{c}_q") for c in YTD_FIELDS])

    # 2) same-date ratios
    p = p.with_columns(
        pl.col("ASSET").log().alias("log_asset"),
        sdiv(pl.col("EQ"), pl.col("ASSET")).alias("equity_ratio"),
        sdiv(pl.col("NCLNLS"), pl.col("LNLSGR")).alias("npl_ratio"),
        sdiv(pl.col("P3ASSET"), pl.col("ASSET")).alias("past_due_30_89"),
        sdiv(pl.col("P9ASSET"), pl.col("ASSET")).alias("past_due_90"),
        sdiv(pl.col("NAASSET"), pl.col("ASSET")).alias("nonaccrual_assets"),
        sdiv(pl.col("ORE"), pl.col("ASSET")).alias("ore_ratio"),
        sdiv(pl.col("LNATRES"), pl.col("LNLSGR")).alias("allowance_cov"),
        sdiv(pl.col("NCLNLS") + pl.col("ORE").fill_null(0), pl.col("EQ") + pl.col("LNATRES")).alias("texas_ratio"),
        sdiv(4 * pl.col("NTLNLS_q"), pl.col("LNLSGR")).alias("nco_rate"),
        sdiv(4 * pl.col("ELNATR_q"), pl.col("LNLSGR")).alias("provision_rate"),
        (4 * pl.col("NETINC_q") / pl.col("ASSET")).alias("roa_q"),
        sdiv(pl.col("NONIX_q"), pl.col("INTINC_q") - pl.col("EINTEXP_q") + pl.col("NONII_q")).alias("efficiency"),
        sdiv(pl.col("NONII_q"), pl.col("INTINC_q") - pl.col("EINTEXP_q") + pl.col("NONII_q")).alias("nonint_income_share"),
        sdiv(pl.col("LNLSGR"), pl.col("ASSET")).alias("loans_assets"),
        sdiv(pl.col("LNLSNET"), pl.col("DEP")).alias("loan_deposit"),
        sdiv(pl.col("LNRE"), pl.col("LNLSGR")).alias("re_loan_share"),
        sdiv(pl.col("LNCI"), pl.col("LNLSGR")).alias("ci_loan_share"),
        sdiv(pl.col("LNCON"), pl.col("LNLSGR")).alias("con_loan_share"),
        sdiv(pl.col("BRO"), pl.col("DEP")).alias("brokered_share"),
        sdiv(pl.col("CHBAL") + pl.col("SC"), pl.col("ASSET")).alias("liquid_assets"),
        sdiv(pl.col("SC"), pl.col("ASSET")).alias("securities_share"),
        pl.col("REPDTE").cum_count().over("CERT").alias("n_quarters_seen"),
    )

    # 3) dynamics: only lags (past) -> point-in-time
    p = p.with_columns(
        (pl.col("equity_ratio") - lag("equity_ratio", 4)).alias("d_equity_ratio_4q"),
        (pl.col("npl_ratio") - lag("npl_ratio", 4)).alias("d_npl_ratio_4q"),
        (pl.col("brokered_share") - lag("brokered_share", 4)).alias("d_brokered_4q"),
        sdiv(pl.col("ASSET"), lag("ASSET", 4)).sub(1).alias("asset_growth_4q"),
        sdiv(pl.col("LNLSGR"), lag("LNLSGR", 4)).sub(1).alias("loan_growth_4q"),
        sdiv(pl.col("DEP"), lag("DEP", 4)).sub(1).alias("deposit_growth_4q"),
        (pl.col("roa_q") - lag("roa_q", 4)).alias("d_roa_4q"),
        pl.mean_horizontal(pl.col("roa_q"), lag("roa_q", 1), lag("roa_q", 2), lag("roa_q", 3)).alias("roa_q_mean4"),
        pl.mean_horizontal(pl.col("roa_q") ** 2, lag("roa_q", 1) ** 2, lag("roa_q", 2) ** 2,
                           lag("roa_q", 3) ** 2).alias("_m2"),
        pl.sum_horizontal(*[x.is_not_null().cast(pl.Int8) for x in
                            (pl.col("roa_q"), lag("roa_q", 1), lag("roa_q", 2), lag("roa_q", 3))]).alias("_n"),
    ).with_columns(
        pl.when(pl.col("_n") >= 3).then((pl.col("_m2") - pl.col("roa_q_mean4") ** 2).clip(lower_bound=0).sqrt())
        .otherwise(None).alias("roa_q_std4"),
        pl.when(pl.col("_n") >= 2).then(pl.col("roa_q_mean4")).otherwise(None).alias("roa_q_mean4"),
    ).drop(["_m2", "_n"])
    return p


def to_feature_table(panel: pl.DataFrame) -> pl.DataFrame:
    f = build_features(panel)
    return f.select("CERT", "REPDTE", "qidx", "y", "label_incomplete", "known_from", "FAILDATE", *FEATURES)


def leakage_test(panel: pl.DataFrame, n_banks: int = 400, cutoffs=((2005, 4), (2012, 3), (2019, 4))) -> dict:
    """Recompute features with all rows AFTER cutoff q0 removed; features at q0 must be identical."""
    random.seed(SEED)
    certs = panel["CERT"].unique().to_list()
    sample = random.sample(certs, min(n_banks, len(certs)))   # python list -> unambiguous is_in
    sub = panel.filter(pl.col("CERT").is_in(sample))
    full = build_features(sub)
    out = {}
    for y, q in cutoffs:
        q0 = qidx(y, q)
        trunc = build_features(sub.filter(pl.col("qidx") <= q0)).filter(pl.col("qidx") == q0)
        ref = full.filter(pl.col("qidx") == q0)
        j = ref.join(trunc, on="CERT", suffix="_trunc")
        worst, nulls_differ = 0.0, 0
        for c in FEATURES:
            a, b = j[c].cast(pl.Float64), j[f"{c}_trunc"].cast(pl.Float64)
            nulls_differ += int((a.is_null() != b.is_null()).sum())
            d = (a - b).abs().drop_nulls()
            if len(d):
                worst = max(worst, float(d.max()))
        out[f"{y}Q{q}"] = {"banks_compared": j.height, "max_abs_diff": worst, "null_pattern_mismatches": nulls_differ,
                           "passed": bool(worst < 1e-9 and nulls_differ == 0)}
    out["all_passed"] = all(v["passed"] for v in out.values())
    return out


def main():
    panel = pl.read_parquet(PROC / "panel.parquet")
    feats = to_feature_table(panel)
    feats.write_parquet(PROC / "features.parquet", compression="zstd")
    lt = leakage_test(panel)
    save_result("leakage_test", lt)
    labelled = feats.filter(~pl.col("label_incomplete"))
    save_result("features", {
        "n_features": len(FEATURES), "features": FEATURES, "rows": feats.height,
        "rows_labelled": labelled.height, "positives": int(labelled["y"].sum()),
        "null_share": {c: round(float(feats[c].is_null().mean()), 4) for c in FEATURES},
        "last_labelled_quarter": f"{LAST_LABELLED_Q[0]}Q{LAST_LABELLED_Q[1]}"})
    print("features:", feats.shape, "| leakage test passed:", lt["all_passed"])
    if not lt["all_passed"]:
        raise SystemExit("LEAKAGE TEST FAILED - fix before modelling")


if __name__ == "__main__":
    main()