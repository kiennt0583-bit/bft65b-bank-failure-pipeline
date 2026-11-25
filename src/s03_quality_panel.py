"""PART 2 - Automated data-quality report + clean, point-in-time bank-quarter panel.

Inputs : data_raw/fdic/{financials,failures.parquet,institutions.parquet}
Outputs: data_proc/panel.parquet  (clean bank-quarter panel with label y)
         data_proc/banks.parquet  (one row per bank: exit type, dates)
         results/dq.json, results/dq_report.md, figures/dq_*.png
Design rules
  * the raw layer is never modified; every rule only flags or drops, and counts what it did;
  * nothing is imputed here (imputation is fitted on training data only, in s07);
  * rows dated on/after the failure date are dropped (a failed bank does not report);
  * label y = 1{FAILDATE in (REPDTE, REPDTE + 1 year]}; merger / other exits get y = 0.
"""
from __future__ import annotations

import polars as pl

from common import save_result, savefig, style_axes
from config import (FIG, LAST_LABELLED_Q, NONNEG_FIELDS, PROC, RAW, RATIO_FIELDS, RES, YTD_FIELDS,
                    qidx)
from s02_storage import load_raw_pl, optimize_schema

QX = (pl.col("REPDTE").dt.year() * 4 + (pl.col("REPDTE").dt.month() - 1) // 3).alias("qidx")


def parse_date(col: str) -> pl.Expr:
    c = pl.col(col).cast(pl.Utf8)
    return pl.coalesce(c.str.to_date("%m/%d/%Y", strict=False), c.str.to_date("%Y-%m-%d", strict=False),
                       c.str.slice(0, 10).str.to_date("%Y-%m-%d", strict=False)).alias(col)


def load_tables():
    fin = optimize_schema(load_raw_pl()).with_columns(QX)
    fails = pl.read_parquet(RAW / "failures.parquet")
    fails = fails.with_columns(parse_date("FAILDATE"), pl.col("CERT").cast(pl.Int32))
    inst = pl.read_parquet(RAW / "institutions.parquet")
    inst = inst.with_columns(pl.col("CERT").cast(pl.Int32), parse_date("ESTYMD"), parse_date("ENDEFYMD"))
    return fin, fails, inst


# ----------------------------------------------------------------------------- audit
def audit(fin: pl.DataFrame, fails: pl.DataFrame, inst: pl.DataFrame) -> dict:
    dq: dict = {}
    n = fin.height
    dq["rows"] = n
    dq["banks"] = fin["CERT"].n_unique()
    dq["duplicates_cert_repdte"] = int(fin.select(pl.struct("CERT", "REPDTE").is_duplicated().sum()).item())

    # --- completeness of fields: null share by field and decade
    fin = fin.with_columns(((pl.col("REPDTE").dt.year() // 10) * 10).alias("decade"))
    value_cols = [c for c in fin.columns if c not in ("CERT", "REPDTE", "STNAME", "qidx", "decade")]
    nulls = fin.group_by("decade").agg([pl.col(c).is_null().mean().alias(c) for c in value_cols]).sort("decade")
    dq["null_share_by_decade"] = {str(r["decade"]): {c: round(r[c], 4) for c in value_cols}
                                  for r in nulls.iter_rows(named=True)}
    dq["null_share_overall"] = {c: round(float(fin[c].is_null().mean()), 4) for c in value_cols}
    dq["first_quarter_with_data"] = {
        c: (str(fin.filter(pl.col(c).is_not_null())["REPDTE"].min())) for c in value_cols}
    ytd = fin.group_by("REPDTE").agg([pl.col(c).is_null().mean().alias(c) for c in value_cols]).sort("REPDTE")

    # --- hard-rule violations
    rules = {
        "ASSET <= 0 or null": pl.col("ASSET").is_null() | (pl.col("ASSET") <= 0),
        "negative value in a non-negative field": pl.any_horizontal(
            [pl.col(c) < 0 for c in NONNEG_FIELDS if c in fin.columns]),
        "DEP > ASSET": pl.col("DEP") > pl.col("ASSET"),
        "DEPDOM > DEP": pl.col("DEPDOM") > pl.col("DEP"),
        "EQ > ASSET": pl.col("EQ") > pl.col("ASSET"),
        "LNLSGR > ASSET": pl.col("LNLSGR") > pl.col("ASSET"),
        "NCLNLS > LNLSGR": pl.col("NCLNLS") > pl.col("LNLSGR"),
        "EQ <= 0 (book insolvent)": pl.col("EQ") <= 0,
        "|ROA| > 50 %": pl.col("ROA").abs() > 50,
        "RBC1AAJ outside [-50, 100] %": (pl.col("RBC1AAJ") < -50) | (pl.col("RBC1AAJ") > 100),
    }
    dq["rule_violations"] = {k: int(fin.select(v.fill_null(False).sum()).item()) for k, v in rules.items()}

    # --- outliers: tails of the ratio fields
    qs = [0.0, 0.001, 0.01, 0.5, 0.99, 0.999, 1.0]
    dq["ratio_quantiles"] = {c: {str(q): (None if fin[c].drop_nulls().len() == 0 else
                                          round(float(fin[c].drop_nulls().quantile(q)), 3)) for q in qs}
                             for c in RATIO_FIELDS}

    # --- YTD evidence: median (Q4 value / Q1 value) of flow fields within a bank-year
    evid = {}
    for c in ("NONIX", "INTINC", "NETINC"):
        wide = (fin.select("CERT", pl.col("REPDTE").dt.year().alias("y"), pl.col("REPDTE").dt.month().alias("m"), c)
                .filter(pl.col("m").is_in([3, 12]) & (pl.col(c) > 0))
                .pivot(on="m", index=["CERT", "y"], values=c, aggregate_function="first"))
        if "3" in wide.columns and "12" in wide.columns:
            r = (wide["12"] / wide["3"]).drop_nulls()
            evid[c] = round(float(r.median()), 2) if len(r) else None
    dq["ytd_evidence_median_Q4_over_Q1"] = evid

    # --- reporting gaps
    g = fin.sort(["CERT", "REPDTE"]).with_columns(
        (pl.col("qidx") - pl.col("qidx").shift(1).over("CERT")).alias("dq"))
    gaps = g.filter(pl.col("dq") > 1)
    dq["gaps"] = {"banks_with_gap": gaps["CERT"].n_unique(), "gap_events": gaps.height,
                  "max_gap_quarters": int(gaps["dq"].max() or 0)}

    # --- failures table
    f = fails
    dq["failures"] = {
        "rows": f.height, "unique_cert": f["CERT"].n_unique(),
        "faildate_unparsed": int(f["FAILDATE"].is_null().sum()),
        "cost_null": int(f["COST"].is_null().sum()),
        "restype": {str(k): int(v) for k, v in f["RESTYPE"].value_counts().iter_rows()} if "RESTYPE" in f.columns else {},
        "per_year": {int(k): int(v) for k, v in f.with_columns(pl.col("FAILDATE").dt.year().alias("y"))
                     .group_by("y").len().sort("y").iter_rows() if k is not None},
        "no_financial_row": int(f.join(fin.select("CERT").unique(), on="CERT", how="anti").height),
    }
    # --- link failures <-> financials
    last = fin.group_by("CERT").agg(pl.col("REPDTE").max().alias("last_rep"), pl.col("qidx").max().alias("last_q"))
    fl = f.join(last, on="CERT", how="inner").with_columns(
        ((pl.col("FAILDATE").dt.year() * 4 + (pl.col("FAILDATE").dt.month() - 1) // 3) - pl.col("last_q")).alias("q_gap"))
    dq["failures"]["last_report_to_failure_quarters"] = {
        str(k): int(v) for k, v in fl.group_by("q_gap").len().sort("q_gap").iter_rows()}
    dq["failures"]["last_report_more_than_4q_before_failure"] = int(fl.filter(pl.col("q_gap") > 4).height)
    post = fin.join(f.select("CERT", "FAILDATE"), on="CERT", how="inner").filter(pl.col("REPDTE") >= pl.col("FAILDATE"))
    dq["rows_dated_on_or_after_failure"] = post.height

    # --- institutions / exits
    dq["institutions"] = {"rows": inst.height,
                          "cert_in_financials_not_in_institutions": int(fin.select("CERT").unique().join(inst.select("CERT").unique(), on="CERT", how="anti").height),
                          "active_counts": {str(k): int(v) for k, v in inst["ACTIVE"].value_counts().iter_rows()}}
    return dq, ytd


# ----------------------------------------------------------------------------- panel
def exit_table(fin, fails, inst) -> pl.DataFrame:
    last_q_all = fin["qidx"].max()
    b = fin.group_by("CERT").agg(pl.col("REPDTE").min().alias("first_rep"), pl.col("REPDTE").max().alias("last_rep"),
                                 pl.col("qidx").max().alias("last_q"), pl.len().alias("n_rows"))
    f1 = fails.sort("FAILDATE").unique(subset="CERT", keep="first").select(
        "CERT", "FAILDATE", "COST", *[c for c in ("QBFASSET", "RESTYPE") if c in fails.columns])
    keep = [c for c in ("CERT", "ACTIVE", "CHANGEC1", "ESTYMD", "ENDEFYMD", "BKCLASS") if c in inst.columns]
    b = b.join(f1, on="CERT", how="left").join(inst.select(keep).unique(subset="CERT"), on="CERT", how="left")
    return b.with_columns(
        pl.when(pl.col("FAILDATE").is_not_null()).then(pl.lit("failure"))
        .when(pl.col("last_q") == last_q_all).then(pl.lit("active"))
        .when(pl.col("ACTIVE") == 0).then(pl.lit("other_exit"))
        .otherwise(pl.lit("stopped_unexplained")).alias("exit_type"))


def build_panel(fin, fails, banks) -> pl.DataFrame:
    n0 = fin.height
    log = {"raw_rows": n0}
    p = fin.unique(subset=["CERT", "REPDTE"], keep="last")
    log["after_dedup"] = p.height
    p = p.filter(pl.col("ASSET").is_not_null() & (pl.col("ASSET") > 0))
    log["after_asset_positive"] = p.height
    p = p.join(banks.select("CERT", "FAILDATE", "exit_type", "ESTYMD"), on="CERT", how="left")
    p = p.filter(pl.col("FAILDATE").is_null() | (pl.col("REPDTE") < pl.col("FAILDATE")))
    log["after_drop_post_failure_rows"] = p.height
    p = p.with_columns(
        pl.col("REPDTE").dt.offset_by("1y").alias("label_end"),
        pl.col("REPDTE").dt.offset_by("45d").alias("known_from"),     # filing + publication lag (conservative)
    ).with_columns(
        ((pl.col("FAILDATE") > pl.col("REPDTE")) & (pl.col("FAILDATE") <= pl.col("label_end")))
        .fill_null(False).cast(pl.Int8).alias("y"),
        ((pl.col("REPDTE") - pl.col("ESTYMD")).dt.total_days() / 365.25).alias("age_years"),
        (pl.col("qidx") > qidx(*LAST_LABELLED_Q)).alias("label_incomplete"),
    )
    p = p.sort(["CERT", "REPDTE"])
    log["final_rows"] = p.height
    log["y_positive_rows_labelled_period"] = int(p.filter(~pl.col("label_incomplete"))["y"].sum())
    log["distinct_failed_banks_labelled"] = int(p.filter(pl.col("y") == 1)["CERT"].n_unique())
    return p, log


# ----------------------------------------------------------------------------- report
def write_md(dq, panel_log, banks):
    L = ["# Automated data-quality report", "",
         f"Rows: {dq['rows']:,} | banks: {dq['banks']:,} | duplicate (CERT, REPDTE): {dq['duplicates_cert_repdte']}", "",
         "## Rule violations", "", "| rule | rows | share |", "|---|---:|---:|"]
    for k, v in dq["rule_violations"].items():
        L.append(f"| {k} | {v:,} | {v / dq['rows']:.4%} |")
    L += ["", "## Null share by field (overall)", "", "| field | null share | first quarter with data |", "|---|---:|---|"]
    for c, v in sorted(dq["null_share_overall"].items(), key=lambda kv: -kv[1]):
        L.append(f"| {c} | {v:.1%} | {dq['first_quarter_with_data'][c]} |")
    L += ["", "## Exits", ""] + [f"- {k}: {v:,}" for k, v in
                                  banks.group_by("exit_type").len().sort("len", descending=True).iter_rows()]
    L += ["", "## Panel construction log", ""] + [f"- {k}: {v:,}" for k, v in panel_log.items()]
    L += ["", "## Failures vs financials", "", f"```\n{dq['failures']}\n```"]
    (RES / "dq_report.md").write_text("\n".join(L), encoding="utf-8")


def plots(dq, ytd, fails):
    plt = style_axes()
    import numpy as np
    cols = [c for c in ytd.columns if c != "REPDTE"]
    M = ytd.select(cols).to_numpy().T
    fig, ax = plt.subplots(figsize=(8, 4.6))
    im = ax.imshow(M, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(cols, fontsize=6)
    yrs = ytd["REPDTE"].dt.year().to_list()
    ticks = [i for i in range(0, len(yrs), 20)]
    ax.set_xticks(ticks)
    ax.set_xticklabels([yrs[i] for i in ticks])
    ax.grid(False)
    fig.colorbar(im, label="share missing")
    ax.set_title("Missing values by field and quarter")
    savefig(fig, "dq_missing")
    per = dq["failures"]["per_year"]
    fig, ax = plt.subplots(figsize=(6.5, 2.6))
    ax.bar(list(per.keys()), list(per.values()), color="#c0392b")
    ax.set_title("Bank failures per year (Failures table)")
    savefig(fig, "dq_failures_per_year")


def main():
    fin, fails, inst = load_tables()
    dq, ytd = audit(fin, fails, inst)
    banks = exit_table(fin, fails, inst)
    dq["exit_types"] = {k: int(v) for k, v in banks.group_by("exit_type").len().iter_rows()}
    if "CHANGEC1" in banks.columns:
        dq["other_exit_changec1_top"] = {str(k): int(v) for k, v in
                                         banks.filter(pl.col("exit_type") == "other_exit").group_by("CHANGEC1").len()
                                         .sort("len", descending=True).head(15).iter_rows()}
    panel, log = build_panel(fin, fails, banks)
    dq["panel_log"] = log
    banks.write_parquet(PROC / "banks.parquet")
    panel.write_parquet(PROC / "panel.parquet", compression="zstd", row_group_size=250_000)
    save_result("dq", dq)
    write_md(dq, log, banks)
    plots(dq, ytd, fails)
    print("OK panel rows:", panel.height, "| positives:", log["y_positive_rows_labelled_period"])


if __name__ == "__main__":
    main()