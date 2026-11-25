"""PART 3b - Reproduce the key Polars steps in PySpark, check equality, read the plans, time vs scale.

Runs locally (local[*]), on Google Colab or on Databricks Free Edition: only the SparkSession line differs.
Steps reproduced in Spark:  (1) group-by per quarter, (2) gap-aware lags per bank with window functions,
de-cumulated NETINC, 4-quarter asset growth, 4-quarter mean ROA,  (3) failure label counts.
Checks: results are compared with the Polars feature table (max abs difference, null pattern).
Outputs: results/spark.json, results/spark_plans.txt
"""
from __future__ import annotations

import contextlib
import io
import time
from functools import reduce

from common import save_result
from config import PROC, RES, SCALE_FACTORS


def get_spark():
    import os
    import sys
    # Windows has no `python3`: tell Spark to use the interpreter running this script
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    from pyspark.sql import SparkSession
    return (SparkSession.builder.master("local[*]").appName("bft65b")
            .config("spark.sql.ansi.enabled", "false")           # Spark 4 default is ANSI (errors on bad casts/division)
            .config("spark.sql.shuffle.partitions", "8")        # default 200 is far too many for ~1.7M rows
            .config("spark.sql.session.timeZone", "UTC")
            .config("spark.driver.memory", "4g").getOrCreate())


def spark_features(df):
    """Same logic as s06 for: equity_ratio, asset_growth_4q, roa_q, roa_q_mean4 (window functions)."""
    from pyspark.sql import Window, functions as F
    w = Window.partitionBy("CERT").orderBy("qidx")

    def lag(c, k):
        return F.when(F.col("qidx") - F.lag("qidx", k).over(w) == k, F.lag(c, k).over(w))

    d = df.select("CERT", "REPDTE", "qidx", "ASSET", "EQ", "NETINC")
    d = d.withColumn("netinc_q", F.when(F.month("REPDTE") == 3, F.col("NETINC"))
                     .when((F.col("qidx") - F.lag("qidx", 1).over(w)) == 1, F.col("NETINC") - F.lag("NETINC", 1).over(w)))
    d = d.withColumn("roa_q", 4 * F.col("netinc_q") / F.col("ASSET"))
    d = d.withColumn("equity_ratio", F.when(F.col("ASSET") > 0, F.col("EQ") / F.col("ASSET")))
    la = lag("ASSET", 4)
    d = d.withColumn("asset_growth_4q", F.when(la > 0, F.col("ASSET") / la - 1))
    return d


def stage2(d):
    """4-quarter mean of roa_q (needs the roa_q column to exist first, hence a second stage)."""
    from pyspark.sql import Window, functions as F
    w = Window.partitionBy("CERT").orderBy("qidx")

    def lag(c, k):
        return F.when(F.col("qidx") - F.lag("qidx", k).over(w) == k, F.lag(c, k).over(w))
    return d.withColumn(
        "_s", F.coalesce(F.col("roa_q"), F.lit(0.0)) + sum(F.coalesce(lag("roa_q", k), F.lit(0.0)) for k in (1, 2, 3))
    ).withColumn(
        "_n", F.col("roa_q").isNotNull().cast("int") + sum(lag("roa_q", k).isNotNull().cast("int") for k in (1, 2, 3))
    ).withColumn("roa_q_mean4", F.when(F.col("_n") >= 2, F.col("_s") / F.col("_n"))).drop("_s", "_n")


def capture_plan(df, mode="formatted") -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        df.explain(mode=mode)
    return buf.getvalue()


def main():
    from pyspark.sql import functions as F
    spark = get_spark()
    out = {"spark_version": spark.version, "cores": spark.sparkContext.defaultParallelism}
    panel = spark.read.parquet(str(PROC / "panel.parquet"))
    out["panel_rows"] = panel.count()

    # ---------- (1) group by quarter
    g = panel.groupBy("REPDTE").agg(F.count("*").alias("n"), F.sum("ASSET").alias("asset"),
                                    F.mean(F.col("EQ") / F.col("ASSET")).alias("er"), F.sum("y").alias("y_pos"))
    g_pd = g.orderBy("REPDTE").toPandas()

    # ---------- (2) features with window functions
    feats = stage2(spark_features(panel))
    t0 = time.perf_counter()
    feats_pd = feats.select("CERT", "REPDTE", "equity_ratio", "asset_growth_4q", "roa_q", "roa_q_mean4").toPandas()
    out["spark_feature_job_s"] = round(time.perf_counter() - t0, 2)

    # ---------- equality with Polars
    import numpy as np
    import pandas as pd
    import polars as pl
    pol = pl.read_parquet(PROC / "features.parquet", columns=["CERT", "REPDTE", "equity_ratio", "asset_growth_4q",
                                                              "roa_q", "roa_q_mean4"]).to_pandas()
    pol_g = (pl.read_parquet(PROC / "panel.parquet", columns=["REPDTE", "ASSET", "EQ", "y"])
             .group_by("REPDTE").agg(pl.len().alias("n"), pl.col("ASSET").sum().alias("asset"),
                                     (pl.col("EQ") / pl.col("ASSET")).mean().alias("er"), pl.col("y").sum().alias("y_pos"))
             .sort("REPDTE").to_pandas())
    g_pd["REPDTE"] = pd.to_datetime(g_pd["REPDTE"]); pol_g["REPDTE"] = pd.to_datetime(pol_g["REPDTE"])
    mg = g_pd.merge(pol_g, on="REPDTE", suffixes=("_spark", "_polars"))
    out["groupby_check"] = {"quarters": len(mg), "count_equal": bool((mg.n_spark == mg.n_polars).all()),
                            "asset_sum_equal": bool((mg.asset_spark == mg.asset_polars).all()),
                            "y_pos_equal": bool((mg.y_pos_spark == mg.y_pos_polars).all()),
                            "max_abs_diff_mean_equity_ratio": float((mg.er_spark - mg.er_polars).abs().max())}
    feats_pd["REPDTE"] = pd.to_datetime(feats_pd["REPDTE"]); pol["REPDTE"] = pd.to_datetime(pol["REPDTE"])
    j = feats_pd.merge(pol, on=["CERT", "REPDTE"], suffixes=("_s", "_p"))
    chk = {}
    for c in ["equity_ratio", "asset_growth_4q", "roa_q", "roa_q_mean4"]:
        a, b = j[f"{c}_s"], j[f"{c}_p"]
        chk[c] = {"max_abs_diff": float(np.nanmax(np.abs(a - b))) if (a.notna() & b.notna()).any() else None,
                  "null_pattern_mismatch": int((a.isna() != b.isna()).sum())}
    out["feature_check"] = {"rows_compared": len(j), "rows_spark": len(feats_pd), "rows_polars": len(pol), **chk}
    out["results_match"] = bool(all(v["null_pattern_mismatch"] == 0 and (v["max_abs_diff"] or 0) < 1e-6 for v in chk.values())
                                and out["groupby_check"]["count_equal"] and out["groupby_check"]["y_pos_equal"])

    # ---------- plans
    plans = ["=== group-by plan ===", capture_plan(g), "=== window-feature plan ===",
             capture_plan(feats.select("CERT", "REPDTE", "asset_growth_4q", "roa_q_mean4"))]
    filt = panel.filter(F.col("REPDTE") == F.lit("2008-12-31").cast("date")).select("CERT", "ASSET")
    plans += ["=== filter + projection plan (look for PushedFilters / ReadSchema) ===", capture_plan(filt)]
    (RES / "spark_plans.txt").write_text("\n".join(plans), encoding="utf-8")
    out["plan_excerpt"] = plans[3][:3500]
    out["plan_has_exchange"] = "Exchange hashpartitioning(CERT" in plans[3]
    out["plan_has_window"] = "Window" in plans[3]
    out["plan_pushed_filters"] = "PushedFilters" in plans[5]

    # ---------- scale: time of one window job vs data size
    scale = {}
    for k in SCALE_FACTORS:
        dfk = reduce(lambda a, b: a.union(b), [panel.withColumn("CERT", F.col("CERT") + i * 1_000_000) for i in range(k)])
        fk = stage2(spark_features(dfk))
        t0 = time.perf_counter()
        fk.agg(F.sum("asset_growth_4q"), F.sum("roa_q_mean4")).collect()
        scale[k] = round(time.perf_counter() - t0, 2)
        print("spark scale", k, scale[k])
    out["scale_seconds"] = scale
    save_result("spark", out)
    spark.stop()
    print("OK results_match =", out["results_match"])


if __name__ == "__main__":
    main()