"""PART 3a - Polars vs pandas benchmark, query-plan inspection and scale test.

Each (task, engine) runs in a FRESH subprocess so that peak memory is attributable and an
out-of-memory crash of pandas is caught instead of killing the whole benchmark.

Tasks (all on the clean panel, ~1.7M rows):
  T1 projection + predicate : 6 columns of one quarter
  T2 group-by               : per-quarter mean equity ratio, total assets, bank count
  T3 per-bank window        : gap-aware 4-quarter asset growth + 4-quarter mean of NETINC/ASSET (lags per CERT)
  T4 join + label           : join the Failures table, build y (failure within next 4 quarters)
Engines: pandas (eager), polars-eager, polars-lazy (optimiser + parallel execution).
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from common import PALETTE, peak_rss_mb, save_result, savefig, style_axes
from config import PROC, RAW, RES, SCALE_FACTORS

PANEL = PROC / "panel.parquet"
COLS = ["CERT", "REPDTE", "qidx", "ASSET", "EQ", "NETINC", "LNLSGR"]


# ----------------------------------------------------------------------------- tasks
def run_pandas(task: str, path: str):
    import numpy as np
    import pandas as pd
    if task == "T1":
        # best-effort pandas: column projection + row-group filter pushed into pyarrow
        r = pd.read_parquet(path, columns=COLS, filters=[("REPDTE", "==", pd.Timestamp("2008-12-31"))])
        return len(r)
    if task == "T2":
        df = pd.read_parquet(path, columns=["REPDTE", "ASSET", "EQ"])
        df["er"] = df["EQ"] / df["ASSET"]
        return len(df.groupby("REPDTE").agg(er=("er", "mean"), asset=("ASSET", "sum"), n=("er", "size")))
    if task == "T3":
        df = pd.read_parquet(path, columns=["CERT", "REPDTE", "qidx", "ASSET", "NETINC"]).sort_values(["CERT", "qidx"])
        g = df.groupby("CERT")
        roa = df["NETINC"] / df["ASSET"]
        df["roa"] = roa
        g = df.groupby("CERT")

        def lag(c, k):
            v = g[c].shift(k)
            ok = (df["qidx"] - g["qidx"].shift(k)) == k
            return v.where(ok)

        df["growth4"] = df["ASSET"] / lag("ASSET", 4) - 1
        df["roa_m4"] = pd.concat([df["roa"], lag("roa", 1), lag("roa", 2), lag("roa", 3)], axis=1).mean(axis=1)
        return float(df["growth4"].sum(skipna=True) + df["roa_m4"].sum(skipna=True))
    if task == "T4":
        df = pd.read_parquet(path, columns=["CERT", "REPDTE", "ASSET"])
        df["REPDTE"] = pd.to_datetime(df["REPDTE"])      # Parquet date32 may arrive as python objects; needed for .dt
        f = pd.read_parquet(RAW / "failures.parquet", columns=["CERT", "FAILDATE"])
        f["FAILDATE"] = pd.to_datetime(f["FAILDATE"], errors="coerce", format="mixed")
        f = f.drop_duplicates("CERT")
        m = df.merge(f, on="CERT", how="left")
        # vectorised "+1 year" (Series + DateOffset falls back to a slow per-element path)
        end = pd.to_datetime({"year": m["REPDTE"].dt.year + 1, "month": m["REPDTE"].dt.month, "day": m["REPDTE"].dt.day})
        m["y"] = ((m["FAILDATE"] > m["REPDTE"]) & (m["FAILDATE"] <= end)).astype("int8")
        return int(m["y"].sum())


def run_polars(task: str, path: str, lazy: bool):
    import polars as pl
    from datetime import date
    scan = pl.scan_parquet(path)
    if not lazy:
        scan = pl.read_parquet(path).lazy()          # eager read, then compute (no pushdown of the read)
    if task == "T1":
        q = scan.filter(pl.col("REPDTE") == date(2008, 12, 31)).select(COLS)
    elif task == "T2":
        q = scan.group_by("REPDTE").agg((pl.col("EQ") / pl.col("ASSET")).mean().alias("er"),
                                        pl.col("ASSET").sum(), pl.len())
    elif task == "T3":
        def lag(c, k):
            ok = (pl.col("qidx") - pl.col("qidx").shift(k).over("CERT")) == k
            return pl.when(ok).then(pl.col(c).shift(k).over("CERT"))
        q = (scan.select("CERT", "qidx", "ASSET", "NETINC").sort("CERT", "qidx")
             .with_columns((pl.col("NETINC") / pl.col("ASSET")).alias("roa"))
             .with_columns((pl.col("ASSET") / lag("ASSET", 4) - 1).alias("growth4"),
                           pl.mean_horizontal(pl.col("roa"), lag("roa", 1), lag("roa", 2), lag("roa", 3)).alias("roa_m4"))
             .select(pl.col("growth4").sum(), pl.col("roa_m4").sum()))
    elif task == "T4":
        f = (pl.scan_parquet(RAW / "failures.parquet").select("CERT", "FAILDATE")
             .with_columns(pl.col("CERT").cast(pl.Int32),
                           pl.coalesce(pl.col("FAILDATE").cast(pl.Utf8).str.to_date("%m/%d/%Y", strict=False),
                                       pl.col("FAILDATE").cast(pl.Utf8).str.slice(0, 10).str.to_date("%Y-%m-%d", strict=False)))
             .unique(subset="CERT"))
        q = (scan.select("CERT", "REPDTE", "ASSET").join(f, on="CERT", how="left")
             .with_columns(((pl.col("FAILDATE") > pl.col("REPDTE")) &
                            (pl.col("FAILDATE") <= pl.col("REPDTE").dt.offset_by("1y"))).fill_null(False).cast(pl.Int8).alias("y"))
             .select(pl.col("y").sum()))
    return q


def worker(task: str, engine: str, path: str):
    # import libraries BEFORE the clock starts: we benchmark the work, not interpreter/library start-up
    if engine == "pandas":
        import numpy, pandas, pyarrow.parquet  # noqa: F401
    else:
        import polars  # noqa: F401
    t0 = time.perf_counter()
    if engine == "pandas":
        run_pandas(task, path)
    else:
        run_polars(task, path, lazy=(engine == "polars_lazy")).collect()
    dt = time.perf_counter() - t0
    print(json.dumps({"seconds": dt, "peak_mb": peak_rss_mb()}))


def launch(task, engine, path, timeout=1800):
    try:
        r = subprocess.run([sys.executable, __file__, "--worker", task, engine, str(path)],
                           capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0:
            return {"seconds": None, "peak_mb": None, "error": (r.stderr.strip().splitlines() or ["failed"])[-1][:200]}
        return json.loads(r.stdout.strip().splitlines()[-1])
    except subprocess.TimeoutExpired:
        return {"seconds": None, "peak_mb": None, "error": f"timeout {timeout}s"}


def explain_plans():
    import polars as pl
    from datetime import date
    q = (pl.scan_parquet(PANEL).filter(pl.col("REPDTE") >= date(2008, 1, 1))
         .group_by("REPDTE").agg((pl.col("EQ") / pl.col("ASSET")).mean().alias("er"), pl.col("ASSET").sum())
         .sort("REPDTE"))
    t = ["=== NAIVE plan (optimised=False) ===", q.explain(optimized=False),
         "\n=== OPTIMISED plan ===", q.explain(),
         f"\nthreads used by Polars: {pl.thread_pool_size()}"]
    return "\n".join(t)


def replicate(k: int) -> Path:
    import polars as pl
    out = PROC / f"scale_x{k}.parquet"
    if k == 1:
        return PANEL
    base = pl.read_parquet(PANEL, columns=["CERT", "REPDTE", "qidx", "ASSET", "EQ", "NETINC", "LNLSGR"])
    pl.concat([base.with_columns(pl.col("CERT") + i * 1_000_000) for i in range(k)]).write_parquet(out)
    return out


def main():
    res = {"tasks": {}, "scale": {}}
    for task in ("T1", "T2", "T3", "T4"):
        res["tasks"][task] = {}
        for eng in ("pandas", "polars_eager", "polars_lazy"):
            runs = [launch(task, eng, PANEL) for _ in range(3)]
            ok = [r for r in runs if r.get("seconds") is not None]
            ok_s = sorted(r["seconds"] for r in ok)
            res["tasks"][task][eng] = {"median_s": ok_s[len(ok_s) // 2] if ok_s else None,
                                       "peak_mb": max((r["peak_mb"] for r in ok), default=None),
                                       "error": runs[0].get("error") if not ok else None}
            print(task, eng, res["tasks"][task][eng])
    plan = explain_plans()
    (RES / "polars_plan.txt").write_text(plan, encoding="utf-8")
    res["plan_excerpt"] = plan
    for k in SCALE_FACTORS:
        p = replicate(k)
        res["scale"][str(k)] = {}
        for eng in ("pandas", "polars_lazy"):
            r = launch("T3", eng, p)
            res["scale"][str(k)][eng] = r
            print("scale", k, eng, r)
        if k > 1:
            Path(p).unlink(missing_ok=True)
    save_result("bench", res)
    plot(res)


def plot(res):
    plt = style_axes()
    import numpy as np
    tasks = list(res["tasks"])
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.2))
    w = 0.26
    for i, (eng, col) in enumerate((("pandas", PALETTE["grey"]), ("polars_eager", PALETTE["orange"]), ("polars_lazy", PALETTE["blue"]))):
        ax[0].bar(np.arange(len(tasks)) + i * w, [res["tasks"][t][eng]["median_s"] or 0 for t in tasks], w, label=eng, color=col)
        ax[1].bar(np.arange(len(tasks)) + i * w, [res["tasks"][t][eng]["peak_mb"] or 0 for t in tasks], w, label=eng, color=col)
    for a, t in zip(ax[:2], ("median time (s)", "peak memory (MB)")):
        a.set_xticks(np.arange(len(tasks)) + w); a.set_xticklabels(tasks); a.set_title(t)
    ax[0].legend(fontsize=6)
    ks = [int(k) for k in res["scale"]]
    for eng, col in (("pandas", PALETTE["grey"]), ("polars_lazy", PALETTE["blue"])):
        ax[2].plot(ks, [res["scale"][str(k)][eng].get("seconds") or np.nan for k in ks], marker="o", color=col, label=eng)
    ax[2].set_xlabel("data replicated x k"); ax[2].set_title("T3 time vs data size (s)"); ax[2].legend(fontsize=6)
    savefig(fig, "bench")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker(sys.argv[2], sys.argv[3], sys.argv[4])
    elif len(sys.argv) > 1 and sys.argv[1] == "--plot-only":
        from common import load_result
        plot(load_result("bench"))
    else:
        main()