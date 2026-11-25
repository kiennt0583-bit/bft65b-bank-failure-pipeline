"""PART 1 - Acquisition check & storage layout, with measurements.

Reads the raw quarterly Parquet files written by fdic_download_starter.py, then
  1. verifies completeness (manifest, expected quarters, row count, duplicates),
  2. builds an optimised schema,
  3. writes several layouts (CSV baseline, Parquet variants, partitioning, codecs, row groups),
  4. times five representative queries on every layout.
Outputs: results/storage.json, figures/storage_*.png, data_proc/financials_opt.parquet
"""
from __future__ import annotations

import json
import shutil
import time
from datetime import date, datetime

import pandas as pd
import polars as pl

from common import median_time, save_result, savefig, size_mb, style_axes
from config import EXPECTED_QUARTERS, PROC, RAW, RATIO_FIELDS

LAB = PROC / "storage_lab"
Q_DATE = date(2008, 12, 31)
BANK = 3510          # a long-lived bank; any CERT with a long history works


# ------------------------------------------------------------------ completeness
def completeness() -> dict:
    manifest = json.loads((RAW / "manifest_financials.json").read_text())
    files = sorted((RAW / "financials").glob("REPDTE=*.parquet"))
    bad = [q for q, m in manifest.items() if not m["complete"]]
    rows = sum(m["rows"] for m in manifest.values())
    api_total = sum(m["api_total"] for m in manifest.values())
    mtime = datetime.fromtimestamp((RAW / "manifest_financials.json").stat().st_mtime)
    fails = pd.read_parquet(RAW / "failures.parquet")
    inst = pd.read_parquet(RAW / "institutions.parquet")
    out = {
        "download_method": "FDIC BankFind public API (https://api.fdic.gov/banks), "
                           "paged limit/offset, one Parquet file per quarter, resumable via manifest",
        "download_date": mtime.strftime("%Y-%m-%d"),
        "quarters_expected": EXPECTED_QUARTERS, "quarters_downloaded": len(manifest),
        "files_on_disk": len(files), "incomplete_quarters": bad,
        "rows_downloaded": rows, "rows_api_total": api_total,
        "rows_match_api": rows == api_total,
        "failures_rows": len(fails), "institutions_rows": len(inst),
    }
    print(json.dumps(out, indent=1))
    return out


# ------------------------------------------------------------------ schema
def load_raw_pl() -> pl.DataFrame:
    files = sorted((RAW / "financials").glob("REPDTE=*.parquet"))
    return pl.concat([pl.read_parquet(f) for f in files], how="diagonal_relaxed")


def optimize_schema(df: pl.DataFrame) -> pl.DataFrame:
    """CERT->Int32, REPDTE->Date, STNAME->Categorical, ratios->Float32,
    money amounts -> Int64 when every value is integral (else Float64)."""
    exprs = []
    for c, dt in df.schema.items():
        if c == "CERT":
            exprs.append(pl.col(c).cast(pl.Int32))
        elif c == "REPDTE":
            exprs.append(pl.col(c).cast(pl.Utf8).str.to_date("%Y%m%d"))
        elif c == "STNAME":
            exprs.append(pl.col(c).cast(pl.Categorical))
        elif c in RATIO_FIELDS:
            exprs.append(pl.col(c).cast(pl.Float64).cast(pl.Float32))
        else:
            s = df[c].cast(pl.Float64, strict=False).drop_nulls()
            integral = len(s) > 0 and bool((s == s.round(0)).all())
            exprs.append(pl.col(c).cast(pl.Int64 if integral else pl.Float64, strict=False))
    return df.with_columns(exprs)


# ------------------------------------------------------------------ query suite
def queries(kind: str, path):
    """Return {name: callable}. kind in csv | file | hive_quarter | hive_year."""
    p = str(path)
    if kind == "csv":
        def scan():
            return pl.scan_csv(p, infer_schema_length=10000, try_parse_dates=True)
    elif kind in ("file", "file_raw"):
        def scan():
            return pl.scan_parquet(p)
    else:
        def scan():
            return pl.scan_parquet(p + "/*.parquet", hive_partitioning=False)

    def q_one_quarter():
        if kind == "hive_quarter":      # partition pruning = open one file only
            return pl.scan_parquet(f"{p}/REPDTE={Q_DATE:%Y%m%d}.parquet").collect()
        if kind == "hive_year":
            return pl.scan_parquet(f"{p}/YEAR={Q_DATE.year}.parquet").filter(
                pl.col("REPDTE") == Q_DATE).collect()
        if kind == "file_raw":          # raw layout keeps REPDTE as the original YYYYMMDD string
            return scan().filter(pl.col("REPDTE").cast(pl.Utf8) == f"{Q_DATE:%Y%m%d}").collect()
        return scan().filter(pl.col("REPDTE") == Q_DATE).collect()

    return {
        "Q1 full table": lambda: scan().collect(),
        "Q2 5 columns": lambda: scan().select("CERT", "REPDTE", "ASSET", "EQ", "NETINC").collect(),
        "Q3 one quarter": q_one_quarter,
        "Q4 one bank": lambda: scan().filter(pl.col("CERT") == BANK).collect(),
        "Q5 mean ROA by quarter": lambda: scan().group_by("REPDTE").agg(
            pl.col("ROA").mean(), pl.col("ASSET").sum()).collect(),
    }


def time_layout(name, kind, path, res):
    row = {"layout": name, "size_mb": round(size_mb(path), 1)}
    for qn, fn in queries(kind, path).items():
        try:
            t, _ = median_time(fn, repeat=3, warmup=True)
            row[qn] = round(t, 3)
        except Exception as e:  # noqa: BLE001
            row[qn] = None
            print("   !", name, qn, type(e).__name__, e)
    res.append(row)
    print(row)


def write_partitions(df, col_expr, name_fn, outdir, keys):
    shutil.rmtree(outdir, ignore_errors=True)
    outdir.mkdir(parents=True)
    for k in keys:
        df.filter(col_expr == k).write_parquet(outdir / name_fn(k), compression="zstd",
                                               compression_level=3, statistics=True)


def main():
    LAB.mkdir(parents=True, exist_ok=True)
    out = {"completeness": completeness()}

    t0 = time.perf_counter()
    raw = load_raw_pl()
    out["raw_files_mb"] = round(size_mb(RAW / "financials"), 1)
    out["raw_load_s"] = round(time.perf_counter() - t0, 2)
    out["raw_schema"] = {k: str(v) for k, v in raw.schema.items()}
    out["raw_rows"], out["raw_cols"] = raw.shape
    out["raw_ram_mb"] = round(raw.estimated_size("mb"), 1)

    opt = optimize_schema(raw).sort(["CERT", "REPDTE"])
    out["opt_schema"] = {k: str(v) for k, v in opt.schema.items()}
    out["opt_ram_mb"] = round(opt.estimated_size("mb"), 1)
    out["dup_key_rows"] = int(opt.select(pl.struct("CERT", "REPDTE").is_duplicated().sum()).item())
    per_q = opt.group_by("REPDTE").len().sort("REPDTE")
    out["banks_per_quarter"] = {str(d): n for d, n in zip(per_q["REPDTE"], per_q["len"])}
    out["quarters_in_data"] = per_q.height
    # pandas in-RAM footprint of the same table with default dtypes (for the schema argument)
    out["pandas_object_ram_mb"] = round(raw.to_pandas().memory_usage(deep=True).sum() / 1e6, 1)

    # ---- layouts
    res = []
    csv_path = LAB / "financials.csv"
    t0 = time.perf_counter()
    opt.write_csv(csv_path)           # what a bulk CSV download looks like (ISO dates)
    out["csv_write_s"] = round(time.perf_counter() - t0, 2)
    time_layout("CSV (baseline)", "csv", csv_path, res)

    p_raw = LAB / "parquet_raw_types_snappy.parquet"
    raw.write_parquet(p_raw, compression="snappy")
    time_layout("Parquet raw types, snappy, unsorted", "file_raw", p_raw, res)

    p_opt = LAB / "parquet_opt_sorted_zstd.parquet"
    t0 = time.perf_counter()
    opt.write_parquet(p_opt, compression="zstd", compression_level=3, row_group_size=250_000,
                      statistics=True)
    out["parquet_write_s"] = round(time.perf_counter() - t0, 2)
    time_layout("Parquet optimised types, sorted CERT/REPDTE, zstd", "file", p_opt, res)

    d_q = LAB / "hive_quarter"
    write_partitions(opt, pl.col("REPDTE"), lambda d: f"REPDTE={d:%Y%m%d}.parquet", d_q,
                     opt["REPDTE"].unique().sort().to_list())
    time_layout("Partitioned by quarter (170 files), zstd", "hive_quarter", d_q, res)

    d_y = LAB / "hive_year"
    opt_y = opt.with_columns(pl.col("REPDTE").dt.year().alias("_y"))
    write_partitions(opt_y, pl.col("_y"), lambda y: f"YEAR={y}.parquet", d_y,
                     opt_y["_y"].unique().sort().to_list())
    time_layout("Partitioned by year (43 files), zstd", "hive_year", d_y, res)
    out["layouts"] = res

    # ---- codecs on the optimised sorted table
    codecs = []
    for codec, level in [("uncompressed", None), ("snappy", None), ("lz4", None), ("gzip", 6),
                         ("zstd", 1), ("zstd", 3), ("zstd", 9)]:
        p = LAB / f"codec_{codec}_{level}.parquet"
        t0 = time.perf_counter()
        opt.write_parquet(p, compression=codec, compression_level=level, row_group_size=250_000)
        tw = time.perf_counter() - t0
        tr, _ = median_time(lambda p=p: pl.read_parquet(p), repeat=3)
        codecs.append({"codec": f"{codec}{'' if level is None else '-' + str(level)}",
                       "size_mb": round(size_mb(p), 1), "write_s": round(tw, 2),
                       "read_s": round(tr, 3)})
        p.unlink()
    out["codecs"] = codecs

    # ---- row-group size
    rgs = []
    for rg in (50_000, 250_000, 1_000_000):
        p = LAB / f"rg_{rg}.parquet"
        opt.write_parquet(p, compression="zstd", compression_level=3, row_group_size=rg)
        t_bank, _ = median_time(lambda p=p: pl.scan_parquet(p).filter(pl.col("CERT") == BANK).collect())
        t_full, _ = median_time(lambda p=p: pl.read_parquet(p), repeat=3)
        rgs.append({"row_group_rows": rg, "size_mb": round(size_mb(p), 1),
                    "one_bank_s": round(t_bank, 3), "full_read_s": round(t_full, 3)})
        p.unlink()
    out["row_groups"] = rgs

    # ---- pandas baseline reading CSV vs Parquet
    t_csv, _ = median_time(lambda: pd.read_csv(csv_path), repeat=2)
    t_csv5, _ = median_time(lambda: pd.read_csv(csv_path, usecols=["CERT", "REPDTE", "ASSET", "EQ", "NETINC"]), repeat=2)
    t_pq, _ = median_time(lambda: pd.read_parquet(p_opt), repeat=3)
    t_pq5, _ = median_time(lambda: pd.read_parquet(p_opt, columns=["CERT", "REPDTE", "ASSET", "EQ", "NETINC"]), repeat=3)
    out["pandas_io"] = {"read_csv_all_s": round(t_csv, 2), "read_csv_5cols_s": round(t_csv5, 2),
                        "read_parquet_all_s": round(t_pq, 2), "read_parquet_5cols_s": round(t_pq5, 2)}

    # keep the chosen layout for the pipeline, free disk space
    shutil.copy(p_opt, PROC / "financials_opt.parquet")
    for p in (csv_path, p_raw):
        p.unlink(missing_ok=True)
    shutil.rmtree(d_y, ignore_errors=True)
    # d_q is kept: it is the layout the Spark demo reads (partition pruning)

    save_result("storage", out)
    plot(out)
    print("OK -> results/storage.json")


def plot(out):
    plt = style_axes()
    L = pd.DataFrame(out["layouts"])
    short = ["CSV", "PQ raw\nsnappy", "PQ opt\nsorted zstd", "by quarter", "by year"]
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    ax[0].bar(short, L["size_mb"], color="#2f6db5")
    ax[0].set_ylabel("size on disk (MB)")
    ax[0].set_title("Storage size")
    qs = [c for c in L.columns if c.startswith("Q")]
    w = 0.16
    for i, lay in enumerate(short):
        ax[1].bar([x + i * w for x in range(len(qs))], L.loc[i, qs].astype(float), w, label=lay.replace("\n", " "))
    ax[1].set_xticks([x + 2 * w for x in range(len(qs))])
    ax[1].set_xticklabels([q.split(" ", 1)[0] for q in qs])
    ax[1].set_yscale("log")
    ax[1].set_ylabel("median time (s, log)")
    ax[1].set_title("Query time (Polars)")
    ax[1].legend(fontsize=6)
    savefig(fig, "storage_layouts")

    s = pd.Series(out["banks_per_quarter"])
    s.index = pd.to_datetime(s.index)
    fig, ax = plt.subplots(figsize=(6, 2.6))
    ax.plot(s.index, s.values, color="#2f6db5")
    ax.set_title("Reporting institutions per quarter")
    savefig(fig, "banks_per_quarter")


if __name__ == "__main__":
    main()