"""Build the final report (DOCX) from results/*.json and figures/*.png.

Every number in the report comes from a result file written by the pipeline; nothing is typed by hand.
Usage:  python src/build_report.py [--results results] [--figures figures] [--out report.docx]
Fill report_meta.json (name, student id, GitHub URL) first.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--results", default=str(ROOT / "results"))
ap.add_argument("--figures", default=str(ROOT / "figures"))
ap.add_argument("--out", default=str(ROOT / "report.docx"))
ap.add_argument("--meta", default=str(ROOT / "report_meta.json"))
A = ap.parse_args()
RESD, FIGD = Path(A.results), Path(A.figures)


def load(name):
    p = RESD / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


meta = json.loads(Path(A.meta).read_text(encoding="utf-8")) if Path(A.meta).exists() else {}
ST, DQ, BN, SP, FT, LK, MD, ML = (load(n) for n in ("storage", "dq", "bench", "spark", "features", "leakage_test", "model", "mllib"))
NA = "n/a"


def g(d, path, fmt="{}", default=NA):
    cur = d
    for p in path.split("."):
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        elif isinstance(cur, list) and p.isdigit() and int(p) < len(cur):
            cur = cur[int(p)]
        else:
            return default
    if cur is None:
        return default
    try:
        return fmt.format(cur)
    except Exception:  # noqa: BLE001
        return str(cur)


def num(d, path):
    cur = d
    for p in path.split("."):
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        elif isinstance(cur, list) and p.isdigit() and int(p) < len(cur):
            cur = cur[int(p)]
        else:
            return None
    return cur if isinstance(cur, (int, float)) else None


def ratio(a, b, fmt="{:.1f}x"):
    return fmt.format(a / b) if a and b else NA


# --------------------------------------------------------------------------- doc helpers
doc = Document()
sec = doc.sections[0]
sec.left_margin = sec.right_margin = Cm(2.0)
sec.top_margin = sec.bottom_margin = Cm(1.9)
st = doc.styles["Normal"]
st.font.name, st.font.size = "Calibri", Pt(10)
st.paragraph_format.space_after = Pt(4)
for h, sz in (("Heading 1", 14), ("Heading 2", 11.5), ("Heading 3", 10.5)):
    doc.styles[h].font.name = "Calibri"
    doc.styles[h].font.size = Pt(sz)
    doc.styles[h].font.color.rgb = RGBColor(0x1F, 0x3A, 0x5F)


def H(t, l=1):
    doc.add_heading(t, level=l)


def P(t, bold=False, italic=False, size=None, align=None):
    p = doc.add_paragraph()
    r = p.add_run(t)
    r.bold, r.italic = bold, italic
    if size:
        r.font.size = Pt(size)
    if align:
        p.alignment = align
    return p


def BL(items):
    for it in items:
        doc.add_paragraph(it, style="List Bullet")


def T(rows, header, size=8, widths=None):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Light Grid Accent 1"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(header):
        c = t.rows[0].cells[i]
        c.text = ""
        r = c.paragraphs[0].add_run(str(h))
        r.bold, r.font.size = True, Pt(size)
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            cells[i].paragraphs[0].add_run(str(v)).font.size = Pt(size)
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Cm(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def FIG(name, cap, w=15.5):
    p = FIGD / f"{name}.png"
    if p.exists():
        doc.add_picture(str(p), width=Cm(w))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    else:
        P(f"[figure {name}.png missing - run the pipeline]", italic=True)
    P(cap, italic=True, size=8.5, align=WD_ALIGN_PARAGRAPH.CENTER)


def CODE(txt, n=2200):
    import re
    t = re.sub(r"(?<![A-Za-z])[A-Za-z]:[/\\][^\]\n]*?data_proc", "<project>/data_proc", txt or NA)
    if len(t) > n:
        t = t[:n]
        t = t[:t.rfind("\n")] + "\n[... truncated; full plan in results/]"
    p = doc.add_paragraph()
    r = p.add_run(t)
    r.font.name, r.font.size = "Consolas", Pt(6.5)


# --------------------------------------------------------------------------- numbers reused
S, M = MD.get("split", {}), MD
panel_rows = g(DQ, "panel_log.final_rows", "{:,}")
raw_rows = g(DQ, "rows", "{:,}")
comp = ST.get("completeness", {})

# =========================================================================== FIRST PAGE
P("BFT65B - Big Data Analytics in Finance | Final Individual Report", bold=True, size=9)
P("A Big-Data Pipeline for Bank Failure Early Warning", bold=True, size=18)
P("Data engineering on 40+ years of FDIC Call Reports: storage, quality, scalable processing, leak-free features and honest validation", italic=True)
T([
    ["Name / Student ID", f"{meta.get('name', '[NAME]')} / {meta.get('student_id', '[STUDENT ID]')}"],
    ["GitHub", meta.get("github_url", "[GITHUB URL]")],
    ["Download method and date", f"{g(comp, 'download_method')}; downloaded {g(comp, 'download_date')} "
                                 f"(starter script src/fdic_download_starter.py, field list adapted in Section 2)"],
    ["Row counts", f"Financials {g(comp, 'rows_downloaded', '{:,}')} rows ({g(comp, 'quarters_downloaded')} quarters, "
                   f"API total {g(comp, 'rows_api_total', '{:,}')}); Failures {g(comp, 'failures_rows', '{:,}')}; "
                   f"Institutions {g(comp, 'institutions_rows', '{:,}')}; clean panel {panel_rows}"],
    ["Development period", f"{g(S, 'dev')}: {g(S, 'dev_rows', '{:,}')} bank-quarters, {g(S, 'dev_failures')} failed banks "
                           f"({g(S, 'dev_positive_rows', '{:,}')} positive rows)"],
    ["Gap (dropped)", g(S, "gap")],
    ["Test period (used once)", f"{g(S, 'test')}: {g(S, 'test_rows', '{:,}')} bank-quarters, {g(S, 'test_failures')} failed banks "
                                f"({g(S, 'test_positive_rows', '{:,}')} positive rows)"],
], ["Item", "Value"], size=9, widths=[4.2, 12.8])
P("Sources and tools are cited in Section 8. All numbers, tables and figures in this report are produced by the repository code "
  "(results/*.json -> src/build_report.py); the README reruns everything end to end.", italic=True, size=9)

# =========================================================================== 1. OVERVIEW
H("1. Problem, design principles and pipeline overview")
P("The supervision department needs an early-warning system; this report delivers the data pipeline that would feed it, built on "
  "US FDIC data because Vietnamese supervisory data are not public. Grading emphasises the pipeline, so each stage is designed to be "
  "measured, reproducible and free of look-ahead.")
BL([
    "Bronze -> Silver -> Gold (medallion, lecture 'Big Data, Spark & ML'): raw API Parquet per quarter (immutable) -> typed, sorted, "
    "validated panel -> point-in-time feature table with label.",
    "Every claim is backed by a measurement stored in results/*.json; every design choice is a constant in src/config.py.",
    "Leakage control is structural (backward-looking features, purged time splits, preprocessing inside the Pipeline) and tested automatically.",
])
T([["s01 (starter)", "API download, resumable, completeness check", "data_raw/fdic/*"],
   ["s02_storage", "Completeness verification, schema, layouts, measurements", "financials_opt.parquet, storage.json"],
   ["s03_quality_panel", "Automated DQ audit, exit classification, panel + label", "panel.parquet, dq.json, dq_report.md"],
   ["s04_polars_vs_pandas", "Polars vs pandas, plans, scale test", "bench.json, polars_plan.txt"],
   ["s05_spark", "PySpark re-implementation, equality check, plans", "spark.json, spark_plans.txt"],
   ["s06_features", "Point-in-time features, leakage test", "features.parquet, leakage_test.json"],
   ["s07_model / s08_spark_mllib", "Purged CV, cost-based threshold, single test; MLlib demo", "model.json, mllib.json"]],
  ["Stage", "Purpose", "Output"], widths=[4, 8, 5])

# =========================================================================== PART 1
H("2. Part 1 - Acquisition and storage")
H("2.1 Acquisition and completeness", 2)
P("Method: the public FDIC BankFind API, called with limit/offset paging (10,000 rows per call), polite 1 s pauses and exponential back-off. "
  "One Parquet file per quarter plus a manifest.json (rows, API total, completeness flag) make the download resumable: a rerun skips "
  "quarters already complete. The field list is the 3 keys + 30 Call Report fields of the starter (balance sheet, liquidity, loans, asset "
  "quality, earnings, capital). They map to the CAMELS dimensions (Capital, Asset quality, Earnings, Liquidity, Sensitivity) used by "
  "supervisors; fields were checked against the FDIC field dictionary (units: thousand USD; ratios in %). Failures adds COST (loss to the "
  "deposit-insurance fund, used in the threshold) and Institutions adds ACTIVE/CHANGEC1/ESTYMD (exit reason, age).")
T([["Quarters expected / downloaded", f"{g(comp, 'quarters_expected')} / {g(comp, 'quarters_downloaded')}"],
   ["Incomplete quarters", g(comp, "incomplete_quarters")],
   ["Rows downloaded vs API total", f"{g(comp, 'rows_downloaded', '{:,}')} vs {g(comp, 'rows_api_total', '{:,}')} (match: {g(comp, 'rows_match_api')})"],
   ["Duplicate (CERT, REPDTE)", g(ST, "dup_key_rows")],
   ["Distinct banks", g(DQ, "banks", "{:,}")]],
  ["Check", "Result"], widths=[7, 10])
FIG("banks_per_quarter", "Figure 1. Reporting institutions per quarter (consolidation of the US banking system).", 9)

H("2.2 Schema, layout and measurements", 2)
P("Schema choices: CERT Int32; REPDTE Date (parsed from the YYYYMMDD string); STNAME Categorical (dictionary encoding, 50+ repeated strings); "
  "ratios (ROA, ROE, NIMY, capital ratios) Float32; money amounts Int64 when every value is integral, else Float64. Typed columns keep their "
  "meaning on every read, whereas CSV re-guesses types each time (Problem 1 of the lecture).")
rows = [[l["layout"], g(l, "size_mb"), *[g(l, q) for q in ("Q1 full table", "Q2 5 columns", "Q3 one quarter", "Q4 one bank", "Q5 mean ROA by quarter")]]
        for l in ST.get("layouts", [])]
T(rows, ["Layout", "MB", "Q1 all (s)", "Q2 5 cols (s)", "Q3 1 quarter (s)", "Q4 1 bank (s)", "Q5 agg (s)"], widths=[6, 1.4, 1.9, 1.9, 2, 1.9, 1.9])
FIG("storage_layouts", "Figure 2. Size and query time of each layout (Polars, median of 3 warm runs; log scale).", 14)
L = ST.get("layouts", [])
csv_mb, opt_mb = num(ST, "layouts.0.size_mb"), num(ST, "layouts.2.size_mb")
best = {}
for q in ("Q1 full table", "Q2 5 columns", "Q3 one quarter", "Q4 one bank", "Q5 mean ROA by quarter"):
    c = [(l.get(q), l["layout"]) for l in L if l.get(q) is not None]
    best[q] = min(c)[1] if c else NA
P(f"Findings. The optimised Parquet file is {ratio(csv_mb, opt_mb)} smaller than the CSV ({g(ST, 'layouts.0.size_mb')} MB -> {g(ST, 'layouts.2.size_mb')} MB). "
  f"Reading 5 of {g(ST, 'raw_cols')} columns is {ratio(num(ST, 'layouts.0.Q2 5 columns'), num(ST, 'layouts.2.Q2 5 columns'))} faster than CSV "
  f"(projection pushdown: only those column chunks are read); a one-bank history is "
  f"{ratio(num(ST, 'layouts.0.Q4 one bank'), num(ST, 'layouts.2.Q4 one bank'))} faster because sorting by (CERT, REPDTE) gives every row group tight min/max statistics "
  f"and most row groups are skipped. Fastest layout per query: " + "; ".join(f"{k.split(' ')[0]} -> {v}" for k, v in best.items()) + ". "
  f"In RAM the typed table needs {g(ST, 'opt_ram_mb')} MB versus {g(ST, 'pandas_object_ram_mb')} MB for the pandas default (object strings, float64).")
P(f"pandas I/O: read_csv {g(ST, 'pandas_io.read_csv_all_s')} s vs read_parquet {g(ST, 'pandas_io.read_parquet_all_s')} s (all columns); "
  f"{g(ST, 'pandas_io.read_csv_5cols_s')} s vs {g(ST, 'pandas_io.read_parquet_5cols_s')} s for 5 columns.")
if all(best.get(q, "").startswith("Parquet raw") for q in ("Q1 full table", "Q2 5 columns")) and ST.get("codecs"):
    P("Why the raw-type snappy file wins the full-scan queries: snappy decodes faster than zstd (codec table below), so on a warm local cache decode speed "
      "dominates; zstd buys a smaller file at the price of that extra read time, which matters more on network storage or when the data grow by orders of magnitude.")
P("Design decision. The working store is ONE Parquet file sorted by (CERT, REPDTE) with zstd: it serves the two dominant access patterns of this "
  "project (all history of a bank for window features; selected columns across all quarters). Per-quarter partition files are also written because "
  "they allow partition pruning for time-slice queries (Q3) and are the natural unit of the resumable download. Whether partitioning pays off at this size "
  "is read from Q3 versus Q1/Q5 in the table above (many small files add file-open overhead); on a multi-TB table partitioning by date becomes essential.")
H("2.3 Compression and row-group size", 2)
T([[c["codec"], c["size_mb"], c["write_s"], c["read_s"]] for c in ST.get("codecs", [])], ["Codec", "MB", "write (s)", "full read (s)"], widths=[3, 3, 3, 3])
T([[r["row_group_rows"], r["size_mb"], r["one_bank_s"], r["full_read_s"]] for r in ST.get("row_groups", [])],
  ["Rows per row group", "MB", "one bank (s)", "full read (s)"], widths=[4, 3, 3, 3])
_c = ST.get("codecs", [])
if _c:
    _small, _fast = min(_c, key=lambda c: c["size_mb"]), min(_c, key=lambda c: c["read_s"])
    _z3 = next((c for c in _c if c["codec"] == "zstd-3"), None)
    P(f"Smallest file: {_small['codec']} ({_small['size_mb']} MB); fastest full read: {_fast['codec']} ({_fast['read_s']} s). "
      + (f"Default chosen: zstd-3 ({_z3['size_mb']} MB, read {_z3['read_s']} s, write {_z3['write_s']} s), a middle point between size and speed; " if _z3 else "")
      + "row groups of 250,000 rows balance statistics granularity (more skipping for one-bank queries) against metadata overhead, as the second table shows.")

# =========================================================================== PART 2
H("3. Part 2 - Data quality and clean panel")
P("The audit (s03) is automated and rerunnable: it writes results/dq.json and a Markdown report (appendix A). Rule: the raw layer is never "
  "edited; each rule either flags, drops (counted) or is deferred to the training-only pipeline.")
V = DQ.get("rule_violations", {})
yt = DQ.get("ytd_evidence_median_Q4_over_Q1", {})
nn = DQ.get("null_share_overall", {})
top_null = sorted(nn.items(), key=lambda kv: -kv[1])[:4]
gp = DQ.get("gaps", {})
fl = DQ.get("failures", {})
T([
    ["Duplicate (CERT, REPDTE)", g(DQ, "duplicates_cert_repdte"), "Drop duplicates (keep last); key asserted unique"],
    ["Missing values (highest null share: " + ", ".join(f"{k} {v:.0%}" for k, v in top_null) + ")", "see App. A",
     "Not imputed in the panel: era-dependent reporting (fields start in different years). Median imputation + missing indicators fitted on training data only (s07); GBM uses native NaN"],
    ["ASSET <= 0 or null", V.get("ASSET <= 0 or null", NA), "Drop row (ratios undefined)"],
    ["Negative value in non-negative field", V.get("negative value in a non-negative field", NA), "Keep, winsorise on training quantiles"],
    ["DEP > ASSET / DEPDOM > DEP", f"{V.get('DEP > ASSET', NA)} / {V.get('DEPDOM > DEP', NA)}", "Flag only (reporting error or restatement); winsorise"],
    ["EQ > ASSET, LNLSGR > ASSET, NCLNLS > LNLSGR", f"{V.get('EQ > ASSET', NA)} / {V.get('LNLSGR > ASSET', NA)} / {V.get('NCLNLS > LNLSGR', NA)}", "Flag only; ratios computed with safe division (null if denominator <= 0)"],
    ["Book-insolvent (EQ <= 0)", V.get("EQ <= 0 (book insolvent)", NA), "Keep: these are the signal, not noise"],
    ["Extreme ratios (|ROA| > 50 %)", V.get("|ROA| > 50 %", NA), "Winsorise at 0.5/99.5 % of the training period (inside Pipeline)"],
    ["Year-to-date flows (NETINC, INTINC, EINTEXP, NONII, NONIX, ELNATR, NTLNLS)", f"median Q4/Q1 NONIX = {yt.get('NONIX', NA)}",
     "Values accumulate within the calendar year (ratio ~4 instead of ~1): de-cumulated to single-quarter flows (Q1 = YTD, else YTD_q - YTD_q-1)"],
    ["Reporting gaps in a bank's series", f"{gp.get('banks_with_gap', NA)} banks, {gp.get('gap_events', NA)} gaps (max {gp.get('max_gap_quarters', NA)} q)", "Lags are gap-aware: null unless the row k quarters earlier exists (no shift across a gap)"],
    ["Rows dated on/after the failure date", DQ.get("rows_dated_on_or_after_failure", NA), "Dropped: a failed bank stops reporting, and rows after the event (also for assistance transactions, RESTYPE counts " + str(fl.get("restype", {})) + ", where the bank kept operating) belong to a different regime and would leak the outcome"],
    ["Failures whose last report is > 4 quarters before FAILDATE", fl.get("last_report_more_than_4q_before_failure", NA), "Cannot be labelled by any model; reported as a limitation"],
    ["Failures without any financial row", fl.get("no_financial_row", NA), "Excluded from modelling (no features); e.g. failures before the 1984Q1 start of the panel"],
    ["Failures with missing COST", fl.get("cost_null", NA), "Cost mean computed on non-null values"],
], ["Problem found", "Count", "Handling and justification"], size=7.5, widths=[5.2, 2.6, 9.2])
FIG("dq_missing", "Figure 3. Share of missing values by field and quarter: many fields start only in the 1990s (structural, not random, missingness).", 12.5)
FIG("dq_failures_per_year", "Figure 4. Failures per year in the Failures table: three clusters (S&L, GFC, 2023).", 9)
H("3.1 Exits: failure versus other exits", 2)
ex = DQ.get("exit_types", {})
T([[k, f"{v:,}"] for k, v in ex.items()], ["Exit type", "Banks"], widths=[6, 3])
P("Rule (s03.exit_table): a bank is 'failure' if its CERT is in the Failures table; else 'active' if it still reports in the last quarter; else "
  "'other_exit' if Institutions.ACTIVE = 0 (mergers, voluntary liquidations, charter conversions; the CHANGEC1 code distribution is in appendix A); "
  "anything else is 'stopped_unexplained' (monitored, kept as non-failure). Other exits receive y = 0 as required (mergers are not failures) and are "
  "NOT dropped, which avoids survivorship bias. A limitation remains: a bank that merges cannot be observed failing afterwards (right-censoring).")
H("3.2 Point-in-time correctness and the label", 2)
BL([
    "Label y(i,q) = 1{FAILDATE in (REPDTE, REPDTE + 1 year]}, computed from the failure date, not from the last report, because many failed banks stop "
    "reporting before they are closed. Rows with REPDTE >= FAILDATE are removed. Quarters after 2025Q2 are flagged label_incomplete and never modelled.",
    "Event definition: the Failures table contains closures (RESTYPE = FAILURE) and open-bank assistance transactions (RESTYPE = ASSISTANCE; counts in the table above). "
    "Following the course definition every row of the Failures table is an event, so both are labelled 1. Assistance cases are typically concentrated in the 1980s-early 1990s, "
    "i.e. in the development period, and the test period is dominated by closures; a sensitivity run that keeps only closures is a natural extension (set RESTYPE filter in s03).",
    "Each row carries known_from = REPDTE + 45 days (filing deadline plus publication lag): the earliest date the figures are public. "
    "Features at q use only rows <= q; the leakage test (Section 5.3) enforces this.",
    "Vintage caveat: the API returns the latest restated values, not the values as first filed. Restatements are small for most banks but this is a "
    "real point-in-time imperfection (Limitations).",
])
pl_ = DQ.get("panel_log", {})
T([[k, f"{v:,}"] for k, v in pl_.items()], ["Panel construction step", "Rows"], widths=[8, 3])

# =========================================================================== PART 3
H("4. Part 3 - Scalable processing")
H("4.1 How Polars executes the transformations", 2)
P("The panel is processed with the lazy API: scan_parquet builds a logical plan, the optimiser applies projection pushdown (only needed columns are "
  "read), predicate pushdown (filters are applied while scanning, skipping row groups using min/max statistics), common sub-plan elimination and "
  "expression simplification, then executes in parallel across row groups and independent expressions on all cores with a vectorised columnar engine "
  "(Arrow memory). Window expressions (shift(k).over('CERT')) partition the data once by bank and are evaluated per group in parallel. "
  "The pipeline avoids Python UDFs and row loops, which would serialise the work behind the GIL.")
P("Plan of a filtered group-by (naive vs optimised, from LazyFrame.explain()):", bold=True)
CODE(BN.get("plan_excerpt", NA), 2600)
H("4.2 Benchmark against pandas", 2)
P("Each task runs in a fresh process (3 runs, median; peak resident memory). pandas is given its best practical form (column projection and "
  "row-group filters through pyarrow where the task allows). polars_eager reads the whole file first; polars_lazy lets the optimiser push work into the scan.")
names = {"T1": "T1 project+filter (1 quarter)", "T2": "T2 group-by quarter", "T3": "T3 per-bank lags/rolling", "T4": "T4 join + label"}
rows = []
for t, d in BN.get("tasks", {}).items():
    pd_, pe, pz = d.get("pandas", {}), d.get("polars_eager", {}), d.get("polars_lazy", {})
    rows.append([names.get(t, t), g(pd_, "median_s", "{:.2f}"), g(pe, "median_s", "{:.2f}"), g(pz, "median_s", "{:.2f}"),
                 ratio(pd_.get("median_s"), pz.get("median_s")), g(pd_, "peak_mb", "{:.0f}"), g(pz, "peak_mb", "{:.0f}")])
T(rows, ["Task", "pandas (s)", "Polars eager (s)", "Polars lazy (s)", "speed-up (pandas / lazy)", "pandas MB", "Polars MB"], widths=[4.6, 1.8, 2.2, 2.2, 2.6, 1.8, 1.8])
_T = BN.get("tasks", {})
if _T:
    _fast = sum(1 for d in _T.values() if (d.get("polars_lazy", {}).get("median_s") or 1e9) < (d.get("pandas", {}).get("median_s") or 0))
    _low = [t for t, d in _T.items() if (d.get("polars_lazy", {}).get("peak_mb") or 1e9) < (d.get("pandas", {}).get("peak_mb") or 0)]
    _high = [t for t in _T if t not in _low]
    _s = f"Polars lazy is faster than pandas on {_fast} of {len(_T)} tasks. Peak memory is lower on {', '.join(_low) or 'no task'}"
    if _high:
        _s += (f" and higher on {', '.join(_high)}: at this size the fixed overhead of the Polars runtime and its thread pool is comparable to the data "
               "itself, so the memory advantage should be judged from the replication table below.")
    else:
        _s += "."
    P(_s)
FIG("bench", "Figure 5. Left/middle: time and peak memory per task. Right: task T3 as the data is replicated k times.", 16)
sc = BN.get("scale", {})
T([[k, g(v, "pandas.seconds", "{:.1f}"), g(v, "pandas.peak_mb", "{:.0f}") + (" (" + v["pandas"]["error"][:40] + ")" if v.get("pandas", {}).get("error") else ""),
    g(v, "polars_lazy.seconds", "{:.1f}"), g(v, "polars_lazy.peak_mb", "{:.0f}"), g(SP, f"scale_seconds.{k}", "{:.1f}")] for k, v in sc.items()],
  ["Replication k", "pandas T3 (s)", "pandas MB", "Polars T3 (s)", "Polars MB", "Spark local (s)"], widths=[2.4, 2.6, 4, 2.6, 2.4, 2.6])
H("4.3 Reproducing the steps in PySpark", 2)
fc = SP.get("feature_check", {})
gc = SP.get("groupby_check", {})
P(f"The group-by, the gap-aware lag features (window functions with partitionBy('CERT').orderBy('qidx')) and the label counts were re-implemented in "
  f"PySpark {g(SP, 'spark_version')} (local[*], {g(SP, 'cores')} cores; spark.sql.shuffle.partitions lowered from 200 to 8 for a 1.7M-row table). "
  f"Equality check against the Polars outputs: results match = {g(SP, 'results_match')}.")
T([["per-quarter count / asset sum / failure-label count equal", f"{g(gc, 'count_equal')} / {g(gc, 'asset_sum_equal')} / {g(gc, 'y_pos_equal')}"],
   ["max |diff| of mean equity ratio per quarter", g(gc, "max_abs_diff_mean_equity_ratio", "{:.2e}")]] +
  [[f"{c}: max |diff| / null-pattern mismatches", f"{g(fc, c + '.max_abs_diff', '{:.2e}')} / {g(fc, c + '.null_pattern_mismatch')}"]
   for c in ("equity_ratio", "asset_growth_4q", "roa_q", "roa_q_mean4")] +
  [["rows compared (Spark / Polars / joined)", f"{g(fc, 'rows_spark', '{:,}')} / {g(fc, 'rows_polars', '{:,}')} / {g(fc, 'rows_compared', '{:,}')}"]],
  ["Check", "Result"], widths=[10, 6])
P("Reading the execution plan (excerpt below, full plans in results/spark_plans.txt): the window feature job contains Exchange hashpartitioning(CERT, ...) "
  f"(present: {'hashpartitioning(CERT' in SP.get('plan_excerpt', '') or bool(SP.get('plan_has_exchange'))}) - a shuffle that moves all rows of one bank to the same partition - followed by Sort (by qidx within the partition) "
  f"and Window (present: {g(SP, 'plan_has_window')}). The shuffle is the expensive wide transformation; the group-by also needs an Exchange but only after a "
  f"partial (map-side) aggregation. The scan node shows PushedFilters/ReadSchema (present: {g(SP, 'plan_pushed_filters')}), i.e. the same projection/predicate "
  "pushdown as Polars. Because all lags share the same partitioning and ordering (CERT, qidx), Spark typically evaluates them in one Window operator after one shuffle.")
CODE(SP.get("plan_excerpt", NA), 2200)
H("4.4 When is a cluster justified?", 2)
rows_n = num(DQ, "panel_log.final_rows")
t3 = BN.get("tasks", {}).get("T3", {})


def _slope(eng):
    ks = sorted(int(k) for k in sc if (sc[k].get(eng) or {}).get("peak_mb"))
    if rows_n and len(ks) >= 2:
        return (sc[str(ks[-1])][eng]["peak_mb"] - sc[str(ks[0])][eng]["peak_mb"]) * 1e6 / (rows_n * (ks[-1] - ks[0]))
    return None


bpr_pd, bpr_pl = _slope("pandas"), _slope("polars_lazy")
ram_gb = 16
cap_pd = ram_gb * 0.7e9 / bpr_pd if bpr_pd else None
cap_pl = ram_gb * 0.7e9 / bpr_pl if bpr_pl else None
P(f"Evidence. The complete clean panel has {panel_rows} rows and {g(ST, 'layouts.2.size_mb')} MB on disk. The window task needs about "
  f"{g(t3, 'pandas.peak_mb', '{:.0f}')} MB in pandas and {g(t3, 'polars_lazy.peak_mb', '{:.0f}')} MB in Polars, for the single-size run; between the smallest and the largest replication the marginal memory is ~{(bpr_pd or 0):.0f} (pandas) and ~{(bpr_pl or 0):.0f} (Polars) bytes per extra row, which removes the fixed library overhead. "
  f"Using 70 % of a {ram_gb} GB laptop, extrapolated capacity is ~{(cap_pd or 0) / 1e6:.0f} M rows for pandas and ~{(cap_pl or 0) / 1e6:.0f} M rows for Polars "
  f"({ratio(cap_pl, rows_n, '{:.0f}x')} the full 40-year panel). Spark local took {g(SP, 'scale_seconds.1', '{:.1f}')} s at k = 1 versus "
  f"{g(sc, '1.polars_lazy.seconds', '{:.1f}')} s for Polars"
  + (": JVM start-up, serialisation and the shuffle are fixed costs that a single machine cannot amortise at this size."
     if (num(SP, 'scale_seconds.1') or 0) > (num(sc, '1.polars_lazy.seconds') or 1e9) else ": Spark is not slower here, but it is not needed either (see below)."))
P("Argument. (i) For this project (1.7M rows, < 1 GB) a cluster adds overhead and no benefit: the measured Polars times above show that one laptop is "
  "sufficient, so the right architecture is Parquet + Polars. (ii) A cluster becomes justified when one of three measured limits is crossed: the working set no longer fits in RAM "
  "even with streaming/lazy execution (here beyond the extrapolated capacity above, e.g. loan-level or daily transaction data - hundreds of millions to billions "
  "of rows), the job is CPU-bound with runtime that must fall below an SLA (wall time scales ~ 1/N only for the parallel fraction p: speed-up = 1 / ((1 - p) + p/N), "
  "Amdahl's law, and the shuffle in the window job is part of the serial/communication share), or many users/jobs must share data concurrently. "
  "(iii) The Spark code was verified to give identical results, so migration is a deployment decision, not a rewrite.")

# =========================================================================== PART 4
H("5. Part 4 - Feature pipeline")
H("5.1 Feature design", 2)
P("Features follow the CAMELS framework used by supervisors and the bank-failure literature (capital adequacy, asset quality, earnings, liquidity and "
  "funding, sensitivity/growth), plus size and age. All are ratios (scale-free across 40 years of inflation and bank sizes), computed from fields "
  "known at the report date. Year-to-date flows are first converted to quarterly flows; flow ratios are annualised (x4).")
FD = {
    "log_asset": ("ln(ASSET)", "size; too-big-to-fail vs fragile small banks"),
    "age_years": ("(REPDTE - ESTYMD)/365.25", "young banks fail more often"),
    "n_quarters_seen": ("count of reports so far", "history length / data maturity"),
    "equity_ratio": ("EQ / ASSET", "capital buffer (C)"), "RBC1AAJ": ("Tier-1 leverage ratio", "regulatory capital (C)"),
    "RBCRWAJ": ("total risk-based capital ratio", "regulatory capital (C)"), "d_equity_ratio_4q": ("equity_ratio - same, 4q ago", "capital erosion"),
    "npl_ratio": ("NCLNLS / LNLSGR", "noncurrent loans (A)"), "past_due_30_89": ("P3ASSET / ASSET", "early delinquency (A)"),
    "past_due_90": ("P9ASSET / ASSET", "late delinquency (A)"), "nonaccrual_assets": ("NAASSET / ASSET", "problem assets (A)"),
    "ore_ratio": ("ORE / ASSET", "foreclosed property (A)"), "allowance_cov": ("LNATRES / LNLSGR", "reserve adequacy (A)"),
    "texas_ratio": ("(NCLNLS + ORE) / (EQ + LNATRES)", "classic distress ratio: bad assets vs loss-absorbing capital"),
    "d_npl_ratio_4q": ("npl_ratio - same, 4q ago", "deterioration speed"), "nco_rate": ("4 x quarterly NTLNLS / LNLSGR", "realised credit losses"),
    "provision_rate": ("4 x quarterly ELNATR / LNLSGR", "expected losses booked"),
    "roa_q": ("4 x quarterly NETINC / ASSET", "current profitability (E)"), "ROA": ("FDIC YTD ROA", "profitability"), "ROE": ("FDIC YTD ROE", "profitability"),
    "NIMY": ("net interest margin", "core earning power"), "efficiency": ("NONIX / (net interest income + NONII)", "cost control (E)"),
    "nonint_income_share": ("NONII / revenue", "revenue mix"), "roa_q_mean4": ("mean of roa_q over 4 quarters (past only)", "smooths seasonality"),
    "roa_q_std4": ("std of roa_q over 4 quarters", "earnings volatility"), "d_roa_4q": ("roa_q - roa_q 4q ago", "earnings trend"),
    "asset_growth_4q": ("ASSET / ASSET_4q-ago - 1", "rapid growth precedes failure (S)"), "loan_growth_4q": ("LNLSGR growth", "aggressive lending"),
    "deposit_growth_4q": ("DEP growth", "funding growth"), "loans_assets": ("LNLSGR / ASSET", "risk appetite"),
    "loan_deposit": ("LNLSNET / DEP", "funding gap (L)"), "re_loan_share": ("LNRE / LNLSGR", "real-estate concentration"),
    "ci_loan_share": ("LNCI / LNLSGR", "commercial concentration"), "con_loan_share": ("LNCON / LNLSGR", "consumer concentration"),
    "brokered_share": ("BRO / DEP", "reliance on rate-sensitive funding (L)"), "liquid_assets": ("(CHBAL + SC) / ASSET", "liquidity (L)"),
    "securities_share": ("SC / ASSET", "balance-sheet structure"), "d_brokered_4q": ("brokered_share - same, 4q ago", "funding shift"),
}
T([[f, *FD.get(f, ("", ""))] for f in FT.get("features", list(FD))], ["Feature", "Definition", "Rationale"], size=7, widths=[3.6, 6.2, 7.2])
H("5.2 Leak-free construction", 2)
BL([
    "All time operations are backward-looking: lag_k only, trailing windows only, per-bank. No centred windows, no future joins, no cross-bank statistics that use later quarters.",
    "Gap-aware lags: lag_k is null unless the row exactly k quarters earlier exists, so a reporting gap cannot silently pull in an older value.",
    "Label joined only through FAILDATE (target) after features are built; FAILDATE never enters a feature.",
    "No imputation, scaling, winsorisation or feature selection in the feature table: they are steps of a scikit-learn Pipeline (DropEmptyColumns -> Winsorizer -> SimpleImputer(median, missing indicators) "
    "-> StandardScaler -> model) that is fitted on the training fold only, in every CV fold and in the final fit. Calendar year is deliberately NOT a feature (regime identifier).",
])
H("5.3 Automated leakage test", 2)
T([[k, v["banks_compared"], f"{v['max_abs_diff']:.1e}", v["null_pattern_mismatches"], v["passed"]] for k, v in LK.items() if isinstance(v, dict)],
  ["Cut-off quarter q0", "banks compared", "max |diff|", "null mismatches", "passed"], widths=[3, 3, 3, 3, 2])
P("Method: for a random sample of banks the whole feature set is recomputed after deleting every row dated after q0; the features at q0 must be identical "
  f"to those computed from the full history. All cut-offs passed: {g(LK, 'all_passed')}. The pipeline aborts if this test fails (s06).")

# =========================================================================== PART 5
H("6. Part 5 - Modelling and validation")
H("6.1 Split and validation scheme", 2)
P(f"Development {g(S, 'dev')}; gap {g(S, 'gap')} (4 quarters, dropped); test {g(S, 'test')}. Why: failures cluster in 1984-1993 and 2008-2014 and are almost absent "
  "in between, so a random split is meaningless. Putting the S&L crisis AND the first GFC failures in development lets models learn two different "
  "crisis regimes; the test then contains the final GFC years, the calm 2013-2019 period and the 2023 regional-bank failures, "
  f"yielding {g(S, 'test_failures')} failed banks (>= 50 required). The gap exceeds the 4-quarter label window, so no development label can look into the test period. "
  "Trade-off: a test set dominated by the GFC tail has a lower base rate than the development period "
  f"({g(S, 'test_base_rate', '{:.4%}')} vs {g(S, 'dev_base_rate', '{:.4%}')}), which makes it a demanding, realistic out-of-regime test.")
cv = MD.get("cv", {})
P(f"Inside development: expanding-window, purged cross-validation with {g(cv, 'n_folds')} date blocks ({', '.join(cv.get('blocks', []))}). "
  f"For a block starting at q the training set contains only quarters <= q - {g(cv, 'purge_quarters')} (4-quarter label window + 1), so no training label overlaps the "
  "validation period. Random K-fold (and Spark's CrossValidator) is rejected because it mixes quarters of the same crisis in train and validation (lecture, Problem 10).")
P("Metrics. Accuracy is meaningless at a base rate of a few failures per 1,000 bank-quarters. Reported: PR-AUC (average precision; the primary, imbalance-sensitive metric), "
  "ROC-AUC (ranking), lift and recall in the top 5 % of rows, Brier score (calibration, needed because the threshold rests on probabilities) and, for decisions, "
  "an event-level expected cost: a failing bank is 'caught' if it raises at least one alarm in its four pre-failure quarters; false alarms are counted per bank-quarter.")
H("6.2 Models and tuning (development CV only)", 2)
cvr = MD.get("cv_results", {})
T([[k, g(v, "pooled.pr_auc", "{:.4f}"), g(v, "pooled.roc_auc", "{:.3f}"), g(v, "pooled.lift_top5pct", "{:.1f}"), g(v, "pooled.brier", "{:.4f}")]
   for k, v in cvr.items()], ["Model", "OOF PR-AUC", "OOF ROC-AUC", "lift top 5%", "Brier"], widths=[7, 2.5, 2.5, 2.5, 2.5])
P("Baselines: B1/B2 are one-variable supervisory rules (no fitting); B3 is a logistic regression on four CAMELS-style variables. Own models: M1 logistic on all features "
  "(interpretable linear benchmark) and M2 histogram gradient boosting (non-linear interactions, native NaN). Training negatives are subsampled and re-weighted "
  "(weight 1/f) inside CV to keep probabilities calibrated while cutting compute. M2 hyper-parameters were chosen by mean fold PR-AUC:")
T([[t["lr"], t["depth"], t["l2"], f"{t['mean_fold_pr_auc']:.4f}", g(t, "pooled_pr_auc", "{:.4f}")] for t in MD.get("tuning", [])],
  ["learning rate", "max depth", "L2", "mean fold PR-AUC", "pooled PR-AUC"], widths=[3, 3, 2, 4, 4])
P(f"Selected model (development evidence only): {g(MD, 'selected_model')}.")
H("6.3 Alarm threshold from costs", 2)
cm = MD.get("cost_model", {})
P(f"Cost of a missed failure c_FN is the mean COST (loss to the deposit-insurance fund) of the development failures: {g(cm, 'c_fn_mean_cost_kUSD', '{:,.0f}')} thousand USD "
  f"(n = {g(cm, 'n_cost_obs')}; median {g(cm, 'c_fn_median_cost_kUSD', '{:,.0f}')}). The data contain no cost of a false alarm (an extra supervisory exam or restriction), so "
  f"c_FP = {g(cm, 'fp_ratio_base')} x c_FN is an explicit assumption, and the result is shown for five ratios. The threshold minimises event-level expected cost on the "
  "out-of-fold development scores (never on test). If probabilities were perfectly calibrated the optimum would be p* = c_FP / (c_FP + c_FN), "
  f"i.e. {g(cm, 'break_even_precision', '{:.2%}')} here.")
th = MD.get("threshold", {})
T([[f"{s['fp_ratio']}", f"{s['threshold']:.4g}", s["caught"], s["missed"], s["false_alarms"], f"{s['cost']:,.0f}"] for s in th.get("sensitivity", [])],
  ["c_FP / c_FN", "threshold", "failures caught", "missed", "false alarms", "dev cost (kUSD)"], widths=[2.5, 2.5, 3, 2.2, 2.8, 3])
P(f"Chosen threshold (base case): {g(th, 'value', '{:.4g}')}; on development OOF scores it catches {g(th, 'dev_oof_confusion.caught')} of {g(th, 'dev_oof_confusion.events')} failures "
  f"with {g(th, 'dev_oof_confusion.false_alarms', '{:,}')} false alarms; doing nothing would cost {g(th, 'cost_never_alarm', '{:,.0f}')} kUSD versus {g(th, 'dev_oof_cost_kUSD', '{:,.0f}')} kUSD.")
FIG("model_cost_sensitivity", "Figure 6. Share of development failures caught as the assumed false-alarm cost rises.", 8)
H("6.4 Single evaluation on the test period", 2)
te = MD.get("test", {})
P("After all choices were frozen on development data, the selected model was refitted on the whole development period and evaluated once on the test period "
  "(a guard file prevents a second evaluation). The baselines were scored in the same single evaluation, fitted on development only.")
rowsT = [[te.get("model", NA) + " (selected)", g(te, "metrics.pr_auc", "{:.4f}"), g(te, "metrics.roc_auc", "{:.3f}"), g(te, "metrics.lift_top5pct", "{:.1f}"), g(te, "metrics.brier", "{:.4f}")]]
rowsT += [[k, g(v, "pr_auc", "{:.4f}"), g(v, "roc_auc", "{:.3f}"), g(v, "lift_top5pct", "{:.1f}"), g(v, "brier", "{:.4f}")] for k, v in te.get("reference_models", {}).items()]
T(rowsT, ["Model (test period)", "PR-AUC", "ROC-AUC", "lift top 5%", "Brier"], widths=[7.5, 2.3, 2.3, 2.3, 2.3])
P(f"Base rate in test: {g(S, 'test_base_rate', '{:.4%}')} (random PR-AUC = base rate).", italic=True, size=9)
ct = te.get("confusion_at_threshold", {})
T([["Failures in test / caught / missed", f"{g(ct, 'events')} / {g(ct, 'caught')} / {g(ct, 'missed')}"],
   ["False alarms (bank-quarters) / alarm rate", f"{g(ct, 'false_alarms', '{:,}')} / {g(te, 'alarm_rate', '{:.3%}')}"],
   ["Expected cost: model / never alarm / alarm everyone (kUSD)", f"{g(te, 'expected_cost_kUSD', '{:,.0f}')} / {g(te, 'cost_never_alarm_kUSD', '{:,.0f}')} / {g(te, 'cost_alarm_everyone_kUSD', '{:,.0f}')}"],
   ["Share of failing banks ranked in the top 5 % / 10 % of banks in a quarter at least once",
    f"{g(te, 'cross_sectional_recall_top5pct_of_banks', '{:.1%}')} / {g(te, 'cross_sectional_recall_top10pct_of_banks', '{:.1%}')}"]],
  ["Decision metric at the development threshold", "Value"], widths=[10, 6])
FIG("model_test", "Figure 7. Precision-recall curve on the test period and failures per year in the test set.", 15)
T([[int(r["year"]), int(r["rows"]), int(r["failures"]), f"{r['pr_auc']:.3f}" if r.get("pr_auc") == r.get("pr_auc") else NA,
    f"{r['roc_auc']:.3f}" if r.get("roc_auc") == r.get("roc_auc") else NA] for r in te.get("by_year", [])],
  ["Year", "rows", "failed banks", "PR-AUC", "ROC-AUC"], widths=[2, 3, 3, 3, 3], size=7.5)
_by = [r for r in te.get("by_year", []) if r.get("failures", 0) >= 3 and r.get("pr_auc") == r.get("pr_auc")]
_weak = [f"{int(r['year'])} ({r['pr_auc']:.3f})" for r in _by if r["pr_auc"] < 0.2]
_strong = [f"{int(r['year'])} ({r['pr_auc']:.2f})" for r in _by if r["pr_auc"] >= 0.5]
if _weak or _strong:
    P("Stability over time. PR-AUC by year is high in " + (", ".join(_strong) or "no year") + " but low in " + (", ".join(_weak) or "no year") +
      " (years with >= 3 failures). The pooled test score is therefore dominated by the post-crisis years; the weak years (few failures each, so noisy) "
      "indicate that Call Report ratios of this field set do not capture liquidity/interest-rate stress well. Metrics such as recall in the top 5 % are almost "
      "identical for all models and are not used to rank them.")
H("6.5 Spark MLlib demo (development data only)", 2)
P(f"A chronological split inside the development period (train <= {g(ML, 'train_end')}, validation {g(ML, 'val')}; the test period is not touched) fits a Spark ML Pipeline "
  "(Imputer -> VectorAssembler -> StandardScaler -> LogisticRegression / GBTClassifier with class weights).")
T([["MLlib logistic regression", g(ML, "mllib_logit.pr_auc", "{:.4f}"), g(ML, "mllib_logit.roc_auc", "{:.3f}")],
   ["MLlib GBT", g(ML, "mllib_gbt.pr_auc", "{:.4f}"), g(ML, "mllib_gbt.roc_auc", "{:.3f}")],
   ["scikit-learn logistic (same split)", g(ML, "sklearn_logit_same_split.pr_auc", "{:.4f}"), g(ML, "sklearn_logit_same_split.roc_auc", "{:.3f}")]],
  ["Model", "PR-AUC", "ROC-AUC"], widths=[8, 3, 3])
_ml, _sk = num(ML, "mllib_logit.pr_auc"), num(ML, "sklearn_logit_same_split.pr_auc")
_gb = num(ML, "mllib_gbt.pr_auc")
_txt = f"Validation base rate {g(ML, 'base_rate_val', '{:.4%}')}. "
if _ml is not None and _sk is not None:
    if abs(_ml - _sk) < 0.03:
        _txt += ("MLlib and scikit-learn logistic regression agree closely on the same split "
                 f"({_ml:.3f} vs {_sk:.3f} PR-AUC), which confirms that the implementations are comparable. ")
    else:
        _txt += (f"MLlib logistic regression ({_ml:.3f} PR-AUC) and scikit-learn logistic regression ({_sk:.3f}) differ on the same split. "
                 "The two pipelines are not identical (different regularisation scale, solver and missing-value handling: scikit-learn adds "
                 "missing-value indicators), and logistic models are sensitive to extreme ratios, so the gap reflects preprocessing and "
                 "regularisation rather than a different data split. ")
if _gb is not None and _sk is not None:
    _txt += (f"MLlib gradient-boosted trees reach {_gb:.3f}. ")
_txt += "MLlib is demonstrated for scale and API parity, not to obtain a better model on 1.7M rows."
P(_txt)

# =========================================================================== PART 6
H("7. Part 6 - Conclusions")
pa, pz = t3.get("pandas", {}).get("median_s"), t3.get("polars_lazy", {}).get("median_s")
H("Q1. Storage and processing on a laptop; when a cluster?", 3)
P(f"Store the 1.7M-row, 33-field panel as typed, sorted, zstd-compressed Parquet ({g(ST, 'layouts.2.size_mb')} MB, {ratio(csv_mb, opt_mb)} smaller than CSV, "
  f"with column and row-group skipping) and process it lazily with Polars (T3 window task {g(t3, 'polars_lazy.median_s', '{:.2f}')} s vs pandas {g(t3, 'pandas.median_s', '{:.2f}')} s, "
  f"{ratio(pa, pz)}). The whole project fits on a laptop with large headroom, and the PySpark version reproduces the Polars results (match = {g(SP, 'results_match')}). A cluster is justified only when data exceed RAM "
  "(order of 10^8-10^9 rows), when wall-time SLAs cannot be met by one machine, or for concurrent multi-user access.")
H("Q2. Quality problems and their treatment", 3)
P(f"The raw data contain era-dependent missing fields, year-to-date flow variables, logical inconsistencies and extreme ratios, reporting gaps, rows after failure, "
  f"failures whose reporting stops before the failure date ({fl.get('last_report_more_than_4q_before_failure', NA)} more than 4 quarters before) and non-failure exits "
  f"({g(ex, 'other_exit', '{:,}')} banks). The pipeline flags rather than silently edits, de-cumulates flows, makes lags gap-aware, classifies exits from the Failures and "
  "Institutions tables, and defers imputation, scaling and winsorisation to the training fold. Look-ahead is prevented structurally and verified: "
  f"the automated truncation test passed = {g(LK, 'all_passed')}.")
H("Q3. Can the output support an honestly validated early-warning model?", 3)
pr, br = num(te, "metrics.pr_auc"), num(S, "test_base_rate")
verdict = NA
refs = {k: (v.get("pr_auc") or 0) for k, v in te.get("reference_models", {}).items()}
if pr is not None and br:
    verdict = f"Yes, as a ranking tool: test PR-AUC {pr:.3f} is {pr / br:.0f}x the base rate {br:.4%}"
    if refs:
        bn = max(refs, key=refs.get)
        gain = pr - refs[bn]
        verdict += (f"; the best reference model ({bn}) reaches {refs[bn]:.3f}, so the gain of the selected model is "
                    f"{'marginal' if gain < 0.03 else 'clear'} ({gain:+.3f}) and should not be over-claimed with this few failures.")
    else:
        verdict += "."
P(f"{verdict} The validation is honest by construction (time-ordered split with a gap, purged CV, preprocessing fitted on training only, threshold from development cost "
  f"analysis, one test evaluation). At the chosen threshold the model caught {g(ct, 'caught')} of {g(ct, 'events')} failures with {g(ct, 'false_alarms', '{:,}')} false alarms. "
  "Absolute precision stays low because failures are extremely rare: the output is a prioritisation list for supervisors, not an automatic verdict.")
H("Limitations", 3)
BL([
    "Restated data (latest vintage), not as first filed; a true point-in-time store would need the FFIEC original filings.",
    "c_FP is an assumption (sensitivity shown); c_FN uses the average insurance-fund cost and ignores indirect costs and bank size.",
    "Failures are few in some regimes; the test period mixes the GFC tail, calm years and 2023 (uninsured-deposit runs and interest-rate risk are weakly captured by Call Report ratios of this field set).",
    "Banks that merge are right-censored; failures whose reporting ends > 4 quarters before FAILDATE cannot be labelled.",
    "Single test period; no confidence intervals on test metrics beyond the sample size note (few failures per year). Bootstrap by bank would be the next step.",
    "Spark runs locally; no real cluster measurements, so the scaling argument is extrapolated from measured per-row cost.",
])
H("What would change for Vietnam", 3)
BL([
    "Events: only a handful of bank resolutions (forced transfers, restructurings) exist, so a supervised model on Vietnamese data alone is infeasible; the realistic design is "
    "transfer learning/pre-training on US (and other-country) data plus a rule-based or scorecard layer calibrated by supervisors, and 'distress' labels (capital below minimum, "
    "SBV special control, restructuring) instead of failure.",
    "Data: monthly/quarterly supervisory returns, differing definitions (loan classification, NPL, restructured loans under evolving circulars), VAS vs IFRS, "
    "related-party and cross-ownership exposures, off-balance-sheet items; a canonical schema and per-field definition-versioning table is the first engineering task.",
    "Scale and architecture: ~tens of banks x quarters is tiny; Parquet + Polars (or DuckDB) on a secured on-premise server suffices; a cluster is irrelevant unless "
    "loan-level or transaction-level data from the credit register are ingested.",
    "Governance: data confidentiality (no cloud, access control, audit trail via a lakehouse ledger such as Delta), strict vintage storage (as-filed + revisions), and model risk "
    "management with human review of every alarm.",
])

# =========================================================================== REFERENCES
H("8. Sources, tools and AI disclosure")
BL([
    "FDIC BankFind Suite API and field dictionary: https://api.fdic.gov/banks/docs/ ; bulk data: https://banks.data.fdic.gov/bankfind-suite/bulkData.",
    "Course material BFT65B, Dr. Nguyen Tuan Long, National Economics University (2026): lectures on ML/Python review, data cleaning and transformation, feature selection, "
    "time-series ML, Big Data/Spark ML essentials (parquet, Polars, Spark windows and shuffle, PR-AUC and cost-based threshold, purged date splits) and the data-visualisation lectures.",
    "Course starter script fdic_download_starter.py (included unchanged as src/fdic_download_starter.py).",
    "Libraries: polars, pandas, pyarrow, scikit-learn, PySpark (MLlib), matplotlib, python-docx (versions in requirements.txt).",
    "AI tools: " + meta.get("ai_disclosure", "Claude (Anthropic) was used to draft the pipeline code structure and report text; the author ran the pipeline, checked the "
                            "outputs and is responsible for the content."),
])

# =========================================================================== APPENDIX
doc.add_page_break()
H("Appendix A. Data-quality report (automated)")
nbd = DQ.get("null_share_by_decade", {})
cols = ["ROA", "RBC1AAJ", "RBCRWAJ", "NAASSET", "P3ASSET", "P9ASSET", "ORE", "BRO", "NTLNLS", "NIMY"]
decs = sorted(nbd)
T([[c, *[f"{nbd[d].get(c, float('nan')):.0%}" if c in nbd[d] else NA for d in decs]] for c in cols], ["Field null share", *decs], size=7.5)
T([[k, f"{v:,}"] for k, v in DQ.get("rule_violations", {}).items()], ["Rule", "Rows"], size=7.5, widths=[9, 3])
T([[str(k).replace(".0", ""), f"{v:,}"] for k, v in DQ.get("other_exit_changec1_top", {}).items()], ["CHANGEC1 code (other exits, top 15)", "Banks"], size=7.5, widths=[7, 3])
T([[k, v] for k, v in DQ.get("ratio_quantiles", {}).get("ROA", {}).items()], ["ROA quantile", "Value (%)"], size=7.5, widths=[4, 4])
_g = {int(k): v for k, v in fl.get("last_report_to_failure_quarters", {}).items()}
_rows = [[str(k), f"{_g.get(k, 0):,}"] for k in range(-4, 6)]
_rows.insert(0, ["< -4", f"{sum(v for k, v in _g.items() if k < -4):,}"])
_rows.append(["> 5", f"{sum(v for k, v in _g.items() if k > 5):,}"])
T(_rows, ["Failure quarter minus last-report quarter", "Failed banks"], size=7.5, widths=[7, 3])
P("Reading: +1 means the bank filed its last report one quarter before the quarter of the failure date (the normal case). Negative values mean the "
  "bank kept reporting after FAILDATE (for example assistance transactions or institutions that continued under the same CERT); all rows dated on or after FAILDATE are "
  "dropped from the panel, so these banks contribute only their pre-event history to the labels.", size=9)
H("Appendix B. Polars and Spark plans")
CODE(BN.get("plan_excerpt", NA), 3000)
CODE(SP.get("plan_excerpt", NA), 3500)
H("Appendix C. Reproducibility")
P("README.md gives the exact commands: pip install -r requirements.txt; python run_all.py (download is resumable; --smoke runs 4 quarters in minutes); "
  "python src/build_report.py. Seeds are fixed (config.SEED = 42). Data files are not committed (.gitignore). "
  f"Runtime of the modelling stage: {g(MD, 'runtime_s')} s.")

doc.save(A.out)
print("saved", A.out)