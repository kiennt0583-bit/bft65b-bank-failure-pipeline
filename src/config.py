"""Central configuration. Every design choice that the report justifies lives here."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data_raw" / "fdic"          # written by fdic_download_starter.py (NOT committed)
PROC = ROOT / "data_proc"                 # derived data (NOT committed)
RES = ROOT / "results"                    # json / md results read by build_report.py
FIG = ROOT / "figures"
for _p in (PROC, RES, FIG):
    _p.mkdir(parents=True, exist_ok=True)

SEED = 42

# ---- scope (course convention) -------------------------------------------------
FIRST_Q = (1984, 1)
LAST_Q = (2026, 2)
EXPECTED_QUARTERS = (LAST_Q[0] - FIRST_Q[0]) * 4 + (LAST_Q[1] - FIRST_Q[1]) + 1   # 170
LAST_LABELLED_Q = (2025, 2)               # q+4 quarters must be <= 2026Q2

# ---- field groups --------------------------------------------------------------
# Flow variables in Call Reports are YEAR-TO-DATE (reset every Q1) -> de-cumulated in s06.
YTD_FIELDS = ["NETINC", "INTINC", "EINTEXP", "NONII", "NONIX", "ELNATR", "NTLNLS"]
RATIO_FIELDS = ["ROA", "ROE", "NIMY", "RBC1AAJ", "RBCRWAJ"]              # percent
NONNEG_FIELDS = ["ASSET", "DEP", "DEPDOM", "BRO", "CHBAL", "SC", "LNLSGR", "LNRE", "LNCI",
                 "LNCON", "LNATRES", "NCLNLS", "P3ASSET", "P9ASSET", "NAASSET", "ORE"]

# ---- time split (justified in the report) --------------------------------------
# Development: 1984Q1-2009Q4 (S&L crisis + the first GFC failures).
# Gap: 2010Q1-2010Q4 (4 quarters, dropped) so that no development label window
#      overlaps the test feature dates.
# Test: 2011Q1-2025Q2 (last GFC failures, 2013-2019 calm, 2023 regional-bank stress).
DEV_END = (2009, 4)
GAP_Q = 4
TEST_START = (2011, 1)
TEST_END = LAST_LABELLED_Q
MIN_TEST_FAILURES = 50

# Expanding-window, purged CV blocks inside the development period (validation blocks).
CV_BLOCKS = [((1988, 1), (1990, 4)), ((1991, 1), (1993, 4)), ((1994, 1), (2001, 4)),
             ((2002, 1), (2005, 4)), ((2006, 1), (2009, 4))]
PURGE_Q = 5          # train quarters must be <= block_start - 5 (4-quarter label window + 1)

# ---- cost model for the alarm threshold ----------------------------------------
# c_FN = mean DIF COST of development failures (from the Failures table, data-derived).
# c_FP = FP_RATIO * c_FN. The Failures table cannot reveal the cost of a false alarm,
#        so FP_RATIO is an explicit assumption with a sensitivity table (see report).
FP_RATIO_BASE = 0.01
FP_RATIO_GRID = [0.002, 0.005, 0.01, 0.02, 0.05]

# ---- scale test ----------------------------------------------------------------
SCALE_FACTORS = [1, 4, 8]


def qidx(year: int, q: int) -> int:
    """Absolute quarter index: 2009Q4 -> 2009*4+3."""
    return year * 4 + (q - 1)


def qlabel(idx: int) -> str:
    return f"{idx // 4}Q{idx % 4 + 1}"
