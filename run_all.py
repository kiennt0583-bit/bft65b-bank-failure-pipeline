"""Run the whole pipeline end to end.  python run_all.py [--smoke] [--skip-download] [--skip-spark]

--smoke : download only 4 quarters (2023) and use a tiny CV grid - minutes instead of hours; for testing the setup.
          NOTE: the smoke run cannot satisfy the split rules (needs >= 50 test failures); full data are required for the report.
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true")
ap.add_argument("--skip-download", action="store_true")
ap.add_argument("--skip-spark", action="store_true")
a = ap.parse_args()


def run(*cmd):
    print("\n>>>", " ".join(map(str, cmd)), flush=True)
    subprocess.run([sys.executable, *map(str, cmd)], cwd=ROOT, check=True)


if not a.skip_download:
    extra = ["--start", "2023-03-31", "--end", "2023-12-31"] if a.smoke else []
    run(SRC / "fdic_download_starter.py", "--out", ROOT / "data_raw" / "fdic", *extra)   # resumable
run(SRC / "s02_storage.py")
run(SRC / "s03_quality_panel.py")
run(SRC / "s06_features.py")
run(SRC / "s04_polars_vs_pandas.py")
if not a.skip_spark:
    run(SRC / "s05_spark.py")
if not a.smoke:
    run(SRC / "s07_model.py")
    if not a.skip_spark:
        run(SRC / "s08_spark_mllib.py")
    run(SRC / "build_report.py")
