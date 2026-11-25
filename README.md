# BFT65B Final Report - Big-Data Pipeline for Bank Failure Early Warning

Data engineering pipeline on FDIC Call Reports (1984Q1-2026Q2): resumable download -> Parquet storage -> automated data-quality audit and
point-in-time panel -> Polars/pandas/PySpark processing -> leak-free features -> purged time-series validation, cost-based threshold, single test.
Every number in the report is produced by this code (`results/*.json` -> `src/build_report.py`).

## Layout
| file | part | what it does |
|---|---|---|
| `src/fdic_download_starter.py` | 1 | course starter: API download, resumable, completeness manifest (unchanged) |
| `src/s02_storage.py` | 1 | completeness verification, optimised schema, CSV vs Parquet layouts / codecs / row groups, timings |
| `src/s03_quality_panel.py` | 2 | automated DQ audit (`results/dq_report.md`), exit classification, clean panel + label |
| `src/s04_polars_vs_pandas.py` | 3 | Polars vs pandas benchmark (subprocess per run, peak memory), plan, scale test |
| `src/s05_spark.py` | 3 | PySpark re-implementation, equality check vs Polars, plans, scale timing |
| `src/s06_features.py` | 4 | point-in-time features, **automated leakage test** (aborts on failure) |
| `src/s07_model.py` | 5 | purged expanding CV, tuning, cost-based threshold, single test (guard file) |
| `src/s08_spark_mllib.py` | 5 | Spark MLlib pipeline demo on development data only |
| `src/build_report.py` | - | builds `report.docx` from `results/` and `figures/` |
| `tests/test_logic.py` | - | unit tests (purge, event cost, threshold, label, YTD) |
| `src/config.py` | - | all design constants (split, CV blocks, cost ratios, seeds) |

## Run
```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python tests/test_logic.py                                # quick logic tests
python run_all.py --smoke                                 # optional: 4-quarter dry run (minutes)
python run_all.py                                         # full run: download ~1 h (resumable), then everything
# edit report_meta.json (name, student id, GitHub URL), then:
python src/build_report.py                                # -> report.docx  (export to PDF from Word)
```
Individual stages can be run alone, in order: `s02 -> s03 -> s06 -> s04 -> s05 -> s07 -> s08 -> build_report`.
The download is resumable: rerun the same command and finished quarters are skipped.

## Notes
* Spark runs locally (`local[*]`). On Windows PySpark needs Java 11/17 (and HADOOP_HOME/winutils); if that is a hassle, run `s05` and `s08` on
  Google Colab or Databricks Free Edition (upload `data_proc/panel.parquet` and `data_proc/features.parquet`, change nothing but the paths in `config.py`)
  and copy `results/spark.json`, `results/mllib.json`, `results/spark_plans.txt` back.
* The test period is evaluated once: `results/test_evaluated.flag` blocks a second run of `s07_model.py`.
* Do not commit `data_raw/` or `data_proc/` (see `.gitignore`).
* Seeds: `config.SEED = 42` everywhere.

## Sources and AI disclosure
FDIC BankFind API (https://api.fdic.gov/banks/docs/); course lectures BFT65B (NEU, Dr. Nguyen Tuan Long); course starter script;
polars, pandas, pyarrow, scikit-learn, PySpark documentation. The pipeline code and report text were drafted with the assistance of Claude
(Anthropic); the author ran, checked and is responsible for them.
