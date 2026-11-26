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
| `src/s09_ablation.py` | 4 | feature-group ablation on development CV only (evidence for the feature choices) |
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
Individual stages can be run alone, in order: `s02 -> s03 -> s06 -> s04 -> s05 -> s07 -> s09 -> s08 -> build_report`.
The download is resumable: rerun the same command and finished quarters are skipped.

## Notes
* Spark runs locally (`local[*]`). On Windows PySpark needs Java 11/17 (and HADOOP_HOME/winutils); if that is a hassle, run `s05` and `s08` on
  Google Colab or Databricks Free Edition (upload `data_proc/panel.parquet` and `data_proc/features.parquet`, change nothing but the paths in `config.py`)
  and copy `results/spark.json`, `results/mllib.json`, `results/spark_plans.txt` back.
* The test period is evaluated once: `results/test_evaluated.flag` blocks a second run of `s07_model.py`.
* Do not commit `data_raw/` or `data_proc/` (see `.gitignore`).
* Seeds: `config.SEED = 42` everywhere.

## Sources and AI disclosure
Data: FDIC BankFind API (https://api.fdic.gov/banks/docs/). Course: BFT65B lectures (NEU, Dr. Nguyen Tuan Long) and the course starter script
`src/fdic_download_starter.py` (included unchanged). Libraries (versions in `requirements.txt`): polars, pandas, pyarrow, scikit-learn, PySpark/MLlib,
matplotlib, python-docx.

AI tools: the pipeline code structure and the report text were drafted with the assistance of Claude (Anthropic, via Claude Code); the author ran the
pipeline, checked the outputs and is responsible for the content.

### References (the same numbering as in the report; each is applied where noted)
1. FDIC. *Risk Management Manual of Examination Policies*, Section 1.1 (UFIRS / CAMELS). https://www.fdic.gov/resources/supervision-and-examinations/examination-policies-manual/section1-1.pdf  - feature groups (`s06_features.py`).
2. Jenkins, S. and Romanko, R. (2025). Banking Analytics: Understanding Credit Risk with the Texas Ratio. Federal Reserve Bank of St. Louis. https://www.stlouisfed.org/on-the-economy/2025/nov/banking-analytics-taking-banks-risk-temperature-texas-ratio  - `texas_ratio` feature and baseline B2.
3. Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley (Ch. 7)  - purged expanding-window CV (`modeling_utils.purged_expanding_folds`).
4. Bergmeir, C. and Benitez, J. M. (2012). On the use of cross-validation for time series predictor evaluation. *Information Sciences* 191, 192-213  - why not random K-fold.
5. Kaufman, S., Rosset, S., Perlich, C. and Stitelman, O. (2012). Leakage in data mining: formulation, detection, and avoidance. *ACM TKDD* 6(4), Art. 15  - leakage definition, learn-predict separation (Pipeline fitted on training only, `s06` leakage test).
6. Saito, T. and Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLoS ONE* 10(3), e0118432  - PR-AUC as primary metric.
7. Brier, G. W. (1950). Verification of forecasts expressed in terms of probability. *Monthly Weather Review* 78(1), 1-3  - Brier score.
8. Ke, G. et al. (2017). LightGBM: a highly efficient gradient boosting decision tree. *NeurIPS 30*  - histogram-based boosting behind scikit-learn's `HistGradientBoostingClassifier` (model M2).
9. Pedregosa, F. et al. (2011). Scikit-learn: machine learning in Python. *JMLR* 12, 2825-2830  - Pipeline, logistic regression, boosting, metrics.
10. McKinney, W. (2010). Data structures for statistical computing in Python. *Proc. 9th Python in Science Conf.*, 56-61  - pandas baseline in `s04`.
11. Zaharia, M. et al. (2016). Apache Spark: a unified engine for big data processing. *Commun. ACM* 59(11), 56-65  - PySpark re-implementation (`s05`).
12. Meng, X. et al. (2016). MLlib: machine learning in Apache Spark. *JMLR* 17(34), 1-7  - Spark ML pipeline (`s08`).
13. Amdahl, G. M. (1967). Validity of the single processor approach to achieving large scale computing capabilities. *AFIPS Spring Joint Computer Conf.*, 483-485  - cluster argument (report Section 4.4).
14. Databricks documentation, medallion lakehouse architecture. https://docs.databricks.com/aws/en/lakehouse/medallion.html  - bronze/silver/gold layering.
15. Apache Parquet documentation, File Format and Page Index. https://parquet.apache.org/docs/file-format/  - row groups, column chunks, min/max skipping (`s02`).
16. Polars user guide, lazy API optimizations. https://docs.pola.rs/user-guide/lazy/optimizations/  - lazy plans in `s04`.
17. Apache Spark documentation, SQL Performance Tuning (`spark.sql.shuffle.partitions`). https://spark.apache.org/docs/latest/sql-performance-tuning.html  - `s05` configuration.