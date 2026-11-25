"""PART 5b - Spark MLlib demo (development data ONLY: the test period is never touched here).

Chronological split inside the development period (no random split, no CrossValidator with random folds):
  train: quarters <= 2002Q4 | gap 4 quarters | validation: 2004Q1-2009Q4 (labels up to 2010Q4 are dev-only).
MLlib Pipeline: Imputer(median) -> VectorAssembler -> StandardScaler -> LogisticRegression (class weights).
Also a GBTClassifier. Metric: area under the PR curve (and ROC). The same split is fitted in scikit-learn.
"""
from __future__ import annotations

import numpy as np

from common import save_result
from config import DEV_END, PROC, SEED, qidx
from s05_spark import get_spark
from s06_features import FEATURES

TRAIN_END, VAL_START = qidx(2002, 4), qidx(2004, 1)


def main():
    from pyspark.ml import Pipeline
    from pyspark.ml.classification import GBTClassifier, LogisticRegression
    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    from pyspark.ml.feature import Imputer, StandardScaler, VectorAssembler
    from pyspark.sql import functions as F

    spark = get_spark()
    df = (spark.read.parquet(str(PROC / "features.parquet")).filter(~F.col("label_incomplete"))
          .filter(F.col("qidx") <= qidx(*DEV_END)))
    tr = df.filter(F.col("qidx") <= TRAIN_END)
    va = df.filter(F.col("qidx") >= VAL_START)
    pos_rate = tr.agg(F.mean("y")).first()[0]
    w_pos = (1 - pos_rate) / pos_rate
    tr = tr.withColumn("w", F.when(F.col("y") == 1, w_pos).otherwise(1.0)).withColumn("label", F.col("y").cast("double"))
    va = va.withColumn("label", F.col("y").cast("double"))

    # Imputer needs double columns; columns that are entirely null in train are dropped from the feature list
    usable = [c for c in FEATURES if tr.filter(F.col(c).isNotNull()).limit(1).count() > 0]
    tr = tr.select("label", "w", *[F.col(c).cast("double").alias(c) for c in usable])
    va = va.select("label", *[F.col(c).cast("double").alias(c) for c in usable])
    # winsorise at the 0.5 / 99.5 % quantiles learned on the TRAINING rows only (same rule as the sklearn Winsorizer)
    qs = tr.approxQuantile(usable, [0.005, 0.995], 0.001)

    def clip(df):
        keep = [F.col(x) for x in df.columns if x not in usable]
        return df.select(*keep, *[F.when(F.col(c).isNotNull(), F.least(F.greatest(F.col(c), F.lit(lo)), F.lit(hi))).alias(c)
                                  for c, (lo, hi) in zip(usable, qs)])
    tr, va = clip(tr), clip(va)
    imp = Imputer(strategy="median", inputCols=usable, outputCols=[f"{c}_i" for c in usable])
    asm = VectorAssembler(inputCols=[f"{c}_i" for c in usable], outputCol="raw")
    sc = StandardScaler(inputCol="raw", outputCol="features", withMean=True, withStd=True)
    ev_pr = BinaryClassificationEvaluator(metricName="areaUnderPR")
    ev_roc = BinaryClassificationEvaluator(metricName="areaUnderROC")
    out = {"train_rows": tr.count(), "val_rows": va.count(), "features_used": len(usable),
           "train_end": "2002Q4", "val": "2004Q1-2009Q4"}

    lr = LogisticRegression(featuresCol="features", labelCol="label", weightCol="w", regParam=0.01, maxIter=50)
    m_lr = Pipeline(stages=[imp, asm, sc, lr]).fit(tr)
    p_lr = m_lr.transform(va)
    out["mllib_logit"] = {"pr_auc": ev_pr.evaluate(p_lr), "roc_auc": ev_roc.evaluate(p_lr)}

    gbt = GBTClassifier(featuresCol="features", labelCol="label", weightCol="w", maxIter=40, maxDepth=4, seed=SEED)
    m_gb = Pipeline(stages=[imp, asm, sc, gbt]).fit(tr)
    p_gb = m_gb.transform(va)
    out["mllib_gbt"] = {"pr_auc": ev_pr.evaluate(p_gb), "roc_auc": ev_roc.evaluate(p_gb)}

    # same split in scikit-learn for a like-for-like comparison
    from sklearn.metrics import average_precision_score, roc_auc_score
    from s07_model import make_pipe
    trp, vap = tr.toPandas(), va.toPandas()
    pipe = make_pipe("logit", usable)
    pipe.fit(trp[usable].values.astype("float32"), trp["label"].values,
             clf__sample_weight=np.where(trp["label"] == 1, w_pos, 1.0))
    s = pipe.predict_proba(vap[usable].values.astype("float32"))[:, 1]
    out["sklearn_logit_same_split"] = {"pr_auc": float(average_precision_score(vap["label"], s)),
                                       "roc_auc": float(roc_auc_score(vap["label"], s))}
    out["base_rate_val"] = float(vap["label"].mean())
    save_result("mllib", out)
    spark.stop()
    print(out)


if __name__ == "__main__":
    main()