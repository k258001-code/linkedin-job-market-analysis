"""
spark_pipeline.py
=================
Full LinkedIn Jobs analysis pipeline - preprocessing, EDA aggregations,
and K-Means clustering - in a single Spark application.

Run on YARN (1 master + 3 workers):
    spark-submit \
        --master yarn \
        --deploy-mode cluster \
        --num-executors 6 \
        --executor-cores 2 \
        --executor-memory 6g \
        --driver-memory 3g \
        --conf spark.executor.memoryOverhead=1g \
        spark_pipeline.py

HDFS paths (created automatically):
    Input  : hdfs:///data/linkedin/raw/linkedin_jobs.csv
    Clean  : hdfs:///data/linkedin/processed/clean_jobs      (Parquet)
    Skills : hdfs:///data/linkedin/processed/skills_exploded (Parquet)
    Results: hdfs:///data/linkedin/results/<table>/          (CSV)

After this script finishes, run:
    bash 04_export_results.sh   →   linkedin_results.zip
Then open colab_dashboard.py in Google Colab for visualisation.
"""

# ─── Imports ──────────────────────────────────────────────────────────────────
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, DoubleType, BooleanType
)
from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, HashingTF, IDF, PCA, StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql import functions as F

# ─── Config ───────────────────────────────────────────────────────────────────
HDFS_RAW       = "hdfs:///data/linkedin/raw/linkedin_jobs.csv"
HDFS_CLEAN     = "hdfs:///data/linkedin/processed/clean_jobs"
HDFS_SKILLS    = "hdfs:///data/linkedin/processed/skills_exploded"
HDFS_RESULTS   = "hdfs:///data/linkedin/results"

RAW_SCHEMA = StructType([
    StructField("job_id",           StringType(),  True),
    StructField("title",            StringType(),  True),
    StructField("company",          StringType(),  True),
    StructField("sector",           StringType(),  True),
    StructField("city",             StringType(),  True),
    StructField("state",            StringType(),  True),
    StructField("region",           StringType(),  True),
    StructField("latitude",         DoubleType(),  True),
    StructField("longitude",        DoubleType(),  True),
    StructField("experience_level", StringType(),  True),
    StructField("employment_type",  StringType(),  True),
    StructField("salary_min",       IntegerType(), True),
    StructField("salary_max",       IntegerType(), True),
    StructField("skills",           StringType(),  True),
    StructField("posted_date",      StringType(),  True),
    StructField("applies",          IntegerType(), True),
    StructField("views",            IntegerType(), True),
    StructField("remote_allowed",   BooleanType(), True),
])

# ─── Spark Session ─────────────────────────────────────────────────────────────
spark = (SparkSession.builder
         .appName("LinkedIn-Full-Pipeline")
         .config("spark.sql.shuffle.partitions", "48")
         .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
         .config("spark.sql.parquet.compression.codec", "snappy")
         .getOrCreate())
spark.sparkContext.setLogLevel("WARN")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 - PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════
def run_preprocessing():
    print("\n" + "="*60)
    print("  STAGE 1 - PREPROCESSING")
    print("="*60)

    # 1. Read
    print("  Reading raw CSV ")
    df = (spark.read
          .option("header", "true")
          .option("inferSchema", "false")
          .schema(RAW_SCHEMA)
          .csv(HDFS_RAW))
    print(f"   Raw rows: {df.count():,}")

    # 2. Drop rows missing critical fields
    df = df.dropna(subset=["job_id", "title", "sector", "posted_date"])

    # 3. Fill nulls
    df = df.fillna({
        "company": "Unknown", "city": "Unknown", "state": "Unknown",
        "region": "Unknown", "experience_level": "Unknown",
        "employment_type": "Full-time", "skills": "", "remote_allowed": False,
    })
    for col in ("salary_min", "salary_max", "applies", "views"):
        median_val = df.approxQuantile(col, [0.5], 0.01)[0]
        df = df.fillna({col: int(median_val)})

    # 4. Date & temporal features
    print("  Parsing dates and building temporal features ")
    df = (df
          .withColumn("posted_date",   F.to_date("posted_date", "yyyy-MM-dd"))
          .withColumn("post_year",     F.year("posted_date"))
          .withColumn("post_month",    F.month("posted_date"))
          .withColumn("post_quarter",  F.quarter("posted_date"))
          .withColumn("post_yearmon",  F.date_format("posted_date", "yyyy-MM")))

    # 5. Text normalisation
    df = (df
          .withColumn("sector",           F.trim(F.lower("sector")))
          .withColumn("title_clean",      F.trim(F.lower(F.regexp_replace("title", r"[^a-zA-Z0-9 ]", ""))))
          .withColumn("experience_level", F.trim("experience_level"))
          .withColumn("employment_type",  F.trim("employment_type")))

    # 6. Salary features
    print("  Engineering salary features ")
    df = (df
          .withColumn("salary_mid",  ((F.col("salary_min") + F.col("salary_max")) / 2).cast("int"))
          .withColumn("salary_band",
              F.when(F.col("salary_mid") < 50000,  "Under $50K")
               .when(F.col("salary_mid") < 80000,  "$50K-$80K")
               .when(F.col("salary_mid") < 120000, "$80K-$120K")
               .when(F.col("salary_mid") < 170000, "$120K-$170K")
               .otherwise("$170K+")))

    # 7. Engagement rate
    df = df.withColumn("apply_rate",
        F.when(F.col("views") > 0, F.col("applies") / F.col("views")).otherwise(0.0))

    # 8. Explode skills (one row per skill per job)
    print("  Exploding skills ...")
    skills_df = (df
        .withColumn("skill", F.explode(F.split(F.col("skills"), r",\s*")))
        .withColumn("skill", F.trim(F.lower("skill")))
        .filter(F.col("skill") != "")
        .select("job_id", "sector", "city", "state", "region",
                "post_year", "post_month", "post_yearmon",
                "experience_level", "salary_mid", "skill"))

    # 9. Write Parquet
    print("  Writing clean Parquet (partitioned by sector / post_year) ...")
    (df.repartition("sector", "post_year")
       .write.mode("overwrite")
       .partitionBy("sector", "post_year")
       .parquet(HDFS_CLEAN))

    print("  Writing exploded skills Parquet")
    (skills_df.repartition("sector")
              .write.mode("overwrite")
              .partitionBy("sector")
              .parquet(HDFS_SKILLS))

    clean_count = spark.read.parquet(HDFS_CLEAN).count()
    print(f"\n  Stage 1 done  -  {clean_count:,} clean rows,  {skills_df.count():,} skill rows")
    return df, skills_df


# ═
# STAGE 2 - EDA AGGREGATIONS
# ══════════════════════════════════════════════════════════════════════════════
def run_eda(df, skills):
    print("\n" + "="*60)
    print("  STAGE 2 - EDA AGGREGATIONS")
    print("="*60)

    def save(table_df, name):
        (table_df.coalesce(1)
                 .write.mode("overwrite")
                 .option("header", "true")
                 .csv(f"{HDFS_RESULTS}/{name}"))
        print(f"     {name}")

    # A. Sector × geo demand
    print(" [A] Sector demand map")
    save(
        df.filter(F.col("city") != "Remote")
          .groupBy("sector", "city", "state", "region", "latitude", "longitude")
          .agg(
              F.count("*").alias("job_count"),
              F.round(F.avg("salary_mid"), 0).alias("avg_salary"),
              F.round(F.avg("apply_rate"), 4).alias("avg_apply_rate"),
          )
          .orderBy(F.desc("job_count")),
        "sector_demand_geo"
    )

    # B. Sector monthly trend
    print("  [B] Sector monthly trend ")
    save(
        df.groupBy("sector", "post_yearmon", "post_year", "post_month")
          .agg(F.count("*").alias("job_count"))
          .orderBy("sector", "post_yearmon"),
        "sector_monthly_trend"
    )

    # C. Overall skill frequency (top 100)
    print(" [C] Skill frequency ")
    save(
        skills.groupBy("skill")
              .agg(F.count("*").alias("freq"))
              .orderBy(F.desc("freq"))
              .limit(100),
        "skill_freq_overall"
    )

    # D. Skill YoY trend - emerging vs declining
    print("  [D] Skill YoY trend ")
    skill_pivot = (skills
        .groupBy("skill", "post_year")
        .agg(F.count("*").alias("count"))
        .groupBy("skill")
        .pivot("post_year", [2022, 2023, 2024])
        .agg(F.first("count"))
        .fillna(0)
        .withColumnRenamed("2022", "cnt_2022")
        .withColumnRenamed("2023", "cnt_2023")
        .withColumnRenamed("2024", "cnt_2024"))

    save(
        skill_pivot
          .withColumn("total", F.col("cnt_2022") + F.col("cnt_2023") + F.col("cnt_2024"))
          .withColumn("yoy_2223",
              F.when(F.col("cnt_2022") > 0,
                     F.round((F.col("cnt_2023") - F.col("cnt_2022")) / F.col("cnt_2022"), 4))
               .otherwise(None))
          .withColumn("yoy_2324",
              F.when(F.col("cnt_2023") > 0,
                     F.round((F.col("cnt_2024") - F.col("cnt_2023")) / F.col("cnt_2023"), 4))
               .otherwise(None))
          .withColumn("trend_label",
              F.when(F.col("yoy_2324") > 0.15,  "Emerging")
               .when(F.col("yoy_2324") < -0.10, "Declining")
               .otherwise("Stable"))
          .filter(F.col("total") >= 20)
          .orderBy(F.desc("total")),
        "skill_trend_yoy"
    )

    # E. Geo heatmap (city-level)
    print(" [E] Geo heatmap ")
    save(
        df.filter(F.col("city") != "Remote")
          .groupBy("city", "state", "region", "latitude", "longitude")
          .agg(
              F.count("*").alias("total_jobs"),
              F.round(F.avg("salary_mid"), 0).alias("avg_salary"),
              F.countDistinct("sector").alias("sector_diversity"),
              F.countDistinct("company").alias("unique_companies"),
          )
          .orderBy(F.desc("total_jobs")),
        "geo_heatmap"
    )

    # F. Salary distribution
    print("  [F] Salary distribution ")
    save(
        df.groupBy("sector", "experience_level", "salary_band")
          .agg(
              F.count("*").alias("job_count"),
              F.round(F.avg("salary_mid"), 0).alias("avg_salary"),
              F.round(F.stddev("salary_mid"), 0).alias("std_salary"),
              F.round(F.percentile_approx("salary_mid", 0.25), 0).alias("p25_salary"),
              F.round(F.percentile_approx("salary_mid", 0.50), 0).alias("median_salary"),
              F.round(F.percentile_approx("salary_mid", 0.75), 0).alias("p75_salary"),
          )
          .orderBy("sector", "experience_level"),
        "salary_distribution"
    )

    # G. Remote vs on-site trend
    print("  [G] Remote trend ")
    save(
        df.groupBy("post_yearmon", "remote_allowed", "sector")
          .agg(F.count("*").alias("job_count"))
          .orderBy("post_yearmon", "sector"),
        "remote_trend"
    )

    print("\n  Stage 2 done  -  7 result tables written")


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 - ML CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════
def run_clustering():
    print("\n" + "="*60)
    print("  STAGE 3 - ML CLUSTERING")
    print("="*60)

    # Load clean data
    df = (spark.read.parquet(HDFS_CLEAN)
          .withColumn(
              "post_year",
              F.substring("post_yearmon", 1, 4).cast("int")
          )
          .select(
              "job_id",
              "title_clean",
              "sector",
              "skills",
              "salary_mid",
              "experience_level",
              "city",
              "company",
              "post_year"
          )
          .filter(F.col("skills") != ""))
    print(f"  Rows for clustering: {df.count():,}")

    # 1. Feature pipeline: skills - TF-IDF → PCA(50) → StandardScaler
    print("  Fitting TF-IDF - PCA - Scaler pipeline ")
    feature_pipeline = Pipeline(stages=[
        RegexTokenizer(inputCol="skills", outputCol="tokens",
                       pattern=r",\s*", toLowercase=True),
        HashingTF(inputCol="tokens", outputCol="raw_features", numFeatures=512),
        IDF(inputCol="raw_features", outputCol="tfidf_features", minDocFreq=2),
        PCA(inputCol="tfidf_features", outputCol="pca_features", k=50),
        StandardScaler(inputCol="pca_features", outputCol="scaled_features",
                       withStd=True, withMean=False),
    ])

    feature_model = feature_pipeline.fit(df)
    features_df   = feature_model.transform(df).cache()
    print("   Feature matrix cached on workers.")

    # 2. Tune K via silhouette (k = 4..12)
    print("  Tuning K (k=4..12) via silhouette score ")
    evaluator = ClusteringEvaluator(
        featuresCol="scaled_features", predictionCol="cluster_id",
        metricName="silhouette", distanceMeasure="squaredEuclidean"
    )

    best_k, best_score, best_model, best_preds = 8, -1.0, None, None
    k_rows = []

    for k in range(4, 13):
        model = KMeans(featuresCol="scaled_features", predictionCol="cluster_id",
                       k=k, seed=42, maxIter=30, tol=1e-4).fit(features_df)
        preds = model.transform(features_df)
        score = evaluator.evaluate(preds)
        k_rows.append((str(k), float(score)))
        print(f"   k={k:2d}  silhouette={score:.4f}")
        if score > best_score:
            best_score, best_k, best_model, best_preds = score, k, model, preds

    print(f"\n     Best k={best_k}  silhouette={best_score:.4f}")

    # Save k-selection table
    (spark.createDataFrame(k_rows, ["k", "silhouette"])
          .coalesce(1).write.mode("overwrite")
          .option("header", "true")
          .csv(f"{HDFS_RESULTS}/kmeans_k_selection"))

    # 3. Cluster assignments
    print("  Writing cluster assignments ")
    (best_preds
        .select("job_id", "sector", "city", "post_year",
                "salary_mid", "experience_level", "cluster_id")
        .coalesce(4).write.mode("overwrite")
        .option("header", "true")
        .csv(f"{HDFS_RESULTS}/cluster_assignments"))

    # 4. Cluster profiles - top skills + dominant sector
    print("  Computing cluster profiles ")
    win_rank = Window.partitionBy("cluster_id").orderBy(F.desc("skill_count"))

    top_skills = (best_preds
        .withColumn("skill", F.explode(F.split(F.lower(F.col("skills")), r",\s*")))
        .withColumn("skill", F.trim("skill"))
        .filter(F.col("skill") != "")
        .groupBy("cluster_id", "skill")
        .agg(F.count("*").alias("skill_count"))
        .withColumn("rank", F.rank().over(win_rank))
        .filter(F.col("rank") <= 10)
        .groupBy("cluster_id")
        .agg(F.concat_ws(" | ", F.collect_list("skill")).alias("top_skills")))

    dom_sector = (best_preds
        .groupBy("cluster_id", "sector")
        .agg(F.count("*").alias("cnt"))
        .withColumn("rank", F.rank().over(
            Window.partitionBy("cluster_id").orderBy(F.desc("cnt"))))
        .filter(F.col("rank") == 1)
        .select("cluster_id", F.col("sector").alias("dominant_sector")))

    cluster_stats = (best_preds
        .groupBy("cluster_id")
        .agg(
            F.count("*").alias("job_count"),
            F.round(F.avg("salary_mid"), 0).alias("avg_salary"),
            F.countDistinct("sector").alias("sector_count"),
            F.countDistinct("company").alias("company_count"),
        ))

    (cluster_stats
        .join(dom_sector, "cluster_id", "left")
        .join(top_skills, "cluster_id", "left")
        .orderBy("cluster_id")
        .coalesce(1).write.mode("overwrite")
        .option("header", "true")
        .csv(f"{HDFS_RESULTS}/cluster_profiles"))

    # 5. 2-D PCA scatter (20% sample for Colab)
    print("  Computing 2-D PCA scatter data ")

    pca2_model = PCA(inputCol="scaled_features", outputCol="pca2", k=2).fit(best_preds)
    pca2_df = pca2_model.transform(best_preds)

# FIX: convert sparse/dense vector safely using SQL expression
    pca2_df = pca2_df.withColumn(

        "pca_x",

        F.expr("pca2.values[0]")
    ).withColumn(
        "pca_y",
        F.expr("pca2.values[1]")

    )

    (pca2_df
     
    .select(
        "job_id", "sector", "cluster_id", "salary_mid",
        "experience_level", "pca_x", "pca_y"
    )
    
    .sample(fraction=0.20, seed=42)
    .coalesce(1)
    .write.mode("overwrite")
    .option("header", "true")
    .csv(f"{HDFS_RESULTS}/cluster_scatter_2d"))

    features_df.unpersist()
    print(f"\n  Stage 3 done  -  best k={best_k},  silhouette={best_score:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + ""*60)
    print("  LinkedIn Jobsb- Full Spark Pipeline")
    print(""*60)

    df_clean, df_skills = run_preprocessing()
    run_eda(df_clean, df_skills)
    run_clustering()

    print("\n" + ""*60)
    print("  PIPELINE COMPLETE")
    print(f"  Results -> {HDFS_RESULTS}/")
    print("  Next: bash 04_export_results.sh  ->  linkedin_results.zip")
    print(""*60)

    spark.stop()
