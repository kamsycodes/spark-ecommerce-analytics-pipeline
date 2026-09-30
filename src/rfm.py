from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window


PROJECT_ROOT = Path(__file__).resolve().parent.parent

SILVER_PATH = PROJECT_ROOT / "data" / "silver" / "transactions"
RFM_PATH = PROJECT_ROOT / "data" / "gold" / "rfm_segments"


def create_spark_session():

    return (
        SparkSession.builder
        .master("local[*]")
        .appName("EcommerceRFM")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )


def build_rfm(spark):

    df = spark.read.parquet(str(SILVER_PATH))

    sales = df.filter(
        (F.col("is_valid_sale") == True)
        & F.col("customer_id").isNotNull()
    )

    # Use one day after the latest transaction as
    # the analysis/reference date.
    max_date = sales.agg(
        F.max("invoice_date")
    ).collect()[0][0]

    print(f"Latest transaction: {max_date}")

    reference_date = F.date_add(
        F.lit(max_date.date()),
        1
    )

    rfm = (
        sales
        .groupBy("customer_id")
        .agg(
            F.max("invoice_date").alias("last_purchase"),

            F.countDistinct(
                "invoice"
            ).alias("frequency"),

            F.round(
                F.sum("revenue"),
                2
            ).alias("monetary")
        )
        .withColumn(
            "recency",
            F.datediff(
                reference_date,
                F.to_date("last_purchase")
            )
        )
    )

    # ---------------------------------------------
    # RFM scoring using quartiles
    # ---------------------------------------------

    recency_window = Window.orderBy(
        F.desc("recency")
    )

    frequency_window = Window.orderBy(
        "frequency"
    )

    monetary_window = Window.orderBy(
        "monetary"
    )

    rfm = (
        rfm

        .withColumn(
            "r_score",
            F.ntile(4).over(recency_window)
        )

        .withColumn(
            "f_score",
            F.ntile(4).over(frequency_window)
        )

        .withColumn(
            "m_score",
            F.ntile(4).over(monetary_window)
        )
    )

    # Combined RFM score

    rfm = rfm.withColumn(
        "rfm_score",
        F.concat(
            F.col("r_score"),
            F.col("f_score"),
            F.col("m_score")
        )
    )

    # ---------------------------------------------
    # Customer segmentation
    # ---------------------------------------------

    rfm = rfm.withColumn(
        "customer_segment",

        F.when(
            (F.col("r_score") >= 3)
            & (F.col("f_score") >= 3)
            & (F.col("m_score") >= 3),
            "Champions"
        )

        .when(
            (F.col("r_score") >= 3)
            & (F.col("f_score") >= 3),
            "Loyal Customers"
        )

        .when(
            (F.col("r_score") >= 3)
            & (F.col("f_score") <= 2),
            "Potential Loyalists"
        )

        .when(
            (F.col("r_score") <= 2)
            & (F.col("f_score") >= 3),
            "At Risk"
        )

        .when(
            (F.col("r_score") <= 2)
            & (F.col("f_score") <= 2),
            "Hibernating"
        )

        .otherwise("Other")
    )

    return rfm


def main():

    spark = create_spark_session()

    try:

        print("Building RFM customer segmentation...")

        rfm = build_rfm(spark)

        print("\nCUSTOMER SEGMENTS")

        segment_summary = (
            rfm
            .groupBy("customer_segment")
            .agg(
                F.count("*").alias("customers"),
                F.round(
                    F.avg("monetary"),
                    2
                ).alias("avg_customer_value"),
                F.round(
                    F.avg("frequency"),
                    2
                ).alias("avg_orders"),
                F.round(
                    F.avg("recency"),
                    2
                ).alias("avg_recency_days")
            )
            .orderBy(
                F.desc("customers")
            )
        )

        segment_summary.show(
            truncate=False
        )

        print("\nTOP CHAMPION CUSTOMERS")

        (
            rfm
            .filter(
                F.col("customer_segment")
                == "Champions"
            )
            .orderBy(
                F.desc("monetary")
            )
            .show(
                10,
                truncate=False
            )
        )

        print("\nWriting RFM Gold table...")

        (
            rfm.write
            .mode("overwrite")
            .parquet(str(RFM_PATH))
        )

        print(
            f"RFM table written: {RFM_PATH}"
        )

    finally:
        spark.stop()


if __name__ == "__main__":
    main()