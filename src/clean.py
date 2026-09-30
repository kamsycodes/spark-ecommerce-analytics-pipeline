from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import LongType


PROJECT_ROOT = Path(__file__).resolve().parent.parent

BRONZE_PATH = PROJECT_ROOT / "data" / "bronze" / "transactions"
SILVER_PATH = PROJECT_ROOT / "data" / "silver" / "transactions"


def create_spark_session():
    """Create a local Spark session."""

    return (
        SparkSession.builder
        .master("local[*]")
        .appName("EcommerceSilverTransformation")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )


def transform_to_silver(spark):
    """Clean and transform Bronze transactions into Silver data."""

    print("Reading Bronze data...")

    df = spark.read.parquet(str(BRONZE_PATH))

    raw_count = df.count()

    print(f"Bronze rows: {raw_count:,}")

    # --------------------------------------------------
    # 1. Standardize column names
    # --------------------------------------------------

    df = (
        df
        .withColumnRenamed("Invoice", "invoice")
        .withColumnRenamed("StockCode", "stock_code")
        .withColumnRenamed("Description", "description")
        .withColumnRenamed("Quantity", "quantity")
        .withColumnRenamed("InvoiceDate", "invoice_date")
        .withColumnRenamed("Price", "unit_price")
        .withColumnRenamed("Customer ID", "customer_id")
        .withColumnRenamed("Country", "country")
    )

    # --------------------------------------------------
    # 2. Remove exact duplicate records
    # --------------------------------------------------

    df = df.dropDuplicates()

    deduplicated_count = df.count()

    print(
        f"Duplicate rows removed: "
        f"{raw_count - deduplicated_count:,}"
    )

    # --------------------------------------------------
    # 3. Standardize important fields
    # --------------------------------------------------

    df = (
        df
        .withColumn(
            "customer_id",
            F.col("customer_id").cast(LongType())
        )
        .withColumn(
            "description",
            F.trim(F.col("description"))
        )
        .withColumn(
            "country",
            F.trim(F.col("country"))
        )
    )

    # --------------------------------------------------
    # 4. Identify cancellations / returns
    # --------------------------------------------------

    df = df.withColumn(
        "is_cancellation",
        F.when(
            F.upper(F.col("invoice")).startswith("C"),
            True
        ).otherwise(False)
    )

    df = df.withColumn(
        "is_return",
        F.when(
            F.col("quantity") < 0,
            True
        ).otherwise(False)
    )

    # --------------------------------------------------
    # 5. Data quality flags
    # --------------------------------------------------

    df = df.withColumn(
        "has_customer_id",
        F.col("customer_id").isNotNull()
    )

    df = df.withColumn(
        "is_valid_sale",
        (
            (F.col("quantity") > 0)
            & (F.col("unit_price") > 0)
            & (~F.col("is_cancellation"))
        )
    )

    # --------------------------------------------------
    # 6. Revenue
    # --------------------------------------------------

    df = df.withColumn(
        "revenue",
        F.round(
            F.col("quantity") * F.col("unit_price"),
            2
        )
    )

    # --------------------------------------------------
    # 7. Time features
    # --------------------------------------------------

    df = (
        df
        .withColumn("year", F.year("invoice_date"))
        .withColumn("month", F.month("invoice_date"))
        .withColumn("day", F.dayofmonth("invoice_date"))
        .withColumn("day_of_week", F.dayofweek("invoice_date"))
        .withColumn("hour", F.hour("invoice_date"))
        .withColumn(
            "year_month",
            F.date_format("invoice_date", "yyyy-MM")
        )
    )

    # --------------------------------------------------
    # 8. Data quality summary
    # --------------------------------------------------

    print("\nData quality summary:")

    df.select(
        F.count("*").alias("silver_rows"),

        F.sum(
            F.col("customer_id").isNull().cast("int")
        ).alias("missing_customer_ids"),

        F.sum(
            F.col("is_cancellation").cast("int")
        ).alias("cancellations"),

        F.sum(
            F.col("is_return").cast("int")
        ).alias("returns"),

        F.sum(
            F.col("is_valid_sale").cast("int")
        ).alias("valid_sales")

    ).show()

    # --------------------------------------------------
    # 9. Display transformed records
    # --------------------------------------------------

    print("\nSilver schema:")
    df.printSchema()

    print("\nSample transformed records:")

    df.select(
        "invoice",
        "stock_code",
        "quantity",
        "unit_price",
        "customer_id",
        "country",
        "revenue",
        "is_cancellation",
        "is_return",
        "is_valid_sale",
        "year_month"
    ).show(10, truncate=False)

    # --------------------------------------------------
    # 10. Write Silver layer
    # --------------------------------------------------

    print("\nWriting Silver layer...")

    (
        df.write
        .mode("overwrite")
        .partitionBy("year")
        .parquet(str(SILVER_PATH))
    )

    print(
        f"Silver layer successfully written to: "
        f"{SILVER_PATH}"
    )

    return df


def main():

    spark = create_spark_session()

    try:
        transform_to_silver(spark)

    finally:
        spark.stop()


if __name__ == "__main__":
    main()