from pathlib import Path

import pandas as pd
from pyspark.sql import SparkSession


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
BRONZE_DIR = PROJECT_ROOT / "data" / "bronze"

EXCEL_FILE = RAW_DIR / "online_retail_II.xlsx"
CSV_FILE = RAW_DIR / "online_retail_II.csv"


def create_spark_session():
    """Create a local Spark session."""
    return (
        SparkSession.builder
        .master("local[*]")
        .appName("EcommerceAnalytics")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.host", "127.0.0.1")
        .getOrCreate()
    )


def convert_excel_to_csv():
    """Convert all Excel sheets into one CSV file."""

    if CSV_FILE.exists():
        print(f"CSV already exists: {CSV_FILE}")
        return

    print("Reading Excel workbook...")

    sheets = pd.read_excel(
        EXCEL_FILE,
        sheet_name=None,
        engine="openpyxl"
    )

    print(f"Found {len(sheets)} sheets.")

    frames = []

    for sheet_name, frame in sheets.items():
        print(f"Processing sheet: {sheet_name} ({len(frame):,} rows)")
        frame["source_sheet"] = sheet_name
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)

    print(f"Combined rows: {len(combined):,}")

    combined.to_csv(CSV_FILE, index=False)

    print(f"CSV created: {CSV_FILE}")


def ingest_bronze(spark):
    """Ingest raw transaction data into the Bronze layer."""

    print("\nLoading raw CSV with Spark...")

    df = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .csv(str(CSV_FILE))
    )

    row_count = df.count()

    print(f"Rows loaded by Spark: {row_count:,}")

    print("\nSchema:")
    df.printSchema()

    print("\nSample records:")
    df.show(10, truncate=False)

    print("\nWriting Bronze layer as Parquet...")

    (
        df.write
        .mode("overwrite")
        .parquet(str(BRONZE_DIR / "transactions"))
    )

    print(
        f"Bronze layer successfully written to: "
        f"{BRONZE_DIR / 'transactions'}"
    )

    return df


def main():
    convert_excel_to_csv()

    spark = create_spark_session()

    try:
        ingest_bronze(spark)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()