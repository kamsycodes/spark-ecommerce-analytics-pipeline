from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SILVER_PATH = PROJECT_ROOT / "data" / "silver" / "transactions"
GOLD_PATH = PROJECT_ROOT / "data" / "gold"


def create_spark_session():
    """Create local Spark session."""

    return (
        SparkSession.builder
        .master("local[*]")
        .appName("EcommerceGoldAnalytics")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )


def build_customer_metrics(sales):
    """Create customer-level analytics."""

    customers = (
        sales
        .filter(F.col("customer_id").isNotNull())
        .groupBy("customer_id")
        .agg(
            F.countDistinct("invoice").alias("total_orders"),
            F.sum("quantity").alias("items_purchased"),
            F.round(F.sum("revenue"), 2).alias("total_revenue"),
            F.round(F.avg("revenue"), 2).alias("avg_line_value"),
            F.min("invoice_date").alias("first_purchase"),
            F.max("invoice_date").alias("last_purchase"),
            F.countDistinct("stock_code").alias("unique_products")
        )
    )

    return customers


def build_product_metrics(sales):
    """Create product-level analytics."""

    products = (
        sales
        .groupBy("stock_code", "description")
        .agg(
            F.sum("quantity").alias("units_sold"),
            F.countDistinct("invoice").alias("order_count"),
            F.round(F.sum("revenue"), 2).alias("total_revenue"),
            F.round(F.avg("unit_price"), 2).alias("avg_unit_price")
        )
    )

    # Rank products by revenue.
    ranking_window = Window.orderBy(F.desc("total_revenue"))

    products = products.withColumn(
        "revenue_rank",
        F.dense_rank().over(ranking_window)
    )

    return products


def build_country_metrics(sales):
    """Create country-level business metrics."""

    countries = (
        sales
        .groupBy("country")
        .agg(
            F.countDistinct("invoice").alias("total_orders"),
            F.countDistinct("customer_id").alias("unique_customers"),
            F.sum("quantity").alias("units_sold"),
            F.round(F.sum("revenue"), 2).alias("total_revenue")
        )
    )

    return countries


def build_monthly_metrics(sales):
    """Create monthly KPI table."""

    monthly = (
        sales
        .groupBy("year_month")
        .agg(
            F.countDistinct("invoice").alias("total_orders"),
            F.countDistinct("customer_id").alias("unique_customers"),
            F.sum("quantity").alias("units_sold"),
            F.round(F.sum("revenue"), 2).alias("total_revenue")
        )
        .withColumn(
            "avg_order_value",
            F.round(
                F.col("total_revenue") / F.col("total_orders"),
                2
            )
        )
        .orderBy("year_month")
    )

    return monthly


def run_spark_sql(sales):
    """Demonstrate Spark SQL analytics."""

    sales.createOrReplaceTempView("sales")

    print("\nTOP 10 COUNTRIES BY REVENUE — SPARK SQL")

    result = sales.sparkSession.sql(
        """
        SELECT
            country,
            COUNT(DISTINCT invoice) AS orders,
            ROUND(SUM(revenue), 2) AS revenue
        FROM sales
        GROUP BY country
        ORDER BY revenue DESC
        LIMIT 10
        """
    )

    result.show(truncate=False)


def show_business_kpis(sales):
    """Calculate overall business KPIs."""

    print("\nOVERALL BUSINESS KPIs")

    kpis = sales.agg(
        F.count("*").alias("transaction_lines"),
        F.countDistinct("invoice").alias("total_orders"),
        F.countDistinct("customer_id").alias("unique_customers"),
        F.sum("quantity").alias("units_sold"),
        F.round(F.sum("revenue"), 2).alias("total_revenue")
    )

    kpis.show(truncate=False)


def write_gold_table(df, name):
    """Write analytics table to Gold layer."""

    output_path = GOLD_PATH / name

    (
        df.write
        .mode("overwrite")
        .parquet(str(output_path))
    )

    print(f"Gold table written: {output_path}")


def main():

    spark = create_spark_session()

    try:
        print("Reading Silver transactions...")

        silver = spark.read.parquet(str(SILVER_PATH))

        # Gold sales analytics should contain actual positive sales.
        sales = silver.filter(F.col("is_valid_sale") == True)

        print(f"Valid sales records: {sales.count():,}")

        show_business_kpis(sales)

        # ------------------------------------------------
        # Customer analytics
        # ------------------------------------------------

        print("\nBuilding customer metrics...")

        customers = build_customer_metrics(sales)

        print("\nTOP 10 CUSTOMERS BY REVENUE")

        customers.orderBy(
            F.desc("total_revenue")
        ).show(10, truncate=False)

        # ------------------------------------------------
        # Product analytics
        # ------------------------------------------------

        print("\nBuilding product metrics...")

        products = build_product_metrics(sales)

        print("\nTOP 10 PRODUCTS BY REVENUE")

        products.filter(
            F.col("revenue_rank") <= 10
        ).orderBy(
            "revenue_rank"
        ).show(10, truncate=False)

        # ------------------------------------------------
        # Country analytics
        # ------------------------------------------------

        print("\nBuilding country metrics...")

        countries = build_country_metrics(sales)

        countries.orderBy(
            F.desc("total_revenue")
        ).show(10, truncate=False)

        # ------------------------------------------------
        # Monthly analytics
        # ------------------------------------------------

        print("\nBuilding monthly metrics...")

        monthly = build_monthly_metrics(sales)

        monthly.show(30, truncate=False)

        # ------------------------------------------------
        # Spark SQL
        # ------------------------------------------------

        run_spark_sql(sales)

        # ------------------------------------------------
        # Write Gold datasets
        # ------------------------------------------------

        print("\nWriting Gold analytics tables...")

        write_gold_table(customers, "customer_metrics")
        write_gold_table(products, "product_metrics")
        write_gold_table(countries, "country_metrics")
        write_gold_table(monthly, "monthly_metrics")

        print("\nGold analytics layer complete.")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()