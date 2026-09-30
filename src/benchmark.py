from pathlib import Path
from time import perf_counter

from pyspark.sql import SparkSession
from pyspark.sql import functions as F


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SILVER_PATH = PROJECT_ROOT / "data" / "silver" / "transactions"


# --------------------------------------------------
# Spark session
# --------------------------------------------------

def create_spark_session():
    """Create a local Spark session for benchmarking."""

    return (
        SparkSession.builder
        .master("local[*]")
        .appName("EcommerceSparkBenchmark")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )


# --------------------------------------------------
# Timing utility
# --------------------------------------------------

def timed_operation(name, function):
    """Measure execution time of a Spark operation."""

    start = perf_counter()

    result = function()

    elapsed = perf_counter() - start

    print(
        f"{name}: {elapsed:.2f} seconds"
    )

    return result, elapsed


# --------------------------------------------------
# Create scale-test dataset
# --------------------------------------------------

def create_scaled_dataset(df):
    """
    Create a controlled scale-test dataset by
    replicating each Silver record 10 times.

    The original dataset contains ~1.05M Silver rows,
    producing ~10.55M rows for the benchmark.
    """

    print("\nCreating approximately 10M rows...")

    multiplier = (
        df.sparkSession
        .range(10)
        .withColumnRenamed(
            "id",
            "replica_id"
        )
    )

    scaled = df.crossJoin(
        F.broadcast(multiplier)
    )

    return scaled


# --------------------------------------------------
# Revenue aggregation workload
# --------------------------------------------------

def revenue_query(df):
    """
    Aggregate valid sales revenue by country.
    """

    return (
        df
        .filter(
            F.col("is_valid_sale") == True
        )
        .groupBy(
            "country"
        )
        .agg(
            F.round(
                F.sum("revenue"),
                2
            ).alias("revenue")
        )
        .orderBy(
            F.desc("revenue")
        )
    )


# --------------------------------------------------
# Monthly aggregation workload
# --------------------------------------------------

def monthly_query(df):
    """
    Aggregate transaction volume and revenue
    by year and month.
    """

    return (
        df
        .filter(
            F.col("is_valid_sale") == True
        )
        .groupBy(
            "year_month"
        )
        .agg(
            F.count("*").alias(
                "transaction_lines"
            ),

            F.round(
                F.sum("revenue"),
                2
            ).alias(
                "revenue"
            )
        )
    )


# --------------------------------------------------
# Main benchmark
# --------------------------------------------------

def main():

    spark = create_spark_session()

    try:

        # ==========================================
        # 1. Load Silver dataset
        # ==========================================

        print("Reading Silver dataset...")

        silver = spark.read.parquet(
            str(SILVER_PATH)
        )

        original_count = silver.count()

        print(
            f"Original rows: "
            f"{original_count:,}"
        )


        # ==========================================
        # 2. Build ~10M-row scale-test dataset
        # ==========================================

        scaled = create_scaled_dataset(
            silver
        )

        scaled_count, generation_time = (
            timed_operation(
                "Materialize scaled dataset",
                scaled.count
            )
        )

        print(
            f"Scaled rows: "
            f"{scaled_count:,}"
        )


        # ==========================================
        # 3. Single uncached query
        # ==========================================

        print(
            "\n--- UNCACHED BENCHMARK ---"
        )

        def run_uncached():

            result = revenue_query(
                scaled
            )

            # collect() forces Spark to actually
            # execute the transformations.
            result.collect()

            return result

        _, uncached_time = timed_operation(
            "Revenue aggregation (uncached)",
            run_uncached
        )


        # ==========================================
        # 4. Cache dataset
        # ==========================================

        print(
            "\nCaching scaled dataset..."
        )

        scaled.cache()

        # count() forces Spark to populate the cache.
        _, cache_time = timed_operation(
            "Cache materialization",
            scaled.count
        )


        # ==========================================
        # 5. Single cached query
        # ==========================================

        print(
            "\n--- CACHED BENCHMARK ---"
        )

        def run_cached():

            result = revenue_query(
                scaled
            )

            result.collect()

            return result

        _, cached_time = timed_operation(
            "Revenue aggregation (cached)",
            run_cached
        )


        # ==========================================
        # 6. Second cached analytical workload
        # ==========================================

        def run_monthly():

            result = monthly_query(
                scaled
            )

            result.collect()

            return result

        _, monthly_time = timed_operation(
            "Monthly aggregation (cached)",
            run_monthly
        )


        # ==========================================
        # 7. Initial benchmark results
        # ==========================================

        print(
            "\n========== BENCHMARK RESULTS =========="
        )

        print(
            f"Original rows: "
            f"{original_count:,}"
        )

        print(
            f"Scaled rows: "
            f"{scaled_count:,}"
        )

        print(
            f"Dataset generation: "
            f"{generation_time:.2f}s"
        )

        print(
            f"Uncached revenue query: "
            f"{uncached_time:.2f}s"
        )

        print(
            f"Cache materialization: "
            f"{cache_time:.2f}s"
        )

        print(
            f"Cached revenue query: "
            f"{cached_time:.2f}s"
        )

        print(
            f"Cached monthly query: "
            f"{monthly_time:.2f}s"
        )

        if cached_time > 0:

            speedup = (
                uncached_time
                / cached_time
            )

            print(
                f"Observed cached-query speedup: "
                f"{speedup:.2f}x"
            )

        print(
            "======================================="
        )


        # ==========================================
        # 8. Repeated workload benchmark
        # ==========================================

        print(
            "\n--- REPEATED WORKLOAD BENCHMARK ---"
        )

        # Remove the existing cache so the uncached
        # test starts from an uncached DataFrame.
        scaled.unpersist()

        print(
            "\nRunning 5 uncached revenue queries..."
        )

        def repeated_uncached():

            for _ in range(5):

                result = revenue_query(
                    scaled
                )

                result.collect()

        _, repeated_uncached_time = (
            timed_operation(
                "5 revenue queries (uncached)",
                repeated_uncached
            )
        )


        # ==========================================
        # 9. Cache dataset again
        # ==========================================

        print(
            "\nCaching dataset for repeated workload..."
        )

        scaled.cache()

        # Force Spark to populate the cache before
        # measuring repeated query execution.
        _, repeated_cache_time = timed_operation(
            "Repeated-test cache materialization",
            scaled.count
        )


        # ==========================================
        # 10. Five cached queries
        # ==========================================

        print(
            "\nRunning 5 cached revenue queries..."
        )

        def repeated_cached():

            for _ in range(5):

                result = revenue_query(
                    scaled
                )

                result.collect()

        _, repeated_cached_time = (
            timed_operation(
                "5 revenue queries (cached)",
                repeated_cached
            )
        )


        # ==========================================
        # 11. Repeated workload results
        # ==========================================

        print(
            "\n====== REPEATED WORKLOAD RESULTS ======"
        )

        print(
            f"5 uncached queries: "
            f"{repeated_uncached_time:.2f}s"
        )

        print(
            f"Repeated cache materialization: "
            f"{repeated_cache_time:.2f}s"
        )

        print(
            f"5 cached queries: "
            f"{repeated_cached_time:.2f}s"
        )

        if repeated_cached_time > 0:

            repeated_speedup = (
                repeated_uncached_time
                / repeated_cached_time
            )

            print(
                f"Repeated-query speedup: "
                f"{repeated_speedup:.2f}x"
            )

        # Include cache creation cost as well.
        cached_total_with_materialization = (
            repeated_cache_time
            + repeated_cached_time
        )

        print(
            f"Cached total including cache creation: "
            f"{cached_total_with_materialization:.2f}s"
        )

        if cached_total_with_materialization > 0:

            end_to_end_speedup = (
                repeated_uncached_time
                / cached_total_with_materialization
            )

            print(
                f"End-to-end speedup including cache cost: "
                f"{end_to_end_speedup:.2f}x"
            )

        print(
            "======================================="
        )


        # ==========================================
        # 12. Clean up cache
        # ==========================================

        scaled.unpersist()


    finally:

        spark.stop()


if __name__ == "__main__":
    main()