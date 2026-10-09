"""causal_data: one row per promoted product/store/week.

The raw file lists only promoted product-store-weeks (every row has display or mailer != '0').
15,245 keys appear twice with conflicting codes; they are collapsed by an OR rule: a key is on
display if any of its rows names a display location, likewise for mailer. When two different
non-zero codes conflict, the alphabetically first is kept. Raw codes are kept in arrays and
every source row of a collapsed key is logged.
"""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from retail_promo import dq
from retail_promo.config import DISPLAY_CODES, MAILER_CODES
from retail_promo.silver._typing import cast_columns, trim_strings

TABLE = "causal"
KEY = ["product_id", "store_id", "week_no"]


def build(bronze: DataFrame, run_id: str) -> tuple[DataFrame, DataFrame]:
    df = dq.init_flags(bronze.drop("_source_file", "_ingested_at", "_run_id"))
    df = cast_columns(df, {"product_id": "int", "store_id": "int", "week_no": "int"})
    df = trim_strings(df, ["display", "mailer"])
    df = dq.add_flag(
        df,
        ~F.coalesce(F.col("display").isin(*sorted(DISPLAY_CODES)), F.lit(False)),
        "invalid_display_code",
    )
    df = dq.add_flag(
        df,
        ~F.coalesce(F.col("mailer").isin(*sorted(MAILER_CODES)), F.lit(False)),
        "invalid_mailer_code",
    )
    row_log = dq.flags_to_log(df, TABLE, [*KEY, "display", "mailer"], run_id)

    def first_nonzero(col: str) -> F.Column:
        return F.coalesce(F.min(F.when(F.col(col) != "0", F.col(col))), F.lit("0"))

    collapsed = df.groupBy(*KEY).agg(
        F.count(F.lit(1)).alias("_n"),
        first_nonzero("display").alias("display_code"),
        first_nonzero("mailer").alias("mailer_code"),
        F.sort_array(F.collect_list("display")).alias("display_codes_raw"),
        F.sort_array(F.collect_list("mailer")).alias("mailer_codes_raw"),
        F.array_distinct(F.flatten(F.collect_list("dq_flags"))).alias("dq_flags"),
    )
    is_dup = F.col("_n") > 1
    silver = (
        collapsed.withColumn("on_display", F.col("display_code") != "0")
        .withColumn("on_mailer", F.col("mailer_code") != "0")
        .withColumn("display_codes_raw", F.when(is_dup, F.col("display_codes_raw")))
        .withColumn("mailer_codes_raw", F.when(is_dup, F.col("mailer_codes_raw")))
        .withColumn(
            "dq_flags",
            F.when(
                is_dup, F.array_union("dq_flags", F.array(F.lit("causal_conflicting_duplicate")))
            ).otherwise(F.col("dq_flags")),
        )
        .drop("_n")
    )

    dup_keys = collapsed.filter(is_dup).select(*KEY, "display_code", "mailer_code")
    collapse_log = dq.log_rows(
        df.join(dup_keys, KEY),
        TABLE,
        "causal_conflicting_duplicate",
        [*KEY, "display", "mailer"],
        run_id,
        detail=F.concat(
            F.lit("Collapsed to display="),
            F.col("display_code"),
            F.lit(", mailer="),
            F.col("mailer_code"),
        ),
    )
    return silver, row_log.unionByName(collapse_log)
