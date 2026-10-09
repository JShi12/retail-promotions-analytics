"""hh_demographic: 801 households. dunnhumby anonymised the columns; values are ordinal.

Ordinal integer versions are added for modelling. The raw labels are kept. No real-world meaning
(income, marital status, ...) is attached to classification_* because none is documented.
"""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from retail_promo import dq
from retail_promo.silver._typing import cast_columns, trim_strings

TABLE = "demographics"
KEY = ["household_key"]

KNOWN_VALUES: dict[str, list[str]] = {
    "classification_1": [f"Age Group{i}" for i in range(1, 7)],
    "classification_2": ["X", "Y", "Z"],
    "classification_3": [f"Level{i}" for i in range(1, 13)],
    "classification_4": ["1", "2", "3", "4", "5+"],
    "classification_5": [f"Group{i}" for i in range(1, 7)],
    "homeowner_desc": ["Homeowner", "Probable Owner", "Probable Renter", "Renter", "Unknown"],
    "kid_category_desc": ["1", "2", "3+", "None/Unknown"],
}

# Columns whose position in KNOWN_VALUES is a meaningful order (1-based).
ORDINAL = ["classification_1", "classification_3", "classification_4", "classification_5"]


def build(bronze: DataFrame, run_id: str) -> tuple[DataFrame, DataFrame]:
    df = dq.init_flags(bronze.drop("_source_file", "_ingested_at", "_run_id"))
    df = cast_columns(df, {"household_key": "int"})
    df = trim_strings(df, list(KNOWN_VALUES))

    unexpected = F.lit(False)
    for col, values in KNOWN_VALUES.items():
        unexpected = unexpected | ~F.coalesce(F.col(col).isin(*values), F.lit(False))
    df = dq.add_flag(df, unexpected, "unexpected_category")

    for col in ORDINAL:
        mapping = F.create_map(
            *[x for i, v in enumerate(KNOWN_VALUES[col], 1) for x in (F.lit(v), F.lit(i))]
        )
        df = df.withColumn(f"{col}_ord", mapping[F.col(col)])
    # Number of kids where stated; "None/Unknown" mixes none and unknown, so it stays null.
    df = df.withColumn(
        "kids_ord",
        F.create_map(F.lit("1"), F.lit(1), F.lit("2"), F.lit(2), F.lit("3+"), F.lit(3))[
            F.col("kid_category_desc")
        ],
    )

    return df, dq.flags_to_log(df, TABLE, KEY, run_id)
