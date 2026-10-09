"""Typing raw strings without crashing (Spark 4 ANSI mode) and without silent coercion."""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from retail_promo import dq

MONEY = "decimal(12,2)"


def _blank(col: str) -> F.Column:
    return F.col(col).isNull() | (F.trim(F.col(col)) == "")


def cast_columns(df: DataFrame, types: dict[str, str]) -> DataFrame:
    """try_cast each column; flag `cast_failed:<col>` where a non-blank value becomes null.

    Money columns (MONEY) are also flagged `money_rounded` when the raw value carries more than
    2 decimals of precision (the data has float noise such as 5.551115E-17).
    """
    for col, sql_type in types.items():
        typed = F.expr(f"try_cast(`{col}` as {sql_type})")
        df = df.withColumn(f"_typed_{col}", typed)
        df = dq.add_flag(df, ~_blank(col) & F.col(f"_typed_{col}").isNull(), f"cast_failed:{col}")
        if sql_type == MONEY:
            exact = F.expr(f"try_cast(`{col}` as decimal(38,18))")
            df = dq.add_flag(df, exact != F.col(f"_typed_{col}"), "money_rounded")
        df = df.withColumn(col, F.col(f"_typed_{col}")).drop(f"_typed_{col}")
    return df


def trim_strings(df: DataFrame, cols: list[str]) -> DataFrame:
    for col in cols:
        df = df.withColumn(col, F.trim(F.col(col)))
    return df
