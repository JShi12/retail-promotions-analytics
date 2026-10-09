"""Table storage. The pipeline only talks to a TableIO, so the same code runs on Databricks
(Delta tables in Unity Catalog) and locally (Parquet files) for tests and dry runs."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from pyspark.sql import DataFrame, SparkSession


class TableIO(Protocol):
    def read(self, layer: str, name: str) -> DataFrame: ...

    def write(self, df: DataFrame, layer: str, name: str, mode: str = "overwrite") -> None: ...

    def raw_path(self, file_stem: str) -> str: ...


class UnityCatalogIO:
    """Delta tables named <catalog>.<layer>.<name>; raw CSVs in a Unity Catalog volume."""

    def __init__(self, spark: SparkSession, catalog: str, landing_dir: str):
        self.spark = spark
        self.catalog = catalog
        self.landing_dir = landing_dir.rstrip("/")

    def fqn(self, layer: str, name: str) -> str:
        return f"{self.catalog}.{layer}.{name}"

    def read(self, layer: str, name: str) -> DataFrame:
        return self.spark.table(self.fqn(layer, name))

    def write(self, df: DataFrame, layer: str, name: str, mode: str = "overwrite") -> None:
        writer = df.write.format("delta").mode(mode)
        if mode == "overwrite":
            writer = writer.option("overwriteSchema", "true")
        else:
            writer = writer.option("mergeSchema", "true")
        writer.saveAsTable(self.fqn(layer, name))

    def raw_path(self, file_stem: str) -> str:
        return f"{self.landing_dir}/{file_stem}.csv"


class LocalParquetIO:
    """Parquet under <root>/<layer>/<name>; raw CSVs read from raw_dir."""

    def __init__(self, spark: SparkSession, root: str | Path, raw_dir: str | Path):
        self.spark = spark
        self.root = Path(root)
        self.raw_dir = Path(raw_dir)

    def _path(self, layer: str, name: str) -> str:
        return str(self.root / layer / name)

    def read(self, layer: str, name: str) -> DataFrame:
        return self.spark.read.parquet(self._path(layer, name))

    def write(self, df: DataFrame, layer: str, name: str, mode: str = "overwrite") -> None:
        df.write.mode(mode).parquet(self._path(layer, name))

    def raw_path(self, file_stem: str) -> str:
        return str(self.raw_dir / f"{file_stem}.csv")
