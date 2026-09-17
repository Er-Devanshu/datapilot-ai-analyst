from __future__ import annotations

from pathlib import Path

from datapilot.data.duckdb import DuckDBDataSource


class DuckDBDatasetLoader:
    """Load DataPilot CSV tables into DuckDB."""

    def __init__(
        self,
        data_source: DuckDBDataSource,
        csv_directory: str | Path = "data/sample",
    ) -> None:
        self.data_source = data_source
        self.csv_directory = Path(csv_directory)

    def load(self) -> None:
        """Load all CSV tables into DuckDB."""

        if not self.csv_directory.exists():
            raise FileNotFoundError(
                f"CSV directory does not exist: "
                f"{self.csv_directory}"
            )

        csv_files = sorted(
            self.csv_directory.glob("*.csv")
        )

        if not csv_files:
            raise FileNotFoundError(
                f"No CSV files found in "
                f"{self.csv_directory}"
            )

        for csv_file in csv_files:
            table_name = csv_file.stem

            escaped_path = (
                str(csv_file.resolve())
                .replace("'", "''")
            )

            self.data_source.execute(
                f"""
                CREATE OR REPLACE TABLE "{table_name}" AS
                SELECT *
                FROM read_csv_auto(
                    '{escaped_path}',
                    HEADER = TRUE
                )
                """
            )

    def list_loaded_tables(self) -> list[str]:
        """Return tables currently loaded into DuckDB."""

        result = self.data_source.fetch_dataframe(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'main'
              AND table_type = 'BASE TABLE'
            ORDER BY table_name
            """
        )

        return result["table_name"].tolist()