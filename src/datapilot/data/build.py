from __future__ import annotations

from datapilot.data.generator import DatasetConfig, SyntheticDataGenerator
from datapilot.data.quality import DatasetQualityValidator


def build_dataset() -> None:
    """Generate, validate, and save the complete DataPilot dataset."""

    config = DatasetConfig()

    print("Starting DataPilot dataset build...")
    print()
    print(f"Date range : {config.start_date} → {config.end_date}")
    print(f"Customers  : {config.customers:,}")
    print(f"Products   : {config.products:,}")
    print(f"Employees  : {config.employees:,}")
    print(f"Locations  : {config.locations:,}")
    print(f"Orders     : {config.orders:,}")
    print()

    generator = SyntheticDataGenerator(config)

    print("Generating synthetic dataset...")
    tables = generator.generate()

    print("Generation complete.")
    print()

    print("Running data-quality validation...")

    validator = DatasetQualityValidator()
    report = validator.validate(tables)

    failed_checks = report.failed_checks

    print(f"Quality checks : {len(report.checks)}")
    print(f"Failed checks  : {len(failed_checks)}")
    print()

    if not report.passed:
        print("DATASET BUILD FAILED")
        print()

        for check in failed_checks:
            print(
                f"FAIL | {check.name} | {check.details}"
            )

        raise RuntimeError(
            "Dataset failed data-quality validation."
        )

    print("Data-quality validation passed.")
    print()

    print("Saving dataset...")

    output_dir = generator.save_csv(tables)

    print(f"Dataset saved to: {output_dir}")
    print()

    print("Dataset summary:")

    for table_name, dataframe in tables.items():
        print(
            f"  {table_name:<18} "
            f"{len(dataframe):>10,} rows"
        )

    print()
    print("DataPilot dataset build: OK")


if __name__ == "__main__":
    build_dataset()