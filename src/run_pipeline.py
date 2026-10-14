
from src.ingestion.stock_ingestion import main as ingest_stock_data
from src.analysis.validate_data import validate_stock_data
from src.ingestion.load_to_duckdb import load_stock_data
from src.analysis.financial_metrics import calculate_metrics
from src.analysis.anomaly_detection import detect_anomalies

from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

METRICS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "stock_metrics.csv"
)

ANOMALIES_PATH = (
    PROJECT_ROOT / "data" / "processed" / "stock_anomalies.csv"
)


def main():
    print("\n" + "=" * 55)
    print("AI-POWERED FINANCIAL INTELLIGENCE PIPELINE")
    print("=" * 55)

    # STEP 1: Download and save stock data
    print("\nSTEP 1: Downloading stock data...")
    ingest_stock_data()

    # STEP 2: Validate the raw data
    print("\nSTEP 2: Validating data quality...")
    validate_stock_data()

    # STEP 3: Load data into DuckDB
    print("\nSTEP 3: Loading data into DuckDB...")
    load_stock_data()

    # STEP 4: Calculate financial metrics
    print("\nSTEP 4: Calculating financial metrics...")
    calculate_metrics()

    if not METRICS_PATH.exists():
        raise FileNotFoundError(
            f"Metrics file was not created: {METRICS_PATH}"
        )

    # STEP 5: Detect unusual price movements
    print("\nSTEP 5: Detecting unusual price movements...")

    metrics_df = pd.read_csv(
        METRICS_PATH,
        parse_dates=["date"]
    )

    anomalies_df = detect_anomalies(metrics_df)

    ANOMALIES_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    anomalies_df.to_csv(
        ANOMALIES_PATH,
        index=False
    )

    anomaly_count = int(anomalies_df["is_anomaly"].sum())

    print(f"Records analyzed: {len(anomalies_df)}")
    print(f"Unusual movements detected: {anomaly_count}")
    print(f"Results saved to: {ANOMALIES_PATH}")

    if anomaly_count > 0:
        print("\nDetected unusual movements:")

        print(
            anomalies_df.loc[
                anomalies_df["is_anomaly"],
                [
                    "date",
                    "ticker",
                    "close",
                    "daily_return_pct",
                    "return_zscore",
                    "anomaly_type",
                ],
            ].to_string(index=False)
        )
    else:
        print("No unusual movements detected.")

    print("\n" + "=" * 55)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 55)


if __name__ == "__main__":
    main()
