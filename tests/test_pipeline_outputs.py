
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "aapl_prices.csv"
METRICS_PATH = PROJECT_ROOT / "data" / "processed" / "stock_metrics.csv"
ANOMALIES_PATH = PROJECT_ROOT / "data" / "processed" / "stock_anomalies.csv"


def test_raw_stock_data_exists():
    assert RAW_PATH.exists()


def test_metrics_file_has_required_columns():
    assert METRICS_PATH.exists()

    df = pd.read_csv(METRICS_PATH)

    required = {
        "date",
        "ticker",
        "close",
        "daily_return_pct",
        "ma_7",
        "ma_30",
    }

    assert required.issubset(df.columns)
    assert not df.empty


def test_anomaly_file_has_required_columns():
    assert ANOMALIES_PATH.exists()

    df = pd.read_csv(ANOMALIES_PATH)

    required = {
        "date",
        "ticker",
        "daily_return_pct",
        "return_zscore",
        "is_anomaly",
        "anomaly_type",
    }

    assert required.issubset(df.columns)
    assert not df.empty
