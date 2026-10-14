
import pandas as pd

from src.analysis.anomaly_detection import detect_anomalies


def create_test_data():
    # Use small but varying returns so the standard deviation is nonzero.
    baseline_returns = [0.1, 0.2, 0.0, 0.15, 0.05] * 6 + [
        0.1, 0.2, 0.0, 0.15
    ]

    returns = baseline_returns + [10.0]

    return pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=35),
        "ticker": ["TEST"] * 35,
        "daily_return_pct": returns,
        "close": [100.0] * 35,
        "volume": [1000] * 35,
    })


def test_anomaly_detection_preserves_rows():
    df = create_test_data()

    result = detect_anomalies(df)

    assert len(result) == len(df)
    assert "is_anomaly" in result.columns
    assert "return_zscore" in result.columns


def test_anomaly_detection_adds_labels():
    df = create_test_data()

    result = detect_anomalies(df)

    assert bool(result.iloc[-1]["is_anomaly"]) is True
    assert (
        result.iloc[-1]["anomaly_type"]
        == "Unusually large gain"
    )
