from pathlib import Path

import pandas as pd

PROJECT_ROOT =  Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "stock_metrics.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "stock_anomalies.csv"

def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["ticker", "date"])

    # Compare each return against the previous 30 trading sessions.
    grouped = df.groupby("ticker")["daily_return_pct"]

    df["rolling_mean"] = grouped.transform(
        lambda s: s.shift(1).rolling(30, min_periods=20).mean()
    )

    df["rolling_std"] = grouped.transform(
        lambda s: s.shift(1).rolling(30, min_periods=20).std()
    )

    # Avoid division by zero when recent returns have no variation.
    valid_std = df["rolling_std"].where(df["rolling_std"] > 0)

    df["return_zscore"] = (
        df["daily_return_pct"] - df["rolling_mean"]
    ) / valid_std

    # Flag returns more than 3 standard deviations from recent history.
    df["is_anomaly"] = df["return_zscore"].abs() > 3

    df["anomaly_type"] = "Normal"
    df.loc[df["return_zscore"] > 3, "anomaly_type"] = "Unusually large gain"
    df.loc[df["return_zscore"] < -3, "anomaly_type"] = "Unusually large loss"

    return df


def main():
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            "Metrics file not found. Run financial_metrics.py first."
        )

    df = pd.read_csv(INPUT_PATH, parse_dates=["date"])
    result = detect_anomalies(df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)

    anomalies = result[result["is_anomaly"] == True]

    print(f"Analyzed {len(result)} stock records.")
    print(f"Unusual movements detected: {len(anomalies)}")
    print(f"Results saved to: {OUTPUT_PATH}")

    if not anomalies.empty:
        print("\nDetected unusual movements:")
        print(
            anomalies[
                ["date", "ticker", "close", "daily_return_pct",
                 "return_zscore", "anomaly_type"]
            ].to_string(index=False)
        )
    else:
        print("No unusual movements detected using this threshold.")


if __name__ == "__main__":
    main()
