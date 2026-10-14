
from pathlib import Path

import duckdb
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "data" / "financial.duckdb"
METRICS_PATH = PROJECT_ROOT / "data" / "processed" / "stock_metrics.csv"
ANOMALIES_PATH = PROJECT_ROOT / "data" / "processed" / "stock_anomalies.csv"


def get_stock_summary(ticker: str = "AAPL") -> str:
    """Return historical price statistics for a stock."""

    if not DB_PATH.exists():
        return "Financial database not found. Load stock data first."

    ticker = ticker.strip().upper()

    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        result = con.execute(
            """
            SELECT
                COUNT(*) AS trading_sessions,
                MIN(date) AS first_date,
                MAX(date) AS latest_date,
                ROUND(MIN(close), 2) AS minimum_close,
                ROUND(MAX(close), 2) AS maximum_close,
                ROUND(AVG(close), 2) AS average_close
            FROM stock_prices
            WHERE ticker = ?
            """,
            [ticker],
        ).fetchone()

    if not result or result[0] == 0:
        return f"No stored price data found for {ticker}."

    sessions, first_date, latest_date, minimum, maximum, average = result

    return (
        f"Stock: {ticker}\n"
        f"Trading sessions: {sessions}\n"
        f"Data range: {first_date} to {latest_date}\n"
        f"Minimum closing price: {minimum}\n"
        f"Maximum closing price: {maximum}\n"
        f"Average closing price: {average}"
    )


def get_recent_anomalies(ticker: str = "AAPL") -> str:
    """Return detected unusual price movements for a stock."""

    if not ANOMALIES_PATH.exists():
        return "Anomaly results not found. Run anomaly_detection.py first."

    df = pd.read_csv(ANOMALIES_PATH)
    ticker = ticker.strip().upper()

    df = df[
        (df["ticker"].str.upper() == ticker)
        & (df["is_anomaly"].astype(str).str.lower() == "true")
    ]

    if df.empty:
        return f"No detected anomalies found for {ticker}."

    columns = [
        "date",
        "daily_return_pct",
        "return_zscore",
        "anomaly_type",
    ]

    return df[columns].tail(10).to_string(index=False)


if __name__ == "__main__":
    print(get_stock_summary("AAPL"))
    print("\nRecent anomalies:\n")
    print(get_recent_anomalies("AAPL"))
