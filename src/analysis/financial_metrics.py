
from pathlib import Path

import duckdb
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "data" / "financial.duckdb"


def calculate_metrics() -> pd.DataFrame:
    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        df = con.execute("""
            SELECT date, ticker, close, volume
            FROM stock_prices
            ORDER BY ticker, date
        """).df()

    if df.empty:
        raise ValueError("No stock data found in DuckDB.")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).copy()

    grouped = df.groupby("ticker", group_keys=False)

    # Daily percentage change in closing price.
    df["daily_return_pct"] = (
        grouped["close"].pct_change() * 100
    )

    # Moving averages over 7 and 30 trading sessions.
    df["ma_7"] = grouped["close"].transform(
        lambda s: s.rolling(7, min_periods=7).mean()
    )

    df["ma_30"] = grouped["close"].transform(
        lambda s: s.rolling(30, min_periods=30).mean()
    )

    # Daily price movement compared with the previous session.
    df["price_change"] = grouped["close"].diff()

    output_path = PROJECT_ROOT / "data" / "processed" / "stock_metrics.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Saved metrics to: {output_path}")
    print("\nLatest calculated metrics:")
    print(df.tail(5).to_string(index=False))

    return df


if __name__ == "__main__":
    calculate_metrics()
