
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = PROJECT_ROOT / "data" / "raw" / "aapl_prices.csv"

REQUIRED_COLUMNS = [
    "date",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "volume",
]


def validate_stock_data():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Stock CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    df.columns = df.columns.str.strip().str.lower()

    # Handle a possible alternative date-column name.
    if "data" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"data": "date"})

    missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    if df.empty:
        raise ValueError("The stock dataset is empty.")

    df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce",
    utc=True
)
    if df["date"].isna().any():
        raise ValueError("Invalid or missing dates detected.")

    numeric_columns = ["open", "high", "low", "close", "volume"]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if df[numeric_columns].isna().any().any():
        raise ValueError("Missing or non-numeric financial values detected.")

    if not df["ticker"].notna().all():
        raise ValueError("Missing ticker symbols detected.")

    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("Stock prices must be greater than zero.")

    if (df["volume"] < 0).any():
        raise ValueError("Trading volume cannot be negative.")

    if (df["high"] < df[["open", "low", "close"]].max(axis=1)).any():
        raise ValueError("Some high prices are lower than other daily prices.")

    if (df["low"] > df[["open", "high", "close"]].min(axis=1)).any():
        raise ValueError("Some low prices are higher than other daily prices.")

    duplicate_count = df.duplicated(subset=["ticker", "date"]).sum()
    if duplicate_count:
        raise ValueError(
            f"Found {duplicate_count} duplicate ticker/date records."
        )

    print("Data quality validation passed.")
    print(f"Rows checked: {len(df)}")
    print(f"Tickers: {df['ticker'].nunique()}")
    print(f"Date range: {df['date'].min().date()} to "
          f"{df['date'].max().date()}")


if __name__ == "__main__":
    validate_stock_data()
