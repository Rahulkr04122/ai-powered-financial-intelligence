
from pathlib import Path

import pandas as pd
import yfinance as yf


# Find the project root automatically
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

# Create the folder if it doesn't exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


def fetch_stock_data(ticker: str, period: str = "1y") -> pd.DataFrame:
    """Download historical daily stock prices for one ticker."""

    print(f"Downloading stock data for {ticker}...")

    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval="1d", auto_adjust=True)

    if df.empty:
        raise ValueError(
            f"No stock data returned for {ticker}. "
            "Check the ticker symbol or your internet connection."
        )

    # Convert the date index into a regular column
    df = df.reset_index()

    # Normalize column names
    df.columns = [
        str(column).strip().lower().replace(" ", "_")
        for column in df.columns
    ]

    # Standardize the date column name
    if "date" not in df.columns and "datetime" in df.columns:
        df = df.rename(columns={"datetime": "date"})

    df["ticker"] = ticker.upper()

    # Keep only the fields needed for our first pipeline
    required_columns = [
        "date",
        "ticker",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    df = df[required_columns]

    # Remove incomplete records and sort chronologically
    df = df.dropna(subset=["date", "close"])
    df = df.sort_values("date").reset_index(drop=True)

    return df


def save_stock_data(df: pd.DataFrame, ticker: str) -> Path:
    """Save downloaded stock data as a CSV file."""

    output_path = RAW_DATA_DIR / f"{ticker.lower()}_prices.csv"

    df.to_csv(output_path, index=False)

    print(f"Saved {len(df)} rows to: {output_path}")

    return output_path


def main():
    ticker = "AAPL"

    df = fetch_stock_data(ticker, period="1y")
    save_stock_data(df, ticker)

    print("\nFirst five rows:")
    print(df.head().to_string(index=False))

    print("\nDataset summary:")
    print(f"Rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")


if __name__ == "__main__":
    main()
