
from pathlib import Path

import duckdb
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CSV_PATH = PROJECT_ROOT / "data" / "raw" / "aapl_prices.csv"
DB_PATH = PROJECT_ROOT / "data" / "financial.duckdb"


def load_stock_data():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    # Standardize dates for consistent database storage.
    df["date"] = (
        pd.to_datetime(df["date"], utc=True)
        .dt.strftime("%Y-%m-%d")
    )

    df["ticker"] = df["ticker"].str.upper()

    with duckdb.connect(str(DB_PATH)) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS stock_prices (
                date DATE,
                ticker VARCHAR,
                open DOUBLE,
                high DOUBLE,
                low DOUBLE,
                close DOUBLE,
                volume BIGINT,
                PRIMARY KEY (ticker, date)
            )
        """)

        # Make repeated runs safe: replace this ticker's records.
        con.execute(
            "DELETE FROM stock_prices WHERE ticker = ?",
            [df["ticker"].iloc[0]],
        )

        con.register("incoming_stock_data", df)

        con.execute("""
            INSERT INTO stock_prices
            SELECT
                CAST(date AS DATE),
                ticker,
                open,
                high,
                low,
                close,
                CAST(volume AS BIGINT)
            FROM incoming_stock_data
        """)

        count = con.execute("""
            SELECT COUNT(*)
            FROM stock_prices
        """).fetchone()[0]

        print(f"Loaded stock data successfully. Total rows: {count}")
        print(f"Database location: {DB_PATH}")


if __name__ == "__main__":
    load_stock_data()
