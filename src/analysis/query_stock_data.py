
from pathlib import Path

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "data" / "financial.duckdb"


def main():
    if not DB_PATH.exists():
        raise FileNotFoundError(
            "Database not found. Run load_to_duckdb.py first."
        )

    with duckdb.connect(str(DB_PATH), read_only=True) as con:

        # 1. Show the latest 10 trading sessions.
        print("\nLATEST STOCK PRICES")
        latest = con.execute("""
            SELECT date, ticker, close, volume
            FROM stock_prices
            WHERE ticker = 'AAPL'
            ORDER BY date DESC
            LIMIT 10
        """).df()

        print(latest.to_string(index=False))

        # 2. Calculate basic price statistics.
        print("\nSTOCK SUMMARY")
        summary = con.execute("""
            SELECT
                ticker,
                COUNT(*) AS trading_sessions,
                ROUND(MIN(close), 2) AS minimum_close,
                ROUND(MAX(close), 2) AS maximum_close,
                ROUND(AVG(close), 2) AS average_close
            FROM stock_prices
            GROUP BY ticker
        """).df()

        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
