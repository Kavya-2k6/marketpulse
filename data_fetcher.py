# data_fetcher.py
# PURPOSE: Download stock prices from Yahoo Finance and save to CSV
# INTERVIEW EXPLAINER: "I used yfinance to pull historical OHLC data,
# then kept only the 'Close' price and reset the index so Date becomes
# a regular column — easier to merge later."

import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Downloads daily closing prices for a given stock ticker.

    Args:
        ticker : Stock symbol, e.g. "AAPL", "TSLA"
        start  : Start date string "YYYY-MM-DD"
        end    : End date string   "YYYY-MM-DD"

    Returns:
        DataFrame with columns: ['Date', 'Close']
    """
    # Step 1: Download raw data from Yahoo Finance
    raw = yf.download(ticker, start=start, end=end)

    # Step 2: Keep only the 'Close' price column
    df = raw[["Close"]].copy()

    # Step 3: 'Date' is currently the INDEX — move it to a normal column
    df = df.reset_index()          # Date index → regular column
    df.columns = ["Date", "Close"] # Rename for clarity

    # Step 4: Make sure Date is a proper date (no time info)
    df["Date"] = pd.to_datetime(df["Date"]).dt.date

    print(f"✅ Fetched {len(df)} rows of stock data for {ticker}")
    return df


if __name__ == "__main__":
    # ── Quick test ──────────────────────────────────────────────────
    df = fetch_stock_data("AAPL", "2024-01-01", "2024-06-30")
    df.to_csv("stock_prices.csv", index=False)
    print(df.head())
    # ────────────────────────────────────────────────────────────────
