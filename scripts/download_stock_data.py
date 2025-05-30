# scripts/download_stock_data.py
import yfinance as yf
import os

def download_stock_data(tickers, start_date, end_date, output_folder="C:/Users/firao/Desktop/PYTHON PROJECTS/KIAM PROJECTS/data/stock_prices"):
    #Download stock price data and save as CSV files.
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for ticker in tickers:
        data = yf.download(ticker, start=start_date, end=end_date)
        data.to_csv(f"{output_folder}/{ticker}.csv")
        print(f"Downloaded data for {ticker}")

if __name__ == "__main__":
    # Example tickers from your dataset
    tickers = ["AAPL", "AMZN", "GOOGL", "META", "MSFT", "NVDA", "TSLA"]
    download_stock_data(tickers, "2020-01-01", "2025-01-01")