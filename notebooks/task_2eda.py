import sys
sys.path.append("C:/Users/firao/Desktop/PYTHON PROJECTS/KIAM PROJECTS/news-sentiment-stock-prediction")  # Add root directory to path

from src.stock_analyzer import StockAnalyzer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

def process_stock(ticker):
    """
    Process stock data for a given ticker.
    """
    analyzer = StockAnalyzer(ticker)
    analyzer.load_data()
    analyzer.calculate_indicators()
    analyzer.save_processed_data()
    return analyzer

# List of tickers for the 7 companies
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA"]

# Process each stock
for ticker in tickers:
    logging.info(f"Processing data for {ticker}")
    analyzer = process_stock(ticker)
    logging.info(f"Finished processing data for {ticker}")