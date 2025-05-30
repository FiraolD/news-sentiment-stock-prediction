# src/stock_analyzer.py
import pandas as pd
import talib
import os
import logging

class StockAnalyzer:
    def __init__(self, ticker, data_folder="C:/Users/firao/Desktop/PYTHON PROJECTS\KIAM PROJECTS/news-sentiment-stock-prediction/data/yfinance_data"):
        
        #Initialize the StockAnalyzer for a specific stock ticker.
        #:param ticker: Stock ticker symbol (e.g., "AAPL", "MSFT").
        #:param data_folder: Folder containing stock price CSV files.
    
        self.ticker = ticker
        self.data_folder = data_folder
        self.filepath = os.path.join(data_folder, f"{ticker}.csv")
        self.data = None
        self.logger = logging.getLogger(self.__class__.__name__)

    def load_data(self):
    
        #Load stock price data from a CSV file.
        
        try:
            self.data = pd.read_csv(self.filepath, parse_dates=["Date"], index_col="Date")
            self.logger.info(f"Successfully loaded data for {self.ticker}")
        except Exception as e:
            self.logger.error(f"Failed to load data for {self.ticker}: {e}")
            raise

    def calculate_indicators(self):
        
        #Calculate technical indicators (SMA, RSI, MACD).
        
        try:
            # Moving Averages
            self.data['SMA_20'] = talib.SMA(self.data['Close'], timeperiod=20)
            self.data['SMA_50'] = talib.SMA(self.data['Close'], timeperiod=50)

            # RSI (Relative Strength Index)
            self.data['RSI'] = talib.RSI(self.data['Close'], timeperiod=14)

            # MACD (Moving Average Convergence Divergence)
            self.data['MACD'], self.data['MACD_signal'], self.data['MACD_hist'] = talib.MACD(
                self.data['Close'], fastperiod=12, slowperiod=26, signalperiod=9
            )

            self.logger.info(f"Calculated technical indicators for {self.ticker}")
        except Exception as e:
            self.logger.error(f"Error calculating indicators for {self.ticker}: {e}")
            raise

    def save_processed_data(self, output_folder="data/processed_stock_prices"):
    
        #Save processed data to a new CSV file.
        
        try:
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            output_path = os.path.join(output_folder, f"{self.ticker}_processed.csv")
            self.data.to_csv(output_path)
            self.logger.info(f"Saved processed data for {self.ticker} to {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save processed data for {self.ticker}: {e}")
            raise