# src/stock_analyzer.py
import pandas as pd
import talib
import os
import logging
import matplotlib.pyplot as plt

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
            # Calculate SMA (Simple Moving Average)
            self.data['SMA_20'] = talib.SMA(self.data['Close'], timeperiod=20)
            self.data['SMA_50'] = talib.SMA(self.data['Close'], timeperiod=50)

            # Calculate RSI (Relative Strength Index)
            self.data['RSI'] = talib.RSI(self.data['Close'], timeperiod=14)

            # Calculate MACD (Moving Average Convergence Divergence)
            self.data['MACD'], self.data['MACD_signal'], self.data['MACD_hist'] = talib.MACD(
                self.data['Close'], fastperiod=12, slowperiod=26, signalperiod=9
            )

            # Drop rows with NaN values for indicators
            self.data.dropna(subset=['SMA_20', 'SMA_50', 'RSI', 'MACD'], inplace=True)

            self.logger.info("Technical indicators calculated successfully")
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {e}")
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
        
        import pandas as pd

# Load a processed file
df = pd.read_csv("data/processed_stock_prices/AAPL_processed.csv")
print(df.head())

def plot_moving_averages(data, ticker):
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['Close'], label='Close Price', color='blue')
    plt.plot(data.index, data['SMA_20'], label='20-Day SMA', color='orange')
    plt.plot(data.index, data['SMA_50'], label='50-Day SMA', color='green')
    plt.title(f"{ticker} Stock Price and Moving Averages")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.show()

# Example for AAPL
df = pd.read_csv("data/processed_stock_prices/AAPL_processed.csv", parse_dates=["Date"], index_col="Date")
plot_moving_averages(df, "AAPL")

def plot_rsi(data, ticker):
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['RSI'], label='RSI', color='purple')
    plt.axhline(70, color='red', linestyle='--', label='Overbought (70)')
    plt.axhline(30, color='green', linestyle='--', label='Oversold (30)')
    plt.title(f"{ticker} RSI Indicator")
    plt.xlabel("Date")
    plt.ylabel("RSI Value")
    plt.legend()
    plt.grid(True)
    plt.show()

# Example for AAPL
plot_rsi(df, "AAPL")

def plot_macd(data, ticker):
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['MACD'], label='MACD', color='blue')
    plt.plot(data.index, data['MACD_signal'], label='Signal Line', color='orange')
    plt.bar(data.index, data['MACD_hist'], label='MACD Histogram', color='gray')
    plt.title(f"{ticker} MACD Indicator")
    plt.xlabel("Date")
    plt.ylabel("MACD Values")
    plt.legend()
    plt.grid(True)
    plt.show()

# Example for AAPL
plot_macd(df, "AAPL")

# Load a processed file
df = pd.read_csv("data/processed_stock_prices/MSFT_processed.csv")
print(df.head())

def plot_moving_averages(data, ticker):
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['Close'], label='Close Price', color='blue')
    plt.plot(data.index, data['SMA_20'], label='20-Day SMA', color='orange')
    plt.plot(data.index, data['SMA_50'], label='50-Day SMA', color='green')
    plt.title(f"{ticker} Stock Price and Moving Averages")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.show()

# Example for MSFT
df = pd.read_csv("data/processed_stock_prices/MSFT_processed.csv", parse_dates=["Date"], index_col="Date")
plot_moving_averages(df, "MSFT")

def plot_rsi(data, ticker):
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['RSI'], label='RSI', color='purple')
    plt.axhline(70, color='red', linestyle='--', label='Overbought (70)')
    plt.axhline(30, color='green', linestyle='--', label='Oversold (30)')
    plt.title(f"{ticker} RSI Indicator")
    plt.xlabel("Date")
    plt.ylabel("RSI Value")
    plt.legend()
    plt.grid(True)
    plt.show()

# Example for MSFT
plot_rsi(df, "MSFT")

def plot_macd(data, ticker):
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['MACD'], label='MACD', color='blue')
    plt.plot(data.index, data['MACD_signal'], label='Signal Line', color='orange')
    plt.bar(data.index, data['MACD_hist'], label='MACD Histogram', color='gray')
    plt.title(f"{ticker} MACD Indicator")
    plt.xlabel("Date")
    plt.ylabel("MACD Values")
    plt.legend()
    plt.grid(True)
    plt.show()

# Example for MSFT
plot_macd(df, "MSFT")

# Load a processed file
df = pd.read_csv("data/processed_stock_prices/AMZN_processed.csv")
print(df.head())

def plot_moving_averages(data, ticker):
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['Close'], label='Close Price', color='blue')
    plt.plot(data.index, data['SMA_20'], label='20-Day SMA', color='orange')
    plt.plot(data.index, data['SMA_50'], label='50-Day SMA', color='green')
    plt.title(f"{ticker} Stock Price and Moving Averages")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.show()

# Example for AMZN
df = pd.read_csv("data/processed_stock_prices/AMZN_processed.csv", parse_dates=["Date"], index_col="Date")
plot_moving_averages(df, "AMZN")

def plot_rsi(data, ticker):
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['RSI'], label='RSI', color='purple')
    plt.axhline(70, color='red', linestyle='--', label='Overbought (70)')
    plt.axhline(30, color='green', linestyle='--', label='Oversold (30)')
    plt.title(f"{ticker} RSI Indicator")
    plt.xlabel("Date")
    plt.ylabel("RSI Value")
    plt.legend()
    plt.grid(True)
    plt.show()

# Example for AMZN
plot_rsi(df, "AMZN")

def plot_macd(data, ticker):
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data['MACD'], label='MACD', color='blue')
    plt.plot(data.index, data['MACD_signal'], label='Signal Line', color='orange')
    plt.bar(data.index, data['MACD_hist'], label='MACD Histogram', color='gray')
    plt.title(f"{ticker} MACD Indicator")
    plt.xlabel("Date")
    plt.ylabel("MACD Values")
    plt.legend()
    plt.grid(True)
    plt.show()

# Example for AMZN
plot_macd(df, "AMZN")