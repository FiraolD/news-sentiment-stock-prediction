# src/stock_analyzer.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import talib as ta
import logging
from src.base_analyzer import BaseAnalyzer

class StockAnalyzer(BaseAnalyzer):
    def __init__(self, filepath=None):
        super().__init__(filepath)
        self.logger = logging.getLogger(self.__class__.__name__)

    def load_data(self):
        
        #Load and preprocess stock price data.
        
        try:
            # Load stock price data
            self.data = pd.read_csv(self.filepath)

            # Ensure required columns exist
            required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            missing_cols = [col for col in required_columns if col not in self.data.columns]
            if missing_cols:
                raise KeyError(f"Missing required columns: {missing_cols}")

            # Convert 'Date' to datetime format
            self.data['Date'] = pd.to_datetime(self.data['Date'], errors='coerce')

            # Drop rows with invalid dates
            self.data = self.data.dropna(subset=['Date'])

            # Set 'Date' as the index
            self.data.set_index('Date', inplace=True)

            # Sort by date
            self.data.sort_index(inplace=True)

            self.logger.info("Stock price data loaded and preprocessed successfully")
        except FileNotFoundError as e:
            self.logger.error(f"Dataset not found. Please check the path: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading dataset: {e}")
            raise

    def calculate_technical_indicators(self):
        
        #Calculate technical indicators using TA-Lib.
        
        try:
            # Calculate Simple Moving Average (SMA) - 10-day and 50-day
            self.data['SMA_10'] = ta.SMA(self.data['Close'], timeperiod=10)
            self.data['SMA_50'] = ta.SMA(self.data['Close'], timeperiod=50)

            # Calculate Relative Strength Index (RSI) - 14-day
            self.data['RSI'] = ta.RSI(self.data['Close'], timeperiod=14)

            # Calculate Moving Average Convergence Divergence (MACD)
            macd, macdsignal, macdhist = ta.MACD(self.data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
            self.data['MACD'] = macd
            self.data['MACD_Signal'] = macdsignal
            self.data['MACD_Hist'] = macdhist

            # Calculate Daily Returns
            self.data['Daily_Return'] = self.data['Close'].pct_change() * 100

            self.logger.info("Technical indicators calculated successfully")
        except Exception as e:
            self.logger.error(f"Error calculating technical indicators: {e}")
            raise

    def visualize_data(self):
        
        #Visualize stock price and technical indicators.
        
        try:
            plt.figure(figsize=(14, 8))

            # Plot Closing Price
            plt.subplot(3, 1, 1)
            plt.plot(self.data.index, self.data['Close'], label='Close Price', color='blue')
            plt.title('Stock Closing Price')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()

            # Plot SMA
            plt.subplot(3, 1, 2)
            plt.plot(self.data.index, self.data['SMA_10'], label='SMA 10-Day', color='green')
            plt.plot(self.data.index, self.data['SMA_50'], label='SMA 50-Day', color='orange')
            plt.title('Simple Moving Averages (SMA)')
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.legend()

            # Plot RSI
            plt.subplot(3, 1, 3)
            plt.plot(self.data.index, self.data['RSI'], label='RSI', color='purple')
            plt.axhline(70, color='red', linestyle='--', label='Overbought (70)')
            plt.axhline(30, color='green', linestyle='--', label='Oversold (30)')
            plt.title('Relative Strength Index (RSI)')
            plt.xlabel('Date')
            plt.ylabel('RSI Value')
            plt.legend()

            plt.tight_layout()
            plt.show()

            self.logger.info("Data visualization completed successfully")
        except Exception as e:
            self.logger.error(f"Error visualizing data: {e}")
            raise

    def save_processed_data(self, output_filepath):
        
        #Save processed stock data to a CSV file.
        
        try:
            self.data.to_csv(output_filepath, index=True)
            self.logger.info(f"Processed stock data saved to {output_filepath}")
        except Exception as e:
            self.logger.error(f"Error saving processed stock data: {e}")
            raise


# Main Execution Block
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # File paths
    input_filepath = "C:/Users/firao/Desktop/PYTHON PROJECTS/KIAM PROJECTS/news-sentiment-stock-prediction/data/raw_stock_prices/AAPL.csv"
    output_filepath = "C:/Users/firao/Desktop/PYTHON PROJECTS/KIAM PROJECTS/news-sentiment-stock-prediction/data/processed_stock_prices/AAPL_processed.csv"

    try:
        # Instantiate StockAnalyzer
        stock_analyzer = StockAnalyzer(filepath=input_filepath)

        # Load and preprocess data
        stock_analyzer.load_data()

        # Calculate technical indicators
        stock_analyzer.calculate_technical_indicators()

        # Visualize data
        stock_analyzer.visualize_data()

        # Save processed data
        stock_analyzer.save_processed_data(output_filepath)

    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
    except Exception as e:
        logging.error(f"An error occurred: {e}")