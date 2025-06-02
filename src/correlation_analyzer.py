# src/correlation_analyzer.py
import pandas as pd
from scipy.stats import pearsonr
from src.base_analyzer import BaseAnalyzer

class CorrelationAnalyzer(BaseAnalyzer):
    def __init__(self, news_data=None, stock_data=None):
        self.news_data = news_data
        self.stock_data = stock_data
        self.logger = logging.getLogger(self.__class__.__name__)

    def align_dates(self):
        """
        Align dates between news and stock data.
        """
        try:
            # Normalize dates and merge datasets
            merged_data = pd.merge(self.news_data, self.stock_data, left_on='date', right_on='Date')
            self.logger.info("Dates aligned successfully")
            return merged_data
        except Exception as e:
            self.logger.error(f"Error aligning dates: {e}")
            raise

    def calculate_correlation(self, sentiment_col, returns_col):
        """
        Calculate Pearson correlation between sentiment scores and stock returns.
        """
        try:
            correlation, p_value = pearsonr(merged_data[sentiment_col], merged_data[returns_col])
            self.logger.info(f"Correlation: {correlation}, P-value: {p_value}")
            return correlation, p_value
        except Exception as e:
            self.logger.error(f"Error calculating correlation: {e}")
            raise