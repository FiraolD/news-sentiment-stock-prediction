# src/base_analyzer.py
import pandas as pd
import os
import sys
sys.path.append("C:/Users/firao/Desktop/PYTHON PROJECTS/KIAM PROJECTS/news-sentiment-stock-prediction")
import logging

class BaseAnalyzer:
    def __init__(self, filepath=None):
        self.filepath = filepath
        self.data = None
        self.logger = logging.getLogger(self.__class__.__name__)

    def load_data(self):
        #Load data from a CSV file."""
        try:
            if not os.path.exists(self.filepath):
                raise FileNotFoundError(f"File not found: {self.filepath}")
            self.data = pd.read_csv(self.filepath)
            self.logger.info(f"Successfully loaded data from {self.filepath}")
        except Exception as e:
            self.logger.error(f"Failed to load data: {e}")
            raise
        