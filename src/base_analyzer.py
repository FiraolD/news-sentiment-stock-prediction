import pandas as pd
import logging

class BaseAnalyzer:
    def __init__(self, filepath=None):
        self.filepath = filepath
        self.data = None
        self.logger = logging.getLogger(self.__class__.__name__)
        pass     
    def load_data(self):
        raise NotImplementedError("Subclasses must implement load_data()")

    def save_data(self, output_path):
        try:
            self.data.to_csv(output_path, index=False)
            self.logger.info(f"Data saved to {output_path}")
        except Exception as e:
            self.logger.error(f"Error saving data: {e}")
            raise
        