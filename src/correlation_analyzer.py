# src/correlation_analyzer.py
import os
import pandas as pd
from scipy.stats import pearsonr
import logging
from src.base_analyzer import BaseAnalyzer

class CorrelationAnalyzer(BaseAnalyzer):
    def __init__(self, news_data=None, stock_data_folder=None):
        super().__init__()
        self.news_data = news_data
        self.stock_data_folder = stock_data_folder
        self.logger = logging.getLogger(self.__class__.__name__)

    def align_dates(self, news_data, stock_data):
    #Align dates between news and stock data.
    
        try:
            # Ensure 'date' columns exist in both datasets
            if 'date' not in news_data.columns or 'Date' not in stock_data.columns:
                raise KeyError("Both datasets must contain 'date' and 'Date' columns.")

            # Convert dates to datetime format
            news_data['date'] = pd.to_datetime(news_data['date'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
            stock_data['Date'] = pd.to_datetime(stock_data['Date'], format='%Y-%m-%d %H:%M:%S', errors='coerce')

            # Merge datasets on date
            merged_data = pd.merge(news_data, stock_data, left_on='date', right_on='Date', how='inner')
            self.logger.info("Dates aligned successfully")
            return merged_data
        except Exception as e:
            self.logger.error(f"Error aligning dates: {e}")
            raise

    def aggregate_sentiment_scores(self, merged_data):
        #Aggregate sentiment scores by date.
    
        try:
            aggregated_data = merged_data.groupby('date').agg({
                'polarity': 'mean',
                'subjectivity': 'mean',
                'Daily_Return': 'first'
            }).reset_index()
            self.logger.info("Sentiment scores aggregated successfully")
            return aggregated_data
        except Exception as e:
            self.logger.error(f"Error aggregating sentiment scores: {e}")
            raise

    def calculate_correlation(self, aggregated_data):
        
        #Calculate Pearson correlation between sentiment scores and stock returns.
        
        try:
            # Calculate Pearson correlation coefficient and p-value
            correlation, p_value = pearsonr(aggregated_data['polarity'], aggregated_data['Daily_Return'])
            self.logger.info(f"Pearson Correlation: {correlation}, P-value: {p_value}")
            return correlation, p_value
        except Exception as e:
            self.logger.error(f"Error calculating correlation: {e}")
            raise

    def analyze_company(self, company_name, news_data, stock_data_folder):
        
        #Perform correlation analysis for a single company.
        
        try:
            # Filter news data for the specific company
            company_news_data = news_data[news_data['stock'] == company_name]

            # Load stock data for the company
            stock_filepath = os.path.join(stock_data_folder, f"{company_name}_processed.csv")
            self.logger.info(f"Loading stock data from: {stock_filepath}")
            stock_data = pd.read_csv(stock_filepath)

            # Align dates
            merged_data = self.align_dates(company_news_data, stock_data)

            # Aggregate sentiment scores
            aggregated_data = self.aggregate_sentiment_scores(merged_data)

            # Calculate correlation
            correlation, p_value = self.calculate_correlation(aggregated_data)

            # Return results
            return {
                "Company": company_name,
                "Correlation": correlation,
                "P-value": p_value
            }
        except FileNotFoundError as e:
            self.logger.error(f"File not found for company {company_name}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error analyzing company {company_name}: {e}")
            return None


# Main Execution Block
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Load preprocessed news data
    news_filepath = "C:/Users/firao/Desktop/PYTHON PROJECTS/KIAM PROJECTS/news-sentiment-stock-prediction/data/raw_analyst_ratings/raw_analyst_ratings.csv"
    #stock_data_folder = "C:/Users/firao/Desktop/PYTHON PROJECTS/KIAM PROJECTS/news-sentiment-stock-prediction/data/yfinance_data"
    stock_data_folder = "C:/Users/firao/Desktop/PYTHON PROJECTS/KIAM PROJECTS/news-sentiment-stock-prediction/data/processed_stock_prices"

    try:
        # Load news data
        news_data = pd.read_csv(news_filepath)

        # List of companies to analyze
        companies = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA", "META"]

        # Instantiate CorrelationAnalyzer
        correlation_analyzer = CorrelationAnalyzer(news_data=news_data, stock_data_folder=stock_data_folder)

        # Analyze all companies
        results = []
        for company in companies:
            result = correlation_analyzer.analyze_company(company, news_data, stock_data_folder)
            if result:
                results.append(result)

        # Print results
        print("Correlation Results:")
        for res in results:
            print(f"Company: {res['Company']}, Correlation: {res['Correlation']}, P-value: {res['P-value']}")

        # Save results to a CSV file
        results_df = pd.DataFrame(results)
        results_df.to_csv("/Users/firao/Desktop/PYTHON PROJECTS/KIAM PROJECTS/news-sentiment-stock-prediction/data/correlation_results.csv", index=False)
        logging.info("Correlation results saved to /Users/firao/Desktop/PYTHON PROJECTS/KIAM PROJECTS/news-sentiment-stock-prediction/data/correlation_results.csv")

    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
    except Exception as e:
        logging.error(f"An error occurred: {e}")