import sys
sys.path.append("C:/Users/firao/Desktop/PYTHON PROJECTS/KIAM PROJECTS/news-sentiment-stock-prediction")  # Add root directory to path

from src.stock_analyzer import StockAnalyzer
import logging
from textblob import TextBlob
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
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

def perform_sentiment_analysis(news_data):

    #Perform sentiment analysis on news headlines.
    
    #Parameters:
    #    news_data (pd.DataFrame): News dataset with 'headline' column.
    
    #Returns:
    #     pd.DataFrame: News dataset with added 'polarity' and 'subjectivity' columns.
    
    news_data['polarity'] = news_data['headline'].apply(lambda x: TextBlob(x).sentiment.polarity)
    news_data['subjectivity'] = news_data['headline'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
    return news_data

def aggregate_sentiment_scores(news_data):
    
    #Aggregate sentiment scores by date.
    
    #Parameters:
        #news_data (pd.DataFrame): News dataset with 'date', 'polarity', and 'subjectivity' columns.
    
    #Returns:
        #pd.DataFrame: Aggregated sentiment scores by date.
    
    aggregated_data = news_data.groupby('date').agg({
        'polarity': 'mean',
        'subjectivity': 'mean'
    }).reset_index()
    return aggregated_data

def calculate_daily_returns(stock_data):
    
    #Calculate daily percentage changes in stock closing prices.
    
    #Parameters:
        #stock_data (pd.DataFrame): Stock dataset with 'Close' column.
    
    #Returns:
    #    pd.DataFrame: Stock dataset with added 'Daily_Return' column.
    
    stock_data['Daily_Return'] = stock_data['Close'].pct_change() * 100
    return stock_data

def calculate_correlation(merged_data):
    
    #Calculate Pearson correlation between sentiment scores and stock returns.
    
    #Parameters:
        #merged_data (pd.DataFrame): Merged dataset with 'polarity' and 'Daily_Return' columns.
    
    #Returns:
        #float, float: Correlation coefficient and p-value.
    
    correlation, p_value = pearsonr(merged_data['polarity'], merged_data['Daily_Return'])
    print(f"Correlation: {correlation}, P-value: {p_value}")
    return correlation, p_value

def plot_trends(merged_data):
    plt.figure(figsize=(12, 6))
    plt.plot(merged_data['date'], merged_data['polarity'], label='Sentiment Polarity', color='blue')
    plt.plot(merged_data['date'], merged_data['Daily_Return'], label='Daily Returns', color='orange')
    plt.title("Trends in Sentiment Polarity and Daily Returns")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_scatter(merged_data):
    plt.figure(figsize=(10, 6))
    plt.scatter(merged_data['polarity'], merged_data['Daily_Return'], alpha=0.5)
    plt.title("Sentiment Polarity vs. Daily Stock Returns")
    plt.xlabel("Sentiment Polarity")
    plt.ylabel("Daily Stock Returns (%)")
    plt.grid(True)
    plt.show()