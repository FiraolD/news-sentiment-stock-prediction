import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import string
import re
import os
sys.path.append("C:/Users/firao/Desktop/PYTHON PROJECTS/KIAM PROJECTS/news-sentiment-stock-prediction")  # So we can access src/
from src.news_analyzer import NewsAnalyzer
# Configure logging
import logging
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    # Load and analyze news data
    news_file = "C:/Users/firao/Desktop/PYTHON PROJECTS/KIAM PROJECTS/data/raw_analyst_ratings/raw_analyst_ratings.csv"

    news = NewsAnalyzer(news_file)
    try:
        news.load_data()
        # Check for missing or empty headlines
        print(news.data['headline'].isnull().sum())  # Count missing values
        print((news.data['headline'] == '').sum())  # Count empty strings

        # Drop rows with missing or empty headlines
        news.data = news.data.dropna(subset=['headline'])
        news.data = news.data[news.data['headline'] != '']

        print("Headline Lengths:")
        news.analyze_headline_lengths()
    
        # Top publishers
        print("\nTop Publishers:")
        print(news.get_top_publishers())
        
        
        # Extract and display top keywords
        print("Top Keywords:")
        keywords = news.extract_keywords() # Changed from analyzer to news
        for word, count in keywords:
            print(f"{word}: {count}")   
        
        #Plot for Distribution of Headline Length
        plt.figure(figsize=(10, 6))
        sns.histplot(news.data['headline_length'], bins=30, kde=True)
        plt.title("Distribution of Headline Lengths")
        plt.xlabel("Length")
        plt.ylabel("Frequency")
        plt.show()

        #plot for Top publishers
        top_publishers = news.get_top_publishers()
        plt.figure(figsize=(12, 6))
        sns.barplot(x=top_publishers.values, y=top_publishers.index, palette="viridis")
        plt.title("Top Publishers")
        plt.xticks(rotation=45)
        plt.xlabel("publishers")
        plt.ylabel("Article Count")
        plt.show()

        #Plot for most frequent keywords in Headline
        keywords = news.extract_keywords()
        words, counts = zip(*keywords)
        sns.barplot(x=counts, y=words)
        plt.title("Most Frequent Keywords in Headlines")
        plt.xlabel("Frequency")
        plt.ylabel("Keyword")
        plt.show()

        # plot for Weekly trends
        time_df = news.analyze_time_series(freq='W')  
        plt.figure(figsize=(12, 6))
        plt.plot(time_df['time'], time_df['count'])
        plt.title("Weekly News Volume Over Time")
        plt.xlabel("Week")
        plt.ylabel("Number of Articles")
        plt.grid(True)
        plt.show()

    except Exception as e:
        (f"[ERROR] {e}")