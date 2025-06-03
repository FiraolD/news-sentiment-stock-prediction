# src/task_1_eda.py
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


class Task1EDA:
    def __init__(self, filepath=None):
        self.news_analyzer = NewsAnalyzer(filepath)

    def run_analysis(self):
    
        #Run all EDA tasks.
        
        try:
            # Step 1: Load and preprocess data
            self.news_analyzer.load_data()

            # Step 2: Analyze headline lengths
            self.analyze_headline_lengths()

            # Step 3: Identify top publishers
            self.analyze_top_publishers()

            # Step 4: Analyze publication trends over time
            self.analyze_publication_trends()

            # Step 5: Extract keywords from headlines
            self.extract_keywords()

            # Step 6: Extract topics using LDA
            self.extract_topics()

            self.logger.info("All EDA tasks completed successfully")
        except Exception as e:
            self.logger.error(f"Error running EDA analysis: {e}")
            raise

    def analyze_headline_lengths(self):
        
        #Analyze the distribution of headline lengths.
        
        try:
            # Ensure the 'headline_length' column exists
            if 'headline_length' not in self.news_analyzer.data.columns:
                self.news_analyzer.data['headline_length'] = self.news_analyzer.data['headline'].apply(len)

            # Plot the distribution of headline lengths
            plt.figure(figsize=(10, 6))
            sns.histplot(self.news_analyzer.data['headline_length'], bins=30, kde=True)
            plt.title("Distribution of Headline Lengths")
            plt.xlabel("Headline Length")
            plt.ylabel("Frequency")
            plt.show()
            self.logger.info("Headline length analysis completed successfully")
        except Exception as e:
            self.logger.error(f"Error analyzing headline length: {e}")
            raise

    def analyze_top_publishers(self):
        
        #Analyze the number of articles per publisher to identify the most active publishers.
        
        try:
            # Get top publishers
            top_publishers = self.news_analyzer.get_top_publishers(top_n=10)

            # Plot the top 10 publishers
            plt.figure(figsize=(10, 6))
            sns.barplot(x=top_publishers.values, y=top_publishers.index, palette="viridis")
            plt.title("Top 10 Publishers by Article Count")
            plt.xlabel("Number of Articles")
            plt.ylabel("Publisher")
            plt.show()
            self.logger.info("Publisher activity analysis completed successfully")
        except Exception as e:
            self.logger.error(f"Error analyzing publisher activity: {e}")
            raise

    def analyze_publication_trends(self):
        
        #Analyze publication dates to see trends over time.
        
        try:
            # Resample data to count articles per day
            daily_article_counts = self.news_analyzer.analyze_time_series(freq='D')

            # Plot the trend of article counts over time
            plt.figure(figsize=(12, 6))
            sns.lineplot(data=daily_article_counts, x='time', y='count')
            plt.title("Daily Article Publication Trends")
            plt.xlabel("Date")
            plt.ylabel("Number of Articles")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
            self.logger.info("Publication trends analysis completed successfully")
        except Exception as e:
            self.logger.error(f"Error analyzing publication trends: {e}")
            raise

    def extract_keywords(self):
        
        #Extract and display the top keywords from headlines.
        
        try:
            # Extract top keywords
            top_keywords = self.news_analyzer.extract_keywords(top_n=10)
            print("Top Keywords:")
            for word, count in top_keywords:
                print(f"{word}: {count}")

            self.logger.info("Keyword extraction completed successfully")
        except Exception as e:
            self.logger.error(f"Error extracting keywords: {e}")
            raise

    def extract_topics(self):
        
        #Extract and display topics from headlines using Latent Dirichlet Allocation (LDA).
        
        try:
            # Extract topics
            topics = self.news_analyzer.extract_topics(num_topics=5, num_words=8)
            print("Extracted Topics:")
            for topic in topics:
                print(topic)

            self.logger.info("Topic extraction completed successfully")
        except Exception as e:
            self.logger.error(f"Error extracting topics: {e}")
            raise