import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from nltk.corpus import stopwords
import string
import re
import os
sys.path.append("C:/Users/firao/Desktop/PYTHON PROJECTS/KIAM PROJECTS/news-sentiment-stock-prediction")  # So we can access src/
from src.news_analyzer import NewsAnalyzer
# Configure logging
import logging
logging.basicConfig(level=logging.INFO)
# Load and analyze news data
news_file = "C:/Users/firao/Desktop/PYTHON PROJECTS/KIAM PROJECTS/data/raw_analyst_ratings/raw_analyst_ratings.csv"
news = NewsAnalyzer(news_file)
try:
    news.load_data()

    print("Headline Lengths:")
    print(news.analyze_headline_lengths())
    print("\nTop Publishers:")
    print(news.get_top_publishers())
    print("\nTop Keywords:")
    print(news.extract_keywords())
except Exception as e:
    print(f"[ERROR] {e}")
#Plot for Distribution of Headline Length
plt.figure(figsize=(10, 6))
sns.histplot(news.data['headline_length'], bins=30, kde=True)
plt.title("Distribution of Headline Lengths")
plt.xlabel("Length")
plt.ylabel("Frequency")
plt.show()
#plot for Top publishers
top_publishers = news.get_top_publishers()
top_publishers.plot(kind='bar', title="Top Publishers", figsize=(12, 6))
plt.xticks(rotation=45)
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
# Extract and display top keywords
print("Top Keywords:")
keywords = news.extract_keywords()  # Changed from analyzer to news
for word, count in keywords:
    print(f"{word}: {count}")

# Extract topics using LDA
print("\nTop Topics (LDA):")
topics = news.extract_topics(num_topics=5, num_words=8)  # Changed from analyzer to news
for i, topic in enumerate(topics):
    print(f"Topic {i+1}: {', '.join(topic)}")