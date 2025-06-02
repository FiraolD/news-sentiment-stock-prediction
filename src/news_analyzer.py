# src/news_analyzer.py
import sys
sys.path.append("C:/Users/firao/Desktop/PYTHON PROJECTS/KIAM PROJECTS/news-sentiment-stock-prediction")
import pandas as pd
import logging
from collections import Counter
from datetime import datetime
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from textblob import TextBlob
from src.base_analyzer import BaseAnalyzer

class NewsAnalyzer(BaseAnalyzer):
    def __init__(self, filepath=None):
        super().__init__(filepath)
        self.top_publishers = None
        self.top_keywords = None
        self.top_topics = None
        nltk.download('stopwords')

    def load_data(self):
        #Load and preprocess the financial news dataset.
        try:
            self.data = pd.read_csv(self.filepath)
            self._preprocess()
            self.logger.info("News data loaded successfully")
        except FileNotFoundError as e:
            self.logger.error("Dataset not found. Please check the path.")
            raise
        except Exception as e:
            self.logger.error(f"Error loading dataset: {e}")
            raise

    def _preprocess(self):
        #Basic preprocessing of news headlines and dates.
        try:
            # Ensure required columns exist
            required_columns = ['headline', 'publisher', 'date', 'stock']
            missing_cols = [col for col in required_columns if col not in self.data.columns]
            if missing_cols:
                raise KeyError(f"Missing required columns: {missing_cols}")

            # Convert date to datetime
            self.data['date'] = pd.to_datetime(self.data['date'], errors='coerce')
            self.data['headline_length'] = self.data['headline'].str.len()

            # Extract domain from email publishers
            def extract_domain(publisher):
                match = re.search(r'@([\w\.-]+)', str(publisher))
                return match.group(1) if match else publisher

            self.data['publisher_domain'] = self.data['publisher'].apply(extract_domain)

        except Exception as e:
            self.logger.error(f"Preprocessing failed: {e}")
            raise

    def get_top_publishers(self, top_n=10):
        #Get most active publishers.
        try:
            self.top_publishers = self.data['publisher_domain'].value_counts().head(top_n)
            return self.top_publishers
        except Exception as e:
            self.logger.error(f"Error calculating top publishers: {e}")
            raise

    def analyze_headline_lengths(self):
        #Analyze headline length distribution.
        try:
            return self.data['headline_length'].describe()
        except Exception as e:
            self.logger.error(f"Error analyzing headline lengths: {e}")
            raise

    def extract_keywords(self, top_n=10):
        #Extract basic keywords from headlines.
        try:
            all_words = []
            for headline in self.data['headline']:
                words = re.findall(r'\b\w+\b', str(headline).lower())
                all_words.extend(words)

            word_counts = Counter(all_words)
            self.top_keywords = word_counts.most_common(top_n)
            return self.top_keywords
        except Exception as e:
            self.logger.error(f"Keyword extraction failed: {e}")
            raise

    def extract_topics(self, num_topics=5, num_words=10):
        #Use LDA to extract topics from headlines.
        try:
            # Custom stopwords + NLTK defaults
            stop_words = set(stopwords.words('english'))
            custom_stopwords = {'said', 'will', 'also', 'reuters', 'bloomberg', 'marketwatch', 'stock', 'stocks', 'report', 'target'}
            stop_words.update(custom_stopwords)

            # Vectorize headlines into word counts
            vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                lowercase=True,
                token_pattern=r'\b[a-zA-Z]{3,}\b'
            )
            tf_matrix = vectorizer.fit_transform(self.data['headline'])

            # Apply LDA
            lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
            lda.fit(tf_matrix)

            # Get feature names and topics
            feature_names = vectorizer.get_feature_names_out()
            topics = []

            for idx, topic in enumerate(lda.components_):
                top_words = [feature_names[i] for i in topic.argsort()[-num_words:]]
                topics.append(top_words)

            self.top_topics = topics
            return topics

        except Exception as e:
            self.logger.error(f"Topic extraction failed: {e}")
            raise
 task-1

    def perform_sentiment_analysis(self):
        
        #Perform sentiment analysis on the 'headline' column.
        #Adds 'polarity' and 'subjectivity' columns to the data.
        

        def perform_sentiment_analysis(self):
        """
        Perform sentiment analysis on the 'headline' column.
        Adds 'polarity' and 'subjectivity' columns to the data.
        """
 main
        try:
            self.data['polarity'] = self.data['headline'].apply(lambda x: TextBlob(x).sentiment.polarity)
            self.data['subjectivity'] = self.data['headline'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
            self.logger.info("Sentiment analysis completed successfully")
        except Exception as e:
            self.logger.error(f"Error during sentiment analysis: {e}")
 task-1
            raise

    def aggregate_sentiment_scores(self):
        #Aggregate sentiment scores by date.
        
        try:
            aggregated_data = self.data.groupby('date').agg({
                'polarity': 'mean',
                'subjectivity': 'mean'
            }).reset_index()
            self.logger.info("Sentiment scores aggregated successfully")
            return aggregated_data
        except Exception as e:
            self.logger.error(f"Error aggregating sentiment scores: {e}")
            raise


# Main Execution Block (Moved Outside the Class Definition)
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Filepath to the news dataset
    filepath = "C:/Users/firao/Desktop/PYTHON PROJECTS/KIAM PROJECTS/news-sentiment-stock-prediction/data/raw_analyst_ratings/raw_analyst_ratings.csv"

    # Instantiate NewsAnalyzer
    news_analyzer = NewsAnalyzer(filepath)

    # Load and preprocess data
    news_analyzer.load_data()

    # Perform EDA tasks
    print("Top Publishers:")
    print(news_analyzer.get_top_publishers(top_n=5))

    print("\nHeadline Length Statistics:")
    print(news_analyzer.analyze_headline_lengths())

    print("\nTop Keywords:")
    print(news_analyzer.extract_keywords(top_n=5))

    print("\nTopics Extracted Using LDA:")
    print(news_analyzer.extract_topics(num_topics=3, num_words=5))

    # Perform sentiment analysis
    news_analyzer.perform_sentiment_analysis()

    # Aggregate sentiment scores
    aggregated_sentiment = news_analyzer.aggregate_sentiment_scores()
    print("\nAggregated Sentiment Scores:")
    print(aggregated_sentiment.head())

            raise
 main
