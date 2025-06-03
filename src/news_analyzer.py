import pandas as pd
import logging
from collections import Counter
from datetime import datetime
import re
import sys
import os
sys.path.append(os.path.abspath("C:/Users/firao/Desktop/PYTHON PROJECTS/KIAM PROJECTS/news-sentiment-stock-prediction"))
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from nltk.corpus import stopwords
import string
import re
import matplotlib.pyplot as plt
from src.base_analyzer import BaseAnalyzer

class NewsAnalyzer(BaseAnalyzer):
    def __init__(self, filepath):
        super().__init__(filepath)
        self.top_publishers = None
        self.top_keywords = None
        self.top_topics = None
        nltk.download('stopwords')

    def load_data(self):
        #Load and preprocess the financial news dataset
        try:
            self.data = pd.read_csv(self.filepath)
            self._preprocess()
            self.logger.info("News data loaded successfully")
            print(self.data.head())
            # Convert date column to datetime
            if 'date' in self.data.columns:
                self.data['date'] = pd.to_datetime(self.data['date'], errors='coerce')
                self.data = self.data.dropna(subset=['date'])  # Remove rows with invalid dates
        except FileNotFoundError as e:
            self.logger.error("Dataset not found. Please check the path.")
            raise
        except Exception as e:
            self.logger.error(f"Error loading dataset: {e}")
            raise
        
    def _preprocess(self, text_column='headline'):
        #Basic preprocessing of news headlines and dates
        try:
            # Ensure required columns exist
            required_columns = ['headline', 'publisher', 'date', 'stock']
            missing_cols = [col for col in required_columns if col not in self.data.columns]
        
            # Convert date to datetime
            self.data['date'] = pd.to_datetime(self.data['date'], errors='coerce')
            #self.data['headline_length'] = self.data['headline'].str.len()
            self.data['headline_length'] = self.data['headline'].apply(len)
        
            # Remove rows with missing or empty headlines
            self.data = self.data.dropna(subset=[text_column])
            self.data = self.data[self.data[text_column] != '']

            # Initialize lemmatizer and stopwords
            stop_words = set(stopwords.words('english'))
            custom_stopwords = {'said', 'will', 'also', 'reuters', 'bloomberg', 'marketwatch', 'stock', 'stocks', 'report', 'target'}
            stop_words.update(custom_stopwords)   
            
            def clean_text(text):
                # Remove special characters, numbers, and extra spaces
                text = re.sub(r'[^a-zA-Z\s]', '', text.lower().strip())
                # Tokenize and remove stopwords
                tokens = text.split()
                tokens = [word for word in tokens if word not in stop_words]
                return ' '.join(tokens)

            # Apply preprocessing to the text column
            self.data['clean_headline'] = self.data[text_column].apply(clean_text)
            self.logger.info("Text preprocessing completed successfully")
        except Exception as e:
            self.logger.error(f"Error preprocessing text: {e}")
            raise KeyError(f"Missing required columns: {missing_cols}")
            
    def perform_sentiment_analysis(self):
        
            #Perform sentiment analysis on the 'headline' column.
            #Adds 'polarity' and 'subjectivity' columns to the data.
        
        try:
            self.data['polarity'] = self.data['headline'].apply(lambda x: TextBlob(x).sentiment.polarity)
            self.data['subjectivity'] = self.data['headline'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
            self.logger.info("Sentiment analysis completed successfully")
        except Exception as e:
            self.logger.error(f"Error during sentiment analysis: {e}")
            raise
            # Extract domain from email publishers
        def extract_domain(publisher):
            match = re.search(r'@([\w\.-]+)', str(publisher))
            return match.group(1) if match else publisher

        self.data['publisher_domain'] = self.data['publisher'].apply(extract_domain)
    def get_top_publishers(self, top_n=10):
        #Get most active publishers
        try:
            self.top_publishers = self.data['publisher'].value_counts().head(top_n)
            return self.top_publishers
        except Exception as e:
            self.logger.error(f"Error calculating top publishers: {e}")
            raise

    def analyze_headline_lengths(self, freq='D'):
        
        """
        Analyze news frequency over time.
        """
        try:
            # Ensure the 'date' column is a DatetimeIndex
            if not isinstance(self.data.index, pd.DatetimeIndex):
                self.data['date'] = pd.to_datetime(self.data['date'], errors='coerce')
                self.data.set_index('date', inplace=True)

            # Resample data to count articles per frequency period
            time_df = self.data.resample(freq).size().reset_index(name='count')
            time_df.rename(columns={'date': 'time'}, inplace=True)
            return time_df
        except Exception as e:
            self.logger.error(f"Time series analysis failed: {e}")
        raise

    def extract_keywords(self, top_n=10):
        #Extract basic keywords from headlines
        
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

    def analyze_time_series(self, freq='D'):
        try:
            # Ensure datetime conversion
            if not pd.api.types.is_datetime64_any_dtype(self.data['date']):
                self.data['date'] = pd.to_datetime(self.data['date'])
                
            # Set date as index for resampling
            temp_df = self.data.set_index('date')
            time_df = temp_df.resample(freq).size().reset_index(name='count')
            return time_df
        except Exception as e:
            print(f"Time series analysis failed: {e}")
            raise
    def extract_topics(self, num_topics=5, num_words=10):
        #Use LDA to extract topics from headlines
        try:
            # step 1:Custom stopwords + NLTK defaults and preprocess data
            self._preprocess()
            stop_words = set(stopwords.words('english'))
            custom_stopwords = {'said', 'will', 'also', 'reuters', 'bloomberg', 'marketwatch', 'stock', 'stocks', 'report', 'target'}
            stop_words.update(custom_stopwords)

            # step 2: Vectorize headlines into word counts
            vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                lowercase=True,
                token_pattern=r'\b[a-zA-Z]{3,}\b'
)
            tf_matrix = vectorizer.fit_transform(self.data['headline'])
            self.logger.info("Headlines vectorized successfully")

            # Step 3: Apply LDA
            lda = LatentDirichletAllocation(
                n_components=num_topics,
                random_state=42,
                learning_decay=0.7,
                n_jobs=1
            )
            lda.fit(tf_matrix)
            self.logger.info("LDA model trained successfully")

            # Get feature names and topics
            feature_names = vectorizer.get_feature_names_out()
            topics = []

            for idx, topic in enumerate(lda.components_):
                top_words = [feature_names[i] for i in topic.argsort()[-num_words:][::-1]]
                topics.append(f"Topic {idx + 1}: {' '.join(top_words)}")
                #self.logger.info(f"Extracted Topic {idx + 1}: {' '.join(top_words)}")
                topics.append(top_words)
                self.top_topics = topics
                return topics

        except Exception as e:
            self.logger.error(f"Topic extraction failed: {e}")
            raise