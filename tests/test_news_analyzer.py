import sys
import os
sys.path.append(os.path.abspath(".."))

from src.news_analyzer import NewsAnalyzer

analyzer = NewsAnalyzer()
print("Success! Class loaded.")