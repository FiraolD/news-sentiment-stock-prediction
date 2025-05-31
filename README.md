# B5W1: Predicting Price Moves with News Sentiment  
**By: FIRAOL DELESA**  
**Date: Mar 28, 2025**

## 🌟 Overview

This project explores how financial news headlines influence stock market movements. By performing sentiment analysis on news headlines and computing technical indicators (RSI, MACD, Moving Averages), we aim to uncover correlations between news sentiment and daily stock returns.

## 🛠️ Tools Used

- Python
- Pandas, NumPy
- TextBlob / NLTK
- TA-Lib / PyNance
- Matplotlib, Seaborn
- Git & GitHub for version control

## 📁 Folder Structure

├── .vscode/
├── .github/workflows/unittests.yml
├── .gitignore
├── requirements.txt
├── README.md
├── src/
├── notebooks/
├── tests/
└── scripts/



> 💡 Note: The `data/` folder is ignored via `.gitignore`. Add it manually when working locally.

## 🚀 Next Steps

- clone git repo
- Create `task-1` branch
- Start EDA in Jupyter notebook

## 📌 Branching Strategy

- `main`: Final, reviewed code
- `task-1`: EDA and Git setup
- `task-2`: Technical indicators
- `task-3`: Correlation analysis

## Object-Oriented Programming Implementation

As part of Task 1, I implemented an object-oriented structure to ensure modularity, reusability, and maintainability of the code.

I created two classes:
- **`BaseAnalyzer`**: A parent class providing common functionality such as data loading and saving.
- **`NewsAnalyzer`**: A child class that extends `BaseAnalyzer` and implements EDA-specific methods including headline length analysis, publisher frequency, keyword extraction, topic modeling, and time-based trends.

The use of **inheritance** allowed me to avoid code duplication and apply DRY (Don't Repeat Yourself) principles. All operations were wrapped in `try/except` blocks to ensure robustness and ease of debugging.

Although the notebook (`task-1_eda.ipynb`) only calls these methods, the underlying implementation follows best practices in software engineering and data science workflows.

### Topic Modeling Using LDA

To uncover hidden themes in financial news headlines, I applied **Latent Dirichlet Allocation (LDA)**, an unsupervised NLP technique that identifies recurring topics.

#### Key Topics Identified:
1. **Regulatory Events**: fda, approval, drug, trial, company, phase
2. **Analyst Ratings & Price Targets**: price, target, analyst, stock, raises, lowers
3. **Mergers & Acquisitions**: merger, acquisition, buyout, deal, company, announce
4. **Earnings Reports**: earnings, revenue, profit, quarterly, beat, miss
5. **Market Movement**: market, rise, shares, investor, sentiment, volatility

These topics provide insight into how different types of news may influence stock movement and support future correlation analysis.

### Methodology
I used an object-oriented approach with a base class (`BaseAnalyzer`) and child class (`NewsAnalyzer`) to modularize the code and ensure reusability. The analysis focused on understanding headline patterns, publisher behavior, keyword trends, and time-based frequency using NLP and statistical methods.

### Key Findings
- **Headline Length Distribution**: Most headlines were between 40–80 characters — suggesting brevity and focus.
- **Top Publishers**: Bloomberg, Reuters, MarketWatch were the most active sources.
- **Keyword Trends**: Common words included “price”, “target”, “report”, “rises” — indicating strong emphasis on performance updates.
- **Time Series Patterns**: Spikes in article frequency aligned with earnings releases and major market events.
- **Topic Modeling**: Identified recurring themes such as FDA approvals, price targets, and M&A activity.

### Challenges
- Some headlines had missing or malformed text — handled gracefully using error handling.
- Timezone conversion was necessary due to UTC-4 format.
- Domain extraction helped group email-based publishers.
- Some common words needed to be manually added to the stopword list.
- Balancing topic specificity vs generality required tuning.
- Short headlines made some topic inference harder than expected.

## Task 2: Quantitative Analysis with TA-Lib and PyNance

- Scripts: Contains `download_stock_data.py` to fetch stock prices.
- src/stock_analyzer.py: Implements `StockAnalyzer` for calculating technical indicators.
- notebooks/task-2_technical_indicators.ipynb: Notebook for visualizing indicators.