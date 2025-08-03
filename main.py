import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
import string

news = pd.read_csv('news.csv')
prices = pd.read_csv('prices.csv')

def clean_text(text):
    tokens = word_tokenize(str(text).lower())
    tokens = [word for word in tokens if word.isalpha()]
    stop = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop]
    return ' '.join(tokens)

news['clean_text'] = news['headline'].apply(clean_text)
analyzer = SentimentIntensityAnalyzer()
news['sentiment'] = news['clean_text'].apply(lambda x: analyzer.polarity_scores(x)['compound'])
daily_sentiment = news.groupby('date')['sentiment'].mean().reset_index()
prices['price_change'] = prices['close'] - prices['close'].shift(1)
prices['direction'] = np.where(prices['price_change'] > 0, 1, 0)
prices['date'] = pd.to_datetime(prices['date'])
daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])
merged = pd.merge(prices, daily_sentiment, on='date', how='inner').dropna()
X = merged[['sentiment']]
y_reg = merged['price_change']
y_clf = merged['direction']
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.2, shuffle=False)
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_clf, test_size=0.2, shuffle=False)
reg = LinearRegression()
reg.fit(X_train_r, y_train_r)
pred_r = reg.predict(X_test_r)
rmse = np.sqrt(mean_squared_error(y_test_r, pred_r))
clf = LogisticRegression()
clf.fit(X_train_c, y_train_c)
pred_c = clf.predict(X_test_c)
acc = accuracy_score(y_test_c, pred_c)
cm = confusion_matrix(y_test_c, pred_c)
plt.figure(figsize=(10,5))
plt.plot(merged['date'], merged['sentiment'], label='Aggregated Sentiment')
plt.plot(merged['date'], merged['price_change'], label='Price Change')
plt.legend()
plt.title('Sentiment vs. Price Change (Time Series)')
plt.xlabel('Date')
plt.ylabel('Value')
plt.show()
plt.figure()
sns.scatterplot(x='sentiment', y='price_change', data=merged)
plt.title('Sentiment vs. Price Change (Scatter)')
plt.xlabel('Sentiment')
plt.ylabel('Price Change')
plt.show()
plt.figure()
plt.bar(['Regression RMSE', 'Classification Accuracy'], [rmse, acc])
plt.title('Model Performance Metrics')
plt.show()
