import numpy as np
import pandas as pd
import sklearn.preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from wordcloud import WordCloud

sns.set()

data = pd.read_csv(r"C:\Users\oshij\OneDrive\Pictures\Desktop\Assignment Data.csv")
df = data.copy()

sns.scatterplot(x="country", y="winery", data=df)
plt.show()

df = df.astype(str, errors='raise')
print(df.dtypes)

missing_values = ["NaN"]
df = pd.read_csv(r"C:\Users\oshij\OneDrive\Pictures\Desktop\Assignment Data.csv", na_values=missing_values)
print(df.isnull().sum())

median = df['price'].median()
df['price'].fillna(median, inplace=True)
print(df[['price']].isnull().sum())

df = df.fillna(df.mode().iloc[0])
print(df.isnull().sum())

print(df.shape)

dfs = pd.DataFrame(data=df, columns=['description', 'variety'])
df = dfs.copy()
print(df)

print(df['variety'].unique())
print(df['description'].unique())

df['sentiment'] = df['description'].apply(lambda x: TextBlob(x).sentiment.polarity)


def get_sentiment(score):
    if score > 0:
        return 'Positive'
    elif score < 0:
        return 'Negative'
    else:
        return 'Neutral'


df['sentiment_class'] = df['sentiment'].apply(get_sentiment)

sns.countplot(x='sentiment_class', data=df)
plt.show()

all_words = ' '.join(df['description'])
wordcloud = WordCloud(width=800, height=400).generate(all_words)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

df['description'] = df['description']
X = df['description']
df['variety'] = df['variety']
y = df['variety']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0, shuffle=True)
from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
print(X_train_counts.shape)

# TF-IDF
from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print(X_train_tfidf.shape)

from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB().fit(X_train_tfidf, y_train)
from sklearn.pipeline import Pipeline

text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])

text_clf = text_clf.fit(X_train, y_train)

import numpy as np

predicted = text_clf.predict(X_test)
print("Accuracy using Naive Bayes Algorithm:")

from sklearn.metrics import accuracy_score

accuracy_dt = accuracy_score(y_test, predicted)
print(format(accuracy_dt))

from sklearn.linear_model import SGDClassifier

text_clf_svm = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                         ('clf-svm', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, max_iter=5,
                                                   random_state=42))])
text_clf_svm = text_clf_svm.fit(X_train, y_train)

predicted_svm = text_clf_svm.predict(X_test)
accuracy_svc = np.mean(predicted_svm == y_test)
print("Accuracy after training dataset using SVM algorithm:", accuracy_svc)

from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("english", ignore_stopwords=True)

# Stemming Code
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("english", ignore_stopwords=True)

class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([stemmer.stem(w) for w in analyzer(doc)])


stemmed_count_vect = StemmedCountVectorizer(stop_words='english')

from sklearn.pipeline import Pipeline

text_clf = Pipeline([('vect', CountVectorizer(stop_words='english')), ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB())])
text_mnb_stemmed = Pipeline([('vect', stemmed_count_vect), ('tfidf', TfidfTransformer()),
                             ('mnb', MultinomialNB(fit_prior=False))])
text_mnb_stemmed = text_mnb_stemmed.fit(X_train, y_train)
predicted_mnb_stemmed = text_mnb_stemmed.predict(X_test)
accuracy = np.mean(predicted_mnb_stemmed == y_test)
print("Accuracy of Naive Bayes algorithm after stemming:", accuracy)

from sklearn.linear_model import SGDClassifier

text_clf_svm = Pipeline([('vect', CountVectorizer(stop_words='english')), ('tfidf', TfidfTransformer()),
                         ('clf-svm', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, max_iter=5,
                                                   random_state=42))])

text_clf_svm = text_clf_svm.fit(X_train, y_train)

predicted_svm = text_clf_svm.predict(X_test)
accuracy_1 = np.mean(predicted_svm == y_test)
print("Accuracy of SVM algorithm after stemming:", accuracy_1)

print("Accuracy Comparison for positive data")
comparison_dict = {"Algorithm": ["Naive Bayes Algorithm", "Support Vector Machine"],
                   "Accuracy": [accuracy_dt, accuracy_svc]}

comparison = pd.DataFrame(comparison_dict)

print(comparison.sort_values(['Accuracy'], ascending=False))

# After removing stop words comparison
comparison_dict = {"Algorithm": ["Naive Bayes Algorithm", "Support Vector Machine"],
                   "Accuracy": [accuracy, accuracy_1]}

comparison = pd.DataFrame(comparison_dict)
print("Accuracy Comparison for Negative Sentiment")
print(comparison.sort_values(['Accuracy'], ascending=False))
sentiment_input = input("Enter sentiment type (positive/negative/neutral): ")

if sentiment_input.lower() == 'positive':
    filtered_df = df[df['sentiment_class'] == 'Positive']
elif sentiment_input.lower() == 'negative':
    filtered_df = df[df['sentiment_class'] == 'Negative']
elif sentiment_input.lower() == 'neutral':
    filtered_df = df[df['sentiment_class'] == 'Neutral']
else:
    print("Invalid sentiment type. Please enter 'positive', 'negative', or 'neutral'.")
    exit()

print(filtered_df.head())

user_input = input("Enter a text: ")

user_sentiment = get_sentiment(TextBlob(user_input).sentiment.polarity)

print("Sentiment: ", user_sentiment)
