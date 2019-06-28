# Part 7: Natural language processing

## Section 27: Natural Language Processing
**Natural Language Processing** is an area of computer science and artificial intelligence concerned with the interactions between computers and human (natural) languages. NLP is used to apply ML models to text and language. Teach machines to understand what is said in spoken and written word is the focus of NLP. Whenever you dictate something to your phone that is then converted to a text, that’s an NLP algorithm in action.

NLP can be used for:
* Sentiment analysis. Identifying the mood or subjective opinions within large amounts of text, including average sentiment and opinion minig.
* Use it to predict the genre of the book.
* Question answering
* Use NLP to build a machine translator or a speech recognition system
* Document summarization

We have an input tsv(tab separated values) file with two colums - Review which contains atext of a review and Liked which tells us if it is a positive or negative review.
The model we are going to implement is called Bag of Words. It doesn’t work with all words, so texts have to be cleaned. For example we have to delete all articles from there, change all tenses of verbs to the present tense and get rid of capitals.
Аfter that we start tokenization process, which will calculate number of times each word appears in a review.
When it is done our data is ready and the problem can be treated as a clssification one.
Usually people use Decision Tree or Naive Bayes models for NLP tasks.

```python
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('../data_files/Restaurant_Reviews.tsv', delimiter='\t', quoting=3)

# Cleaning the texts
import re
import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []
for i in range(0, len(dataset['Review'])):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    stopword_set = set(stopwords.words('english'))
    review = [ps.stem(word) for word in review if not word in stopword_set]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Fitting classifier to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
```
[Next Part >>>](8_deep_learning.md)
