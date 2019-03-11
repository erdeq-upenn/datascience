# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# Generate two clusters with labels using GaussianNB
import seaborn as sns
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


X, y = make_blobs(100, 2, centers=2, random_state=2, cluster_std=1.5)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='viridis');

#GaussianNB fit

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X, y);

#Generate some new random data

rng = np.random.RandomState(0)
Xnew = [-6, -14] + [14, 18] * rng.rand(2000, 2)
#Xnew = [-6,-10]+[20,20]*rng.rand(2000, 2)

ynew = model.predict(Xnew)

# plot new dataa and decide boundary
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='viridis')
lim = plt.axis()
plt.scatter(Xnew[:, 0], Xnew[:, 1], c=ynew, s=20, cmap='RdBu', alpha=0.1)
plt.axis(lim);

yprob = model.predict_proba(Xnew)
yprob[-8:].round(2)


#==============================================================================
# multinomial Bayes are good for counts in text classification where features
# are related to word counts for frequencies


from sklearn.datasets import fetch_20newsgroups

data = fetch_20newsgroups()
data.target_names
# For simplicity here, we will select just a few of these categories, 
#and download the training and testing set:
categories = ['talk.religion.misc', 'soc.religion.christian',
              'sci.space', 'comp.graphics']
train = fetch_20newsgroups(subset='train', categories=categories)
test = fetch_20newsgroups(subset='test', categories=categories)

# convert th content of each stirng into a vector of mumbers using TF-IDF vecterizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

model = make_pipeline(TfidfVectorizer(), MultinomialNB()) # pipeline

# fit and predict using this pipeliend model
model.fit(train.data, train.target)
labels = model.predict(test.data)

# visualization of test result on testdata and predicted lables
from sklearn.metrics import confusion_matrix
mat = confusion_matrix(test.target, labels)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=train.target_names, yticklabels=train.target_names)
plt.xlabel('true label')
plt.ylabel('predicted label');

# quick utility fucntion reture the prediction of a single string 
def predict_category(s, train=train, model=model):
    pred = model.predict([s])
    return train.target_names[pred[0]]

predict_category('sending a payload to the ISS')