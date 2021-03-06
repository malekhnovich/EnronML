#!/usr/bin/python

import pickle
import numpy
numpy.random.seed(42)


### The words (features) and authors (labels), already largely processed.
### These files should have been created from the previous (Lesson 10)
### mini-project.
words_file = "../text_learning/your_word_data.pkl"
authors_file = "../text_learning/your_email_authors.pkl"
word_data = pickle.load( open(words_file, "r"))
authors = pickle.load( open(authors_file, "r") )



### test_size is the percentage of events assigned to the test set (the
### remainder go into training)
### feature matrices changed to dense representations for compatibility with
### classifier functions in versions 0.15.2 and earlier
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(word_data, authors, test_size=0.1, random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
features_train = vectorizer.fit_transform(features_train)
features_test  = vectorizer.transform(features_test).toarray()


### a classic way to overfit is to use a small number
### of data points and a large number of features;
### train on only 150 events to put ourselves in this regime
features_train = features_train[:150].toarray()
labels_train   = labels_train[:150]



### your code goes here

from sklearn import tree

from sklearn.metrics import accuracy_score

tree = tree.DecisionTreeClassifier()


tree.fit(features_train,labels_train)

pred = tree.predict(features_test)

print("The amount of training points is", len(features_train))
print("The amount of labels points is ",len(labels_train))

print("The accuracy score is ",accuracy_score(labels_test,pred))

features_list = tree.feature_importances_
importantFeaturesDict = {}
featureCount = 0
for feature in features_list:
    if feature>0.2:
        importantFeaturesDict[featureCount]=feature
    featureCount=featureCount+1
print(importantFeaturesDict)


from sklearn.feature_extraction.text import TfidfVectorizer

featureNames = vectorizer.get_feature_names()
print(featureNames[30898])