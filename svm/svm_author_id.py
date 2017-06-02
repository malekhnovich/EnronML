#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


from sklearn import svm

#########################################################
### your code goes here ###

#creating the classifier for the SVM

features_train = features_train[:len(features_train)/100]
labels_train = labels_train[:len(labels_train)/100]

clf=svm.SVC(kernel="linear")

#fit the svm with the features (X) and labels(Y)
clf.fit(features_train, labels_train)

#making the prediction
prediction = clf.predict(features_test)

t0 = time()
#importing accuracy

from sklearn.metrics import accuracy_score
#printing the accuracy value
print("The accuracy score is "+str(((accuracy_score(prediction,labels_test)))))

print("training time:", round(time()-t0, 3), "s")

#########################################################


