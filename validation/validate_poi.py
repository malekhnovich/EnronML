#!/usr/bin/python


"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

from sklearn.cross_validation import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### first element is our labels, any added elements are predictor
### features. Keep this the same for the mini-project, but you'll
### have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)


features_train, features_test, labels_train, labels_test =  train_test_split(features,labels,test_size=0.3, random_state=42)

#NOT THE WAY TO DO THINGS PROPERLY
clf =DecisionTreeClassifier()
clf.fit(features_train,labels_train)
prediction = clf.predict(features_test)
print("the accuracy score is ",accuracy_score(labels_test,prediction))

poiCount = 0
for element in prediction,labels_test:
    print(element)
print("number of poi is ",poiCount)
print("the number of elements in the set",len(prediction))
### it's all yours from here forward!  
#print(labels_test)

