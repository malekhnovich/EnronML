#!/usr/bin/python

"""
    Starter code for the regression mini-project.
    
    Loads up/formats a modified version of the dataset
    (why modified?  we've removed some trouble points
    that you'll find yourself in the outliers mini-project).

    Draws a little scatterplot of the training/testing data

    You fill in the regression code where indicated:
"""    


import sys
import pickle
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
dictionary = pickle.load( open("../final_project/final_project_dataset_modified.pkl", "r") )





### list the features you want to look at--first item in the 
### list will be the "target" feature
features_list = ["bonus", "salary"]
#features_list = ["bonus","long_term_incentive"]

data = featureFormat( dictionary, features_list, remove_any_zeroes=True)
target, features = targetFeatureSplit( data )

### training-testing split needed in regression, just like classification
from sklearn.cross_validation import train_test_split
feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.5, random_state=42)
train_color = "b"
test_color = "r"



### Your regression goes here!
### Please name it reg, so that the plotting code below picks it up and 
### plots it correctly. Don't forget to change the test_color above from "b" to
### "r" to differentiate training points from test points.

from sklearn import linear_model
#creating the regression

reg = linear_model.LinearRegression()
#fitting the regression
reg.fit(feature_train,target_train)

#making a prediction correctly
prediction = reg.predict(feature_test)

#print("The correct prediction on the test data is ",prediction)

#making a prediction on the training data --INCORRECTLY
incorrectPrediction = reg.predict(feature_train)


#print("The result from incorrectly making a prediction on the training data is ", incorrectPrediction)

#finding the slope
slope =reg.coef_

#the slope with

print("The slope is ",slope)

#the y-intercept, y-value when x equals zero

y_intercept = reg.intercept_

print("The y-intercept is ",y_intercept)

from sklearn.metrics import r2_score

#correct score
sumOfSquaredError = r2_score(target_test,prediction)


#incorrect score
#sumOfIncorrectSquaredError = r2_score(target_test,incorrectPrediction)


print("The sum of squared error is",sumOfSquaredError)

#print("The sum of the incorrect squared error is ",sumOfIncorrectSquaredError)





### draw the scatterplot, with color-coded training and testing points
import matplotlib.pyplot as plt
for feature, target in zip(feature_test, target_test):
    plt.scatter( feature, target, color=test_color ) 
for feature, target in zip(feature_train, target_train):
    plt.scatter( feature, target, color=train_color ) 

### labels for the legend
plt.scatter(feature_test[0], target_test[0], color=test_color, label="test")
plt.scatter(feature_test[0], target_test[0], color=train_color, label="train")




### draw the regression line, once it's coded
try:
    plt.plot( feature_test, reg.predict(feature_test) )
except NameError:
    pass
reg.fit(feature_test, target_test)
slope_outliersIncluded = reg.coef_


print("The slope with outliers is ",slope_outliersIncluded)


plt.plot(feature_train, reg.predict(feature_train), color="r")

plt.xlabel(features_list[1])
plt.ylabel(features_list[0])
plt.legend()
plt.show()
