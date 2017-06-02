#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
data_dict.pop("TOTAL",0)
features = ["salary", "bonus"]

data = featureFormat(data_dict, features)

#seeing what the data looks like
# print(data)

#print(data_dict)
### your code below
max = 0
for point in data:

    salary = point[0]

    bonus = point[1]
    matplotlib.pyplot.scatter(salary,bonus)
#print(max)
matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()



#print(data_dict)

for entry in data_dict:
    if data_dict[entry]["bonus"]!="NaN" and data_dict[entry]["salary"]!="NaN":
        if data_dict[entry]["bonus"] >=5000000 and data_dict[entry]["salary"]>1000000:
            print(entry)
        #data_dict.pop("entry",0)


#print(data_dict["GRAMM WENDY L"])