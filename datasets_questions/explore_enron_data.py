#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))


dataLength = len(enron_data)
print("The length of the dataset is "+str(dataLength))

#amount of attributes

def  poiTotal():
    poiSum = 0
    for person in enron_data:
        if enron_data[person]["poi"]==1:
            poiSum+=1
    return poiSum


def unknownSalaryTotal():
    ust = 0
    for person in enron_data:
        if enron_data[person]['salary']!='NaN':
            ust +=1
    return ust

def unknownEmailTotal():
    uet = 0
    for person in enron_data:
        if enron_data[person]['email_address']!='NaN':
            uet+=1
    return uet




print("The unknown salary total is "+str(unknownSalaryTotal()))

print("The unknown email total is "+str(unknownEmailTotal()))

#print(enron_data)


def prenticeStock():
    print(enron_data["PRENTICE JAMES"]['total_stock_value'])
#prenticeStock()


def colwellEmails():
    print(enron_data["COLWELL WESLEY"]["from_this_person_to_poi"])
#colwellEmails()

#print(enron_data)

def skillingOptions():
    print(enron_data["SKILLING JEFFREY K"]["exercised_stock_options"])


#skillingOptions()
import sys

sys.path.append("../final_project/")

poiNames = open("../final_project/poi_names.txt",'r')


def read_poiFile():
    poiTotal = 0
    for line in poiNames:
        if line[0]=="(":
            poiTotal+=1
    return poiTotal

print("The poi total is "+str(read_poiFile()))

#print(str(poiTotal()))