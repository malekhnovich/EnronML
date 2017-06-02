#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
import matplotlib.pyplot
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import numpy as np

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".



#features_list = ['poi','stock_sum','bonus','salary','shared_receipt_with_poi'] # You will need to use more features

#variations of features_list that I tried
features_list = ['poi','total_poi_messages','salary','restricted_stock_deferred','restricted_stock','exercised_stock_options','bonus'] # You will need to use more features





#all features related to emails
email_features = ['poi','shared_receipt_with_poi','from_poi_to_this_person','from_this_person_to_poi']

new_features = ['total_poi_messages','total_comp','stock_sum']

### Load the dictionary containing the dataset

with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)




def basicStats():
    basicStats = {}
    poiCount = 0
    nonPoiCount = 0
    numberEmployees = len(data_dict)
    basicStats['number_employees'] = numberEmployees
    print("Number of employees INITIALLY ",numberEmployees)
    numberFeatures = len(data_dict['GRAMM WENDY L'])
    basicStats['number_features'] = numberFeatures
    print("Number of features for each employee is ",numberFeatures)
    for entry in data_dict:
        if data_dict[entry]['poi']:
            poiCount+=1
        else:
            nonPoiCount+=1
    print("There are "+str(poiCount)+" pois")




basicStats()




#test user

all_features = []


for feature in data_dict['LAY KENNETH L']:
    if feature!='email_address':

        all_features.append(feature)


    # want to plot the NaN values
    def findNanFeatures():
        NanEntries = {}
        for feature in all_features:
            NanEntries[feature] = 0
            for entry in data_dict:
                if data_dict[entry][feature]=='NaN':
                    NanEntries[feature]+=1
        return NanEntries



print("NaN count for each entry ",findNanFeatures())


### Task 2: Remove outliers
'''
def plotPoints():
    for entry in data_dict:
        salary = data_dict[entry]['salary']
        bonus = data_dict[entry]['bonus']

        matplotlib.pyplot.scatter(salary, bonus)
    # print(max)
    matplotlib.pyplot.xlabel("salary")
    matplotlib.pyplot.ylabel("bonus")
    matplotlib.pyplot.show()
#plotPoints()
'''


#plotting the data to see outliers


#from previous outlier mini-project, we know that one of the outliers is located at the key 'TITLE'
data_dict.pop('TOTAL')
#removing THE TRAVEL AGENCY IN THE PARK, too many 'NaN' values
data_dict.pop('THE TRAVEL AGENCY IN THE PARK')


#print(data_dict)


#trying to find out which entries have a lot of NaN
def getNanPeople():
    peopleNan = {}
    for entry in data_dict:
        nanCount = 0
        for criteria in data_dict[entry]:
            value =  data_dict[entry][criteria]
            if value=="NaN":
                nanCount =nanCount+1
        if nanCount>17:
            peopleNan[entry]=nanCount
    return peopleNan







NanPeople = getNanPeople()

print("there are "+str(len(NanPeople))+" who have too many NaN values")

#removing people who had more than 17 'NaN' in their data


def removeNanPeople():
    for entry in NanPeople:
        print("removing ",entry)
        data_dict.pop(entry)
    return data_dict

print("People being removed for having 18 or more NaN values",NanPeople)
print("\n")
data_dict = removeNanPeople()





### Task 3: Create new feature(s)
#creating a feature to cover the multiple emails being sent out
def totalPoiMessages():
    for entry in data_dict:
        if data_dict[entry]['from_poi_to_this_person']=="NaN"\
                                                 or data_dict[entry]['from_this_person_to_poi']=="NaN"\
                                                 or data_dict[entry]['shared_receipt_with_poi']=="NaN":
            data_dict[entry]['total_poi_messages'] = 'NaN'
        else:

            data_dict[entry]['total_poi_messages']  =data_dict[entry]['from_poi_to_this_person']\
                                                 +data_dict[entry]['from_this_person_to_poi']\
                                                 +data_dict[entry]['shared_receipt_with_poi']



#feature to measure compensation
#lowers the precision amount
def totalComp():
    for entry in data_dict:
        if data_dict[entry]['total_payments']=='NaN' or data_dict[entry]['total_stock_value']=='NaN':
            data_dict[entry]['total_comp'] = 'NaN'
        else:
            data_dict[entry]['total_comp'] = data_dict[entry]['total_payments']+data_dict[entry]['total_stock_value']


def stockValuesSum():
    for entry in data_dict:
        if data_dict[entry]['restricted_stock_deferred']== 'NaN'\
        or data_dict[entry]['total_stock_value'] =='NaN'\
        or data_dict[entry]['exercised_stock_options']=='NaN' \
        or data_dict[entry]['restricted_stock']=='NaN':
            data_dict[entry]['stock_sum'] = 'NaN'
        else:
            data_dict[entry]['stock_sum'] = data_dict[entry]['restricted_stock_deferred']+data_dict[entry]['total_stock_value']+\
            data_dict[entry]['exercised_stock_options']+data_dict[entry]['restricted_stock']




totalPoiMessages()
totalComp()
stockValuesSum()



### Store to my_dataset for easy export below.
my_dataset = data_dict



### Extract features and labels from dataset for local testing
#features_list = email_features



#features such as loan_advances, restricted_stock_deferred, and director_fees have too many NaN values

#putting all features into freatures_list




data = featureFormat(my_dataset, features_list,remove_NaN = True,remove_all_zeroes = True, sort_keys = True)
labels, features = targetFeatureSplit(data)




### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest,f_classif
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler



from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
train_test_split(features, labels, test_size=0.3, random_state=42)

def pcaResults():
    pca_x_values = []
    n_components=len(features_list)-1
    pca = PCA(n_components=n_components,whiten=True).fit(features_train)
    x_train_pca = pca.transform(features_train)
    x_test_pca = pca.transform(features_test)
    pca_x_values.append(x_train_pca)
    pca_x_values.append(x_test_pca)
    return pca_x_values

pca_x_values = pcaResults()



decisionTreeParams = {}
decisionTreeParams['dt_min_samples_split'] = [14,15,16]
decisionTreeParams['dt_presort'] = [True,False]
decisionTreeParams['dt_max_leaf_nodes'] = [None]
#decisionTreeParams['dt_max_depth'] = [None,10,8,6,4,2]

classifiers = {}
classifiers["dt"]=DecisionTreeClassifier(random_state=42)
classifiers["kneighbors"] = KNeighborsClassifier()
classifiers["svc"]=SVC(kernel='liner')
classifiers["gnb"] = GaussianNB()


classifierResults = {}



#using the decision Tree's 'tree.feature_importances_/gini Index to determine which features are important'
def findImportantFeatures(features_list):

    data = featureFormat(my_dataset, features_list, remove_NaN=True, remove_all_zeroes=True, sort_keys=True)
    labels, features = targetFeatureSplit(data)

    from sklearn.cross_validation import train_test_split
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.3, random_state=42)

    tree =  DecisionTreeClassifier(random_state=42)
    tree.fit(features_train, labels_train)
    features_list_importance = tree.feature_importances_

    importantFeaturesDict = {}
    featureCount = 0
    for feature in features_list_importance:
        if feature > 0.05:
            importantFeaturesDict[featureCount] =features_list[featureCount]

        featureCount = featureCount + 1

    print("\n")
    print(" using tree.feature_importance and ",features_list)
    print("features that have a value above 0.5", importantFeaturesDict)
    print("\n")
    return importantFeaturesDict


findImportantFeatures(all_features)
importantFeatures = findImportantFeatures(features_list)





def kBestResults():
    print(all_features)
    data = featureFormat(my_dataset, features_list, remove_NaN=True, remove_all_zeroes=True, sort_keys=True)
    labels, features = targetFeatureSplit(data)

    kbest = SelectKBest(f_classif, k='all')
    kbest.fit(features, labels)
    scores = -np.log10(kbest.pvalues_)
    valueIndex = {}
    valueCount = 0
    for score in scores:
        feat = features_list[valueCount]
        valueIndex[feat]=score
        valueCount = valueCount+1

    print("kbest results: ", scores)

kBestResults()






def checkClassifiers(trainedFeatures,testFeatures):
    classifier = 'dt'
    scaler = MinMaxScaler()
    kbest = SelectKBest(f_classif, k='all')
    if classifier =='dt':
        parameters = {'min_samples_split':[10,15,20,25],
                      'presort':[True,False],
                      'max_features':['auto','sqrt']
                      }
    elif classifier == "svc":
        parameters = {'C':[5,7,9],
                      'gamma':[5,7,9],
                      'kernel':['linear','rbf']
                      }
    elif classifier =="kneighbors":
        parameters = {'n_neighbors':[5],
                      'weights':['uniform','distance']
                      }
    elif classifier =="gnb":
        parameters = { 'priors':[None]
                     }
    print("Classifier : ",classifiers[classifier])
    print("parameter : ",parameters)
    gs = Pipeline([('scaling', scaler), ('kbest', kbest), (classifier, classifiers[classifier])])
    gclf = GridSearchCV(gs,parameters,scoring='f1')
    gclf.fit(features,labels)
    clf = gclf.best_estimator_
    prediction = gs.predict(features)
    print("The accuracy score of "+classifier+" is "+str(accuracy_score(prediction,labels)))
    print()

    print("best estimator returns",clf)


    return clf


print("Now lets see the results with Non-pca Set")
print("\n")
#classifierResults = checkClassifiers(features_train,features_test)



#clf = classifierResults["kneighbors"]



def checkKNeighbors():
    anova_filter = SelectKBest(k = 5)
    clf = SVC(kernel = 'linear')
    anova_svm = Pipeline([('anova',anova_filter),('svc',clf)])
    anova_svm.fit(features_train,labels_train)
    prediction = anova_svm.predict(features_test)
    print (anova_svm.score(labels_test,prediction))
    print("the accuracy score using Kneighbors is",accuracy_score(labels,prediction))

#checkKNeighbors()





#getting the accuracy with KNeighbors




### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)







def accuracyKNeighbors():
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(features_train,labels_train)
    prediction = clf.predict(features_test)
    #
    print("Using K Means Neighbors the accuracy on the labels is ",accuracy_score(labels_test,prediction))
    return clf


params = {}
'''
def accuracyGNB(features_train,features_test,labels_train,labels_test):
    clf = GaussianNB()
    clf.fit(features_train,labels_train)

    prediction=clf.predict(features_test)

    print("Using gaussian NB the accuracy on the labels is ",accuracy_score(labels_test,prediction))
    return clf








def accuracySVC(features_train,features_test,labels_train,labels_test):
    from sklearn import svm, grid_search
    svr = svm.SVC(kernel='linear')

    parameters = {'kernel': ('linear'), 'C': [3,5,]}


    svr.fit(features_train, labels_train)
    prediction = clf.predict(features_test)
    print("Using SVC the accuracy on the labels is ",accuracy_score(labels_test,prediction))

    return clf






#accuracyGNB(features_train,features_test,labels_train,labels_test)
#clf = accuracyKNeighbors(features_train,features_test,labels_train,labels_test)
#clf = accuracySVC(features_train,features_test,labels_train,labels_test)



'''



### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

clf = accuracyKNeighbors()


dump_classifier_and_data(clf, my_dataset, features_list)