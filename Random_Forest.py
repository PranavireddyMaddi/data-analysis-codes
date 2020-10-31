import pandas as pd
import numpy as np
# Reading the Iris Data #################
data = pd.read_csv("D:\\CQ\\iris.csv")
data.head()
data.columns
colnames = list(data.columns)
predictors = colnames[:4]
target = colnames[4]

X = data[predictors]
Y = data[target]

from sklearn.model_selection import train_test_split

trainx,testx,trainy,testy=train_test_split(X,Y,test_size=0.3)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_jobs=4,oob_score=True,n_estimators=15,criterion="entropy")
# n_estimators -> Number of trees ( you can increase for better accuracy)
# n_jobs -> Parallelization of the computing and signifies the number of jobs 
# running parallel for both fit and predict
# oob_score = True means model has done out of box sampling to make predictions


#### Attributes that comes along with RandomForest function
rf.fit(trainx,trainy) # Fitting RandomForestClassifier model from sklearn.ensemble 
rf.estimators_ # 
rf.classes_ # class labels (output)
rf.n_classes_ # Number of levels in class labels 
rf.n_features_  # Number of input features in model 8 here.

rf.n_outputs_ # Number of outputs when fit performed

rf.oob_score_  # 0.72916

np.mean(rf.predict(trainx)==trainy)# training accuracy
np.mean(rf.predict(testx)==testy)#testing accuracy

##############################

rf.fit(X,Y)
np.mean(rf.predict(X)==Y)













####################### SALARY Data #################

salary_train = pd.read_csv("D:\\ML\\Python\\Python-ML\\Random Forests\\SalaryData_Train.csv")
salary_test = pd.read_csv("D:\\ML\\Python\\Python-ML\\Random Forests\\SalaryData_Test.csv")

colnames = salary_train.columns
len(colnames[0:13])
trainX = salary_train[colnames[0:13]]
trainY = salary_train[colnames[13]]
rfsalary = RandomForestClassifier(n_jobs=2,oob_score=True,n_estimators=15,criterion="entropy")
rfsalary.fit(trainX,trainY) # Error Can not convert a string into float means we have to use LabelEncoder()

# Considering only the string data type columns and 
string_columns=["workclass","education","maritalstatus","occupation","relationship","race","sex","native"]
from sklearn import preprocessing
for i in string_columns:
    number = preprocessing.LabelEncoder()
    trainX[i] = number.fit_transform(trainX[i])

rfsalary.fit(trainX,trainY)
# Training Accuracy
trainX["rf_pred"] = rfsalary.predict(trainX)
confusion_matrix(trainY,trainX["rf_pred"]) # Confusion matrix
# Accuracy
print ("Accuracy",(22321+6954)/(22321+332+554+6954)) # 97.06

# Accuracy on testing data 
testX = salary_test[colnames[0:13]]
testY = salary_test[colnames[13]]
# Converting the string values in testing data into float
for i in string_columns:
    number = preprocessing.LabelEncoder()
    testX[i] = number.fit_transform(testX[i])
testX["rf_pred"] = rfsalary.predict(testX)
confusion_matrix(testY,testX["rf_pred"])
# Accuracy 
print ("Accuracy",(10359+2283)/(10359+1001+1417+2283)) # 83.94

import pandas as pd 
from sklearn import datasets






