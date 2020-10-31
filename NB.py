import pandas as pd
import numpy as np
######### Iris Data Set ########################
iris = pd.read_csv("D:\\CQ\\iris.csv")
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split


ip_columns = ["Sepal.Length","Sepal.Width","Petal.Length","Petal.Width"]
op_column  = ["Species"]

# Splitting data into train and test
Xtrain,Xtest,ytrain,ytest = train_test_split(iris[ip_columns],iris[op_column],test_size=0.3)

ignb = GaussianNB()

# Building and predicting at the same time 



predtrain_gnb=ignb.fit(train.iloc[:,0:4],train.iloc[:,4]).predict(train.iloc[0:4])

predtest_gnb=ignb.predict(test.iloc[:,0:4])
train_acc=np.mean(predtrain_gnb==train.iloc[:,4])
test_acc=np.mean(predtest_gnb==test["Species"])

ignb.fit(iris.iloc[:,0:4],iris.iloc[:,4])

np.mean(ignb.predict(iris.iloc[:,0:4])==iris["Species"])













############# Reading the Diabetes Data #################
Diabetes = pd.read_csv("E:\\Bokey\\Excelr Data\\Python Codes\\all_py\\Naive Bayes\\Diabetes_RF.csv")
colnames = list(Diabetes.columns)
predictors = colnames[:8]
target = colnames[8]
# Splitting data into training and testing 
DXtrain,DXtest,Dytrain,Dytest = train_test_split(Diabetes[predictors],Diabetes[target],test_size=0.3, random_state=0)
# Creating GaussianNB and MultinomialNB functions
Dgnb = GaussianNB()
Dmnb = MultinomialNB()
# Building and predicting at the same time 
Dpred_gnb = Dgnb.fit(DXtrain,Dytrain).predict(DXtest)
Dpred_mnb = Dmnb.fit(DXtrain,Dytrain).predict(DXtest)
# Confusion matrix 
confusion_matrix(Dytest,Dpred_gnb) 
print ("Accuracy",(138+38)/(138+38+19+36)) # 76.19 

confusion_matrix(Dytest,Dpred_mnb)
print ("Accuracy",(114+36)/(114+43+38+36)) # 64.93


################## Reading the Salary Data 
salary_train = pd.read_csv("E:\\Bokey\\Excelr Data\\Python Codes\\all_py\\Naive Bayes\\SalaryData_Train.csv")
salary_test = pd.read_csv("E:\\Bokey\\Excelr Data\\Python Codes\\all_py\\Naive Bayes\\SalaryData_Test.csv")
string_columns=["workclass","education","maritalstatus","occupation","relationship","race","sex","native"]

from sklearn import preprocessing
number = preprocessing.LabelEncoder()
for i in string_columns:
    salary_train[i] = number.fit_transform(salary_train[i])
    salary_test[i] = number.fit_transform(salary_test[i])

colnames = salary_train.columns
len(colnames[0:13])
trainX = salary_train[colnames[0:13]]
trainY = salary_train[colnames[13]]
testX  = salary_test[colnames[0:13]]
testY  = salary_test[colnames[13]]

sgnb = GaussianNB()
smnb = MultinomialNB()
spred_gnb = sgnb.fit(trainX,trainY).predict(testX)
confusion_matrix(testY,spred_gnb)
print ("Accuracy",(10759+1209)/(10759+601+2491+1209)) # 80%

spred_mnb = smnb.fit(trainX,trainY).predict(testX)
confusion_matrix(testY,spred_mnb)
print("Accuracy",(10891+780)/(10891+780+2920+780))  # 75%
