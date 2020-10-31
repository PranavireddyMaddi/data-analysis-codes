
###Logistic Regression - User_ data

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 


#Loading Dataset

dataset = pd.read_csv('D:\\CQ\\User_Data.csv') 

#removing the 1st column

dataset.drop("User ID",axis=1,inplace=True)

x=dataset.dropna()

# checking for missing values

dataset.isnull().sum()

#impute the missing values
dataset.Age= dataset.Age.fillna(dataset.Age.mean())
dataset.EstimatedSalary=dataset.EstimatedSalary.fillna(dataset.EstimatedSalary.mean())

#dummy variables

dataset=pd.get_dummies(dataset,drop_first=True)

dataset.describe()



# input 
x = dataset.iloc[:,[0,1,3]]

from sklearn.preprocessing import StandardScaler 
sc_x = StandardScaler() 
x_scaled = pd.DataFrame(sc_x.fit_transform(x.iloc[:,[0,1]]))
x_scaled.columns=["Age","EstimatedSalary"]
x_scaled["Gender"]=x.Gender_Male


# output 
y = dataset.iloc[:,2]

from sklearn.model_selection import train_test_split 
xtrain, xtest, ytrain, ytest = train_test_split( x_scaled, y, test_size = 0.30) 

from sklearn.linear_model import LogisticRegression 
classifier = LogisticRegression() 
classifier.fit(xtrain, ytrain) 

#After training the model, it time to use it to do prediction on testing data.


y_pred = classifier.predict(xtrain) 


#Let’s test the performance of our model – Confusion Matrix

from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(ytrain, y_pred) 

print ("Confusion Matrix : \n", cm) 

#Performance measure – Accuracy

from sklearn.metrics import accuracy_score 
print ("Accuracy : ", accuracy_score(ytrain, y_pred)) 




## using statsmodels

dataset.dropna(inplace=True)

from sklearn.model_selection import train_test_split

train,test=train_test_split(dataset,test_size=0.3)

#model
dataset.columns

import statsmodels.formula.api as sm
logit_model = sm.logit('Purchased~Age+EstimatedSalary',data = train).fit()

#summary
logit_model.summary()
y_pred = logit_model.predict(train)


train["pred_prob"] = y_pred
# Creating new column for storing predicted class

# filling all the cells with zeroes
train["pred_purchased"] = 0

# taking threshold value as 0.5 and above the prob value will be treated 
# as correct value 
train.loc[y_pred>=0.345,"pred_purchased"] = 1



from sklearn.metrics import classification_report
classification_report(train.Purchased,train.pred_purchased)



# confusion matrix 
confusion_matrix = pd.crosstab(train.Purchased,train.pred_purchased)

confusion_matrix
accuracy = (137+93)/(275) 
accuracy


# ROC curve 
from sklearn import metrics
# fpr => false positive rate
# tpr => true positive rate
fpr, tpr, threshold = metrics.roc_curve(train.Purchased,train.pred_prob)


# the above function is applicable for binary classification class 

plt.plot(fpr,tpr);plt.xlabel("False Positive");plt.ylabel("True Positive")

d1={"FPR":fpr,"TPR":tpr,"Threshold":threshold}

df1=pd.DataFrame(d1)

 
roc_auc = metrics.auc(fpr, tpr) # area under ROC curve 


