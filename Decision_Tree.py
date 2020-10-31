import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("D:\\CQ\\iris.csv")
data.head()
data['Species'].unique()
data.Species.value_counts()
colnames = list(data.columns)

predictors = colnames[:4]
target = colnames[4]

# Splitting data into training and testing data set

import numpy as np

from sklearn.model_selection import train_test_split
train,test = train_test_split(data,test_size = 0.3)

from sklearn.tree import  DecisionTreeClassifier
help(DecisionTreeClassifier)

model = DecisionTreeClassifier(criterion = 'entropy')
model.fit(train.iloc[:,0:4],train.iloc[:,4])

pred_test = model.predict(test.iloc[:,0:4])
pred_train = model.predict(train.iloc[:,0:4])
pd.Series(test["Species"]).value_counts()
pd.crosstab(test["Species"],pred_test)
pd.crosstab(train["Species"],pred_train)
# Accuracy = train
np.mean(train.Species == pred_train)

# Accuracy = Test
np.mean(pred_test==test.Species) # 1

model.fit(iris.iloc[:,0:4],iris.iloc[:,4])
np.mean(model.predict(iris.iloc[:,0:4])==iris.iloc[:,4])
