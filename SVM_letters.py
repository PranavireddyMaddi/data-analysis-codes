import pandas as pd 
import numpy as np 

letters = pd.read_csv("D:\\CQ\\Python\\datasets\\letters.csv")
letters.head()
letters.describe()
letters.columns
letters.lettr.nunique()
letters.lettr.value_counts()
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
train,test = train_test_split(letters,test_size = 0.3)

train_X = train.iloc[:,1:]
train_y = train.iloc[:,0]
test_X  = test.iloc[:,1:]
test_y  = test.iloc[:,0]

# Create SVM classification object 
# 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'

# kernel = linear
help(SVC)
model_linear = SVC(kernel = "linear")
model_linear.fit(train_X,train_y)
pred_test_linear = model_linear.predict(test_X)
pred_train_linear = model_linear.predict(train_X)
np.mean(pred_test_linear==test_y) # Accuracy 
np.mean(pred_train_linear==train_y)
# Kernel = poly
model_poly = SVC(kernel = "poly")
model_poly.fit(train_X,train_y)
pred_test_poly = model_poly.predict(test_X)
np.mean(model_poly.predict(train_X)==train_y)
np.mean(pred_test_poly==test_y) # 

# kernel = rbf
model_rbf = SVC()
model_rbf.fit(train_X,train_y)
pred_test_rbf = model_rbf.predict(test_X)

np.mean(pred_test_rbf==test_y) # Accuracy
np.mean(model_rbf.predict(train_X)==train_y)


model_rbf.fit(letters.iloc[:,1:],letters.iloc[:,0])
np.mean(model_rbf.predict(letters.iloc[:,1:])==letters.iloc[:,0])
