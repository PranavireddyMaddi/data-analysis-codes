#simple linear regression

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sb
import statsmodels.formula.api as smf

marketing = pd.read_csv("D:\\CQ\\Python\\datasets\\marketing.csv")

############## Exploratory Data Analysis ##################
marketing.shape
marketing.describe()
marketing.columns

plt.hist(marketing.youtube,color="r")
plt.boxplot(marketing.youtube)
help(plt.boxplot)

plt.hist(marketing["sales"])
plt.boxplot(marketing.sales)

marketing.isna().sum()

from statsmodels.graphics.gofplots import qqplot

qqplot(marketing.youtube,line = 's')
qqplot(marketing.sales,line = 's')
help(qqplot)

### Scatter Plot ###############
plt.scatter(marketing.youtube,marketing.sales);plt.xlabel("youtube");plt.ylabel("sales")

np.corrcoef(marketing.youtube,marketing.sales)



#### Model ####
model1 = smf.ols("sales~youtube",data=marketing).fit()
model1.summary()


# Co-efficients of the model equation 
model1.params


######## Prediction ##############
pred = model1.predict(marketing) # Predicted values of sales using the model
errors=marketing.sales-pred
np.mean(errors)
# Visualization of regresion line over the scatter plot of sales and Youtube

plt.scatter(x=marketing['youtube'],y=marketing['sales'],color='red');
plt.plot(marketing['youtube'],pred,color='black');
plt.xlabel('youtube');plt.ylabel('sales')

pred.corr(marketing.sales) # 0.99


# Transforming variables for accuracy
model2 = smf.ols('sales~np.log(youtube)',data=marketing).fit()
model2.params
model2.summary()
print(model2.conf_int(0.01)) # 99% confidence level
pred2 = model2.predict(marketing.youtube)
errors2= marketing.sales-pred2
np.mean(errors2)

pred2.corr(marketing.sales)
# pred2 = model2.predict(marketing.iloc[:,0])
pred2

plt.scatter(x=marketing['youtube'],y=marketing['sales'],color='red');
plt.plot(marketing['youtube'],pred2,color='black');
plt.xlabel('youtube');plt.ylabel('sales')

pred.corr(marketing.youtube) # 0.99

# Exponential transformation
model3 = smf.ols('np.log(sales)~youtube',data=marketing).fit()
model3.params
model3.summary()
print(model3.conf_int(0.01)) # 99% confidence level
pred_log = model3.predict(pd.DataFrame(marketing['youtube']))
pred_log
pred3=np.exp(pred_log)  # as we have used log(sales) in preparing model so we need to convert it back
pred3
pred3.corr(marketing.sales)
plt.scatter(x=marketing['youtube'],y=marketing['sales'],color='green');
plt.plot(marketing.youtube,pred3,color='blue');
plt.xlabel('youtube');plt.ylabel('TISSUE')
resid_3 = pred3-marketing.sales
np.mean(resid_3)

# so we will consider the model having highest R-Squared value which is the log transformation - model3
# getting residuals of the entire data set
student_resid = model3.resid_pearson 
student_resid
plt.plot(model3.resid_pearson,'o');
plt.axhline(y=0,color='green');
plt.xlabel("Observation Number");
plt.ylabel("Standardized Residual")

# Predicted vs actual values
plt.scatter(x=pred3,y=marketing.sales);plt.xlabel("Predicted");plt.ylabel("Actual")

plt.scatter(marketing.youtube,np.log(marketing.sales))

# Quadratic model
marketing["youtube_Sq"] = marketing.youtube*marketing.youtube
model_quad = smf.ols("np.log(sales)~youtube+youtube_Sq",data=marketing).fit()
model_quad.params
model_quad.summary()

pred= model_quad.predict(pd.DataFrame(marketing))
pred_quad=np.exp(pred)
pred_quad.corr(marketing.sales)


model_quad.conf_int(0.05) # 
plt.scatter(marketing.youtube,marketing.sales);
plt.plot(marketing.youtube,pred_quad,"r");
plt.show()

plt.scatter(marketing.sales,pred_quad.predictions)
np.corrcoef(marketing.sales,pred_quad)


def RMSE(y,ypred):
    err=y-ypred
    temp=np.sqrt(np.mean(err*err))
    print(temp)
    
    
RMSE(marketing.sales,pred)
RMSE(marketing.sales,pred2)
RMSE(marketing.sales,pred3)
RMSE(marketing.sales,pred_quad)


plt.scatter(np.arange(109),model_quad.resid_pearson);plt.axhline(y=0,color='red');plt.xlabel("Observation Number");plt.ylabel("Standardized Residual")
plt.show()
plt.hist(model_quad.resid_pearson) # histogram for residual values 






from sklearn.model_selection import train_test_split

train,test=train_test_split(marketing,test_size=0.3)


model_train=smf.ols("np.log(sales)~youtube+youtube_Sq",data=train).fit()


model_train.summary()

train_pred= np.exp(model_train.predict(train))

train_errors= train.sales-train_pred

RMSE(train.sales,train_pred)

test_pred= np.exp(model_train.predict(test))


RMSE(test.sales,test_pred)
