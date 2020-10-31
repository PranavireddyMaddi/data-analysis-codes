
## Multiple linear Regression
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
from sklearn.model_selection import train_test_split#splitting the data
import statsmodels.formula.api as smf# regression
import statsmodels.api as sm  # influence index and added variable plot

df=pd.read_csv("D:\\CQ\\Linear Regression\\KPL.csv")

df.head(50)
df.size
df.shape
df.ndim
df.describe()

# scatter plot

sns.pairplot(df)

#correlation coefficient matrix

df.corr()


#Building the model with all variables - backward elimination

df.columns

mlr1 = smf.ols('KPL~BHP+bootspace+torque+weight',data=df).fit() # regression model

# Summary
mlr1.summary()

# preparing model based only on Bootspace
mlr_bp=smf.ols('KPL~bootspace',data = df).fit()  
mlr_bp.summary() 

# p-value <0.05 .. It is significant 

# Preparing model based only on Weight
mlr_w=smf.ols('KPL~weight',data = df).fit()  
mlr_w.summary() 

# p-value <0.05 .. It is significant 

# Preparing model based only on bootspace& Weight
ml_wv=smf.ols('KPL~bootspace+weight',data = df).fit()  
ml_wv.summary() 

# Both coefficients p-value became insignificant... 
#which says that they are involved in collinearity proble and one needs to be elimated.


#checking VIF values of input variables

# calculating VIF's values of independent variables
rsq_BHP = smf.ols('BHP~bootspace+torque+weight',data=df).fit().rsquared  
vif_BHP = 1/(1-rsq_BHP) 

rsq_wt = smf.ols('weight~BHP+bootspace+torque',data=df).fit().rsquared  
vif_wt = 1/(1-rsq_wt) 

rsq_bootspace = smf.ols('bootspace~BHP+weight+torque',data=df).fit().rsquared  
vif_bootspace = 1/(1-rsq_bootspace)

rsq_torque = smf.ols('torque~weight+bootspace+BHP',data=df).fit().rsquared  
vif_torque = 1/(1-rsq_torque) 

# VIF values in a data frame
d1 = {'Variables':['BHP', 'bootspace', 'torque', 'weight'],'VIF':[vif_BHP,vif_bootspace,vif_torque,vif_wt]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame
# As weight is having higher VIF value, we are not going to include this prediction model

# Added varible plot 
sm.graphics.plot_partregress_grid(mlr1)

#weight is not showing any contribution- it makes sense to remove weight and retain bootspace


# Checking whether data has any influential values 
# influence index plots

import statsmodels.api as sm
sm.graphics.influence_plot(mlr1)
# index 76 AND 78 is showing high influence so we can exclude that entire row

# Studentized Residuals = Residual/standard deviation of residuals

df_new = df.drop(df.index[[83,76]],axis=0) # ,inplace=False)

# Preparing final model                  

final_ml= smf.ols('KPL~BHP+bootspace+torque',data = df_new).fit()
final_ml.params
final_ml.summary() 

KPL_pred = final_ml.predict(df_new)

residuals= df_new.KPL-KPL_pred

residuals.mean()

import statsmodels.api as sm

# added variable plot for the final model
sm.graphics.plot_partregress_grid(final_ml)


######  Linearity #########
# Observed values VS Fitted values
plt.scatter(df_new.KPL,KPL_pred,c="r");plt.xlabel("observed_values");plt.ylabel("fitted_values")

# Residuals VS Fitted Values 
plt.scatter(KPL_pred,final_ml.resid_pearson,c="r");plt.axhline(y=0,color='blue');plt.xlabel("fitted_values");plt.ylabel("residuals")


########    Normality plot for residuals ######
# histogram
plt.hist(residuals) # Checking the standardized residuals are normally distributed






### Splitting the data into train and test data 

from sklearn.model_selection import train_test_split
df_train,df_test  = train_test_split(df_new,test_size = 0.2) # 20% size

# preparing the model on train data 

model_train = smf.ols("KPL~BHP+bootspace+torque",data=df_train).fit()

# train_data prediction
train_pred = model_train.predict(df_train)

# train residual values 
train_resid  = train_pred - df_train.KPL

# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid*train_resid))

# prediction on test data set 
test_pred = model_train.predict(df_test)

# test residual values 
test_resid  = test_pred - df_test.KPL

# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid*test_resid))

print(train_rmse,test_rmse)
