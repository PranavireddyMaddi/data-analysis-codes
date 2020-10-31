import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

from statsmodels.tsa.holtwinters import SimpleExpSmoothing # SES
from statsmodels.tsa.holtwinters import Holt # Holts Exponential Smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing # 


# load the ts
bluejet = pd.read_csv("D:\\CQ\\Forecasting\\bluejet.csv")

#visualize the series
bluejet.flyers.plot() # time series plot 


# Centering moving average for the time series to understand better about the trend character in bluejet
bluejet.flyers.plot(label="org")
for i in range(2,24,6):
    bluejet["flyers"].rolling(14).mean().plot(label=str(14))

plt.legend(loc=6)

# splitting the data into Train and Test data and considering the last 12 months data as 
# Test data and left over data as train data 

Train = bluejet.head(132)
Test = bluejet.tail(12)
# Creating a function to calculate the MAPE value for test data 
def MAPE(pred,org):
    temp = np.abs((pred-org)/org)*100
    return np.mean(temp)

Train["flyers"]=Train.flyers.astype('double')
bluejet["flyers"]=bluejet.flyers.astype('double')
# Simple Exponential Method
help(SimpleExpSmoothing)
ses_model = SimpleExpSmoothing(Train["flyers"]).fit()
ses_model.summary()
pred_ses = ses_model.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_ses,Test.flyers) #14.25

# Holt method 
hw_model = Holt(Train["flyers"]).fit()
hw_model.summary()
pred_hw = hw_model.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hw,Test.flyers) #12.41

# Holts winter exponential smoothing with additive seasonality
hwe_model_add_add = ExponentialSmoothing(Train["flyers"],seasonal="add",trend="add",seasonal_periods=12).fit()
hwe_model_add_add.summary()
pred_hwe_add_add = hwe_model_add_add.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hwe_add_add,Test.flyers)#2.31

# Holts winter exponential smoothing with multiplicative seasonality and additive trend
hwe_model_mul_add = ExponentialSmoothing(Train["flyers"],seasonal="mul",trend="add",seasonal_periods=12).fit()
pred_hwe_mul_add = hwe_model_mul_add.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hwe_mul_add,Test.flyers)#2.37

#Final Model

hwe_model_add_add = ExponentialSmoothing(bluejet["flyers"],seasonal="add",trend="add",seasonal_periods=12).fit()

bluejet.set_index(np.arange(1,145),inplace=True)

pred_hwe_add_add = hwe_model_add_add.predict(start = bluejet.index[0],end = bluejet.index[-1])

MAPE(pred_hwe_add_add,bluejet.flyers)

#forecasts for next 12 months

pred_data= pd.DataFrame()

pred_data["month"]=pd.date_range(start="Jan-1961", periods=12,freq="MS")
#%B- complete month name
#%b- abb of month
#%Y- year
#%y- last two digits of years
#%d- day of the month

pred_data.month=pred_data.month.dt.strftime('%b-%y')
    
pred_data.set_index(np.arange(145,157),inplace=True)


pred_data["forecasted_flyers"]=pd.Series(hwe_model_add_add.predict(start = pred_data.index[0],end = pred_data.index[-1]))

















