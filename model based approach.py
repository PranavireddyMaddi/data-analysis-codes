import pandas as pd
bluejet = pd.read_csv("D:\\CQ\\Forecasting\\bluejet.csv")
month =['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'] 
import numpy as np
p = bluejet["Month"][0]
p[0:3]
bluejet['months']= 0

for i in range(144):
    p = bluejet["Month"][i]
    bluejet['months'][i]= p[0:3]
    
month_dummies = pd.DataFrame(pd.get_dummies(bluejet['months']))
bluejet1 = pd.concat([bluejet,month_dummies],axis = 1)

bluejet1["t"] = np.arange(1,145)

bluejet1["t_squared"] = bluejet1["t"]*bluejet1["t"]
bluejet1.columns
bluejet1["log_flyers"] = np.log(bluejet1["flyers"])

bluejet1.flyers.plot()
Train = bluejet1.head(132)
Test = bluejet1.tail(12)

# to change the index value in pandas data frame 
# Test.set_index(np.arange(1,13))

####################### L I N E A R ##########################
import statsmodels.formula.api as smf 

linear_model = smf.ols('flyers~t',data=Train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['flyers'])-np.array(pred_linear))**2))
rmse_linear

##################### Exponential ##############################

Exp = smf.ols('log_flyers~t',data=Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test['flyers'])-np.array(np.exp(pred_Exp)))**2))
rmse_Exp

#################### Quadratic ###############################

Quad = smf.ols('flyers~t+t_squared',data=Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t","t_squared"]]))
rmse_Quad = np.sqrt(np.mean((np.array(Test['flyers'])-np.array(pred_Quad))**2))
rmse_Quad

################### Additive seasonality ########################

add_sea = smf.ols('flyers~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['flyers'])-np.array(pred_add_sea))**2))
rmse_add_sea

################## Additive Seasonality Quadratic ############################

add_sea_Quad = smf.ols('flyers~t+t_squared+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','t','t_squared']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['flyers'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad 

################## Multiplicative Seasonality ##################

Mul_sea = smf.ols('log_flyers~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','t','t_squared']]))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['flyers'])-np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea

##################Multiplicative Seasonality with linear trend ###########

Mul_Add_sea = smf.ols('log_flyers~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Test['flyers'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea 

################## Testing #######################################

data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
table_rmse=pd.DataFrame(data)
table_rmse
# so rmse_add_sea has the least value among the models prepared so far 
# Predicting new values 


model_full = smf.ols('flyers~t+t_squared+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=bluejet1).fit()

pred_data=pd.DataFrame()
pred_data["t"]=np.arange(145,157)
pred_data["t_squared"]=pred_data.t*pred_data.t
dummies=pd.get_dummies(month)
pred_data=pd.concat((pred_data,dummies),axis=1)

pred_new  = pd.Series(model_full.predict(pred_data))
pred_new

pred_data["forecasted_flyers"] = pred_new



#aurto regression


import statsmodels.graphics.tsaplots as tsa_plots
from statsmodels.tsa.arima_model import ARIMA


#calculate errors

errors=model_full.predict(bluejet1)-bluejet1.flyers


#ACF plot to look significance of errors
tsa_plots.plot_acf(errors,lags=12)

# from the acf plot we can see that lags of the errors having significant association, hence we can use errors for forecsting errors for next 12 timeperiods.

# Auto regreesion(p), from the ACF considering principal of parcimony we can take P=1


model_AR=ARIMA(errors,order=(1,0,0)).fit(disp=0)

model_AR.summary()

pred_data["forecasted_errors"]=pd.Series(model_AR.forecast(12)[0])
pred_data["improved_forecast"]=pred_data.forecasted_flyers+pred_data.forecasted_errors




#decomposition


import pandas as pd
from pandas import read_csv
from matplotlib import pyplot
from statsmodels.tsa.seasonal import seasonal_decompose
series = read_csv('bluejet.csv', header=0, index_col=0)
series.reset_index(inplace= True)
series["date"]=pd.to_datetime(series["Month"])
series=series.set_index("date")
result = seasonal_decompose(series.flyers, model='additive')
result.plot()
pyplot.show()
