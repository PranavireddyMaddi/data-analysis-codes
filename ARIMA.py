import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.graphics.tsaplots as tsa_plots
from statsmodels.tsa.arima_model import ARIMA

bluejet = pd.read_csv("D:\\CQ\\Forecasting\\bluejet.csv")


tsa_plots.plot_acf(bluejet.flyers,lags=12)
tsa_plots.plot_pacf(bluejet.flyers,lags=12)


model1=ARIMA(bluejet.flyers,order=(1,1,1)).fit(disp=0)
model2=ARIMA(bluejet.flyers,order=(1,1,10)).fit(disp=0)
model1.aic
model2.aic

p=1
q=0
d=1
pdq=[]
aic=[]
for q in range(1,13):
    try:
        model=ARIMA(bluejet.flyers,order=(p,d,q)).fit(disp=0)

        x=model.aic

        x1= p,d,q
               
        aic.append(x)
        pdq.append(x1)
    except:
        pass
            
keys = pdq
values = aic


d = dict(zip(keys, values))
df= pd.DataFrame(d)
