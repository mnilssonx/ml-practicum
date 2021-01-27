import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import math
from fbprophet import Prophet
import matplotlib.pyplot as plt

data = pd.read_csv('airpassengers.csv', parse_dates=['Month'])
data.rename(columns={'Month': 'ds', 'Passengers': 'y'}, inplace=True)
#print(data.head())

train, test = train_test_split(data, shuffle=False)

m = Prophet()
m.fit(train)
#future = m.make_future_dataframe(len(test.index), freq='M')
#forecast = m.predict(future)
#forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
#m.plot(forecast)
#plt.show()

test_forecast = m.predict(test[['ds']])
mse = mean_squared_error(test_forecast['yhat'].values, test['y'].values)
print('\nRMSE: %f' % math.sqrt(mse))