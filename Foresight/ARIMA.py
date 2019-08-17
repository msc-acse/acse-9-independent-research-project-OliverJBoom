import os

if "preprocessing.py" in os.listdir():
    print("In correct directory")
elif "preprocessing.py" in os.listdir("../"):
    print("Moving up a layer")
    os.chdir("../")
else:
    print("Have a deeper look at where the notebook is situated"
         "relative to the python files")

import numpy as np
import os
import pandas as pd
from pmdarima import auto_arima
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from preprocessing import *
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima_model import ARIMA
import time
from eval_inspect import *
import warnings

warnings.filterwarnings("ignore")

def arima_forecaster(history, P, D, Q, forecast_length):
    """Fits an ARIMA model and makes a predicion one
    forecast length into the future

    :param history:      The training data the ARIMA model is fit to
    :type  history:      list

    :param P:            The autoregressive parameter for an ARIMA model
    :type  P:            int

    :param D:            The differencing parameter for an ARIMA model
    :type  D:            int

    :param Q:            The moving average parameter for an ARIMA model
    :type  Q:            int

    :return:             A predicition one time step into the future
    :rtype:              float
    """
    model = ARIMA(history, order=(P, D, Q))
    model_fit = model.fit(disp=0)
    prediction = model_fit.forecast(forecast_length)[0]
    return prediction


path = "Data/Commodity_Data/"
universe_dict = universe_select(path, "Cu", custom_list=None)
use_lg_returns = False
use_PCA = True
autoregressive = True

# Renaming the columns to price
universe_dict = price_rename(universe_dict)

# Cleaning the dataset of any erroneous datapoints
universe_dict = clean_dict_gen(universe_dict)

# Making sure that all the points in the window have consistent length
universe_dict = truncate_window_length(universe_dict)

# Generating the dataset
if use_lg_returns:
    # Lg Returns Only
    df_full = generate_dataset(universe_dict, lg_only=True, price_only=False)
    target_col = "cu_lme"

else:
    # Price Only
    df_full = generate_dataset(universe_dict, lg_only=False, price_only=True)
    target_col = "price_cu_lme"

if autoregressive:
    df_full = df_full[[target_col]]

# Data scaling
scaler_data_X = MinMaxScaler()

# Transforming the data
data_X = scaler_data_X.fit_transform(df_full)
# Making data 1D
data_X = data_X[:, 0]
print("Data X shape", data_X.shape)

auto = auto_arima(data_X)
P, D, Q = auto.order

# Training and test dataset
Train = data_X[0:int(len(data_X) * 0.6)]
Test = data_X[int(len(data_X) * 0.6):len(data_X)]

for forecast_length in [5, 22, 66, 132]:

    # The rolling window that will grow
    history = [x for x in Train]
    # Array to store predictions
    predictions = list()

    # Rolling Window Predictions
    for i in range(len(Test)-forecast_length+1):
        # Making the predictions
        pred = arima_forecaster(history, 0, 1, 2, forecast_length=forecast_length)[forecast_length-1]
        predictions.append(pred)
        # Keeping constant window size
        history.append(Test[i])
        del(history[0])

    # Matching the observed and predicted array sizes
    observed = Test[forecast_length - 1:]

    # Rescaling the results
    predictions = np.array(predictions).reshape(len(predictions), 1)
    observed  = np.array(observed).reshape(len(observed), 1)

    observed_scaled = scaler_data_X.inverse_transform(observed)
    predictions_scaled = scaler_data_X.inverse_transform(predictions)

    plt.plot(observed_scaled, label='observed')
    plt.plot(predictions_scaled, label='predictions')
    plt.legend()
    plt.grid()
    plt.show()

    mse, mae, mda = evaluate(predictions_scaled, observed_scaled, log_ret=False)
    print("Forecast Length:", forecast_length)
    print("Price Metrics: %i %i %.3f" % (mse, mae, mda))
