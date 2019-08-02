import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error


def check_length(universe_dict):
    """Checks the name of all the dataframes in the 
    dictionary of instruments

    :param universe_dict: a dictionary of financial time series
    :type universe_dict: dict
    """
    for df_name in universe_dict:
        print(len(universe_dict[df_name]))
    
    
def visualise_df(df):
  """Visualises the features for an instrument

  :param df: the time series to visualise
  :type df: pd.DataFrame
  """
  fig, axarr = plt.subplots(int(len(df.columns) / 2), 2, figsize=(4 * 10, 4 * len(df.columns)))

  for ax, df_name in zip(axarr.flatten(), df.columns):
        ax.set_title(df_name)
        ax.plot(df.index, df[df_name])
        ax.grid()
        ax.legend()

plt.show()

  
def visualise_universe(universe_dict):
    """Plots the price and log return for every 
    instrument in the univese dictionary

    :param universe_dict: a dictionary of financial time series to visualise
    :type universe_dict: dict
    """
    for df_name in universe_dict:
        visualise_df(universe_dict[df_name])
            
    
def check_day_frequency(df, day_col_name='ds'):
    """Returns a barchart showing the frequency of the 
    days of the week within a dataframe

    :param df: the time series to visualise
    :type df: pd.DataFrame
    """
    df["day"] = df[day_col_name].apply(lambda x: x.weekday_name)
    print(df['day'].value_counts())
    df['day'].value_counts().plot(kind='bar')
    
    
def df_std(df, col_name):
    """Returns the standard deviation of a dataframes column

    :param df: a dataframe of time series
    :type df: pd.DataFrame

    :param col_name: the column of interest
    :type df: pd.DataFrame

    :return: the standard deviation of the series on interest
    :rtype: float
    """
    return df[[col_name]].stack().std()


def inverse_log_returns(original_prices, log_returns, lag=5, shift=0):
    """Takes a dataframes of predicted log returns and original 
    prices and returns an array of predicted absolute prices

    :param original_prices: a dataframe of absolute prices
    :type original_prices: pd.DataFrame

    :param log_returns: a dataframe of log returns
    :type log_returns: pd.DataFrame

    :param lag: the lag duration of the log returns
    :type lag: int

    :param shift: whether to offset the series forwards of backwards
    :type shift: int

    :return: the raw prices indicated by the log returns
    :rtype:  pd.Series
    """
    assert isinstance(log_returns, pd.DataFrame)
    assert isinstance(original_prices, pd.DataFrame)
    # shift is for 
    if shift == 0:
        return (original_prices.shift(shift).values[:-lag] * np.exp(log_returns[:-lag])).values.ravel()
    else: 
        return (original_prices.shift(shift).values * np.exp(log_returns)).values.ravel()


def mean_absolute_percentage_error(y_true, y_pred):
    """Calculated the mean absolute percentage error metric 
    between two arrays

    :param y_true: The observed values
    :type y_true:  pd.Series

    :param y_pred: The predicted values
    :type y_pred: pd.Series

    :return: The mean absolute percentage error of the series
    :rtype:  float
    """
    assert isinstance(y_true, pd.core.series.Series)
    assert isinstance(y_pred, pd.core.series.Series)
    return 100 * np.mean(np.abs(y_true - y_pred) / y_true)


def mean_directional_accuracy(y_true, y_pred):
    """Calculated the mean directional accuracy
    error metric between two series

    :param y_true: The observed values
    :type y_true:  pd.Series

    :param y_pred: The predicted values
    :type y_pred: pd.Series

    :return: The mean direcional accuracy of the series
    :rtype:  float
    """
    assert isinstance(y_true, pd.core.series.Series)
    assert isinstance(y_pred, pd.core.series.Series)
    return np.mean(np.sign(y_true - y_true.shift(1)) == np.sign(y_pred - y_pred.shift(1)))


def evaluate(df, y_orig_col, y_pred_col):
    """Calculated the error metric for a dataframe 
    of predictions and observed values

    :param df: The dataframe containing the predicted and observed values
    :type df: pd.DataFrame

    :param y_orig_col: The observed values column name
    :type y_orig_col:  pd.Series

    :param y_pred_col: The predicted values column name
    :type y_pred_col: pd.Series

    :return mse, mae, mde: Returns the mean squared error, mean absolute accuracy and mean directional accuracy metrics
    :rtype: float
    """
    assert isinstance(df, pd.DataFrame)
    assert isinstance(y_orig_col, str)
    assert isinstance(y_pred_col, str)
    # Dropping first NA column. Needs to be dropped because
    # evaluate criteria can't deal with nans or zeros
    df = df.dropna()
    
    mse = mean_squared_error(df[y_orig_col], df[y_pred_col])
    mae = mean_absolute_error(df[y_orig_col], df[y_pred_col])
    mde = mean_directional_accuracy(df[y_orig_col], df[y_pred_col])
    return mse, mae, mde



