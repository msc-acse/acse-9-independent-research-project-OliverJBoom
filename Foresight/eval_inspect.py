"""
This module include a set of functions that are used to evaluate and
inspect the time series in the dataset.

Author: Oliver Boom
Github Alias: OliverJBoom
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error


def check_length(universe_dict):
    """Checks the name of all the DataFrames in the dictionary of time series.

    :param universe_dict:       The dictionary of time series
    :type  universe_dict:       dict
    """
    for df_name in universe_dict:
        print(len(universe_dict[df_name]))


def visualise_df(df):
    """Visualises each time series in a DataFrame.

    :param df:                  The DataFrame of time series to visualise
    :type  df:                  pd.DataFrame
    """
    _, ax_arr = plt.subplots(int(len(df.columns) / 2), 2,
                             figsize=(4 * 10, 4 * len(df.columns)))

    for axes, df_name in zip(ax_arr.flatten(), df.columns):
        axes.set_title(df_name)
        axes.plot(df.index, df[df_name])
        axes.grid()
        axes.legend()

    plt.show()


def check_day_frequency(df, col_name='ds'):
    """Creates a bar chart showing the frequency of the days of the week.

    Used to check that only business days are included in the dataset, and
    that there is a roughly equal distribution of entries across the week.

    :param df:               A DataFrame containing the time series to check
    :type  df:               pd.DataFrame

    :param col_name:     The name of the column of interest
    :type  col_name:     string
    """
    df["day"] = df[col_name].apply(lambda x: x.weekday_name)
    print(df['day'].value_counts())
    df['day'].value_counts().plot(kind='bar')


def df_std(df, col_name):
    """Calculates standard deviation of a DataFrames column.

    :param df:                    A DataFrame of time series
    :type  df:                    pd.DataFrame

    :param col_name:              The column of interest
    :type  col_name:              string

    :return:                      The standard deviation of the series
    :rtype:                       float
    """
    return df[[col_name]].stack().std()


def inverse_log_returns(original_prices, log_returns, lag=5, offset=0):
    """Takes a DataFrame of predicted log returns and original
    prices and returns an array of predicted absolute prices

    The offset parameter moves the series forwards or backwards to
    align the series with the DataFrame it might be appended to.

    :param original_prices:  A DataFrame of absolute prices
    :type  original_prices:  pd.DataFrame

    :param log_returns:      A DataFrame of log returns
    :type  log_returns:      pd.DataFrame

    :param lag:              The lag in days between series
    :type  lag:              int

    :param offset:          Amount to offset the series forwards of backwards
    :type  offset:          int

    :return:                The raw prices given by the log returns
    :rtype:                 pd.Series
    """
    assert isinstance(log_returns, pd.DataFrame)
    assert isinstance(original_prices, pd.DataFrame)
    if offset == 0:
        return (original_prices.shift(offset).values[:-lag] *
                np.exp(log_returns[:-lag])).values.ravel()

    return (original_prices.shift(offset).values *
            np.exp(log_returns)).values.ravel()


def mean_absolute_percentage_error(y_true, y_pred):
    """Calculates the mean absolute percentage error between two arrays.

    :param y_true:            The observed values
    :type  y_true:            np.array

    :param y_pred:            The predicted values
    :type  y_pred:            np.array

    :return:                  The mean absolute percentage error of the series
    :rtype:                   float
    """
    return 100 * np.mean(np.abs(y_true - y_pred) / y_true)


def evaluate(y_true, y_pred, log_ret=False):
    """Calculates the error metrics for between two arrays.

    The error metrics calculated are:
        Means Squared Error
        Mean Absolute Error
        Mean Directional Accuracy

    For a log returns series the definition of mean directional accuracy
    changes. This is as for a log return series it is the signum values of the
    series that details which direction the series has moved. This is as a log
    return series is the first difference of the original series. For raw
    price. The signal needs to be differenced before the signum function
    is applied.


    :param y_true:            The observed values
    :type  y_true:            np.array

    :param y_pred:            The predicted values
    :type  y_pred:            np.array

    :param log_ret:           Whether the series compared are log returns
    :type  log_ret:           bool

    :return error_metrics:    The error metrics of the series
    :rtype:                   List
    """
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    if log_ret:
        mda = mean_directional_accuracy_log_ret(y_true, y_pred)
    else:
        mda = mean_directional_accuracy(y_true, y_pred)

    error_metrics = [mse, mae, mda]
    return error_metrics


def mean_directional_accuracy_log_ret(y_true, y_pred):
    """Calculates the mean directional accuracy error metric between
    two series of log returns.

    :param y_true:           The observed values
    :type  y_true:           np.array

    :param y_pred:           The predicted values
    :type  y_pred:           np.array

    :return:                 The mean directional accuracy of the series
    :rtype:                  float
    """
    return np.mean(np.sign(y_true) == np.sign(y_pred))


def mean_directional_accuracy(y_true, y_pred):
    """Calculated the mean directional accuracy error metric
    between two series.

    :param y_true:           The observed values
    :type  y_true:           np.array

    :param y_pred:           The predicted values
    :type  y_pred:           np.array

    :return:                 The mean directional accuracy of the series
    :rtype:                  float
    """
    return np.mean(np.sign(y_pred[1:, :] - y_pred[:-1, :])
                   == np.sign(y_true[1:, :] - y_true[:-1, :]))
