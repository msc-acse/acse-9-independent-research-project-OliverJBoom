"""
This module includes functions relating to the pre-processing of raw price
time series. They are used to create a dataset that can be used for deep
learning using long short term memory networks.

Author: Oliver Boom
Github Alias: OliverJBoom
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def universe_select(path, commodity_name, custom_list=None):
    """Selects the financial time series relevant for the commodities selected.

    :param path:            path to the folder containing csvs
    :type  path:            string

    :param commodity_name:  the name of the metal/s being inspected
    :type  commodity_name:  string

    :param custom_list:     the names of csvs to be included in the dataset
    :type  custom_list:     list

    :return:                The time series relevant to the commodities
    :rtype:                 dict
    """
    universe_dict = {}

    # Time series relevant to Aluminium
    if commodity_name == "Al":
        aluminium_list = ["al_shfe", "al_lme", "al_comex_p",
                          "al_comex_s", "al_lme_s", "yuan",
                          "bdi", "ted", "vix", "skew", "gsci"]

        for instrument in aluminium_list:
            # Sort by date
            df = pd.read_csv(path + instrument + ".csv",
                             index_col='date', parse_dates=['date'],
                             dayfirst=True).sort_index(ascending=True)

            universe_dict[instrument] = df

    # Time series relevant to Copper
    elif commodity_name == "Cu":
        copper_list = ["cu_shfe", "cu_lme", "cu_comex_p",
                       "cu_comex_s", "peso", "sol",
                       "bdi", "ted", "vix", "skew", "gsci"]

        for instrument in copper_list:
            df = pd.read_csv(path + instrument + ".csv",
                             index_col='date', parse_dates=['date'],
                             dayfirst=True).sort_index(ascending=True)

            universe_dict[instrument] = df

    # Multi Task Learning relevant to:
    # Copper, Aluminium, Tin, Lead, Nickle
    elif commodity_name == "MTL":
        metals_list = ["al_shfe", "al_lme", "al_comex_p",
                       "al_comex_s", "al_lme_s", "yuan",
                       "cu_shfe", "cu_lme", "cu_comex_p",
                       "cu_comex_s", "peso", "sol",
                       "bdi", "ted", "vix", "skew", "gsci",
                       "sn_lme", "pb_lme", "ni_lme"]

        for instrument in metals_list:
            df = pd.read_csv(path + instrument + ".csv",
                             index_col='date', parse_dates=['date'],
                             dayfirst=True).sort_index(ascending=True)

            universe_dict[instrument] = df

    elif commodity_name == "custom" and custom_list is not None:
        for instrument in custom_list:
            df = pd.read_csv(path + instrument + ".csv",
                             index_col='date', parse_dates=['date'],
                             dayfirst=True).sort_index(ascending=True)

            universe_dict[instrument] = df

    else:
        print("Select an appropriate commodity")
    return universe_dict


def price_rename(universe_dict):
    """Renaming the column of the DataFrame values to price.
    This is actually the market closing price of the time series.

    :param universe_dict:       The dictionary of time series
    :type  universe_dict:       dict

    :return:                    The dictionary of renamed time series
    :rtype:                     dict
    """
    for df_name in universe_dict:
        df = universe_dict[df_name]
        df.sort_index(inplace=True)
        df = df.rename(columns={'value': "price"})
        universe_dict[df_name] = df
    return universe_dict


def clean_data(df, n_std=20):
    """Removes any outliers that are further than a chosen
    number of standard deviations from the mean.

    These values are most likely wrongly inputted data,
    and so are forward filled.

    :param df:          A time series
    :type  df:          pd.DataFrame

    :param n_std:       The number of standard deviations from the mean
    :type  n_std:       int

    :return:            The cleaned time series
    :rtype:             pd.DataFrame
    """
    # Find the upper and lower bounds
    upper = df.price.mean() + n_std * (df.price.std())
    lower = df.price.mean() - n_std * (df.price.std())

    # Removing erroneous datapoints
    df.loc[((df.price > upper) | (df.price < lower)), 'price'] = None
    # Reporting the points cleaned
    if df.price.isnull().sum() > 0:
        print("Rows removed:", df.price.isnull().sum())

    # Replacing them with the previous days price
    df.ffill(inplace=True)

    # Only want to keep business days
    df = df[df.index.weekday_name != "Saturday"]
    df = df[df.index.weekday_name != "Sunday"]
    return df


def clean_dict_gen(universe_dict, verbose=True):
    """Generates a dictionary of cleaned DataFrames

    :param universe_dict:       The dictionary of time series
    :type  universe_dict:       dict

    :param verbose:             Whether to display the included instruments
    :type  verbose:             bool

    :return:                    The cleaned dictionary of time series
    :rtype:                     dict
    """
    cleaned_dict = {}
    if verbose:
        print("Included Instrument:")

    for df_name in universe_dict:
        if verbose:
            print(df_name)

        cleaned_dict[df_name] = clean_data(universe_dict[df_name])

    return cleaned_dict


def truncate_window_length(universe_dict):
    """Chopping the length of all of the DataFrames to ensure
    that they are all between the same dates.

    :param universe_dict:           The dictionary of time series
    :type  universe_dict:           dict

    :return:                        the dictionary of truncated time series
    :rtype:                         dict
    """
    start_date_arr = []
    end_date_arr = []

    for df_name in universe_dict:
        # Finding the latest of the start dates for each series
        start_date_arr.append(universe_dict[df_name].index[0])
        # Finding the earliest of the end dates
        end_date_arr.append(universe_dict[df_name].index[-1])

    for df_name in universe_dict:
        df = universe_dict[df_name]
        # Filters the DataFrames between these dates
        universe_dict[df_name] = df.loc[((df.index <= min(end_date_arr))
                                         & (df.index >= max(start_date_arr)))]

    return universe_dict


def column_rename(universe_dict):
    """Appends the name of the instrument to the columns.
    To help keep track of the instruments in the full dataset.

    :param universe_dict:               The dictionary of time series
    :type  universe_dict:               dict

    :return:                            The dictionary of time series
    :rtype:                             dict
    """
    for df_name in universe_dict:
        for col in universe_dict[df_name].columns:
            universe_dict[df_name].rename(
                columns={col: col + "_" + df_name}, inplace=True)

    return universe_dict


def log_returns(series, lag=1):
    """Calculates the log returns between adjacent close prices.
    A constant lag is used across the whole series.
    E.g a lag of one means a day to day log return.

    :param  series:                   Prices to calculate the log returns on
    :type   series:                   np.array

    :param  lag:                      The lag between the series (in days)
    :type   lag:                      int

    :return:                          The series of log returns
    :rtype:                           np.array
    """
    return np.log(series) - np.log(series.shift(lag))


def generate_lg_return(df_full, lag=1):
    """Creates the log return series for each column in the DataFrame
    and returns the full dataset with log returns.

    :param df_full:              The time series
    :type  df_full:              pd.DataFrame

    :param lag:                  The lag between the series (in days)
    :type  lag:                  int

    :return:                     The DataFrame of time series with log returns
    :rtype:                      pd.DataFrame
    """
    for col in df_full.columns:
        # Selecting out the dataframe of interest
        df = df_full[[col]]
        if lag == 1:
            df_full[col.replace('price_', "")] = log_returns(df[col], lag=lag)
            df_full.dropna(inplace=True)
        else:
            if "price" in col:
                df_full[col.replace('price', str(lag) + "_day_lg_return")] \
                    = log_returns(df[col], lag=lag)
                df_full.dropna(inplace=True)

    return df_full


def generate_dataset(universe_dict, price_only=True, lg_only=False):
    """Generates the full dataset.

    :param universe_dict:      The dictionary of time series
    :type  universe_dict:      dict

    :param lag:                The lag in days between series
    :type  lag:                int

    :param lg_only:            Whether to return a dataset of log returns only
    :type  lg_only:            bool

    :param price_only:         Whether to return a dataset of raw prices only
    :type  price_only:         bool

    :return:                   The time series
    :rtype:                    pd.DataFrame
    """
    # Can't have both log returns only and price only
    if lg_only:
        assert lg_only != price_only

    if price_only:
        assert lg_only != price_only

    # Renames the columns with the name of the time series
    # Naming pre-processing is done using a dict because of naming convenience
    # But then moved to DataFrame for further operations
    universe_dict = column_rename(universe_dict)
    universe = []

    for df_name in universe_dict:
        universe.append(universe_dict[df_name])

    # Creating the full dataset
    df_full = pd.concat(universe, axis=1)

    # Doing a forward fill to ensure that every time series has entries on
    # every day. (Different series might have gaps on different holidays)
    df_full.ffill(inplace=True)

    # Generating the log returns of each series
    df_full = generate_lg_return(df_full)

    # Dropping the price or log returns columns depending on preference
    if lg_only:
        df_full = df_full[df_full.columns.drop(
            list(df_full.filter(regex='price')))]

    if price_only:
        df_full = df_full[list(df_full.filter(regex='price'))]

    return df_full


def dimension_reduce(data_X, n_dim, verbose=True):
    """Performing PCA to reduce the dimensionality of the data.

    :param data_X:                  The dataset to perform reduction on
    :type  data_X:                  np.array

    :param n_dim:                   Number of dimensions to reduce to
    :type  n_dim:                   int

    :param verbose:                 Whether to display the explained variance
    :type  verbose:                 bool

    :return:                        The reduced dataset
    :rtype:                         np.array
    """
    # Determining how many dimensions will be reduced down to
    pca = PCA(n_components=n_dim)
    # Performing reduction
    data_X = pca.fit_transform(data_X)
    if verbose:
        print("Explained Variance Sum: %.3f\nExplained Variance Composition"
              % sum(pca.explained_variance_ratio_), pca.explained_variance_ratio_)

    return data_X


def dimension_selector(data_X, thresh=0.98, verbose=True):
    """Calculated the number of dimensions required to reach a threshold level
    of variance.

    Completes a PCA reduction to an increasing number of dimensions
    and calculates the total variance achieved for each reduction. If the
    reduction is above the threshold then that number of dimensions is returned

    :param data_X:                  The dataset to perform reduction on
    :type  data_X:                  np.array

    :param thresh:                  The amount of variance that must be
                                    contained the in reduced dataset

    :type  thresh:                  float

    :param verbose:                 Whether to display the number of dimensions
    :type  verbose:                 bool

    :return:                        The column dimensionality required to
                                    contain the threshold variance
    :rtype:                         int
    """
    n_dim = 1
    # data_X.shape[0] is the number of time series in the dataset
    while n_dim < data_X.shape[0]:
        # Completing a PCA reduction
        pca = PCA(n_components=n_dim)
        pca.fit(data_X)
        # Discerning if the total variance post reduction is adequate
        if sum(pca.explained_variance_ratio_) > thresh:
            if verbose:
                print("Number of dimensions:", n_dim)

            return n_dim

        n_dim += 1

    print("No level of dimensionality reaches threshold variance level %.3f"
          % sum(pca.explained_variance_ratio_))

    return None


def slice_series(data_X, data_y, series_len, dataset_pct=1.0):
    """Slices the train and target dataset time series.

    Turns each time series into a series of time series, with each series
    displaced by one step forward to the previous series. And for each
    of these windows there is an accompanying target value

    The effect of this is to create an array of time series (which is the depth
    equal to the amount of instruments in the dataset) with each entry in this
    array having a target series in the data_y array

    The resulting data_X array shape:
    [amount of rolling windows, length of each series, number of instruments]

    The resulting data_y array shape:
    [amount of rolling windows, number of instruments]

    :param data_X:              The dataset of time series
    :type  data_X:              np.array

    :param data_y:              The target dataset of time series
    :type  data_y:              np.array

    :param series_len:          The length of each time series window
    :type  series_len:          int

    :param dataset_pct:        The percentage of the full dataset to include
    :type  dataset_pct:        float

    :return:
    :rtype:
    """
    # Selecting the length of the full dataset
    length = int(len(data_X) * dataset_pct)

    data_X_ = []
    data_y_ = []

    for i in range(series_len, length):
        data_X_.append(data_X[i-series_len:i, :])
        data_y_.append(data_y[i])

    return np.array(data_X_), np.array(data_y_)


def feature_spawn(df):
    """Takes a time series and spawns several new features that explicitly
    detail information about the series.

    The DataFrame spawned contains the following features
    spawned for each column in the input DataFrame:

        Exponentially Weighted Moving Average of various Half Lives:
            1 day,
            1 week,
            1 month,
            1 quarter,
            6 months,
            1 year

        Rolling vol of different window sizes:
            1 week,
            1 month,
            1 quarter

    :param df:              The dataset of independent variables
    :type  df:              pd.DataFrame

    :return:                The DataFrame containing spawned features
    :rtype:                 pd.DataFrame
    """
    # The half live span to calculate the EWMA with
    hlf_dict = {"week": 5, "month": 22, "quarter": 66,
                "half_year": 132, "year": 260}

    # Spawning the signals for each column in the DataFrame
    for col in df.columns:
        for half_life in hlf_dict:
            df[col + "_ema_" + half_life] = \
                df[col].ewm(span=hlf_dict[half_life]).mean()

    for i, half_life in enumerate(hlf_dict):
        if i < 3:
            df[col + "_roll_vol_" + half_life] = \
                df[col].rolling(window=hlf_dict[half_life]).std(ddof=0)

    df.dropna(inplace=True)
    return df
