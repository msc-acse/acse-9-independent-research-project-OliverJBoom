import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def universe_select(path, commodity_name):
    """Selects the instruments believed to be of
    interest for the commodity selected
    
    :param path: path to the csv folder
    :param commodity_name: the name of the metal being inspected
    
    :type path: type string
    :type commodity_name: type string
    
    :return: financial time series relevant to the commodity
    :rtype: dict
    """
    
    universe_dict = {}

    # If commodity is aluminium
    if commodity_name == "Al":
        aluminium_list = ["al_shfe", "al_lme", "al_comex_p",
                          "al_comex_s", "al_lme_s", "yuan",
                          "bdi", "ted", "vix", "skew", "gsci"]

        for instrument in aluminium_list:
            df = pd.read_csv(path + instrument + ".csv",
                             index_col='date', parse_dates=['date'],
                             dayfirst=True).sort_index(ascending=True)

            universe_dict[instrument] = df

    # If commodity is copper
    elif commodity_name == "Cu":
        copper_list = ["cu_shfe", "cu_lme", "cu_comex_p",
                       "cu_comex_s", "peso", "sol",
                       "bdi", "ted", "vix", "skew", "gsci"]

        for instrument in copper_list:
            df = pd.read_csv(path + instrument + ".csv",
                             index_col='date', parse_dates=['date'],
                             dayfirst=True).sort_index(ascending=True)

            universe_dict[instrument] = df

    else:
        print("Select an appropriate commodity")
    return universe_dict


def price_rename(universe_dict):
    """
    Renaming the column of the dataframe values to price
    
    :param universe_dict: financial time series
    :type universe_dict: dict
    
    :return: financial time series
    :rtype: dict 
    """
    for df_name in universe_dict:
        df = universe_dict[df_name]
        df.sort_index(inplace=True)
        df = df.rename(columns={'value': "price"})
        universe_dict[df_name] = df
    return universe_dict


def clean_data(df, n_std=20):
    """
    Removes any outliers that are further than a chosen
    number of standard deviations from the mean
    
    :param df: the finacial time series
    :type df: dataframe
    
    :param n_std: the number of standard deviations from the mean
    :type n_std: int
    
    :return: the cleaned financial time series
    :rtype: dataframe
    """
    upper = df.price.mean() + n_std * (df.price.std())
    lower = df.price.mean() - n_std * (df.price.std())
    df.loc[((df.price > upper) | (df.price < lower)), 'price'] = None
    df.ffill(inplace=True)

    if df.price.isnull().sum() > 0:
        print("Rows removed:", df.price.isnull().sum())

    # Only want to keep business days
    df = df[df.index.weekday_name != "Saturday"]
    df = df[df.index.weekday_name != "Sunday"]
    return df


def clean_dict_gen(universe_dict):
    """
    Returns a dictionary of cleaned dataframes
    
    :param universe_dict: the financial time series
    :type universe_dict: dict
    
    :return: the cleaned financial time series
    :rtype: dict
    """
    cleaned_dict = {}
    print("Included Instrument:")

    for df_name in universe_dict:
        print(df_name)
        cleaned_dict[df_name] = clean_data(universe_dict[df_name])

    return cleaned_dict


def truncate_window_length(universe_dict):
    """
    Chopping the length of all of the dataframes to ensure
    that they are all between the same dates
    
    :param universe_dict: the financial time series
    :type universe_dict: dict
    
    :return: the truncated financial time series
    :rtype: dict
    """
    start_date_arr = []
    end_date_arr = []

    for df_name in universe_dict:
        # Finding the latest of the start dates
        start_date_arr.append(universe_dict[df_name].index[0])
        # Finding the earliest of the end dates
        end_date_arr.append(universe_dict[df_name].index[-1])

    for i, df_name in enumerate(universe_dict):
        df = universe_dict[df_name]
        # Filters the dataframe between these dates
        universe_dict[df_name] = df.loc[((df.index <= min(end_date_arr))
                                         & (df.index >= max(start_date_arr)))]

    return universe_dict


def column_rename(universe_dict):
    """Appends the name of the instrument
    name to the columns
    
    :param universe_dict: the financial time series
    :type universe_dict: dict
    
    :return: the financial time series
    :rtype: dict
    """
    for df_name in universe_dict:
        for col in universe_dict[df_name].columns:
            universe_dict[df_name].rename(
                columns={col: col + "_" + df_name}, inplace=True)

    return universe_dict


def log_returns(series, lag=1):
    """Calculate log returns between adjacent close prices
    
    :param series: prices to calculate the log returns on
    :type series: numpy array
    
    :param lag: the amount of days the returns are calculated between
    :type lag: int
    
    :return: the series of log returns
    :rtype: numpy array
    """
    return np.log(series) - np.log(series.shift(lag))


def generate_target(df_full, target_col="price_cu_lme", lag=5):
    """Generate the target variable"""
    df_target = df_full[[target_col]].apply(log_returns, lag=lag)
    df_target = df_target.shift(-lag)
    df_target.rename(columns={"price_cu_lme": target_col.replace(
        "price_", str(lag) + "_day_forecast_")}, inplace=True)

    return df_target


def generate_lg_return(df_full, lag=1):
    """Returns a dictionary containing dataframes
    with the additional log returns column"""
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


def generate_dataset(universe_dict, lag=5,
                     lg_returns_only=True, price_only=False):
    """Generates the full dataset"""
    if lg_returns_only:
        assert (lg_returns_only != price_only)

    if price_only:
        assert (lg_returns_only != price_only)

    # Renames the columns with the name of the instrument series
    universe_dict = column_rename(universe_dict)
    universe = []

    for df_name in universe_dict:
        universe.append(universe_dict[df_name])

    df_full = pd.concat(universe, axis=1)
    # Must do log returns calculations after this forwards fill
    df_full.ffill(inplace=True)
    # Calculating the log returns
    df_full = generate_lg_return(df_full)

    # Fill in nan to allow inverse calculations
    df_full["target"] = np.nan
    # As target is forecast backdated the first row values should have
    # value and the last should have nulls
    df_full["target"][:-lag] = generate_target(df_full,
                                               target_col="price_cu_lme",
                                               lag=5)[:-lag].values.ravel()

    if lg_returns_only:
        df_full = df_full[df_full.columns.drop(list(df_full.filter(regex='price')))]

    if price_only:
        df_full = df_full[list(df_full.filter(regex='price')) + ['target']]

    return df_full


def drop_prices(df):
    """Drops the prices column from the training dataset"""
    for col in df.columns:
        if "price" in col:
            df.drop(columns=col, inplace=True)
    return df


def dimension_reduce(df, n_dim):
    """Performing PCA to reduce the amount of"""
    pca = PCA(n_components=n_dim)
    pca.fit(df)
    df_reduced = pca.transform(df)
    print("Explained Variance:", pca.explained_variance_ratio_,
          "\nExplained Variance Sum:", sum(pca.explained_variance_ratio_))
    return pd.DataFrame(df_reduced, index=df.index)


def dimension_selector(df, thresh=0.98):
    """Returns the number of dimensions that reaches the 
    threshold level of desired variance"""
    for n_dim in range(1, 11):
        pca = PCA(n_components=n_dim)
        pca.fit(df)
        if sum(pca.explained_variance_ratio_) > thresh:
            print("Number of dimensions:", n_dim)
            return n_dim
    print("No level of dimensionality reaches threshold variance level %.3f"
          % sum(pca.explained_variance_ratio_))
    return None
