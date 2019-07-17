import numpy as np
import pandas as pd


def universe_select(path, commodity_name):
    """Selects the instruments believed to be of
    interest for the commodity selected
    Returns: A dictionary of dataframes which are
    intruments of interest"""
    universe_dict = {}
    
    # If commodity is aluminium
    if commodity_name == "Al": 
        aluminium_list = ["al_shfe", "al_lme", "al_comex_p", "al_comex_s", "al_lme_s", "yuan",
                 "bdi", "ted", "vix", "skew", "gsci"]
        
        for instrument in aluminium_list:
            df = pd.read_csv(path + instrument + ".csv", index_col='date', parse_dates=['date'], dayfirst=True).sort_index(ascending=True)
            universe_dict[instrument] = df
     
    # If commodity is copper
    elif commodity_name == "Cu":
        copper_list = ["cu_shfe", "cu_lme", "cu_comex_p", "cu_comex_s", "peso", "sol",
                 "bdi", "ted", "vix", "skew", "gsci"]
        
        for instrument in copper_list:
            df = pd.read_csv(path + instrument + ".csv", index_col='date', parse_dates=['date'], dayfirst=True).sort_index(ascending=True)
            universe_dict[instrument] = df
    
    else: print("Select an appropriate commodity")
    return universe_dict


def price_rename(universe_dict):
  """Renaming the column of the dataframe values to price"""
  for df_name in universe_dict:
    df = universe_dict[df_name]
    df.sort_index(inplace=True)
    df = df.rename(columns={'value':"price"})
    universe_dict[df_name] = df
  return universe_dict


def clean_data(df, n_std = 20):
  """Removes any outliers that are further than a chosen
  number of standard deviations from the mean"""
  upper = df.price.mean() + n_std * (df.price.std())
  lower = df.price.mean() - n_std * (df.price.std())
  df.loc[((df.price > upper) | (df.price < lower)), 'price'] = None
  df.ffill(inplace=True)
  if df.price.isnull().sum() > 0: print("Rows removed:", df.price.isnull().sum())
  # Only want to keep business days
  df = df[df.index.weekday_name != "Saturday"]
  df = df[df.index.weekday_name != "Sunday"]
  return df


def clean_dict_gen(universe_dict):
  """Returns a dictionary of cleaned dataframes"""
  cleaned_dict = {}
  print("Included Instrument:")
  for df_name in universe_dict:
    print(df_name)
    cleaned_dict[df_name] = clean_data(universe_dict[df_name])
  return cleaned_dict


def truncate_window_length(universe_dict):
  """Chopping the length of all of the dataframes to ensure
  that they are all between the same dates
  Returns: A dictionary of the dataframes between equal dates"""
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
    universe_dict[df_name] = df.loc[((df.index <= min(end_date_arr)) & (df.index >= max(start_date_arr)))]
  
  return universe_dict


def column_rename(universe_dict):
    """Appends the name of the instrument
    name to the columns"""
    for df_name in universe_dict:
        for col in universe_dict[df_name].columns:
            universe_dict[df_name].rename(
            columns={col:col + "_" + df_name}, inplace=True)
    return universe_dict


def log_returns(x, lag=1):
    """Calculate log returns between adjacent close prices"""
    return np.log(x) - np.log(x.shift(lag))


def generate_lg_return(df_full, lag, target=True):
    """Returns a dictionary containing dataframes
    with the additional log returns column"""
    for col in df_full.columns:
        # Selecting out the dataframe of interest
        df = df_full[[col]]
        # Generating the lg returns for that dataframe
        df_returns = df.apply(log_returns, lag=lag)
        # Shifting the log returns to the training data
        df_returns_offset = df_returns.shift(-lag)
        if target: df_full[col.replace('price','target')] = df_returns_offset
        else: 
            if ("price" in col) == True:
                df_full[col.replace('price', str(lag) + "day_lg_return")] = log_returns(df[col], lag=lag)
                df_full.dropna(inplace=True)

    return df_full


def generate_dataset(universe_dict, lag=5):
    """Generates the full dataset"""
    # Renames the columns with the name of the instrument series
    universe_dict = column_rename(universe_dict)
    universe = [] 
    for df_name in universe_dict: universe.append(universe_dict[df_name])
    df_full = pd.concat(universe, axis = 1)
    # Must do log returns calculations after this forwards fill
    df_full.ffill(inplace=True)
    # Calculating the log returns
    df_full = generate_lg_return(df_full, lag)
    return df_full


def drop_prices(df):
    """Drops the prices column from the training dataset"""
    for col in df.columns: 
        if ("price" in col) == True: df.drop(columns=col, inplace=True)
    return df