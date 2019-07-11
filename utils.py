import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error


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


def generate_lg_return(df):
    """Returns a dictionary containing dataframes
    with the additional log returns column"""
    for col in df.columns:
        df[col.replace('price','lg_return')] = np.log(df[col]) - np.log(df[col].shift(1))
        df[col.replace('price','lg_return')].fillna(0, inplace=True)
    return df


def column_rename(universe_dict):
    """Appends the name of the instrument
    name to the columns"""
    for df_name in universe_dict:
        for col in universe_dict[df_name].columns:
            universe_dict[df_name].rename(
            columns={col:col + "_" + df_name}, inplace=True)
    return universe_dict


def generate_dataset(universe_dict, lg_returns_only=False):
    """Generates the full dataset"""
    # Renames the columns with the name of the instrument series
    universe_dict = column_rename(universe_dict)
    universe = [] 
    for df_name in universe_dict: universe.append(universe_dict[df_name])
    df_full = pd.concat(universe, axis = 1)
    # Must do log returns calculations after this forwards fill
    df_full.ffill(inplace=True)
    # Calculating the log returns
    df_full = generate_lg_return(df_full)
    
    if lg_returns_only == True:
        for col in df_full.columns: 
            if ("price" in col) == True: df_full.drop(columns=col, inplace=True)
            
    return df_full


def check_length(universe_dict):
    """Checks the name of all the dataframes in the 
    dictionary of instruments"""
    for df_name in universe_dict: print(len(universe_dict[df_name]))
    
    
def visualise_df(df):
  """Visualises the features for an instrument"""
  fig, axarr = plt.subplots(int(len(df.columns) / 2), 2, figsize=(4 * 10, 4 * len(df.columns)))

  for ax, df_name in zip(axarr.flatten(), df.columns):
      ax.set_title(df_name)
      ax.plot(df.index, df[df_name])
      ax.grid()
      ax.legend()
      
  plt.show()
  
  
def visualise_universe(universe_dict):
    """Plots the price and log return for every 
    instrument in the univese dictionary"""
    for df_name in universe_dict: visualise_df(universe_dict[df_name])
            
    
def check_day_frequency(df, day_col_name='ds'):
    """Returns a barchart showing the frequency of the 
    days of the week within a dataframe"""
    df["day"] = df[day_col_name].apply(lambda x: x.weekday_name)
    print(df['day'].value_counts())
    df['day'].value_counts().plot(kind='bar')
    
    
def df_std(df, col_name):
    """Returns the standard deviation of a dataframes column"""
    return df[[col_name]].stack().std()


def inverse_log_returns(log_returns, starting_val):
    """Takes an array of log returns and returns the
    prices for a given starting point"""
    factor = np.exp(np.roll(log_returns, -1))
    inv_prices = [starting_val]
    
    for i in range(len(factor) - 1):
        inv_prices.append(factor[i] * inv_prices[-1])
    return inv_prices


def mean_absolute_percentage_error(y_true, y_pred):
    """Calculated the mean absolute percentage error metric 
    between two arrays"""
    return 100 * np.mean(np.abs(y_true - y_pred) / y_true)


def mean_directional_accuracy(y_true, y_pred):
    """Calculated the mean directional accuracy
    error metric between two arrays"""
    return np.mean((np.sign(y_true) == np.sign(y_pred)))


def evaluate(df):
    """Calculated the error metric for a dataframe 
    of predictions and observed values
    Returns: Dataframe with the following additional columns
    MSE: Mean Squared Error
    MAE: Mean Absolute Error
    MDE: Mean Directional Error"""
    df["MSE"] = mean_squared_error(df["y_orig"], df["y_pred"])
    df["MAE"] = mean_absolute_error(df["y_orig"], df["y_pred"])
    df["MDE"] = mean_directional_accuracy(df["y_orig"], df["y_pred"])
    return df