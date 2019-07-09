import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def price_rename(universe_dict):
  """Renaming the column of the dataframe values to price"""
  for df_name in universe_dict:
    df = universe_dict[df_name]
    df.sort_index(inplace=True)
    df = df.rename(columns={'value':"price"})
    df["lg_return"] = np.log(df.price) - np.log(df.price.shift(1))
    df["lg_return"].fillna(0, inplace=True)
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


def generate_dataset(universe_dict, lg_returns_only=False):
    """Generates the full dataset"""
    # Renames the columns with the name of the instrument series
    universe_dict = column_rename(universe_dict)
    universe = [] 
    for df_name in universe_dict: universe.append(universe_dict[df_name])
    df_full = pd.concat(universe, axis = 1)
    df_full.ffill(inplace=True)
    
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
        
    