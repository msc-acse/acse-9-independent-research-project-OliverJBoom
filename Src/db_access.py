# Importing database access information
from config import USER, PASSWORD, HOST, DB_NAME
import datetime
import logging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Postgre sql library
import psycopg2
from sqlalchemy import create_engine


class ChaiDB:
    """Class for accessing Chai database"""

    def __init__(self):
        """Set up the db connection upon initialisation."""
        logging.info('Creating DB connection')
        engine_string = 'postgresql+psycopg2://%s:%s@%s:5432/%s' % \
                        (USER, PASSWORD, HOST, DB_NAME)
        self.engine = create_engine(engine_string, pool_size=5)
        self.engine.raw_connection().set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
        self.connection = self.engine.connect()

    def get_instrument_data(self, instrument):
        """
        Get all instrument data and return in a dataframe.

        :param instrument Name of instrument to get data for e.g. ethylene_nymex_prices
        :return Dataframe containing instrument data, column names date and value
        """
        query = """SELECT date, value 
                    FROM new_instrument_data 
                    WHERE instrument = '%s' 
                    AND source = '%s'"""  % (instrument, 'DATASCOPE')
                    
        df = pd.read_sql_query(query, self.engine, index_col='date')
        df.columns = ['value']
        return df


    def close_db_connection(self):
        """Close the db connection"""
        logging.info('Closing DB connection')
        self.engine.connect().close()


    def get_list_datascope_instruments(self):
           """
           Gets the set of instruments we collect data for from Datascope
    
           :return: list of those instruments
           """
           query = """SELECT distinct(instrument) 
                      FROM new_instrument_data 
                      WHERE source = 'DATASCOPE'"""
                      
           df = pd.read_sql_query(query, self.engine)
           return df['instrument'].tolist()
       
        
def plot_instrument(universe_dict):
    """Plots the instrument of interest"""
    fig, axarr = plt.subplots(len(universe_dict), 1, figsize=(4, len(universe_dict) * 2))
    
    for ax, df_name in zip(axarr.flatten(), universe_dict):
        df = universe_dict[df_name]
        df = df.sort_index()
        ax.plot(df.index, df["value"])
        ax.set_title(df_name)
        ax.grid()
        ax.set_xlim([datetime.date(2018, 1, 1), datetime.date(2019, 1, 1)])
        ax.set_ylim([0.8 * df["value"].min(), 1.2 * df["value"].max()])
    plt.show()
      
    
def covariance_matrix(*args):
    """
    Args: Dataframes
        The time series data that are being cross-correlated
    Returns:
        The covariance matrix of the input dataframes
    Raises:
        Assertion error: If the dataframe do not contain the same
        quantity of points
    """
    for df in args[1:]:
        assert(len(df))==len(args[0])
        
    stacked_arrays = np.vstack(args)
    cov = np.corrcoef(stacked_arrays)
    return cov
    
def df_save(universe_dict):
    for df_name in universe_dict:
        universe_dict[df_name].to_csv("Data/" + df_name + ".csv")
    
    
if __name__ == "__main__":
    """Example of how to user the methods in ChaiDB"""
    db = ChaiDB()
    print(db.get_list_datascope_instruments())
    
    # Copper specific instruments
    cu_shfe = db.get_instrument_data('cu_shfe_prices')
    cu_lme = db.get_instrument_data('cu_lme_prices')
    cu_comex_p = db.get_instrument_data('cu_comex_prices')
    cu_comex_s = db.get_instrument_data('cu_comex_stocks')
    peso = db.get_instrument_data('chile_peso_spot')
    sol = db.get_instrument_data('peru_sol_spot')
    
    # Aluminium specific insturments
    al_shfe = db.get_instrument_data('al_shfe_prices')
    al_lme = db.get_instrument_data('al_lme_prices')
    al_comex_p = db.get_instrument_data('al_comex_prices')
    al_comex_s = db.get_instrument_data('al_comex_stocks')
    al_lme_s = db.get_instrument_data('al_lme_stocks')
    yuan = db.get_instrument_data('china_yuan_spot')
    
    # Generic instruments
    bdi = db.get_instrument_data('bdi')
    ted = db.get_instrument_data('ted')
    vix = db.get_instrument_data('vix')
    skew = db.get_instrument_data('skew')
    gsci = db.get_instrument_data('gsci')
    
    # Instruments for Cu
    copper_dict = {"cu_shfe":cu_shfe, "cu_lme":cu_lme, "cu_comex_p":cu_comex_p,
                   "cu_comex_s":cu_comex_s, "peso":peso, "sol":sol}
    
    # Instruments for Al
    aluminium_dict = {"al_shfe":al_shfe, "al_lme":al_lme, "al_comex_p":al_comex_p,
                      "al_comex_s":al_comex_s, "al_lme_s":al_lme_s, "yuan":yuan}
    
    # Instruments common to both cu and al
    generic_dict =  {"bdi":bdi, "ted":ted, "vix":vix, "skew":skew, "gsci":gsci}
    
    # Saving the dataframe
    df_save(copper_dict)
    df_save(aluminium_dict)
    df_save(generic_dict)
    
    # Visualising the different instuments
    plot_instrument(copper_dict)
    plot_instrument(aluminium_dict)
    plot_instrument(generic_dict)
    
#    covariance_matrix(df_cu_1["value"], df_cu_lme["value"])
    
    db.close_db_connection()
