# Importing database access information
from config import USER, PASSWORD, HOST, DB_NAME
import logging
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
       

if __name__ == "__main__":
    """Example of how to user the methods in ChaiDB"""
    db = ChaiDB()
    print(db.get_list_datascope_instruments())
    df = db.get_instrument_data('al_lme_prices')
    db.close_db_connection()
