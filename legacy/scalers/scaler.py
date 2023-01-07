import pandas as pd

'''
Parent class for the various scalers. Not meant to be used.
'''
class Scaler:
    def __init__(self):
        pass
    
    def fit(self, df: pd.DataFrame, column: str):
        '''
        Gets parameters used for scaling on a given dataset.

        Args:
            df: DataFrame to get scaling parameters.
            column: Name of column to scale by.
        
        Returns:
            None
        '''
        pass

    def transform(self, df: pd.DataFrame, column: str):
        '''
        Scales a dataset based on a previously fitted dataset.

        Args:
            df: DataFrame to scale on.
            column: Name of column to scale on.
        
        Returns:
            df: pd.Dataframe, scaled DataFrame
        '''
        pass

    def fit_transform(self, df: pd.DataFrame, column: str):
        '''
        Fits a DataFrame then scales it.

        Args:
            df: DataFrame to fit/scale on.
            column: Name of column to fit/scale on.
        
        Returns:
            df: pd.Dataframe, scaled DataFrame
        '''
        pass