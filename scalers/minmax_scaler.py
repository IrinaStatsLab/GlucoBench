from .scaler import Scaler
import pandas as pd


'''
Scaler that performs min/max scaling.
'''
class MinMaxScaler(Scaler):
    def __init__(self):
        super().__init__()
    
    def fit(self, df: pd.DataFrame, column: str):
        self.min = df[column].min()
        self.max = df[column].max()
        
    def transform(self, df: pd.DataFrame, column: str):
        df[column] = [(value - self.min) / (self.max - self.min) for value in df[column]]
        return df

    def fit_transform(self, df: pd.DataFrame, column: str):
        self.fit(df, column)
        return self.transform(df, column)