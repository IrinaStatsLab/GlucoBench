from .scaler import Scaler
import pandas as pd
from statistics import stdev, mean


'''
Scaler that scales off standard deviation and mean.
'''
class StandardScaler(Scaler):
    def __init__(self):
        super().__init__()
        self.standard_deviation = None
        self.mean = None
    
    def fit(self, df: pd.DataFrame, column: str):
        self.standard_deviation = stdev(df[column].tolist())
        self.mean = mean(df[column].tolist())
        
    def transform(self, df: pd.DataFrame, column: str):
        df[column] = [(value - self.mean) / self.standard_deviation for value in df[column]]
        return df

    def fit_transform(self, df: pd.DataFrame, column: str):
        self.fit(df, column)
        return self.transform(df, column)