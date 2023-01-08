'''Defines a generic data formatter for CGM data sets.'''

import numpy as np
import pandas as pd
import sklearn.preprocessing
import data_formatter.types as types
import data_formatter.utils as utils

DataTypes = types.DataTypes
InputTypes = types.InputTypes

dict_data_type = {'categorical': DataTypes.CATEGORICAL,
                  'real_valued': DataTypes.REAL_VALUED,
                  'date': DataTypes.DATE}
dict_input_type = {'target': InputTypes.TARGET,
                   'observed_input': InputTypes.OBSERVED_INPUT,      
                   'known_input': InputTypes.KNOWN_INPUT,
                   'static_input': InputTypes.STATIC_INPUT,
                   'id': InputTypes.ID,
                   'time': InputTypes.TIME}


class DataFormatter():
  # Defines and formats data for the IGLU dataset.

  def __init__(self, cnf):
    """Initialises formatter."""
    # load parameters from the config file
    self.params = cnf
    
    # load column definition
    self.process_column_definition()

    # check that column definition is valid
    self.check_column_definition()

    # load data
    # check if data table has index col: -1 if not, index >= 0 if yes
    self.params['index_col'] = False if self.params['index_col'] == -1 else self.params['index_col']
    # read data table
    self.data = pd.read_csv(self.params['data_csv_path'], index_col=self.params['index_col'], na_filter=False)

    # check NA values
    self.check_nan()

    # set data types in DataFrame to match column definition
    self.set_data_types()

    # drop columns / rows
    self.drop()

    # encode
    self._encoding_params = self.params['encoding_params']
    self.encode()

    # interpolate
    self._interpolation_params = self.params['interpolation_params']
    self._interpolation_params['interval_length'] = self.params['observation_interval']
    self.interpolate()

    # split data
    self._split_params = self.params['split_params']
    self.split_data()

    # scale
    # self.train_data, self.val_data, self.test_data, self.scalers = self.scale()

  def process_column_definition(self):
    self._column_definition = []
    for col in self.params['column_definition']:
      self._column_definition.append((col['name'], 
                                      dict_data_type[col['data_type']], 
                                      dict_input_type[col['input_type']]))

  def check_column_definition(self):
    # check that there is unique ID column
    assert len([col for col in self._column_definition if col[2] == InputTypes.ID]) == 1, 'There must be exactly one ID column.'
    # check that there is unique time column
    assert len([col for col in self._column_definition if col[2] == InputTypes.TIME]) == 1, 'There must be exactly one time column.'
    # check that there is at least one target column
    assert len([col for col in self._column_definition if col[2] == InputTypes.TARGET]) >= 1, 'There must be at least one target column.'
  
  def set_data_types(self):
    # set time column as datetime format in pandas
    for col in self._column_definition:
      if col[1] == DataTypes.DATE:
        self.data[col[0]] = pd.to_datetime(self.data[col[0]])
      if col[1] == DataTypes.CATEGORICAL:
        self.data[col[0]] = self.data[col[0]].astype('category')
      if col[1] == DataTypes.REAL_VALUED:
        self.data[col[0]] = self.data[col[0]].astype('float')

  def check_nan(self):
    if self.params['nan_vals'] is not None:
      # replace NA values with pd.np.nan
      self.data = self.data.replace(self.params['nan_vals'], np.nan)
    # delete rows where target, time, or id are na
    self.data = self.data.dropna(subset=[col[0] 
                                  for col in self._column_definition 
                                  if col[2] in [InputTypes.TARGET, InputTypes.TIME, InputTypes.ID]])

  def drop(self):
    # drop columns that are not in the column definition
    self.data = self.data[[col[0] for col in self._column_definition]]
    # drop rows based on conditions set in the formatter
    if self.params['drop'] is not None:
      for col in self.params['drop'].keys():
        self.data = self.data.loc[~self.data[col].isin(self.params['drop'][col])].copy()
  
  def interpolate(self):
    self.data, self._column_definition = utils.interpolate(self.data, self._column_definition, **self._interpolation_params)

  def split_data(self):
    self.train_idx, self.val_idx, self.test_idx = utils.split(self.data, self._column_definition, **self._split_params)
    self.train_data, self.val_data, self.test_data = self.data.iloc[self.train_idx], self.data.iloc[self.val_idx], self.data.iloc[self.test_idx]

  def encode(self):
    self.data, self._column_definition, self.encoders = utils.encode(self.data, self._column_definition, **self._encoding_params)
  
  def scale(self):
    return utils.scale(self.data, self._column_definition, self.train_idx, self.val_idx, self.test_idx, **self.params['scaling_params'])
