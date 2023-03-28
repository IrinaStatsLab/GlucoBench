'''Defines a generic data formatter for CGM data sets.'''
import sys
import warnings
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

  def __init__(self, cnf, study_file = None):
    """Initialises formatter."""
    # load parameters from the config file
    self.params = cnf
    # write progress to file if specified
    self.study_file = study_file
    stdout = sys.stdout
    f = open(study_file, 'a') if study_file is not None else sys.stdout
    sys.stdout = f

    # load column definition
    print('-'*32)
    print('Loading column definition...')
    self.__process_column_definition()

    # check that column definition is valid
    print('Checking column definition...')
    self.__check_column_definition()

    # load data
    # check if data table has index col: -1 if not, index >= 0 if yes
    print('Loading data...')
    self.params['index_col'] = False if self.params['index_col'] == -1 else self.params['index_col']
    # read data table
    self.data = pd.read_csv(self.params['data_csv_path'], index_col=self.params['index_col'])

    # drop columns / rows
    print('Dropping columns / rows...')
    self.__drop()

    # check NA values
    print('Checking for NA values...')
    self.__check_nan()

    # set data types in DataFrame to match column definition
    print('Setting data types...')
    self.__set_data_types()

    # drop columns / rows
    print('Dropping columns / rows...')
    self.__drop()

    # encode
    print('Encoding data...')
    self._encoding_params = self.params['encoding_params']
    self.__encode()

    # interpolate
    print('Interpolating data...')
    self._interpolation_params = self.params['interpolation_params']
    self._interpolation_params['interval_length'] = self.params['observation_interval']
    self.__interpolate()
    
    # split data
    print('Splitting data...')
    self._split_params = self.params['split_params']
    self._split_params['max_length_input'] = self.params['max_length_input']
    self.__split_data()

    # scale
    print('Scaling data...')
    self._scaling_params = self.params['scaling_params']
    self.__scale()

    print('Data formatting complete.')
    print('-'*32)
    if study_file is not None:
      f.close()
      sys.stdout = stdout


  def __process_column_definition(self):
    self._column_definition = []
    for col in self.params['column_definition']:
      self._column_definition.append((col['name'], 
                                      dict_data_type[col['data_type']], 
                                      dict_input_type[col['input_type']]))

  def __check_column_definition(self):
    # check that there is unique ID column
    assert len([col for col in self._column_definition if col[2] == InputTypes.ID]) == 1, 'There must be exactly one ID column.'
    # check that there is unique time column
    assert len([col for col in self._column_definition if col[2] == InputTypes.TIME]) == 1, 'There must be exactly one time column.'
    # check that there is at least one target column
    assert len([col for col in self._column_definition if col[2] == InputTypes.TARGET]) >= 1, 'There must be at least one target column.'
  
  def __set_data_types(self):
    # set time column as datetime format in pandas
    for col in self._column_definition:
      if col[1] == DataTypes.DATE:
        self.data[col[0]] = pd.to_datetime(self.data[col[0]])
      if col[1] == DataTypes.CATEGORICAL:
        self.data[col[0]] = self.data[col[0]].astype('category')
      if col[1] == DataTypes.REAL_VALUED:
        self.data[col[0]] = self.data[col[0]].astype(np.float32)

  def __check_nan(self):
    # delete rows where target, time, or id are na
    self.data = self.data.dropna(subset=[col[0] 
                                  for col in self._column_definition 
                                  if col[2] in [InputTypes.TARGET, InputTypes.TIME, InputTypes.ID]])
    # assert that there are no na values in the data
    assert self.data.isna().sum().sum() == 0, 'There are NA values in the data even after dropping with missing time, glucose, or id.'

  def __drop(self):
    # drop columns that are not in the column definition
    self.data = self.data[[col[0] for col in self._column_definition]]
    # drop rows based on conditions set in the formatter
    if self.params['drop'] is not None:
      if self.params['drop']['rows'] is not None:
        # drop row at indices in the list self.params['drop']['rows']
        self.data = self.data.drop(self.params['drop']['rows'])
        self.data = self.data.reset_index(drop=True)
      if self.params['drop']['columns'] is not None:
        for col in self.params['drop']['columns'].keys():
          # drop rows where specified columns have values in the list self.params['drop']['columns'][col]
          self.data = self.data.loc[~self.data[col].isin(self.params['drop']['columns'][col])].copy()
  
  def __interpolate(self):
    self.data, self._column_definition = utils.interpolate(self.data, 
                                                           self._column_definition, 
                                                           **self._interpolation_params)

  def __split_data(self):
    if self.params['split_params']['test_percent_subjects'] == 0 or \
        self.params['split_params']['length_segment'] == 0:
      print('\tNo splitting performed since test_percent_subjects or length_segment is 0.')
      self.train_idx, self.val_idx, self.test_idx, self.test_idx_ood = None, None, None, None
      self.train_data, self.val_data, self.test_data = self.data, None, None
    else:
      assert self.params['split_params']['length_segment'] > self.params['length_pred'], \
        'length_segment for test / val must be greater than length_pred.'
      self.train_idx, self.val_idx, self.test_idx, self.test_idx_ood = utils.split(self.data, 
                                                                                  self._column_definition, 
                                                                                  **self._split_params)
      self.train_data, self.val_data, self.test_data = self.data.iloc[self.train_idx], \
                                                        self.data.iloc[self.val_idx], \
                                                          self.data.iloc[self.test_idx + self.test_idx_ood]

  def __encode(self):
    self.data, self._column_definition, self.encoders = utils.encode(self.data, 
                                                                     self._column_definition,
                                                                     **self._encoding_params)
  
  def __scale(self):
    self.train_data, self.val_data, self.test_data, self.scalers = utils.scale(self.train_data, 
                                                                               self.val_data, 
                                                                               self.test_data, 
                                                                               self._column_definition, 
                                                                               **self.params['scaling_params'])

  def reshuffle(self, seed):
    stdout = sys.stdout
    f = open(self.study_file, 'a')
    sys.stdout = f
    self.params['split_params']['random_state'] = seed
    # split data
    self.train_idx, self.val_idx, self.test_idx, self.test_idx_ood = utils.split(self.data, 
                                                                                  self._column_definition, 
                                                                                  **self._split_params)
    self.train_data, self.val_data, self.test_data = self.data.iloc[self.train_idx], \
                                                      self.data.iloc[self.val_idx], \
                                                        self.data.iloc[self.test_idx+self.test_idx_ood]
    # re-scale data
    self.train_data, self.val_data, self.test_data, self.scalers = utils.scale(self.train_data, 
                                                                               self.val_data, 
                                                                               self.test_data, 
                                                                               self._column_definition, 
                                                                               **self.params['scaling_params'])
    sys.stdout = stdout
    f.close()
    
  def get_column(self, column_name):
    # write cases for time, id, target, future, static, dynamic covariates
    if column_name == 'time':
      return [col[0] for col in self._column_definition if col[2] == InputTypes.TIME][0]
    elif column_name == 'id':
      return [col[0] for col in self._column_definition if col[2] == InputTypes.ID][0]
    elif column_name == 'sid':
      return [col[0] for col in self._column_definition if col[2] == InputTypes.SID][0]
    elif column_name == 'target':
      return [col[0] for col in self._column_definition if col[2] == InputTypes.TARGET]
    elif column_name == 'future_covs':
      future_covs = [col[0] for col in self._column_definition if col[2] == InputTypes.KNOWN_INPUT] 
      return future_covs if len(future_covs) > 0 else None
    elif column_name == 'static_covs':
      static_covs = [col[0] for col in self._column_definition if col[2] == InputTypes.STATIC_INPUT]
      return static_covs if len(static_covs) > 0 else None
    elif column_name == 'dynamic_covs':
      dynamic_covs = [col[0] for col in self._column_definition if col[2] == InputTypes.OBSERVED_INPUT]
      return dynamic_covs if len(dynamic_covs) > 0 else None
    else:
      raise ValueError('Column {} not found.'.format(column_name))
  
