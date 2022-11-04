# coding=utf-8
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
# Custom formatting functions for IGLU dataset.

import data_formatters.base
import data_formatters.utils as utils
import pandas as pd
import sklearn.preprocessing

GenericDataFormatter = data_formatters.base.GenericDataFormatter
DataTypes = data_formatters.base.DataTypes
InputTypes = data_formatters.base.InputTypes


class WeinstockFormatter(GenericDataFormatter):
  # Defines and formats data for the IGLU dataset.

  _column_definition = [
      ('id', DataTypes.CATEGORICAL, InputTypes.ID),
      ('time', DataTypes.DATE, InputTypes.TIME),
      ('gl', DataTypes.REAL_VALUED, InputTypes.TARGET) # Glycemic load
  ]

  _interpolation_params = {
    'id_col': 'id',
    'time_col': 'time',
    'interpolation_columns': ['gl'], 
    'constant_columns': [],
    'gap_threshold': 45, 
    'min_drop_length': 192, 
    'interval_length': 5
  }

  _split_params = {
    'test_percent_subjects': 0.1,
    'test_length_segment': 192,
    'val_length_segment': 192,
    'min_drop_length': 192,
    'id_col': 'id',
    'id_segment_col': 'id_segment',
  }

  _encoding_params = {
    'id_col': 'id',
    'time_col': 'time',
  }

  _drop_ids = []

  def __init__(self, cnf):
    """Initialises formatter."""
    # load parameters from the config file
    self.params = cnf.all_params
    # check if data table has index col: -1 if not, index >= 0 if yes
    self.params['index_col'] = False if self.params['index_col'] == -1 else self.params['index_col']
    # read data table
    self.data = pd.read_csv(self.params['data_csv_path'], index_col=self.params['index_col'], na_filter=False)
    # set time as datetime format in pandas
    self.data.time = pd.to_datetime(self.data.time)
    # round time to nearest 5 minutes
    self.data.time = self.data.time.dt.round('5min') 
    # within each id, average any duplicates (come from rounding time to 5 minutes)
    self.data = self.data.groupby(['id', 'time']).mean().reset_index().rename(columns={'index': 'time'})

    # start formatting the data:
    # 1. drop columns that are not in the column definition
    # 2. drop rows based on conditions set in the formatter
    # 3. interpolate missing values
    # 4. split data into train, val, test
    # 5. normalize / encode features and targets
    self.train_idx, self.val_idx, self.test_idx = None, None, None
    self.drop()
    self.interpolate()
    self.split_data()
    self.encode()

  def drop(self):
    # drop columns that are not in the column definition
    self.data = self.data[[col[0] for col in self._column_definition]]
    # drop rows based on conditions set in the formatter
    self.data = self.data.loc[~self.data.id.isin(self._drop_ids)].copy()
  
  def interpolate(self):
    self.data = utils.interpolate(self.data, **self._interpolation_params)

    # create new column with unique id for each subject-segment pair
    self.data['id_segment'] = self.data.id.astype('str') + '_' + self.data.segment.astype('str')
    # set subject-segment column as ID and set subject id column as KNOWN_INPUT
    self._column_definition[0] = ('id', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT)
    self._column_definition += [('id_segment', DataTypes.CATEGORICAL, InputTypes.ID)]

  def split_data(self):
    self.train_idx, self.val_idx, self.test_idx = utils.split(self.data, **self._split_params)

  def encode(self):
    self.data, self.id_encoder = utils.encode(self.data, **self._encoding_params)

    # set column definitions for real-value encoded time
    self._column_definition += [('year', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT)]
    self._column_definition += [('month', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT)]
    self._column_definition += [('day', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT)]
    self._column_definition += [('hour', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT)]
    self._column_definition += [('minute', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT)]

  def set_scalers(self, df):
    pass

  def transform_inputs(self, df):
    pass

  def format_predictions(self, df):
    pass