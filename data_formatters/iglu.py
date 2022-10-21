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

GenericDataFormatter = data_formatters.base.GenericDataFormatter
DataTypes = data_formatters.base.DataTypes
InputTypes = data_formatters.base.InputTypes


class IGLUFormatter(GenericDataFormatter):
  # Defines and formats data for the IGLU dataset.

  _column_definition = [
      ('id', DataTypes.CATEGORICAL, InputTypes.ID),
      ('time', DataTypes.DATE, InputTypes.TIME),
      ('gl', DataTypes.REAL_VALUED, InputTypes.TARGET) # Glycemic load
  ]

  _time_measurement_columns = [
      ('year', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
      ('month', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
      ('day', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
      ('hour', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
      ('minute', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT)
  ]

  _interpolation_params = {
    'id_col': 'id',
    'time_col': 'time',
    'interpolation_columns': ['gl'], 
    'constant_columns': [],
    'gap_threshold': 45, 
    'min_drop_length': 144, 
    'interval_length': 5
  }

  _split_params = {
  'test_percent_subjects': 0.1,
  'test_length_segment': 144,
  'val_length_segment': 144,
  'min_drop_length': 144,
  'id_col': 'id',
  'id_segment_col': 'id_segment',
  }

  _encoding_params = {
    'id_col': 'id',
    'time_col': 'time',
  }

  def __init__(self, cnf):
    """Initialises formatter."""
    # Get parameters
    self.params = cnf.all_params
    self.params['index_col'] = False if self.params['index_col'] == -1 else self.params['index_col']
    # Read from input csv file
    self.data = pd.read_csv(self.params['data_csv_path'], index_col=self.params['index_col'])
    # Convert timestamps into datetime objects
    self.data['time'] = pd.to_datetime(self.data['time'])
    # Round to nearest 5 mins
    self.data.time = self.data.time.dt.round('5min')
    # Fix indexing issues when rounding
    self.data = self.data.groupby(['id', 'time']).mean().reset_index().rename(columns={'index': 'time'})
    self.drop_invalid_columns()
    self.interpolate()
    self.train_idx, self.val_idx, self.test_idx = self.split_data()
    self.encode()
  
  def interpolate(self):
    self.data = utils.interpolate(self.data, **self._interpolation_params)
    # create new column with unique id for each subject-segment pair
    self.data['id_segment'] = self.data.id.astype('str') + '_' + self.data.segment.astype('str')
    # set subject-segment column as ID and set subject id column as KNOWN_INPUT
    self._column_definition[0] = ('id', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT)
    self._column_definition += [('id_segment', DataTypes.CATEGORICAL, InputTypes.ID)]

  def split_data(self):
    return utils.split(self.data, **self._split_params)
  
  def encode(self):
    self._column_definition.extend(self._time_measurement_columns)
    self.data, self.id_encoder = utils.encode(self.data, **self._encoding_params)


  def set_scalers(self, df):
    pass

  def transform_inputs(self, df):
    pass

  def format_predictions(self, df):
    pass

  def drop_invalid_columns(self):
    valid_cols = [col[0] for col in self._column_definition]
    self.data = self.data.drop(columns=[col for col in self.data if col not in valid_cols])