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


class IGLUFormatter(GenericDataFormatter):
  # Defines and formats data for the IGLU dataset.

  _column_definition = [
      ('id', DataTypes.CATEGORICAL, InputTypes.ID),
      ('time', DataTypes.REAL_VALUED, InputTypes.TIME),
      ('gl', DataTypes.REAL_VALUED, InputTypes.TARGET) # Glycemic load
  ]

  _interpolation_params = {
      'gap_threshold': 45,
      'min_drop_length': 12,
      'interval_length': 5
  }

  def __init__(self):
    """Initialises formatter."""
    pass
  
  def interpolate(self, df):
    # TODO: implement interpolation in utils
    df.sort_values(by=['id', 'time'], inplace=True)
    df = utils.interpolate(df, **self._interpolation_params)
    # create new column with unique id for each subject-segment pair
    df['segment_id'] = df.id.astype('str') + '_' + df.segment.astype('str')
    # set subject-segment column as ID and set subject id column as KNOWN_INPUT
    self._column_definition[0] = ('id', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT)
    self._column_definition += [('segment_id', DataTypes.CATEGORICAL, InputTypes.ID)]
    return df

  def split_data(self, df):
    pass

  def set_scalers(self, df):
    pass

  def transform_inputs(self, df):
    pass

  def format_predictions(self, df):
    pass