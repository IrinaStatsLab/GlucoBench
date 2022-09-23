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
"""Custom formatting functions for Electricity dataset.

Defines dataset specific column definitions and data transformations. Uses
entity specific z-score normalization.
"""

import data_formatters.base
import data_formatters.utils as utils
import pandas as pd
import sklearn.preprocessing

GenericDataFormatter = data_formatters.base.GenericDataFormatter
DataTypes = data_formatters.base.DataTypes
InputTypes = data_formatters.base.InputTypes


class ElectricityFormatter(GenericDataFormatter):
  """Defines and formats data for the electricity dataset.

  Note that per-entity z-score normalization is used here, and is implemented
  across functions.

  Attributes:
    column_definition: Defines input and data type of column used in the
      experiment.
    identifiers: Entity identifiers used in experiments.
  """

  _column_definition = [
      ('id', DataTypes.CATEGORICAL, InputTypes.ID),
      ('hours_from_start', DataTypes.REAL_VALUED, InputTypes.TIME),
      ('power_usage', DataTypes.REAL_VALUED, InputTypes.TARGET),
      ('hour', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
      ('day_of_week', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
      ('hours_from_start', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
      ('categorical_id', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT),
  ]

  _interpolation_params = {
      'gap_threshold': 45,
      'min_drop_length': 5
  }

  def __init__(self):
    """Initialises formatter."""
    pass

  def interpolate(self, df):
    # TODO: implement interpolation in utils
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

