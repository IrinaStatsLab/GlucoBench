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
"""Generic helper functions used across codebase."""

from collections import namedtuple
from datetime import datetime
import os
import pathlib
import torch
import numpy as np
import pandas as pd
import data_formatters
from typing import List, Tuple


MINUTE = 60

# Loss functions.
def pytorch_quantile_loss(y, y_pred, quantile):
  """Computes quantile loss for tensorflow.

  Standard quantile loss as defined in the "Training Procedure" section of
  the main TFT paper

  Args:
    y: Targets
    y_pred: Predictions
    quantile: Quantile to use for loss calculations (between 0 & 1)

  Returns:
    Tensor for quantile loss.
  """

  # Checks quantile
  if quantile < 0 or quantile > 1:
    raise ValueError(
        'Illegal quantile value={}! Values should be between 0 and 1.'.format(
            quantile))

  prediction_underflow = y - y_pred
  q_loss = quantile * torch.max(prediction_underflow, torch.zeros_like(prediction_underflow)) + (
      1. - quantile) * torch.max(-prediction_underflow, torch.zeros_like(prediction_underflow))

  return torch.sum(q_loss, axis=-1)



# Generic.
def get_single_col_by_input_type(input_type, column_definition):
  """Returns name of single column.

  Args:
    input_type: Input type of column to extract
    column_definition: Column definition list for experiment
  """

  l = [tup[0] for tup in column_definition if tup[2] == input_type]

  if len(l) != 1:
    raise ValueError('Invalid number of columns for {}'.format(input_type))

  return l[0]


def extract_cols_from_data_type(data_type, column_definition,
                                excluded_input_types):
  """Extracts the names of columns that correspond to a define data_type.

  Args:
    data_type: DataType of columns to extract.
    column_definition: Column definition to use.
    excluded_input_types: Set of input types to exclude

  Returns:
    List of names for columns with data type specified.
  """
  return [
      tup[0]
      for tup in column_definition
      if tup[1] == data_type and tup[2] not in excluded_input_types
  ]


def numpy_normalised_quantile_loss(y, y_pred, quantile):
  """Computes normalised quantile loss for numpy arrays.

  Uses the q-Risk metric as defined in the "Training Procedure" section of the
  main TFT paper.

  Args:
    y: Targets
    y_pred: Predictions
    quantile: Quantile to use for loss calculations (between 0 & 1)

  Returns:
    Float for normalised quantile loss.
  """
  prediction_underflow = y - y_pred
  weighted_errors = quantile * np.maximum(prediction_underflow, 0.) \
      + (1. - quantile) * np.maximum(-prediction_underflow, 0.)

  quantile_loss = weighted_errors.mean()
  normaliser = y.abs().mean()

  return 2 * quantile_loss / normaliser


# OS related functions.
def create_folder_if_not_exist(directory):
  """Creates folder if it doesn't exist.

  Args:
    directory: Folder path to create.
  """
  # Also creates directories recursively
  pathlib.Path(directory).mkdir(parents=True, exist_ok=True)


def csv_path_to_folder(path: str):
    return "/".join(path.split('/')[:-1]) + "/"


def interpolate(df: pd.DataFrame, interpolation_columns: List[str], gap_threshold: int = 0, min_drop_length: int = 0, interval_length: int = 0):
  """Interpolates missing values in data.

  Args:
    df: Dataframe to interpolate on. Sorted by id and then time (a DateTime object).
    interpolation_columns: Columns that we want to interpolate on.
    gap_threshold: Maximum allowed gap for interpolation.
    min_drop_length: Minimum number of points needed within an interval to interpolate.
    interval_length: Length of interpolation intervals in minutes.

  Returns:
    data: DataFrame with missing values interpolated and 
          additional column ('segment') indicating continuous segments.
  """
  # returned_columns = df.columns + ['segment']
  # new_dataframe = pd.DataFrame(columns=returned_columns)

  # segments = []
  # prev_segment_start_time: datetime = new_dataframe.iloc[1]['time']
  # for row in df.itertuples():
  #   time_difference = (row['time'] - prev_segment_start_time).total_seconds()

  #   # Must make a new segment if there exists a gap larger than the gap threshold.
  #   if time_difference > gap_threshold * MINUTE or row['id'] != segments[0]['id']:
  #     segment_index += 1

  #     if len(segments) >= min_drop_length:
  #       first_instance, last_instance = segments[0], segments[-1]
  #       number_of_intervals = (row['time'] - first_instance['time']).total_seconds() // (interval_length * MINUTE)

  #       # Interpolate over the total number of intervals within our segment array.
  #       for interval_idx in range(1, number_of_intervals):
  #         temp_row = {}
  #         for column in interpolation_columns:
  #           # Use linear interpolation to get data point every interval_length minutes.
  #           predicted_data_pt = first_instance[column] + interval_idx * (interval_length * MINUTE) * (last_instance[column] - first_instance[column]) / (last_instance['time'] - first_instance['time']).total_seconds()
  #           temp_row[column] = predicted_data_pt

  #         temp_row['id'] = segments[0]['id']
  #         temp_row['segment'] = segment_index

  #         new_dataframe.append(temp_row)  
      
  #     segments = []
  #   segments.append(row)
  #   prev_segment_start_time = row['time']
    
  # return new_dataframe

  ##### Lizzie's temp version while trying to bugfix Urjeet's
  # convert to datetime, float in case
  df['time'] = pd.to_datetime(df['time'])
  df['gl'] = pd.to_numeric(df['gl'], errors = 'coerce')

  # get unique ids and sort 
  ids = list(set(df['id'].tolist()))
  ids.sort()

  subnum = 0 # define subject counter
  for i in ids:
      subnum += 1 # start loop by incrementing subject

      # subset data to this subject and add lag column in minutes
      data = df.loc[df['id'] == i].copy()
      data['lag'] = data['time'].diff().astype('timedelta64[m]')

      # find gap indices for this subject where gaps are lag > 45 minutes
      idx = np.where(data['lag'] > gap_threshold)[0].tolist() # index is the location following gap (i.e. start of next segment)
      # add zero and ending index for taking the difference
      idx = [0] + idx + [len(data['lag'])]

      for j in range(1, len(idx)):
          ## if segment is less than an hour of data, then skip interpolation (data will not be appended)
          # needs to be adjusted bc could have fewer readings but > desired overall length
          if idx[j] - idx[j-1] < min_drop_length: continue # assumes min_drop_length is counting readings, not time
          ## otherwise proceed with interpolation
          # slice and copy the data corresponding to this segment
          segment = data.iloc[idx[j-1]:idx[j]].copy()
          # create index as time for resampling
          segment.index = segment['time']
          # resample to minute and take mean to round to nearest minute
          segment = segment.resample('T').mean()
          # resample to secondly,  interpolate, then resample to 5 minutes to get grid points
          segment = segment.resample('s').mean().interpolate(method = 'linear') \
              .resample('5T').interpolate(method = 'linear')
          # add segment counter, id, and time because time gets dropped above
          segment['segment'] = j
          segment['id'] = i
          segment['time'] = segment.index

          # if first segment, then df1 is just this segment; else df1 needs to append the current segment
          if j == 1:
              df1 = segment.copy()
          else:
              df1 = df1.append(segment)
      # if first subject, then copy; else append
      if subnum == 1:
          dfsubject = df1.copy()
      else:
          dfsubject = dfsubject.append(df1)

  # reorder columns
  dfsubject = dfsubject[['id', 'segment', 'time', 'gl']].reset_index(drop = True)   

  return dfsubject


