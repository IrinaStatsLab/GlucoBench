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
import math
import pathlib
import torch
import numpy as np
import pandas as pd
import data_formatters
from scalers import scaler
from typing import List, Tuple
from sklearn import preprocessing


MINUTE = 60

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


def interpolate(data: pd.DataFrame, 
                id_col: str,
                time_col: str,
                interpolation_columns: List[str], 
                constant_columns: List[str],
                gap_threshold: int = 0, 
                min_drop_length: int = 0, 
                interval_length: int = 0):
  """Interpolates missing values in data.

  Args:
    df: Dataframe to interpolate on. Sorted by id and then time (a DateTime object).
    id_col: Name of the column containing the id of the subject.
    time_col: Name of the time column.
    interpolation_columns: Columns that we want to interpolate on.
    constant_columns: Columns that we want to fill with the constant value.
    gap_threshold: Maximum allowed gap for interpolation.
    min_drop_length: Minimum number of points needed within an interval to interpolate.
    interval_length: Length of interpolation intervals in minutes.

  Returns:
    data: DataFrame with missing values interpolated and 
          additional column ('segment') indicating continuous segments.
  """
  # add id_col, segment as constant (do not need to be interpolated)
  constant_columns += [id_col, 'segment']
  # count dropped segments
  dropped_segments = 0
  # store final output
  output = []
  for id, id_data in data.groupby(id_col):
    # sort values 
    id_data.sort_values(time_col, inplace=True)
    # get time difference between consecutive rows
    lag = (id_data[time_col].diff().dt.total_seconds().fillna(0) / 60.0).astype(int)
    # if lag > gap_threshold, then we have a gap and need to split into segments
    id_data['segment'] = (lag > gap_threshold).cumsum()
    for segment, segment_data in id_data.groupby('segment'):
      # if segment is too short, then we don't interpolate
      if len(segment_data) < min_drop_length:
        dropped_segments += 1
        continue
      # reindex at interval_length minute intervals
      segment_data = segment_data.set_index(time_col).reindex(pd.date_range(start=segment_data[time_col].iloc[0], 
                                                                            end=segment_data[time_col].iloc[-1], 
                                                                            freq=str(interval_length)+'min'))
      # interpolate
      segment_data[interpolation_columns] = segment_data[interpolation_columns].interpolate(method='linear')
      # fill constant columns with last value
      segment_data[constant_columns] = segment_data[constant_columns].fillna(method='ffill')
      # reset index, make the time a column with name time_col
      segment_data = segment_data.reset_index().rename(columns={'index': time_col})
      # add to output
      output.append(segment_data)
  # print number of dropped segments and number of segments
  print('Dropped segments: {}'.format(dropped_segments))
  print('Extracted segments: {}'.format(len(output)))
  # concat all segments and reset index
  output = pd.concat(output)
  output.reset_index(drop=True, inplace=True)
  return output

def split(df: pd.DataFrame, 
          test_percent_subjects: float, 
          length_segment: int,
          id_col: str,
          id_segment_col: str,):
  """Splits data into train, validation and test sets.

  Args: 
    df: Dataframe to split.
    test_percent_subjects: Percentage of subjects to use for test set.
    length_segment: Length of validation segments in number of intervals.
    id_col: Name of the column containing the id of the subject (NOTE: note id_segment).
    id_segment_col: Name of the column containing the id and segment.

  Returns:
    train_idx: Training set indices.
    val_idx: Validation set indices.
    test_idx: Test set indices.
  """

  # get unique ids
  ids = df[id_col].unique()
  # select some subjects for test data set
  test_ids = np.random.choice(ids, math.ceil(len(ids) * test_percent_subjects), replace=False)
  test_idx = list(df[df[id_col].isin(test_ids)].index)
  # get the remaning data for training and validation
  df = df[~df[id_col].isin(test_ids)]

  # iterate through segments and split into train, val and test
  train_idx = []; val_idx = []
  for id, segment_data in df.groupby(id_segment_col):
    if len(segment_data) >= length_segment + length_segment + length_segment:
      # get indices for train, val and test
      train_idx += list(segment_data.iloc[:-length_segment-length_segment].index)
      val_idx += list(segment_data.iloc[-length_segment-length_segment:-length_segment].index)
      test_idx += list(segment_data.iloc[-length_segment:].index)
    elif len(segment_data) >= length_segment + length_segment:
      # get indices for train and test
      train_idx += list(segment_data.iloc[:-length_segment].index)
      val_idx += list(segment_data.iloc[-length_segment:].index)
    elif len(segment_data) >= length_segment:
      # get indices for train
      train_idx += list(segment_data.index)
    else:
      # segment is too short, skip
      continue
  return train_idx, val_idx, test_idx

def encode(df: pd.DataFrame, 
          id_col: str,
          time_col: str,):
  """Encodes time and id as real-valued input.

  Args: 
    df: Dataframe to split.
    id_col: Name of the column containing the id of the subject.
    time_col: Name of the column containing time

  Returns:
    df: Dataframe with time and id encoded as real-values
    id_encoder: Fitted encoder for use in inverse_transform later
  """ 

  # encode time as real-valued columns for year, month, day, hour, and minute
  date = {
      'year': [],
      'month': [],
      'day': [],
      'hour': [],
      'minute': []
      }
  for datetime in df[time_col]:
      date['year'].append(datetime.year)
      date['month'].append(datetime.month)
      date['day'].append(datetime.day)
      date['hour'].append(datetime.hour)
      date['minute'].append(datetime.minute)
  # Create new columns
  df['year'] = date['year']
  df['month'] = date['month']
  df['day'] = date['day']
  df['hour'] = date['hour']
  df['minute'] = date['minute']

  # encode id as real-value
  id_encoder =  preprocessing.LabelEncoder()
  id_encoder.fit(df[id_col])
  df[id_col] = id_encoder.transform(df[id_col])
  
  return df, id_encoder
  
def scale(df: pd.DataFrame, train_idx: List[int], val_idx: List[int], test_idx: List[int], columns_to_scale: List[str], scaler: scaler.Scaler, scale_off_curve: bool):
  """Scales numerical data.

  Args:
    df: DataFrame to scale.
    train_idx: Indexes of rows to train on
    val_idx: Indexes of rows to validate on
    test_idx: Indexes of rows to test on
    columns_to_scale: Columns to independently scale.
    scaler: Scaler used for fitting and scaling.
    scale_off_curve: Scale off a curve vs other parameters.
  
  Returns:
    train_data: pd.Dataframe, DataFrame of scaled training data.
    val_data: pd.Dataframe, DataFrame of scaled validation data.
    test_data: pd.Dataframe, DataFrame of scaled testing data.
  """
  train_data = df.iloc[train_idx, :].copy()
  val_data = df.iloc[val_idx, :].copy()
  test_data = df.iloc[test_idx, :].copy()

  for column in columns_to_scale:
    # Fit scaler on column.
    train_data = scaler.fit_transform(train_data, column)
    # Scale the columns in the datasets.
    val_data = scaler.transform(val_data, column)
    test_data = scaler.transform(test_data, column)

  return train_data, val_data, test_data