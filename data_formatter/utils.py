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
from typing import List, Tuple
from sklearn import preprocessing

import data_formatter
from data_formatter import types

DataTypes = types.DataTypes
InputTypes = types.InputTypes
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
                column_definition: List[Tuple[str, DataTypes, InputTypes]],
                gap_threshold: int = 0, 
                min_drop_length: int = 0, 
                interval_length: int = 0):
  """Interpolates missing values in data.

  Args:
    df: Dataframe to interpolate on. Sorted by id and then time (a DateTime object).
    column_definition: List of tuples (column_name, data_type, input_type).
    gap_threshold: Maximum allowed gap for interpolation.
    min_drop_length: Minimum number of points needed within an interval to interpolate.
    interval_length: Length of interpolation intervals in minutes.

  Returns:
    data: DataFrame with missing values interpolated and 
          additional column ('segment') indicating continuous segments.
    column_definition: Updataed list of tuples (column_name, data_type, input_type).
  """
  # select all real-valued columns that are not id, time, or static
  interpolation_columns = [column_name for column_name, data_type, input_type in column_definition if 
    data_type == DataTypes.REAL_VALUED and 
    input_type not in set([InputTypes.ID, InputTypes.TIME, InputTypes.STATIC_INPUT])]
  # select all other columns except time
  constant_columns = [column_name for column_name, data_type, input_type in column_definition if
    input_type not in set([InputTypes.TIME])]
  constant_columns += ['id_segment']

  # get id and time columns
  id_col = [column_name for column_name, data_type, input_type in column_definition if input_type == InputTypes.ID][0]
  time_col = [column_name for column_name, data_type, input_type in column_definition if input_type == InputTypes.TIME][0]

  # count dropped segments
  dropped_segments = 0
  # store final output
  output = []
  num_segments = 0
  for id, id_data in data.groupby(id_col):
    # sort values 
    id_data.sort_values(time_col, inplace=True)
    # get time difference between consecutive rows
    lag = (id_data[time_col].diff().dt.total_seconds().fillna(0) / 60.0).astype(int)
    # if lag > gap_threshold, then we have a gap and need to split into segments
    id_data['id_segment'] = num_segments + (lag > gap_threshold).cumsum()
    for segment, segment_data in id_data.groupby('id_segment'):
      # update number of segments
      num_segments += 1
      # if segment is too short, then we don't interpolate
      if len(segment_data) < min_drop_length:
        dropped_segments += 1
        continue
      # reindex at interval_length minute intervals
      segment_data = segment_data.set_index(time_col).reindex(pd.date_range(start=segment_data[time_col].iloc[0], 
                                                                            end=segment_data[time_col].iloc[-1], 
                                                                            freq=interval_length))
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
  # add id_segment column to column_definition as ID
  column_definition += [('id_segment', DataTypes.CATEGORICAL, InputTypes.SID)]

  return output, column_definition

def split(df: pd.DataFrame, 
          column_definition: List[Tuple[str, DataTypes, InputTypes]],
          test_percent_subjects: float, 
          length_segment: int):
  """Splits data into train, validation and test sets.

  Args: 
    df: Dataframe to split.
    column_definition: List of tuples (column_name, data_type, input_type).
    test_percent_subjects: Percentage of subjects to use for test set.
    length_segment: Length of validation segments in number of intervals.

  Returns:
    train_idx: Training set indices.
    val_idx: Validation set indices.
    test_idx: Test set indices.
  """
  # get id and id_segment columns
  id_col = [column_name for column_name, data_type, input_type in column_definition if input_type == InputTypes.ID][0]
  id_segment_col = [column_name for column_name, data_type, input_type in column_definition if input_type == InputTypes.SID][0]
  # get unique ids
  ids = df[id_col].unique()

  # select some subjects for test data set
  test_ids = np.random.choice(ids, math.ceil(len(ids) * test_percent_subjects), replace=False)
  test_idx = list(df[df[id_col].isin(test_ids)].index)
  # get the remaning data for training and validation
  df = df[~df[id_col].isin(test_ids)]

  # iterate through segments and split into train, val and test
  train_idx = []; val_idx = []
  for id_segment, segment_data in df.groupby(id_segment_col):
    if len(segment_data) >= 3 * length_segment:
      # get indices for train, val and test
      train_idx += list(segment_data.iloc[:-length_segment-length_segment].index)
      val_idx += list(segment_data.iloc[-length_segment-length_segment:-length_segment].index)
      test_idx += list(segment_data.iloc[-length_segment:].index)
    elif len(segment_data) >= 2 * length_segment:
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
          column_definition: List[Tuple[str, DataTypes, InputTypes]],
          date: List,):
  """Encodes time and id as real-valued input.

  Args: 
    df: Dataframe to split.
    column_definition: list of tuples containing column name and types.
    date: list containing date columns to extract.

  Returns:
    df: Dataframe with encoded columns.
    column_definition: Updated list of tuples containing column name and types.
    encoders: dictionary containing encoders.
  """ 
  encoders = {}
  new_columns = []
  for i in range(len(column_definition)):
    column, column_type, input_type = column_definition[i]
    if column_type == DataTypes.DATE:
      for extract_col in date:
        df[column + '_' + extract_col] = getattr(df[column].dt, extract_col)
        new_columns.append((column + '_' + extract_col, DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT))
    elif column_type == DataTypes.CATEGORICAL:
      encoders[column] = preprocessing.LabelEncoder()
      df[column] = encoders[column].fit_transform(df[column])
      column_definition[i] = (column, DataTypes.REAL_VALUED, input_type)
    else:
      continue
  column_definition += new_columns
  return df, column_definition, encoders
  
def scale(df: pd.DataFrame, 
          column_definition: List[Tuple[str, DataTypes, InputTypes]],
          train_idx: List[int], 
          val_idx: List[int], 
          test_idx: List[int], 
          scale_by: str,
          scaler: str):
  """Scales numerical data.

  Args:
    df: DataFrame to scale.
    train_idx: Indexes of rows to train on
    val_idx: Indexes of rows to validate on
    test_idx: Indexes of rows to test on
    scale_by: Column to use as id.
    scaler: Scaler to use.
  
  Returns:
    train_data: pd.Dataframe, DataFrame of scaled training data.
    val_data: pd.Dataframe, DataFrame of scaled validation data.
    test_data: pd.Dataframe, DataFrame of scaled testing data.
  """
  train_data = df.iloc[train_idx, :].copy()
  val_data = df.iloc[val_idx, :].copy()
  test_data = df.iloc[test_idx, :].copy()

  # select all real-valued columns
  columns_to_scale = [column for column, data_type, input_type in column_definition if data_type == DataTypes.REAL_VALUED]
  
  scalers = {}
  for group, data_group in train_data.groupby(scale_by):
    scalers[group] = {}
    for column in columns_to_scale:
      # scale data
      scaler_class = getattr(preprocessing, scaler)()
      # train, val, test index where scale_by == group
      train_idx = train_data[scale_by] == group
      val_idx = val_data[scale_by] == group
      test_idx = test_data[scale_by] == group
      # fit scaler on column.
      train_data.loc[train_idx, column] = scaler_class.fit_transform(data_group[column].values.reshape(-1, 1))
      # scale the columns in the datasets.
      val_data.loc[val_idx, column] = scaler_class.transform(val_data.loc[val_idx, column].values.reshape(-1, 1))
      test_data.loc[test_idx, column] = scaler_class.transform(test_data.loc[test_idx, column].values.reshape(-1, 1))
      # save scaler.
      scalers[group][column] = scaler

  return train_data, val_data, test_data, scalers