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
import warnings
from collections import namedtuple
from datetime import datetime
import os
import math
import pathlib
import torch
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
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
    column_definition: List of tuples describing columns (column_name, data_type, input_type).
    gap_threshold: Number in minutes, maximum allowed gap for interpolation.
    min_drop_length: Number of points, minimum number within an interval to interpolate.
    interval_length: Number in minutes, length of interpolation.

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
  
  # round to minute
  data[time_col] = data[time_col].dt.round('1min')
  # count dropped segments
  dropped_segments = 0
  # count number of values that are interpolated
  interpolation_count = 0
  # store final output
  output = []
  for id, id_data in data.groupby(id_col):
    # sort values 
    id_data.sort_values(time_col, inplace=True)
    # get time difference between consecutive rows
    lag = (id_data[time_col].diff().dt.total_seconds().fillna(0) / 60.0).astype(int)
    # if lag > gap_threshold
    id_segment = (lag > gap_threshold).cumsum()
    id_data['id_segment'] = id_segment
    for segment, segment_data in id_data.groupby('id_segment'):
      # if segment is too short, then we don't interpolate
      if len(segment_data) < min_drop_length:
        dropped_segments += 1
        continue
      
      # find and print duplicated times
      duplicates = segment_data.duplicated(time_col, keep=False)
      if duplicates.any():
        print(segment_data[duplicates])
        raise ValueError('Duplicate times in segment {} of id {}'.format(segment, id))

      # reindex at interval_length minute intervals
      segment_data = segment_data.set_index(time_col)
      index_new = pd.date_range(start = segment_data.index[0], 
                            end = segment_data.index[-1], 
                            freq = interval_length)
      index_union = index_new.union(segment_data.index)
      segment_data = segment_data.reindex(index_union)
      # count nan values in interpolation columns
      interpolation_count += segment_data[interpolation_columns[0]].isna().sum()
      # interpolate
      segment_data[interpolation_columns] = segment_data[interpolation_columns].interpolate(method='index')
      # fill constant columns with last value
      segment_data[constant_columns] = segment_data[constant_columns].fillna(method='ffill')
      # delete rows not conforming to frequency
      segment_data = segment_data.reindex(index_new)
      # reset index, make the time a column with name time_col
      segment_data = segment_data.reset_index().rename(columns={'index': time_col})
      # set the id_segment to position in output
      segment_data['id_segment'] = len(output)
      # add to output
      output.append(segment_data)
  # print number of dropped segments and number of segments
  print('\tDropped segments: {}'.format(dropped_segments))
  print('\tExtracted segments: {}'.format(len(output)))
  # concat all segments and reset index
  output = pd.concat(output)
  output.reset_index(drop=True, inplace=True)
  # count number of interpolated values
  print('\tInterpolated values: {}'.format(interpolation_count))
  print('\tPercent of values interpolated: {:.2f}%'.format(interpolation_count / len(output) * 100))
  # add id_segment column to column_definition as ID
  column_definition += [('id_segment', DataTypes.CATEGORICAL, InputTypes.SID)]

  return output, column_definition

def create_index(time_col: pd.Series, interval_length: int):
  """Creates a new index at interval_length minute intervals.

  Args:
    time_col: Series of times.
    interval_length: Number in minutes, length of interpolation.

  Returns:
    index: New index.
  """
  # margin of error
  eps = pd.Timedelta('1min')
  new_time_col = [time_col.iloc[0]]
  for time in time_col.iloc[1:]:
    if time - new_time_col[-1] <= pd.Timedelta(interval_length) + eps:
      new_time_col.append(time)
    else:
      filler = new_time_col[-1] + pd.Timedelta(interval_length)
      while filler < time:
        new_time_col.append(filler)
        filler += pd.Timedelta(interval_length)
      new_time_col.append(time)
  return pd.to_datetime(new_time_col)

def split(df: pd.DataFrame, 
          column_definition: List[Tuple[str, DataTypes, InputTypes]],
          test_percent_subjects: float, 
          length_segment: int,
          max_length_input: int,
          random_state: int = 42):
  """Splits data into train, validation and test sets.

  Args: 
    df: Dataframe to split.
    column_definition: List of tuples describing columns (column_name, data_type, input_type).
    test_percent_subjects: Float number from [0, 1], percentage of subjects to use for test set.
    length_segment: Number of points, length of segments saved for validation / test sets.
    max_length_input: Number of points, maximum length of input sequences for models.
    random_state: Number, Random state for reproducibility.

  Returns:
    train_idx: Training set indices.
    val_idx: Validation set indices.
    test_idx: Test set indices.
  """
  # set random state
  np.random.seed(random_state)
  # get id and id_segment columns
  id_col = [column_name for column_name, data_type, input_type in column_definition if input_type == InputTypes.ID][0]
  id_segment_col = [column_name for column_name, data_type, input_type in column_definition if input_type == InputTypes.SID][0]
  # get unique ids
  ids = df[id_col].unique()

  # select some subjects for test data set
  test_ids = np.random.choice(ids, math.ceil(len(ids) * test_percent_subjects), replace=False)
  test_idx_ood = list(df[df[id_col].isin(test_ids)].index)
  # get the remaning data for training and validation
  df = df[~df[id_col].isin(test_ids)]

  # iterate through subjects and split into train, val and test
  train_idx = []; val_idx = []; test_idx = []
  for id, id_data in df.groupby(id_col):
    segment_ids = id_data[id_segment_col].unique()
    if len(segment_ids) >= 2:
      train_idx += list(id_data.loc[id_data[id_segment_col].isin(segment_ids[:-2])].index)
      penultimate_segment = id_data[id_data[id_segment_col] == segment_ids[-2]]
      last_segment = id_data[id_data[id_segment_col] == segment_ids[-1]]
      if len(last_segment) >= max_length_input + 3 * length_segment:
        train_idx += list(penultimate_segment.index)
        train_idx += list(last_segment.iloc[:-2*length_segment].index)
        val_idx += list(last_segment.iloc[-2*length_segment-max_length_input:-length_segment].index)
        test_idx += list(last_segment.iloc[-length_segment-max_length_input:].index)
      elif len(last_segment) >= max_length_input + 2 * length_segment:
        train_idx += list(penultimate_segment.index)
        val_idx += list(last_segment.iloc[:-length_segment].index)
        test_idx += list(last_segment.iloc[-length_segment-max_length_input:].index)
      else:
        test_idx += list(last_segment.index)
        if len(penultimate_segment) >= max_length_input + 2 * length_segment:
          val_idx += list(penultimate_segment.iloc[-length_segment-max_length_input:].index)
          train_idx += list(penultimate_segment.iloc[:-length_segment].index)
        else:
          train_idx += list(penultimate_segment.index)
    else:
      if len(id_data) >= max_length_input + 3 * length_segment:
        train_idx += list(id_data.iloc[:-2*length_segment].index)
        val_idx += list(id_data.iloc[-2*length_segment-max_length_input:-length_segment].index)
        test_idx += list(id_data.iloc[-length_segment-max_length_input:].index)
      elif len(id_data) >= max_length_input + 2 * length_segment:
        train_idx += list(id_data.iloc[:-length_segment].index)
        test_idx += list(id_data.iloc[-length_segment-max_length_input:].index)
      else:
        train_idx += list(id_data.index)
  total_len = len(train_idx) + len(val_idx) + len(test_idx) + len(test_idx_ood)
  print('\tTrain: {} ({:.2f}%)'.format(len(train_idx), len(train_idx) / total_len * 100))
  print('\tVal: {} ({:.2f}%)'.format(len(val_idx), len(val_idx) / total_len * 100))
  print('\tTest: {} ({:.2f}%)'.format(len(test_idx), len(test_idx) / total_len * 100))
  print('\tTest OOD: {} ({:.2f}%)'.format(len(test_idx_ood), len(test_idx_ood) / total_len * 100))
  return train_idx, val_idx, test_idx, test_idx_ood

def encode(df: pd.DataFrame, 
          column_definition: List[Tuple[str, DataTypes, InputTypes]],
          date: List,):
  """Encodes categorical columns.

  Args: 
    df: Dataframe to split.
    column_definition: List of tuples describing columns (column_name, data_type, input_type).
    date: List of str, list containing date info to extract.

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
        df[column + '_' + extract_col] = df[column + '_' + extract_col].astype(np.float32)
        new_columns.append((column + '_' + extract_col, DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT))
    elif column_type == DataTypes.CATEGORICAL:
      encoders[column] = preprocessing.LabelEncoder()
      df[column] = encoders[column].fit_transform(df[column]).astype(np.float32)
      column_definition[i] = (column, DataTypes.REAL_VALUED, input_type)
    else:
      continue
  column_definition += new_columns
  # print updated column definition
  print('\tUpdated column definition:')
  for column, column_type, input_type in column_definition:
    print('\t\t{}: {} ({})'.format(column, 
                                   DataTypes(column_type).name, 
                                   InputTypes(input_type).name))
  return df, column_definition, encoders
  
def scale(train_data: pd.DataFrame,
          val_data: pd.DataFrame,
          test_data: pd.DataFrame,
          column_definition: List[Tuple[str, DataTypes, InputTypes]],
          scaler: str):
  """Scales numerical data.

  Args:
    train_data: pd.Dataframe, DataFrame of training data.
    val_data: pd.Dataframe, DataFrame of validation data.
    test_data: pd.Dataframe, DataFrame of testing data.
    column_definition: List of tuples describing columns (column_name, data_type, input_type).
    scaler: String, scaler to use.
  
  Returns:
    train_data: pd.Dataframe, DataFrame of scaled training data.
    val_data: pd.Dataframe, DataFrame of scaled validation data.
    test_data: pd.Dataframe, DataFrame of scaled testing data.
    scalers: dictionary index by column names containing scalers.
  """
  # select all real-valued columns
  columns_to_scale = [column for column, data_type, input_type in column_definition if data_type == DataTypes.REAL_VALUED]
  # handle no scaling case
  if scaler == 'None':
    print('\tNo scaling applied')
    return train_data, val_data, test_data, None
  scalers = {}
  for column in columns_to_scale:
    scaler_column = getattr(preprocessing, scaler)()
    train_data[column] = scaler_column.fit_transform(train_data[column].values.reshape(-1, 1))
    # handle empty validation and test sets
    val_data[column] = scaler_column.transform(val_data[column].values.reshape(-1, 1)) if val_data.shape[0] > 0 else val_data[column]
    test_data[column] = scaler_column.transform(test_data[column].values.reshape(-1, 1)) if test_data.shape[0] > 0 else test_data[column]
    scalers[column] = scaler_column
  # print columns that were scaled
  print('\tScaled columns: {}'.format(columns_to_scale))
  return train_data, val_data, test_data, scalers