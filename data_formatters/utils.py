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
from typing import List, Tuple


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
  return output

def split(df: pd.DataFrame, 
          test_percent_subjects: float, 
          val_length_segment: int, 
          test_length_segment: int,
          min_drop_length: int,
          id_col: str,
          id_segment_col: str,):
  """Splits data into train, validation and test sets.

  Args: 
    df: Dataframe to split.
    test_percent_subjects: Percentage of subjects to use for test set.
    val_length_segment: Length of validation segments in minutes.
    test_length_segment: Length of test segments in minutes.
    min_drop_length: Minimum number of points needed within an interval.
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
    if len(segment_data) >= min_drop_length + test_length_segment + val_length_segment:
      # get indices for train, val and test
      train_idx += list(segment_data.iloc[:-test_length_segment-val_length_segment].index)
      val_idx += list(segment_data.iloc[-test_length_segment-val_length_segment:-test_length_segment].index)
      test_idx += list(segment_data.iloc[-test_length_segment:].index)
    elif len(segment_data) >= min_drop_length + test_length_segment:
      # get indices for train and test
      train_idx += list(segment_data.iloc[:-test_length_segment].index)
      test_idx += list(segment_data.iloc[-test_length_segment:].index)
    else:
      # get indices for train
      train_idx += list(segment_data.index)
  return train_idx, val_idx, test_idx


 


