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
        continue
      # reindex at interval_length minute intervals
      segment_data = segment_data.set_index(time_col).reindex(pd.date_range(start=segment_data[time_col].iloc[0], 
                                                                            end=segment_data[time_col].iloc[-1], 
                                                                            freq=str(60*interval_length)+'S'))
      # interpolate
      segment_data[interpolation_columns] = segment_data[interpolation_columns].interpolate(method='linear')
      # fill constant columns with last value
      segment_data[constant_columns] = segment_data[constant_columns].fillna(method='ffill')
      # reset index and make the time column a column again
      segment_data = segment_data.reset_index(names=[time_col])
      # add to output
      output.append(segment_data)
  # concat all segments and reset index
  output = pd.concat(output)
  return output

def flatten(l):
  return [item for sublist in l for item in sublist]

def split(df: pd.DataFrame, test_percent_subjects: float, val_length_segment: int, test_length_segment: int, min_drop_length: int):
  L = dict()
  segment = []
  for i in range(len(df)):
      # get row information
      row = df.iloc[i]
      prev_row = df.iloc[0] if i == 0 else df.iloc[i-1]
      
      # check for change in subject or segment
      if row.id != prev_row.id or row.segment != prev_row.segment:
          if prev_row.id not in L:
              L[prev_row.id] = [segment]
          else:
              L[prev_row.id].append(segment)
          segment = []
      
      segment.append(i)
      
      # edge case: once at end of data, need to append final segment
      if i == len(df)-1:
          if prev_row.id not in L:
              L[prev_row.id] = [segment]
          else:
            L[prev_row.id].append(segment)

  test_set = []
  number_of_subject = len(L)
  test_count = int(number_of_subject * test_percent_subjects)

  # add test set data
  for i in range(test_count):
    subjects = list(L.keys())
    curr_subject = L[subjects[i]]
    for segment in curr_subject:
        for idx in segment:
            test_set.append(idx)

  # remove test set from L
  subjects = list(L.keys())
  for i in range(test_count):
      del L[subjects[i]]
  
  train_set = []
  validation = []

  # iterate through L
  # check that segment has length >= min_drop_length + val_length_segment + test_length_segment
  for subject in L:
      for segment in L[subject]:
          if len(segment) >= min_drop_length + val_length_segment + test_length_segment:
              train_set.append(segment[:(-(val_length_segment + test_length_segment))])
              validation.append(segment[-(val_length_segment + test_length_segment):-test_length_segment])
              test_set.append(segment[-test_length_segment:])
          else:
              train_set.append(segment)
  
  train_set = flatten(train_set)
  validation = flatten(validation)

  # Add to dictionary for access
  return {'test': test_set, 'train': train_set, 'validation': validation}


 


