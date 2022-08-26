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

import torch

import os
import pathlib
import numpy as np
import pandas as pd
import itertools as it
from random import triangular
from scipy.stats import kendalltau
from scipy.stats import norm as normal

import dataset
from utils.nearcorr import *

# data extraction functions
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

# OS related functions.
def create_folder_if_not_exist(directory):
  """Creates folder if it doesn't exist.

  Args:
    directory: Folder path to create.
  """
  # Also creates directories recursively
  pathlib.Path(directory).mkdir(parents=True, exist_ok=True)


def make_data_formatter(exp_name):
    """Gets a data formatter object for experiment.

    Returns:
      Default DataFormatter per experiment.
    """

    data_formatter_class = {
        'volatility': dataset.volatility.VolatilityFormatter,
        'electricity': dataset.electricity.ElectricityFormatter,
        'traffic': dataset.traffic.TrafficFormatter,
        'favorita': dataset.favorita.FavoritaFormatter,
    }

    return data_formatter_class[exp_name]()


def csv_path_to_folder(path: str):
    return "/".join(path.split('/')[:-1]) + "/"


def data_csv_path(exp_name):
    csv_map = {
        'volatility': './data/volatility/formatted_omi_vol.csv',
        'electricity': './data/electricity/hourly_electricity.csv',
        'traffic': './data/traffic/hourly_data.csv',
        'favorita': './data/favorita/favorita_consolidated.csv',
    }

    return csv_map[exp_name]

# data statistics functions
def compute_corr(outputs):
    # sample series
    sample = np.random.choice(outputs.shape[0], outputs.shape[0] // 5, replace=False)
    X = outputs[sample, :, 0]
    length = X.shape[1]

    values= []
    print('Correlation estimation progress...')
    for i, j in it.combinations(X.T, 2):
        values.append(kendalltau(i, j)[0])
    print('Correlation estimated.')
    print('Enforcing pd for correlation matrix...')
    est_corr = np.empty((length, length))
    iu = np.triu_indices(length, 1)
    il = np.tril_indices(length, -1)
    dg = np.diag_indices(length)
    est_corr[iu] = values
    est_corr[dg] = 1
    est_corr[il] = est_corr.T[il]
    est_corr = np.sin((est_corr * np.pi / 2))
    est_corr = nearcorr(est_corr, max_iterations=1000)
    est_corr = est_corr*0.9+np.eye(length)*0.1
    print('Pd enforced.')

    return est_corr

# data transforms
class IntegralTransform:
  def __init__(self, data):
    self.x = None
    self.y = None
    self.set(data)

  def set(self, data):
    _x = data.flatten()
    _x.sort()
    _x = np.r_[-np.inf, _x, np.inf]
    _y = np.linspace(0, 1, len(_x))
    self.x = _x
    self.y = _y

  def transform(self, x):
    ind = np.searchsorted(self.x, x, side='right') - 1
    out = normal.ppf(self.y[ind])
    return out

  def inverse_transform(self, y):
    ind = np.searchsorted(self.y, y, side='right') - 1
    out = normal.cdf(self.x[ind])
    return out

def unnormalize_tensor(data_formatter, data, identifier):
  data = pd.DataFrame(
    data.detach().cpu().numpy(),
    columns=[
        't+{}'.format(i)
        for i in range(data.shape[1])
    ])

  data['identifier'] = np.array(identifier)
  data = data_formatter.format_predictions(data)

  return data.drop(columns=['identifier']).values

# loss functions
class QuantileLoss(torch.nn.Module):
  # From: https://medium.com/the-artificial-impostor/quantile-regression-part-2-6fdbc26b2629

  def __init__(self, quantiles):
      # takes a list of quantiles
      super().__init__()
      self.quantiles = quantiles

  def numpy_normalised_quantile_loss(self, y_pred, y, quantile):
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
      if isinstance(y_pred, torch.Tensor):
          y_pred = y_pred.detach().cpu().numpy()

      if len(y_pred.shape) == 3:
          ix = self.quantiles.index(quantile)
          y_pred = y_pred[..., ix]

      if isinstance(y, torch.Tensor):
          y = y.detach().cpu().numpy()

      prediction_underflow = y - y_pred
      weighted_errors = quantile * np.maximum(prediction_underflow, 0.) \
                        + (1. - quantile) * np.maximum(-prediction_underflow, 0.)

      quantile_loss = weighted_errors.mean()
      normaliser = np.abs(y).mean()

      return 2 * quantile_loss / normaliser

  def forward(self, preds, target, ret_losses=True):
      assert not target.requires_grad
      assert preds.size(0) == target.size(0)
      losses = []

      for i, q in enumerate(self.quantiles):
          errors = target - preds[:, :, i]
          losses.append(
              torch.max(
                  (q - 1) * errors,
                  q * errors
              ).unsqueeze(1))
      loss = torch.mean(
          torch.sum(torch.cat(losses, dim=1), dim=1))
      if ret_losses:
          return loss, losses
      return loss

def smape_loss(forecast, actual):
  # Symmetric Mean Absolute Percentage Error (SMAPE)
  sequence_length = forecast.shape[1]
  sumf = np.sum(np.abs(forecast - actual) / (np.abs(actual) + np.abs(forecast)), axis=1)
  return np.mean((2 * sumf) / sequence_length)