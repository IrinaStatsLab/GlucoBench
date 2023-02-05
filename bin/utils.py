import sys
import os
import yaml
from typing import Any, BinaryIO, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import darts

from darts import models
from darts import metrics
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler

def make_series(data: Dict[str, pd.DataFrame],
                time_col: str,
                group_col: str,
                value_cols: List[str]
                ) -> Dict[str, darts.TimeSeries]:
    """
    Make TimeSeries from data.
    data: dict of train, val, test dataframes
    time_col: name of time column
    group_col: name of group column
    value_cols: list of value columns

    Returns: dict of TimeSeries
    """
    series = {i: {j: None for j in value_cols} for i in data.keys()}
    scalers = {i: {j: None for j in value_cols} for i in data.keys()}
    for key, df in data.items():
        for name, cols in value_cols.items():
            series[key][name] = TimeSeries.from_group_dataframe(df = df,
                                                                group_cols = group_col,
                                                                time_col = time_col,
                                                                value_cols = cols) if cols is not None else None
            if cols is not None: 
                if key == 'train':
                    scalers[name] = Scaler()
                    series[key][name] = scalers[name].fit_transform(series[key][name])
                else:
                    series[key][name] = scalers[name].transform(series[key][name])
            else:
                scalers[name] = None
    return series, scalers

def early_stopping_check(study, 
                         trial, 
                         study_file, 
                         early_stopping_rounds=10):
      """
      Early stopping callback for Optuna.
      This function checks the current trial number and the best trial number.
      """
      current_trial_number = trial.number
      best_trial_number = study.best_trial.number
      should_stop = (current_trial_number - best_trial_number) >= early_stopping_rounds
      if should_stop:
          with open(study_file, 'a') as f:
              f.write('Early stopping at trial {} (best trial: {})'.format(current_trial_number, best_trial_number))
          study.stop()

def rescale_and_backtest(series: Union[TimeSeries, Sequence[TimeSeries]],
                         forecasts: Union[TimeSeries, Sequence[TimeSeries]], 
                         metric: Union[
                                    Callable[[TimeSeries, TimeSeries], float],
                                    List[Callable[[TimeSeries, TimeSeries], float]],
                                ], 
                         scaler: Callable[[TimeSeries], TimeSeries] = None,
                         reduction: Union[Callable[[np.ndarray], float], None] = np.mean,
                        ):
    """
    Backtest the forecasts on the series.

    Parameters
    ----------
    series
        The target time series.
    forecasts
        The forecasts.
    scaler
        The scaler used to scale the series.
    metric
        The metric to use for backtesting.
    reduction
        The reduction to apply to the metric.

    Returns
    -------
    float or List[float] or List[List[float]]
        The (sequence of) error score on a series, or list of list containing error scores for each
        provided series and each sample.
    """
    series = [series] if isinstance(series, TimeSeries) else series
    if len(series) == 1:
        forecasts = [forecasts]
    if not isinstance(metric, list):
        metric = [metric]

    # reverse scaling, forecasts and true values, compute errors
    backtest_list = []
    for idx, target_ts in enumerate(series):
        if scaler is not None:
            target_ts = scaler.inverse_transform(target_ts)
            for idxf, f in enumerate(forecasts[idx]):
                f = scaler.inverse_transform(f)
        errors = [
            [metric_f(target_ts, f) for metric_f in metric]
            if len(metric) > 1
            else metric[0](target_ts, f)
            for f in forecasts[idx]
        ]
        if reduction is None:
            backtest_list.append(errors)
        else:
            backtest_list.append(reduction(np.array(errors), axis=0))
    return backtest_list if len(backtest_list) > 1 else backtest_list[0]