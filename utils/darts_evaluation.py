import sys
import os
import yaml
import random
from typing import Any, BinaryIO, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np 
from scipy import stats
import pandas as pd
import darts

from darts import models
from darts import metrics
from darts import TimeSeries

# import data formatter
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data_formatter.base import *
from utils.darts_processing import *

def _get_values(
    series: TimeSeries, stochastic_quantile: Optional[float] = 0.5
) -> np.ndarray:
    """
    Returns the numpy values of a time series.
    For stochastic series, return either all sample values with (stochastic_quantile=None) or the quantile sample value
    with (stochastic_quantile {>=0,<=1})
    """
    if series.is_deterministic:
        series_values = series.univariate_values()
    else:  # stochastic
        if stochastic_quantile is None:
            series_values = series.all_values(copy=False)
        else:
            series_values = series.quantile_timeseries(
                quantile=stochastic_quantile
            ).univariate_values()
    return series_values

def _get_values_or_raise(
    series_a: TimeSeries,
    series_b: TimeSeries,
    intersect: bool,
    stochastic_quantile: Optional[float] = 0.5,
    remove_nan_union: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Returns the processed numpy values of two time series. Processing can be customized with arguments
    `intersect, stochastic_quantile, remove_nan_union`.

    Raises a ValueError if the two time series (or their intersection) do not have the same time index.

    Parameters
    ----------
    series_a
        A univariate deterministic ``TimeSeries`` instance (the actual series).
    series_b
        A univariate (deterministic or stochastic) ``TimeSeries`` instance (the predicted series).
    intersect
        A boolean for whether or not to only consider the time intersection between `series_a` and `series_b`
    stochastic_quantile
        Optionally, for stochastic predicted series, return either all sample values with (`stochastic_quantile=None`)
        or any deterministic quantile sample values by setting `stochastic_quantile=quantile` {>=0,<=1}.
    remove_nan_union
        By setting `remove_non_union` to True, remove all indices from `series_a` and `series_b` which have a NaN value
        in either of the two input series.
    """
    series_a_common = series_a.slice_intersect(series_b) if intersect else series_a
    series_b_common = series_b.slice_intersect(series_a) if intersect else series_b

    series_a_det = _get_values(series_a_common, stochastic_quantile=stochastic_quantile)
    series_b_det = _get_values(series_b_common, stochastic_quantile=stochastic_quantile)

    if not remove_nan_union:
        return series_a_det, series_b_det

    b_is_deterministic = bool(len(series_b_det.shape) == 1)
    if b_is_deterministic:
        isnan_mask = np.logical_or(np.isnan(series_a_det), np.isnan(series_b_det))
    else:
        isnan_mask = np.logical_or(
            np.isnan(series_a_det), np.isnan(series_b_det).any(axis=2).flatten()
        )
    return np.delete(series_a_det, isnan_mask), np.delete(
        series_b_det, isnan_mask, axis=0
    )

def rescale_and_backtest(series: Union[TimeSeries, 
                                    Sequence[TimeSeries]],
                         forecasts: Union[TimeSeries, 
                                            Sequence[TimeSeries],
                                            Sequence[Sequence[TimeSeries]]], 
                         metric: Union[
                                    Callable[[TimeSeries, TimeSeries], float],
                                    List[Callable[[TimeSeries, TimeSeries], float]],
                                ], 
                         scaler: Callable[[TimeSeries], TimeSeries] = None,
                         reduction: Union[Callable[[np.ndarray], float], None] = np.mean,
                         likelihood: str = "GaussianMean",
                         cal_thresholds: Optional[np.ndarray] = np.linspace(0, 1, 11),
                        ):
    """
    Backtest the historical forecasts (as provided by Darts) on the series.

    Parameters
    ----------
    series
        The target time series.
    forecasts
        The forecasts.
    scaler
        The scaler used to scale the series.
    metric
        The metric or metrics to use for backtesting.
    reduction
        The reduction to apply to the metric.
    likelihood
        The likelihood to use for evaluating the model.
    cal_thresholds
        The thresholds to use for computing the calibration error.

    Returns
    -------
    np.ndarray
        Error array. If the reduction is none, array is of shape (n, p)
        where n is the total number of samples (forecasts) and p is the number of metrics.
        If the reduction is not none, array is of shape (k, p), where k is the number of series. 
    float
        The estimated log-likelihood of the model on the data.
    np.ndarray
        The ECE for each time point in the forecast.
    """
    series = [series] if isinstance(series, TimeSeries) else series
    if len(series) == 1:
        forecasts = [forecasts]
    if not isinstance(metric, list):
        metric = [metric]

    # compute errors: 1) reverse scaling forecasts and true values, 2)compute errors
    backtest_list = []
    for idx in range(len(series)):
        if scaler is not None:
            series[idx] = scaler.inverse_transform(series[idx])
            forecasts[idx] = [scaler.inverse_transform(f) for f in forecasts[idx]]
        errors = [
            [metric_f(series[idx], f) for metric_f in metric]
            if len(metric) > 1
            else metric[0](series[idx], f)
            for f in forecasts[idx]
        ]
        if reduction is None:
            backtest_list.append(np.array(errors))
        else:
            backtest_list.append(reduction(np.array(errors), axis=0))
    backtest_list = np.vstack(backtest_list)
    
    if likelihood == "GaussianMean":
        # compute likelihood
        est_var = []
        for idx, target_ts in enumerate(series):
            est_var += [metrics.mse(target_ts, f) for f in forecasts[idx]]
        est_var = np.mean(est_var)
        forecast_len = forecasts[0][0].n_timesteps
        log_likelihood = -0.5*forecast_len - 0.5*np.log(2*np.pi*est_var)

        # compute calibration error: 1) cdf values 2) compute calibration error
        # compute the cdf values
        cdf_vals = []
        for idx in range(len(series)):
            for forecast in forecasts[idx]:
                y_true, y_pred = _get_values_or_raise(series[idx], 
                                                      forecast, 
                                                      intersect=True, 
                                                      remove_nan_union=True)
                y_true, y_pred = y_true.flatten(), y_pred.flatten()
                cdf_vals.append(stats.norm.cdf(y_true, loc=y_pred, scale=np.sqrt(est_var)))
        cdf_vals = np.vstack(cdf_vals)
        # compute the prediction calibration
        cal_error = np.zeros(forecasts[0][0].n_timesteps)
        for p in cal_thresholds:
            est_p = (cdf_vals <= p).astype(float)
            est_p = np.mean(est_p, axis=0)
            cal_error += (est_p - p) ** 2

    return backtest_list, log_likelihood, cal_error

def rescale_and_test(series: Union[TimeSeries, 
                                   Sequence[TimeSeries]],
                    forecasts: Union[TimeSeries, 
                                    Sequence[TimeSeries]], 
                    metric: Union[
                            Callable[[TimeSeries, TimeSeries], float],
                            List[Callable[[TimeSeries, TimeSeries], float]],
                        ], 
                    scaler: Callable[[TimeSeries], TimeSeries] = None,
                    likelihood: str = "GaussianMean",
                    cal_thresholds: Optional[np.ndarray] = np.linspace(0, 1, 11),
                ):
    """
    Test the forecasts on the series.

    Parameters
    ----------
    series
        The target time series.
    forecasts
        The forecasts.
    scaler
        The scaler used to scale the series.
    metric
        The metric or metrics to use for backtesting.
    reduction
        The reduction to apply to the metric.
    likelihood
        The likelihood to use for evaluating the likelihood and calibration of model.
    cal_thresholds
        The thresholds to use for computing the calibration error.

    Returns
    -------
    np.ndarray
        Error array. If the reduction is none, array is of shape (n, p)
        where n is the total number of samples (forecasts) and p is the number of metrics.
        If the reduction is not none, array is of shape (k, p), where k is the number of series. 
    float
        The estimated log-likelihood of the model on the data.
    np.ndarray
        The ECE for each time point in the forecast.
    """
    series = [series] if isinstance(series, TimeSeries) else series
    forecasts = [forecasts] if isinstance(forecasts, TimeSeries) else forecasts
    metric = [metric] if not isinstance(metric, list) else metric

    # compute errors: 1) reverse scaling forecasts and true values, 2)compute errors
    series = scaler.inverse_transform(series)
    forecasts = scaler.inverse_transform(forecasts)
    errors = [
        [metric_f(t, f) for metric_f in metric]
        if len(metric) > 1
        else metric[0](t, f)
        for (t, f) in zip(series, forecasts)
        ]
    errors = np.array(errors)

    if likelihood == "GaussianMean":        
        # compute likelihood
        est_var = [metrics.mse(t, f) for (t, f) in zip(series, forecasts)]
        est_var = np.mean(est_var)
        forecast_len = forecasts[0].n_timesteps
        log_likelihood = -0.5*forecast_len - 0.5*np.log(2*np.pi*est_var)

        # compute calibration error: 1) cdf values 2) compute calibration error
        # compute the cdf values
        cdf_vals = []
        for t, f in zip(series, forecasts):
            t, f = _get_values_or_raise(t, f, intersect=True, remove_nan_union=True)
            t, f = t.flatten(), f.flatten()
            cdf_vals.append(stats.norm.cdf(t, loc=f, scale=np.sqrt(est_var)))
        cdf_vals = np.vstack(cdf_vals)
        # compute the prediction calibration
        cal_error = np.zeros(forecasts[0].n_timesteps)
        for p in cal_thresholds:
            est_p = (cdf_vals <= p).astype(float)
            est_p = np.mean(est_p, axis=0)
            cal_error += (est_p - p) ** 2

    if likelihood == "Quantile":
        # no likelihood since we don't have a parametric model
        log_likelihood = 0

        # compute calibration error: 1) get quantiles 2) compute calibration error
        cal_error = np.zeros(forecasts[0].n_timesteps)
        for p in cal_thresholds:
            est_p = 0
            for t, f in zip(series, forecasts):
                q = f.quantile(p)
                t, q = _get_values_or_raise(t, q, intersect=True, remove_nan_union=True)
                t, q = t.flatten(), q.flatten()
                est_p += (t <= q).astype(float)
            est_p = (est_p / len(series)).flatten()
            cal_error += (est_p - p) ** 2
        
    return errors, log_likelihood, cal_error