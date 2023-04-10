import sys
import os
import yaml
import random
from typing import Any, \
                   BinaryIO, \
                   Callable, \
                   Dict, \
                   List, \
                   Optional, \
                   Sequence, \
                   Tuple, \
                   Union

import numpy as np 
import scipy as sp
import pandas as pd
import torch

# import data formatter
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def test(series: np.ndarray, 
         forecasts: np.ndarray, 
         var: np.ndarray,
         cal_thresholds: Optional[np.ndarray] = np.linspace(0, 1, 11),
    ):
    """
    Test the (rescaled to original scale) forecasts on the series.

    Parameters
    ----------
    series
        The target time series of shape (n, t), 
        where t is length of prediction.
    forecasts
        The forecasted means of mixture components of shape (n, t, k),
        where k is the number of mixture components.
    var
        The forecasted variances of mixture components of shape (n, 1, k),
        where k is the number of mixture components.
    metric
        The metric or metrics to use for backtesting.
    cal_thresholds
        The thresholds to use for computing the calibration error.

    Returns
    -------
    np.ndarray
        Error array. Array of shape (n, p) 
        where n = series.shape[0] = forecasts.shape[0] and p = len(metric). 
    float
        The estimated log-likelihood of the model on the data.
    np.ndarray
        The ECE for each time point in the forecast.
    """
    # compute errors: 1) get samples 2) compute errors using median
    samples = np.random.normal(loc=forecasts[..., None],
                               scale=np.sqrt(var)[..., None],
                               size=(forecasts.shape[0], 
                                     forecasts.shape[1], 
                                     forecasts.shape[2],
                                     30))
    samples = samples.reshape(samples.shape[0], samples.shape[1], -1)
    mse = np.mean((series.squeeze() - forecasts.mean(axis=-1))**2, axis=-1)
    mae = np.mean(np.abs(series.squeeze() - forecasts.mean(axis=-1)), axis=-1)
    errors = np.stack([mse, mae], axis=-1)
     
    # compute likelihood
    log_likelihood = sp.special.logsumexp((forecasts - series)**2 / (2 * var) - 
                                           0.5 * np.log(2 * np.pi * var), axis=-1)
    log_likelihood = np.mean(log_likelihood)
    
    # compute calibration error: 
    cal_error = np.zeros(forecasts.shape[1])
    for p in cal_thresholds:
        q = np.quantile(samples, p, axis=-1)
        est_p = np.mean(series.squeeze() <= q, axis=0)
        cal_error += (est_p - p) ** 2
        
    return errors, log_likelihood, cal_error