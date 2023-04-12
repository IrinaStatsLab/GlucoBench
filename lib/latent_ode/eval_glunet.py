import sys
import os
import yaml
import random
from typing import Optional

import numpy as np 
import scipy as sp
import pandas as pd
import torch

# import likelihood evaluation
from .likelihood_eval import masked_gaussian_log_density

def test(series: np.ndarray, 
         forecasts: np.ndarray, 
         obsrv_std: np.ndarray,
         cal_thresholds: Optional[np.ndarray] = np.linspace(0, 1, 11),
    ):
    """
    Test the (rescaled to original scale) forecasts on the series.

    Parameters
    ----------
    series
        The target time series of shape (n_traj, n_tp, n_dim), 
        where t is length of prediction.
    forecasts
        The forecasted means of mixture components of shape (n_traj_samples, n_traj, n_tp, n_dim)
        where k is the number of mixture components.
    obsrv_std
        The forecasted std of mixture components of shape (1).
    cal_thresholds
        The thresholds to use for computing the calibration error.

    Returns
    -------
    np.ndarray
        Error array. Array of shape (n_traj, 2), where 
        along last dimension, we have MSE and MAE.
    float
        The estimated log-likelihood of the model on the data.
    np.ndarray
        The ECE for each time point in the forecast.
    """
    mse = np.mean((series - forecasts.mean(axis=0))**2, axis=-2)
    mae = np.mean(np.abs(series - forecasts.mean(axis=0)), axis=-2)
    errors = np.stack([mse.squeeze(), mae.squeeze()], axis=-1)

    # compute likelihood
    series, forecasts = torch.tensor(series), torch.tensor(forecasts)
    obsrv_std = torch.Tensor(obsrv_std)

    series_repeated = series.repeat(forecasts.size(0), 1, 1, 1)            
    log_density_data = masked_gaussian_log_density(forecasts, series_repeated, 
                                                    obsrv_std = obsrv_std, mask = None)
    log_density_data = log_density_data.permute(1,0)
    log_density = torch.mean(log_density_data, 1)
    log_likelihood = torch.mean(log_density).item()

    # compute calibration error
    samples = torch.distributions.Normal(loc=forecasts, scale=obsrv_std).sample((100, ))
    samples = samples.view(samples.shape[0] * samples.shape[1], 
                           samples.shape[2], 
                           samples.shape[3])
    series = series.squeeze()
    cal_error = torch.zeros(series.shape[1])
    for p in cal_thresholds:
        q = torch.quantile(samples, p, dim=0)
        est_p = torch.mean((series <= q).float(), dim=0)
        cal_error += (est_p - p) ** 2
    cal_error = cal_error.numpy()
        
    return errors, log_likelihood, cal_error