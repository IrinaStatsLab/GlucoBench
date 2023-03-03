from typing import List, Union, Dict
import sys
import os
import yaml
import datetime
from functools import partial

import seaborn as sns
sns.set_style('whitegrid')
import matplotlib.pyplot as plt
import statsmodels.api as sm
import sklearn
import optuna
import darts

from darts import models
from darts import metrics
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler

# import data formatter
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data_formatter.base import *
from bin.utils import *

# define data loader
def load_data(seed = 0, study_file = None):
    # load data
    with open('./config/weinstock.yaml', 'r') as f:
        config = yaml.safe_load(f)
    config['split_params']['random_state'] = seed
    formatter = DataFormatter(config, study_file = study_file)

    # convert to series
    time_col = formatter.get_column('time')
    group_col = formatter.get_column('sid')
    target_col = formatter.get_column('target')
    static_cols = formatter.get_column('static_covs')
    static_cols = static_cols + [formatter.get_column('id')] if static_cols is not None else [formatter.get_column('id')]
    dynamic_cols = formatter.get_column('dynamic_covs')
    future_cols = formatter.get_column('future_covs')

    # build series
    series, scalers = make_series({'train': formatter.train_data,
                                    'val': formatter.val_data,
                                    'test': formatter.test_data.loc[~formatter.test_data.index.isin(formatter.test_idx_ood)],
                                    'test_ood': formatter.test_data.loc[formatter.test_data.index.isin(formatter.test_idx_ood)]},
                                    time_col,
                                    group_col,
                                    {'target': target_col,
                                    'static': static_cols,
                                    'dynamic': dynamic_cols,
                                    'future': future_cols})
    
    # attach static covariates to series
    for i in range(len(series['train']['target'])):
        static_covs = series['train']['static'][i][0].pd_dataframe()
        series['train']['target'][i] = series['train']['target'][i].with_static_covariates(static_covs)
    for i in range(len(series['val']['target'])):
        static_covs = series['val']['static'][i][0].pd_dataframe()
        series['val']['target'][i] = series['val']['target'][i].with_static_covariates(static_covs)
    for i in range(len(series['test']['target'])):
        static_covs = series['test']['static'][i][0].pd_dataframe()
        series['test']['target'][i] = series['test']['target'][i].with_static_covariates(static_covs)
    for i in range(len(series['test_ood']['target'])):
        static_covs = series['test_ood']['static'][i][0].pd_dataframe()
        series['test_ood']['target'][i] = series['test_ood']['target'][i].with_static_covariates(static_covs)
    
    return formatter, series, scalers

def reshuffle_data(formatter, seed):
    # reshuffle
    formatter.reshuffle(seed)

    # convert to series
    time_col = formatter.get_column('time')
    group_col = formatter.get_column('sid')
    target_col = formatter.get_column('target')
    static_cols = formatter.get_column('static_covs')
    static_cols = static_cols + [formatter.get_column('id')] if static_cols is not None else [formatter.get_column('id')]
    dynamic_cols = formatter.get_column('dynamic_covs')
    future_cols = formatter.get_column('future_covs')

    # build series
    series, scalers = make_series({'train': formatter.train_data,
                                    'val': formatter.val_data,
                                    'test': formatter.test_data.loc[~formatter.test_data.index.isin(formatter.test_idx_ood)],
                                    'test_ood': formatter.test_data.loc[formatter.test_data.index.isin(formatter.test_idx_ood)]},
                                    time_col,
                                    group_col,
                                    {'target': target_col,
                                    'static': static_cols,
                                    'dynamic': dynamic_cols,
                                    'future': future_cols})
    
    # attach static covariates to series
    for i in range(len(series['train']['target'])):
        static_covs = series['train']['static'][i][0].pd_dataframe()
        series['train']['target'][i] = series['train']['target'][i].with_static_covariates(static_covs)
    for i in range(len(series['val']['target'])):
        static_covs = series['val']['static'][i][0].pd_dataframe()
        series['val']['target'][i] = series['val']['target'][i].with_static_covariates(static_covs)
    for i in range(len(series['test']['target'])):
        static_covs = series['test']['static'][i][0].pd_dataframe()
        series['test']['target'][i] = series['test']['target'][i].with_static_covariates(static_covs)
    for i in range(len(series['test_ood']['target'])):
        static_covs = series['test_ood']['static'][i][0].pd_dataframe()
        series['test_ood']['target'][i] = series['test_ood']['target'][i].with_static_covariates(static_covs)
    
    return formatter, series, scalers

# define objective function
def objective(trial):
    # select input and output chunk lengths
    out_len = formatter.params["length_pred"]
    in_len = trial.suggest_int("in_len", 12, formatter.params["max_length_input"], step=12) # at least 2 hours of predictions left
    max_samples_per_ts = trial.suggest_int("max_samples_per_ts", 50, 200, step=50)
    if max_samples_per_ts < 100:
        max_samples_per_ts = None # unlimited

    # build the Linear Regression model
    model = models.LinearRegressionModel(lags = in_len,
                                         lags_future_covariates = (in_len, formatter.params['length_pred']),
                                         output_chunk_length = formatter.params['length_pred'])

    # train the model
    model.fit(series['train']['target'],
            future_covariates=series['train']['future'],
            max_samples_per_ts=max_samples_per_ts)

    # backtest on the validation set
    errors = model.backtest(series['val']['target'],
                            future_covariates = series['val']['future'],
                            forecast_horizon=out_len,
                            stride=out_len,
                            retrain=False,
                            verbose=False,
                            metric=metrics.rmse,
                            last_points_only=False,
                            )
    avg_error = np.mean(errors)

    return avg_error

if __name__ == '__main__':
    # Optuna study 
    study_file = './output/linreg_covariates_weinstock.txt'
    # check that file exists otherwise create it
    if not os.path.exists(study_file):
        with open(study_file, "w") as f:
            # write current date and time
            f.write(f"Optimization started at {datetime.datetime.now()}\n")
    # load data
    formatter, series, scalers = load_data(study_file=study_file)
    study = optuna.create_study(direction="minimize")
    print_call = partial(print_callback, study_file=study_file)
    study.optimize(objective, n_trials=50, 
                   callbacks=[print_call], 
                   catch=(np.linalg.LinAlgError, KeyError))
    
    # Select best hyperparameters
    best_params = study.best_trial.params
    in_len = best_params['in_len']
    out_len = formatter.params["length_pred"]
    stride = out_len // 2
    max_samples_per_ts = best_params['max_samples_per_ts']
    if max_samples_per_ts < 100:
        max_samples_per_ts = None # unlimited

    # Test on ID and OOD data
    seeds = list(range(10, 20))
    id_errors_stats = {'mean': [], 'std': [], 'quantile25': [], 'quantile75': [], 'median': [], 'min': [], 'max': []}
    ood_errors_stats = {'mean': [], 'std': [], 'quantile25': [], 'quantile75': [], 'median': [], 'min': [], 'max': []}
    for seed in seeds:
        formatter, series, scalers = reshuffle_data(formatter, seed)
        # build the model
        model = models.LinearRegressionModel(lags = in_len,
                                             lags_future_covariates = (in_len, formatter.params['length_pred']),
                                             output_chunk_length = formatter.params['length_pred'])
        # train the model
        model.fit(series['train']['target'],
                  future_covariates=series['train']['future'],
                  max_samples_per_ts=max_samples_per_ts)

        # backtest on the test set
        forecasts = model.historical_forecasts(series['test']['target'],
                                               future_covariates = series['test']['future'],
                                               forecast_horizon=out_len, 
                                               stride=stride,
                                               retrain=False,
                                               verbose=False,
                                               last_points_only=False,
                                               start=formatter.params["max_length_input"])
        id_errors_sample = rescale_and_backtest(series['test']['target'],
                                    forecasts,  
                                    [metrics.mse, metrics.mae],
                                    scalers['target'],
                                    reduction=None)
        id_errors_sample = np.vstack(id_errors_sample)
        id_error_stats_sample = compute_error_statistics(id_errors_sample)
        for key in id_errors_stats.keys():
            id_errors_stats[key].append(id_error_stats_sample[key])
        with open(study_file, "a") as f:
            f.write(f"\tSeed: {seed} ID errors (MSE, MAE) stats: {id_error_stats_sample}\n")

        # backtest on the ood test set
        forecasts = model.historical_forecasts(series['test_ood']['target'],
                                               future_covariates = series['test_ood']['future'],
                                               forecast_horizon=out_len, 
                                               stride=stride,
                                               retrain=False,
                                               verbose=False,
                                               last_points_only=False,
                                               start=formatter.params["max_length_input"])
        ood_errors_sample = rescale_and_backtest(series['test_ood']['target'],
                                    forecasts,  
                                    [metrics.mse, metrics.mae],
                                    scalers['target'],
                                    reduction=None)
        ood_errors_sample = np.vstack(ood_errors_sample)
        ood_errors_stats_sample = compute_error_statistics(ood_errors_sample)
        for key in ood_errors_stats.keys():
            ood_errors_stats[key].append(ood_errors_stats_sample[key])
        with open(study_file, "a") as f:
            f.write(f"\tSeed: {seed} OOD errors (MSE, MAE) stats: {ood_errors_stats_sample}\n")

    # report estimation error for each statistic
    with open(study_file, "a") as f:
        for key in id_errors_stats.keys():
            id_errors_stats[key] = np.mean(id_errors_stats[key], axis=0)
        f.write(f"RS ID (MSE, MAE) errors stats: {id_errors_stats}\n")
        
        for key in ood_errors_stats.keys():
            ood_errors_stats[key] = np.mean(ood_errors_stats[key], axis=0)
        f.write(f"RS OOD (MSE, MAE) errors stats: {ood_errors_stats}\n")