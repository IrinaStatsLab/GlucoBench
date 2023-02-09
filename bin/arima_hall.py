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
    with open('./config/hall.yaml', 'r') as f:
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
    
    return formatter, series, scalers


# define objective function
def objective(trial):
    # select input and output chunk lengths
    out_len = 12 # 1 hour
    in_len = trial.suggest_int("in_len", 96, 240 - 2 * out_len, step=12) # at least 2 hours of predictions left
    
    # build the ARIMA model
    model = models.AutoARIMA(autoarima_args={'start_p': 0,
                                         'start_q': 0,
                                         'max_p': 10,
                                         'max_q': 10,
                                         'start_P': 0,
                                         'start_Q': 0,
                                         'max_P': 10,
                                         'max_Q': 10,
                                         'max_order': None,})

    # backtest on the validation set
    errors = model.backtest(series['val']['target'],
                            train_length=in_len,
                            forecast_horizon=out_len,
                            stride=out_len,
                            retrain=True,
                            verbose=False,
                            metric=metrics.rmse,
                            last_points_only=False,
                            )
    avg_error = np.mean(errors)

    return avg_error

# for convenience, print some optimization trials information
def print_callback(study, trial, study_file=None):
    # write output to a file
    with open(study_file, "a") as f:
        f.write(f"\nCurrent value: {trial.value}, Current params: {trial.params}")
        f.write(f"\nBest value: {study.best_value}, Best params: {study.best_trial.params}")

if __name__ == '__main__':
    # Optuna study 
    study_file = './output/arima_hall.txt'
    # check that file exists otherwise create it
    if not os.path.exists(study_file):
        with open(study_file, "w") as f:
            # write current date and time
            f.write(f"Optimization started at {datetime.datetime.now()}")
    # load data
    formatter, series, scalers = load_data(study_file=study_file)
    study = optuna.create_study(direction="minimize")
    early_stopping = partial(early_stopping_check, 
                             early_stopping_rounds=5,
                             study_file=study_file)
    print_call = partial(print_callback, study_file=study_file)
    study.optimize(objective, n_trials=100, 
                   callbacks=[print_call, early_stopping], 
                   catch=(np.linalg.LinAlgError, KeyError))
    
    # Select best hyperparameters #
    best_params = study.best_trial.params
    seeds = list(range(10, 20))
    id_errors_stats = {'mean': [], 'std': [], 'quantile25': [], 'quantile75': [], 'median': [], 'min': [], 'max': []}
    ood_errors_stats = {'mean': [], 'std': [], 'quantile25': [], 'quantile75': [], 'median': [], 'min': [], 'max': []}
    for seed in seeds:
        formatter, series, scalers = reshuffle_data(formatter, seed)
        in_len = best_params['in_len']
        out_len = 12

        # build the model
        model = models.AutoARIMA(autoarima_args={'start_p': 0,
                                         'start_q': 0,
                                         'max_p': 10,
                                         'max_q': 10,
                                         'start_P': 0,
                                         'start_Q': 0,
                                         'max_P': 10,
                                         'max_Q': 10,
                                         'max_order': None,})

        # backtest on the test set
        forecasts = model.historical_forecasts(series['test']['target'],
                                               train_length=in_len,
                                               forecast_horizon=out_len, 
                                               stride=out_len,
                                               retrain=True,
                                               verbose=False,
                                               last_points_only=False)
        errors = rescale_and_backtest(series['test']['target'],
                                      forecasts,  
                                      [metrics.mse, metrics.mae],
                                      scalers['target'],
                                      reduction=None)
        errors = np.vstack(errors)
        with open(study_file, "a") as f:
            mean = errors.mean(axis=0); id_errors_stats['mean'].append(mean)
            median = np.median(errors, axis=0); id_errors_stats['median'].append(median)
            std = errors.std(axis=0); id_errors_stats['std'].append(std)
            quantile25 = np.quantile(errors, 0.25, axis=0); id_errors_stats['quantile25'].append(quantile25)
            quantile75 = np.quantile(errors, 0.75, axis=0); id_errors_stats['quantile75'].append(quantile75)
            minn = errors.min(axis=0); id_errors_stats['min'].append(minn)
            maxx = errors.max(axis=0); id_errors_stats['max'].append(maxx)
            f.write(f"\tSeed: {seed}, Std MSE: {std[0]}, MAE: {std[1]}\n")
            f.write(f"\tSeed: {seed}, Min MSE: {minn[0]}, MAE: {minn[1]}\n")
            f.write(f"\tSeed: {seed}, 25% quantile MSE: {quantile25[0]}, MAE: {quantile25[1]}\n")
            f.write(f"\tSeed: {seed}, Median MSE: {median[0]}, MAE: {median[1]}\n")
            f.write(f"\tSeed: {seed}, Mean MSE: {mean[0]}, MAE: {mean[1]}\n")
            f.write(f"\tSeed: {seed}, 75% quantile MSE: {quantile75[0]}, MAE: {quantile75[1]}\n")
            f.write(f"\tSeed: {seed}, Max MSE: {maxx[0]}, MAE: {maxx[1]}\n")
        # backtest on the ood test set
        forecasts = model.historical_forecasts(series['test_ood']['target'],
                                               train_length=in_len,
                                               forecast_horizon=out_len, 
                                               stride=out_len,
                                               retrain=True,
                                               verbose=False,
                                               last_points_only=False)
        errors = rescale_and_backtest(series['test_ood']['target'],
                                      forecasts,  
                                      [metrics.mse, metrics.mae],
                                      scalers['target'],
                                      reduction=None)
        errors = np.vstack(errors)
        with open(study_file, "a") as f:
            mean = errors.mean(axis=0); ood_errors_stats['mean'].append(mean)
            median = np.median(errors, axis=0); ood_errors_stats['median'].append(median)
            std = errors.std(axis=0); ood_errors_stats['std'].append(std)
            quantile25 = np.quantile(errors, 0.25, axis=0); ood_errors_stats['quantile25'].append(quantile25)
            quantile75 = np.quantile(errors, 0.75, axis=0); ood_errors_stats['quantile75'].append(quantile75)
            minn = errors.min(axis=0); ood_errors_stats['min'].append(minn)
            maxx = errors.max(axis=0); ood_errors_stats['max'].append(maxx)
            f.write(f"\tSeed: {seed}, Std OOD MSE: {std[0]}, MAE: {std[1]}\n")
            f.write(f"\tSeed: {seed}, Min OOD MSE: {minn[0]}, MAE: {minn[1]}\n")
            f.write(f"\tSeed: {seed}, 25% quantile OOD MSE: {quantile25[0]}, MAE: {quantile25[1]}\n")
            f.write(f"\tSeed: {seed}, Median OOD MSE: {median[0]}, MAE: {median[1]}\n")
            f.write(f"\tSeed: {seed}, Mean OOD MSE: {mean[0]}, MAE: {mean[1]}\n")
            f.write(f"\tSeed: {seed}, 75% quantile OOD MSE: {quantile75[0]}, MAE: {quantile75[1]}\n")
            f.write(f"\tSeed: {seed}, Max OOD MSE: {maxx[0]}, MAE: {maxx[1]}\n")

    # report estimation error for each statistic
    with open(study_file, "a") as f:
        for key in id_errors_stats.keys():
            mean = np.mean(id_errors_stats[key], axis=0)
            std = np.std(id_errors_stats[key], axis=0)
            f.write(f"ID Mean of {key} of MSE: {mean[0]}, MAE: {mean[1]}\n")
            f.write(f"ID Std of {key} of MSE: {std[0]}, MAE: {std[1]}\n")
        for key in ood_errors_stats.keys():
            mean = np.mean(ood_errors_stats[key], axis=0)
            std = np.std(ood_errors_stats[key], axis=0)
            f.write(f"OOD Mean of {key} of MSE: {mean[0]}, MAE: {mean[1]}\n")
            f.write(f"OOD Std of {key} of MSE: {std[0]}, MAE: {std[1]}\n")

