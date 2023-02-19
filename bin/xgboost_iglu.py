# append proper directories into path
import sys
import os
sys.path.append("/content/drive/MyDrive/Colab Notebooks")
sys.path.append("/content/drive/MyDrive/Colab Notebooks/GitHub/GluNet")
sys.path.append("/content/drive/MyDrive/Colab Notebooks/GitHub/GluNet/bin")

# GluNet imports
from data_formatter.base import DataFormatter # in "/Glunet"
import utils # in "/Glunet/bin"

# installed in "MyDrive/Colab Notebooks"
import optuna

import darts
from darts import models, metrics, TimeSeries
from darts.dataprocessing.transformers import Scaler

# built-in packages
import numpy as np
from typing import List, Union, Dict

import yaml
import datetime
from functools import partial

import seaborn as sns
sns.set_style('whitegrid')
import matplotlib.pyplot as plt
import statsmodels.api as sm
import sklearn

# --------------------------------------------------------------------

# define data loader
def load_data(seed = 0, study_file = None):
    # load data
    with open('./GitHub/GluNet/config/iglu.yaml', 'r') as f:
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
    series, scalers = utils.make_series({'train': formatter.train_data,
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
    series, scalers = utils.make_series({'train': formatter.train_data,
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
    in_len = trial.suggest_int("in_len", 144, 200 - 3*out_len, step=12) # at least 3 hours of predictions left
    lr = trial.suggest_float("lr", 0.001, 1.0, step=0.001)
    subsample = trial.suggest_float("subsample", 0.6, 1.0, step=0.1)
    min_child_weight = trial.suggest_float("min_child_weight", 1.0, 5.0, step=1.0)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0, step=0.1)
    max_depth = trial.suggest_int("max_depth", 4, 10, step=1)
    gamma = trial.suggest_float("gamma", 0.5, 10, step=0.5)
    alpha = trial.suggest_float("alpha", 0.001, 0.3, step=0.001)
    lambda_ = trial.suggest_float("lambda_", 0.001, 0.3, step=0.001)
    n_estimators = trial.suggest_int("n_estimators", 256, 512, step=32)
    
    # build the XGBoost model
    model = models.XGBModel(  lags=in_len, 
                              learning_rate=lr,
                              subsample=subsample,
                              min_child_weight=min_child_weight,
                              colsample_bytree=colsample_bytree,
                              max_depth=max_depth,
                              gamma=gamma,
                              reg_alpha=alpha,
                              reg_lambda=lambda_,
                              n_estimators=n_estimators,
                              seed=0
                            )

    # train the model
    model.fit(series['train']['target'],
              max_samples_per_ts=100)

    # backtest on the validation set
    errors = model.backtest(series['val']['target'],
                            forecast_horizon=out_len,
                            stride=out_len,
                            retrain=False,
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
    study_file = './GitHub/GluNet/output/xgboost_iglu.txt'
    # check that file exists otherwise create it
    if not os.path.exists(study_file):
        with open(study_file, "w") as f:
            # write current date and time
            f.write(f"Optimization started at {datetime.datetime.now()}")
    # load data
    formatter, series, scalers = load_data(study_file=study_file)
    # study = optuna.create_study(direction="minimize")
    # print_call = partial(print_callback, study_file=study_file)
    # study.optimize(objective, n_trials=400, 
                  #  callbacks=[print_call], 
                  #  catch=(np.linalg.LinAlgError, KeyError))
    
    # Select best hyperparameters #
    # best_params = study.best_trial.params
    best_params = {'in_len': 144, 'lr': 1.0, 'subsample': 0.8, 'min_child_weight': 1.0, 'colsample_bytree': 1.0, 'max_depth': 7, 'gamma': 0.5, 'alpha': 0.092, 'lambda_': 0.077, 'n_estimators': 480}
    seeds = list(range(10, 20))
    id_errors_stats = {'mean': [], 'std': [], 'quantile25': [], 'quantile75': [], 'median': [], 'min': [], 'max': []}
    ood_errors_stats = {'mean': [], 'std': [], 'quantile25': [], 'quantile75': [], 'median': [], 'min': [], 'max': []}
    for seed in seeds:
        formatter, series, scalers = reshuffle_data(formatter, seed)
        in_len = best_params['in_len']
        lr = best_params['lr']
        subsample = best_params['subsample']
        min_child_weight = best_params['min_child_weight']
        colsample_bytree = best_params['colsample_bytree']
        max_depth = best_params['max_depth']
        gamma = best_params['gamma']
        alpha = best_params['alpha']
        lambda_ = best_params['lambda_']
        n_estimators = best_params['n_estimators']
        out_len = 12

        # build the model
        model = models.XGBModel(lags=in_len, 
                                    learning_rate=lr,
                                    subsample=subsample,
                                    min_child_weight=min_child_weight,
                                    colsample_bytree=colsample_bytree,
                                    max_depth=max_depth,
                                    gamma=gamma,
                                    reg_alpha=alpha,
                                    reg_lambda=lambda_,
                                    n_estimators=n_estimators,
                                    seed=0)
        # train the model
        model.fit(series['train']['target'],
                  max_samples_per_ts=100)

        # backtest on the test set
        forecasts = model.historical_forecasts(series['test']['target'],
                                               forecast_horizon=out_len, 
                                               stride=out_len,
                                               retrain=False,
                                               verbose=False,
                                               last_points_only=False)
        errors = utils.rescale_and_backtest(series['test']['target'],
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
                                               forecast_horizon=out_len, 
                                               stride=out_len,
                                               retrain=False,
                                               verbose=False,
                                               last_points_only=False)
        errors = utils.rescale_and_backtest(series['test_ood']['target'],
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