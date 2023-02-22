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

import statsforecast as sf

print(sf.__version__)

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
def test_model(test_data, scaler, in_len, out_len, stride, target_col, group_col):
    errors = []
    # print("error array", errors)
    # print(in_len, out_len, stride)
    for group, data in test_data.groupby(group_col):
        train_set = data[target_col].iloc[:in_len].values.flatten()
        # fit model
        model = sf.models.AutoARIMA(start_p = 0,
                        max_p = 10,
                        start_q = 0,
                        max_q = 10,
                        start_P = 0,
                        max_P = 10,
                        start_Q=0,
                        max_Q=10,
                        allowdrift=True,
                        allowmean=True,
                        parallel=False)
        model.fit(train_set)
        # get valid sampling locations for future prediction
        start_idx = np.arange(start=stride, stop=len(data) - in_len - out_len + 1, step=stride)
        end_idx = start_idx + in_len

        # print("start id", start_idx, "end id", end_idx)

        # iterate and collect predictions
        for i in range(len(start_idx)):
            input = data[target_col].iloc[start_idx[i]:end_idx[i]].values.flatten()
            true = data[target_col].iloc[end_idx[i]:(end_idx[i]+out_len)].values.flatten()
            prediction = model.forecast(input, h=out_len)['mean']
            # unscale true and prediction
            true = scaler.inverse_transform(true.reshape(-1, 1)).flatten()
            prediction = scaler.inverse_transform(prediction.reshape(-1, 1)).flatten()
            # collect errors
            errors.append(np.array([np.mean((true - prediction)**2), np.mean(np.abs(true - prediction))]))
            print(errors)

    errors = np.vstack(errors)
    return errors

if __name__ == '__main__':
    # study file
    study_file = './GitHub/GluNet/output/arima_iglu.txt'
    # check that file exists otherwise create it
    if not os.path.exists(study_file):
        with open(study_file, "w") as f:
            # write current date and time
            f.write(f"Optimization started at {datetime.datetime.now()}\n")
    # load data
    with open('./GitHub/GluNet/config/iglu.yaml', 'r') as f:
        config = yaml.safe_load(f)
    config['scaling_params']['scaler'] = 'MinMaxScaler'
    formatter = DataFormatter(config, study_file = study_file)

    # set params
    in_len = formatter.params['max_length_input']
    out_len = formatter.params['length_pred']
    stride = formatter.params['length_pred'] // 2
    target_col = formatter.get_column('target')
    group_col = formatter.get_column('sid')

    seeds = list(range(10, 20))
    id_errors_stats = {'mean': [], 'std': [], 'quantile25': [], 'quantile75': [], 'median': [], 'min': [], 'max': []}
    ood_errors_stats = {'mean': [], 'std': [], 'quantile25': [], 'quantile75': [], 'median': [], 'min': [], 'max': []}
    for seed in seeds:
        formatter.reshuffle(seed)
        test_data = formatter.test_data.loc[~formatter.test_data.index.isin(formatter.test_idx_ood)]
        test_data_ood = formatter.test_data.loc[formatter.test_data.index.isin(formatter.test_idx_ood)]
        
        # backtest on the ID test set
        id_errors_sample = test_model(test_data, formatter.scalers[target_col[0]], in_len, out_len, stride, target_col, group_col)
        id_errors_sample = np.vstack(id_errors_sample)
        id_error_stats_sample = utils.compute_error_statistics(id_errors_sample)
        for key in id_errors_stats.keys():
            id_errors_stats[key].append(id_error_stats_sample[key])
        with open(study_file, "a") as f:
            f.write(f"\tSeed: {seed} ID errors (MSE, MAE) stats: {id_error_stats_sample}\n")
        
        # backtest on the ood test set
        ood_errors_sample = test_model(test_data_ood, formatter.scalers[target_col[0]], in_len, out_len, stride, target_col, group_col)
        ood_errors_sample = np.vstack(ood_errors_sample)
        ood_errors_stats_sample = utils.compute_error_statistics(ood_errors_sample)
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

# if __name__ == '__main__':
#     # Optuna study 
#     study_file = './output/arima_iglu.txt'
#     # check that file exists otherwise create it
#     if not os.path.exists(study_file):
#         with open(study_file, "w") as f:
#             # write current date and time
#             f.write(f"Optimization started at {datetime.datetime.now()}")
#     # load data
#     formatter, series, scalers = load_data(study_file=study_file)
#     study = optuna.create_study(direction="minimize")
#     early_stopping = partial(early_stopping_check, 
#                              early_stopping_rounds=5,
#                              study_file=study_file)
#     print_call = partial(print_callback, study_file=study_file)
#     study.optimize(objective, n_trials=100, 
#                    callbacks=[print_call, early_stopping], 
#                    catch=(np.linalg.LinAlgError, KeyError))
    
#     # Select best hyperparameters #
#     best_params = study.best_trial.params
#     # best_params = { "in_len": 120 } # HARD-CODED: got from training
#     seeds = list(range(10, 20))
#     id_errors_stats = {'mean': [], 'std': [], 'quantile25': [], 'quantile75': [], 'median': [], 'min': [], 'max': []}
#     ood_errors_stats = {'mean': [], 'std': [], 'quantile25': [], 'quantile75': [], 'median': [], 'min': [], 'max': []}
#     for seed in seeds:
#         print("Ran seed: ", seed)
#         formatter, series, scalers = reshuffle_data(formatter, seed)
#         in_len = best_params['in_len']
#         out_len = 12

#         # build the model
#         model = models.AutoARIMA(autoarima_args={'start_p': 0,
#                                          'start_q': 0,
#                                          'max_p': 10,
#                                          'max_q': 10,
#                                          'start_P': 0,
#                                          'start_Q': 0,
#                                          'max_P': 10,
#                                          'max_Q': 10,
#                                          'max_order': None,})

#         # backtest on the test set
#         forecasts = model.historical_forecasts(series['test']['target'],
#                                                train_length=in_len,
#                                                forecast_horizon=out_len, 
#                                                stride=out_len,
#                                                retrain=True,
#                                                verbose=False,
#                                                last_points_only=False)
#         errors = rescale_and_backtest(series['test']['target'],
#                                       forecasts,  
#                                       [metrics.mse, metrics.mae],
#                                       scalers['target'],
#                                       reduction=None)
#         errors = np.vstack(errors)
#         with open(study_file, "a") as f:
#             mean = errors.mean(axis=0); id_errors_stats['mean'].append(mean)
#             median = np.median(errors, axis=0); id_errors_stats['median'].append(median)
#             std = errors.std(axis=0); id_errors_stats['std'].append(std)
#             quantile25 = np.quantile(errors, 0.25, axis=0); id_errors_stats['quantile25'].append(quantile25)
#             quantile75 = np.quantile(errors, 0.75, axis=0); id_errors_stats['quantile75'].append(quantile75)
#             minn = errors.min(axis=0); id_errors_stats['min'].append(minn)
#             maxx = errors.max(axis=0); id_errors_stats['max'].append(maxx)
#             f.write(f"\tSeed: {seed}, Std MSE: {std[0]}, MAE: {std[1]}\n")
#             f.write(f"\tSeed: {seed}, Min MSE: {minn[0]}, MAE: {minn[1]}\n")
#             f.write(f"\tSeed: {seed}, 25% quantile MSE: {quantile25[0]}, MAE: {quantile25[1]}\n")
#             f.write(f"\tSeed: {seed}, Median MSE: {median[0]}, MAE: {median[1]}\n")
#             f.write(f"\tSeed: {seed}, Mean MSE: {mean[0]}, MAE: {mean[1]}\n")
#             f.write(f"\tSeed: {seed}, 75% quantile MSE: {quantile75[0]}, MAE: {quantile75[1]}\n")
#             f.write(f"\tSeed: {seed}, Max MSE: {maxx[0]}, MAE: {maxx[1]}\n")
#         # backtest on the ood test set
#         forecasts = model.historical_forecasts(series['test_ood']['target'],
#                                                train_length=in_len,
#                                                forecast_horizon=out_len, 
#                                                stride=out_len,
#                                                retrain=True,
#                                                verbose=False,
#                                                last_points_only=False)
#         errors = rescale_and_backtest(series['test_ood']['target'],
#                                       forecasts,  
#                                       [metrics.mse, metrics.mae],
#                                       scalers['target'],
#                                       reduction=None)
#         errors = np.vstack(errors)
#         with open(study_file, "a") as f:
#             mean = errors.mean(axis=0); ood_errors_stats['mean'].append(mean)
#             median = np.median(errors, axis=0); ood_errors_stats['median'].append(median)
#             std = errors.std(axis=0); ood_errors_stats['std'].append(std)
#             quantile25 = np.quantile(errors, 0.25, axis=0); ood_errors_stats['quantile25'].append(quantile25)
#             quantile75 = np.quantile(errors, 0.75, axis=0); ood_errors_stats['quantile75'].append(quantile75)
#             minn = errors.min(axis=0); ood_errors_stats['min'].append(minn)
#             maxx = errors.max(axis=0); ood_errors_stats['max'].append(maxx)
#             f.write(f"\tSeed: {seed}, Std OOD MSE: {std[0]}, MAE: {std[1]}\n")
#             f.write(f"\tSeed: {seed}, Min OOD MSE: {minn[0]}, MAE: {minn[1]}\n")
#             f.write(f"\tSeed: {seed}, 25% quantile OOD MSE: {quantile25[0]}, MAE: {quantile25[1]}\n")
#             f.write(f"\tSeed: {seed}, Median OOD MSE: {median[0]}, MAE: {median[1]}\n")
#             f.write(f"\tSeed: {seed}, Mean OOD MSE: {mean[0]}, MAE: {mean[1]}\n")
#             f.write(f"\tSeed: {seed}, 75% quantile OOD MSE: {quantile75[0]}, MAE: {quantile75[1]}\n")
#             f.write(f"\tSeed: {seed}, Max OOD MSE: {maxx[0]}, MAE: {maxx[1]}\n")

#     # report estimation error for each statistic
#     with open(study_file, "a") as f:
#         for key in id_errors_stats.keys():
#             mean = np.mean(id_errors_stats[key], axis=0)
#             std = np.std(id_errors_stats[key], axis=0)
#             f.write(f"ID Mean of {key} of MSE: {mean[0]}, MAE: {mean[1]}\n")
#             f.write(f"ID Std of {key} of MSE: {std[0]}, MAE: {std[1]}\n")
#         for key in ood_errors_stats.keys():
#             mean = np.mean(ood_errors_stats[key], axis=0)
#             std = np.std(ood_errors_stats[key], axis=0)
#             f.write(f"OOD Mean of {key} of MSE: {mean[0]}, MAE: {mean[1]}\n")
#             f.write(f"OOD Std of {key} of MSE: {std[0]}, MAE: {std[1]}\n")

