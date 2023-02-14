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

from statsforecast.models import AutoARIMA

# import data formatter
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data_formatter.base import *
from bin.utils import *

def test_model(test_data, scaler, in_len, out_len, stride, target_col, group_col):
    errors = []
    for group, data in test_data.groupby(group_col):
        train_set = data[target_col].iloc[:in_len].values.flatten()
        # fit model
        model = AutoARIMA(start_p = 0,
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
        # iterate and collect predictions
        for i in range(len(start_idx)):
            input = data[target_col].iloc[start_idx[i]:end_idx[i]].values.flatten()
            true = data[target_col].iloc[end_idx[i]:(end_idx[i]+out_len)].values.flatten()
            prediction = model.forward(input, h=out_len)['mean']
            # unscale true and prediction
            true = scaler.inverse_transform(true.reshape(-1, 1)).flatten()
            prediction = scaler.inverse_transform(prediction.reshape(-1, 1)).flatten()
            # collect errors
            errors.append(np.array([np.mean((true - prediction)**2), np.mean(np.abs(true - prediction))]))
    errors = np.vstack(errors)
    return errors

if __name__ == '__main__':
    # study file
    study_file = './output/arima_weinstock.txt'
    # check that file exists otherwise create it
    if not os.path.exists(study_file):
        with open(study_file, "w") as f:
            # write current date and time
            f.write(f"Optimization started at {datetime.datetime.now()}")
    # load data
    with open('./config/weinstock.yaml', 'r') as f:
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
        errors = test_model(test_data, formatter.scalers[target_col[0]], in_len, out_len, stride, target_col, group_col)
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
        errors = test_model(test_data_ood, formatter.scalers[target_col[0]], in_len, out_len, stride, target_col, group_col)
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

