from typing import List, Union, Dict
import sys
import os
import yaml
import datetime
import argparse

from statsforecast.models import AutoARIMA

# import data formatter
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data_formatter.base import *

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

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='weinstock')
parser.add_argument('--reduction1', type=str, default='mean')
parser.add_argument('--reduction2', type=str, default='median')
parser.add_argument('--reduction3', type=str, default=None)
args = parser.parse_args()
reductions = [args.reduction1, args.reduction2, args.reduction3]
if __name__ == '__main__':
    # study file
    study_file = f'./output/arima_{args.dataset}.txt'
    # check that file exists otherwise create it
    if not os.path.exists(study_file):
        with open(study_file, "w") as f:
            # write current date and time
            f.write(f"Optimization started at {datetime.datetime.now()}\n")
    # load data
    with open(f'./config/{args.dataset}.yaml', 'r') as f:
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
    id_errors_cv = {key: [] for key in reductions if key is not None}
    ood_errors_cv = {key: [] for key in reductions if key is not None}
    for seed in seeds:
        formatter.reshuffle(seed)
        test_data = formatter.test_data.loc[~formatter.test_data.index.isin(formatter.test_idx_ood)]
        test_data_ood = formatter.test_data.loc[formatter.test_data.index.isin(formatter.test_idx_ood)]
        
        # backtest on the ID test set
        id_errors_sample = test_model(test_data, 
                                      formatter.scalers[target_col[0]], 
                                      in_len, 
                                      out_len, 
                                      stride, 
                                      target_col, 
                                      group_col)
        # backtest on the ood test set
        ood_errors_sample = test_model(test_data_ood, 
                                       formatter.scalers[target_col[0]], 
                                       in_len, 
                                       out_len, 
                                       stride, 
                                       target_col, 
                                       group_col)
        # compute, save, and print results
        with open(study_file, "a") as f:
            for reduction in reductions:  
                if reduction is not None:
                    # compute
                    reduction_f = getattr(np, reduction)
                    id_errors_sample_red = reduction_f(id_errors_sample, axis=0)
                    ood_errors_sample_red = reduction_f(ood_errors_sample, axis=0)
                    # save
                    id_errors_cv[reduction].append(id_errors_sample_red)
                    ood_errors_cv[reduction].append(ood_errors_sample_red)
                    # print
                    f.write(f"\tSeed: {seed} ID {reduction} of (MSE, MAE): {id_errors_sample_red.tolist()}\n")
                    f.write(f"\tSeed: {seed} OOD {reduction} of (MSE, MAE) stats: {ood_errors_sample_red.tolist()}\n")
    # compute, save, and print results
    with open(study_file, "a") as f:
        for reduction in reductions:
            if reduction is not None:
                # compute
                id_errors_cv[reduction] = np.vstack(id_errors_cv[reduction])
                ood_errors_cv[reduction] = np.vstack(ood_errors_cv[reduction])
                id_errors_cv[reduction] = np.mean(id_errors_cv[reduction], axis=0)
                ood_errors_cv[reduction] = np.mean(ood_errors_cv[reduction], axis=0)
                # print
                f.write(f"ID {reduction} of (MSE, MAE): {id_errors_cv[reduction].tolist()}\n")
                f.write(f"OOD {reduction} of (MSE, MAE): {ood_errors_cv[reduction].tolist()}\n")