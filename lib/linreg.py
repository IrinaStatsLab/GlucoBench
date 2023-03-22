from typing import List, Union, Dict
import sys
import os
import yaml
import datetime
import argparse
from functools import partial
import optuna

from darts import models
from darts import metrics
from darts import TimeSeries

# import data formatter
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data_formatter.base import *
from utils.darts_processing import load_data, reshuffle_data
from utils.darts_evaluation import rescale_and_backtest
from utils.darts_training import print_callback

# lag setter for covariates
def set_lags(in_len, args):
    lags_past_covariates = None
    lags_future_covariates = None
    if args.use_covs == 'True':
        if series['train']['dynamic'] is not None:
            lags_past_covariates = in_len
        if series['train']['future'] is not None:
            lags_future_covariates = (in_len, formatter.params['length_pred'])
    return lags_past_covariates, lags_future_covariates

# define objective function
def objective(trial):
    # select input and output chunk lengths
    out_len = formatter.params["length_pred"]
    in_len = trial.suggest_int("in_len", 12, formatter.params["max_length_input"], step=12) # at least 2 hours of predictions left
    lags_past_covariates, lags_future_covariates = set_lags(in_len, args)

    # build the Linear Regression model
    model = models.LinearRegressionModel(lags = in_len,
                                        lags_past_covariates = lags_past_covariates,
                                        lags_future_covariates = lags_future_covariates,
                                        output_chunk_length = out_len)

    # train the model
    model.fit(series['train']['target'],
              past_covariates=series['train']['dynamic'],
              future_covariates=series['train']['future'])

    # backtest on the validation set
    errors = model.backtest(series['val']['target'],
                            past_covariates=series['val']['dynamic'],
                            future_covariates=series['val']['future'],
                            forecast_horizon=out_len,
                            stride=out_len,
                            retrain=False,
                            verbose=False,
                            metric=metrics.rmse,
                            last_points_only=False,
                            )
    avg_error = np.mean(errors)

    return avg_error

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='weinstock')
parser.add_argument('--use_covs', type=str, default='False')
parser.add_argument('--optuna', type=str, default='True')
parser.add_argument('--reduction1', type=str, default='mean')
parser.add_argument('--reduction2', type=str, default='median')
parser.add_argument('--reduction3', type=str, default=None)
args = parser.parse_args()
reductions = [args.reduction1, args.reduction2, args.reduction3]
if __name__ == '__main__':
    # load data
    study_file = f'./output/linreg_{args.dataset}.txt' if args.use_covs == 'False' \
        else f'./output/linreg_covariates_{args.dataset}.txt'
    if not os.path.exists(study_file):
        with open(study_file, "w") as f:
            f.write(f"Optimization started at {datetime.datetime.now()}\n")
    formatter, series, scalers = load_data(study_file=study_file, 
                                           dataset=args.dataset,
                                           use_covs=True if args.use_covs == 'True' else False,
                                           cov_type='mixed',
                                           use_static_covs=True)
    
    # hyperparameter optimization
    best_params = None
    if args.optuna == 'True':
        study = optuna.create_study(direction="minimize")
        print_call = partial(print_callback, study_file=study_file)
        study.optimize(objective, n_trials=50, 
                    callbacks=[print_call], 
                    catch=(np.linalg.LinAlgError, KeyError))
        best_params = study.best_trial.params
    else:
        key = "linreg_covariates" if args.use_covs == 'True' else "linreg"
        assert formatter.params[key] is not None, "No saved hyperparameters found for this model"
        best_params = formatter.params[key]

    # select best hyperparameters
    in_len = best_params['in_len']
    out_len = formatter.params["length_pred"]
    stride = out_len // 2
    lags_past_covariates, lags_future_covariates = set_lags(in_len, args)

    # test on ID and OOD data
    seeds = list(range(10, 20))
    id_errors_cv = {key: [] for key in reductions if key is not None}
    ood_errors_cv = {key: [] for key in reductions if key is not None}
    id_likelihoods_cv = []; ood_likelihoods_cv = []
    id_cal_errors_cv = []; ood_cal_errors_cv = []
    for seed in seeds:
        formatter, series, scalers = reshuffle_data(formatter, 
                                                    seed, 
                                                    use_covs=True if args.use_covs == 'True' else False,
                                                    cov_type='mixed',
                                                    use_static_covs=True)
        # build the model
        model = models.LinearRegressionModel(lags = in_len,
                                             lags_past_covariates = lags_past_covariates,
                                             lags_future_covariates = lags_future_covariates,
                                             output_chunk_length = formatter.params['length_pred'])
        # train the model
        model.fit(series['train']['target'],
                  past_covariates=series['train']['dynamic'],
                  future_covariates=series['train']['future'])

        # backtest on the test set
        forecasts = model.historical_forecasts(series['test']['target'],
                                               past_covariates = series['test']['dynamic'],
                                               future_covariates = series['test']['future'],
                                               forecast_horizon=out_len, 
                                               stride=stride,
                                               retrain=False,
                                               verbose=False,
                                               last_points_only=False,
                                               start=formatter.params["max_length_input"])
        id_errors_sample, \
            id_likelihood_sample, \
                id_cal_errors_sample = rescale_and_backtest(series['test']['target'],
                                                            forecasts,  
                                                            [metrics.mse, metrics.mae],
                                                            scalers['target'],
                                                            reduction=None)
        # backtest on the OOD set
        forecasts = model.historical_forecasts(series['test_ood']['target'],
                                               past_covariates = series['test_ood']['dynamic'],
                                               future_covariates = series['test_ood']['future'],
                                               forecast_horizon=out_len, 
                                               stride=stride,
                                               retrain=False,
                                               verbose=False,
                                               last_points_only=False,
                                               start=formatter.params["max_length_input"])
                                               
        ood_errors_sample, \
                ood_likelihood_sample, \
                    ood_cal_errors_sample = rescale_and_backtest(series['test_ood']['target'],
                                                                 forecasts,  
                                                                 [metrics.mse, metrics.mae],
                                                                 scalers['target'],
                                                                 reduction=None)
            
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
            # save
            id_likelihoods_cv.append(id_likelihood_sample)
            ood_likelihoods_cv.append(ood_likelihood_sample)
            id_cal_errors_cv.append(id_cal_errors_sample)
            ood_cal_errors_cv.append(ood_cal_errors_sample)
            # print
            f.write(f"\tSeed: {seed} ID likelihoods: {id_likelihood_sample}\n")
            f.write(f"\tSeed: {seed} OOD likelihoods: {ood_likelihood_sample}\n")
            f.write(f"\tSeed: {seed} ID calibration errors: {id_cal_errors_sample.tolist()}\n")
            f.write(f"\tSeed: {seed} OOD calibration errors: {ood_cal_errors_sample.tolist()}\n")

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
        # compute
        id_likelihoods_cv = np.mean(id_likelihoods_cv)
        ood_likelihoods_cv = np.mean(ood_likelihoods_cv)
        id_cal_errors_cv = np.vstack(id_cal_errors_cv)
        ood_cal_errors_cv = np.vstack(ood_cal_errors_cv)
        id_cal_errors_cv = np.mean(id_cal_errors_cv, axis=0)
        ood_cal_errors_cv = np.mean(ood_cal_errors_cv, axis=0)
        # print
        f.write(f"ID likelihoods: {id_likelihoods_cv}\n")
        f.write(f"OOD likelihoods: {ood_likelihoods_cv}\n")
        f.write(f"ID calibration errors: {id_cal_errors_cv.tolist()}\n")
        f.write(f"OOD calibration errors: {ood_cal_errors_cv.tolist()}\n")