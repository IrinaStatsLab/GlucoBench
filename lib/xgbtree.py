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
    out_len = formatter.params['length_pred']
    # suggest hyperparameters
    in_len = trial.suggest_int("in_len", 24, formatter.params['max_length_input'], step=12)
    lr = trial.suggest_float("lr", 0.001, 1.0, step=0.001)
    subsample = trial.suggest_float("subsample", 0.6, 1.0, step=0.1)
    min_child_weight = trial.suggest_float("min_child_weight", 1.0, 5.0, step=1.0)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.8, 1.0, step=0.1)
    max_depth = trial.suggest_int("max_depth", 4, 10, step=1)
    gamma = trial.suggest_float("gamma", 0.5, 10, step=0.5)
    alpha = trial.suggest_float("alpha", 0.001, 0.3, step=0.001)
    lambda_ = trial.suggest_float("lambda_", 0.001, 0.3, step=0.001)
    n_estimators = trial.suggest_int("n_estimators", 256, 512, step=32)
    lags_past_covariates, lags_future_covariates = set_lags(in_len, args)
    
    # build the XGBoost model
    model = models.XGBModel(lags=in_len, 
                            lags_past_covariates = lags_past_covariates,
                            lags_future_covariates = lags_future_covariates,
                            learning_rate=lr,
                            subsample=subsample,
                            min_child_weight=min_child_weight,
                            colsample_bytree=colsample_bytree,
                            max_depth=max_depth,
                            gamma=gamma,
                            reg_alpha=alpha,
                            reg_lambda=lambda_,
                            n_estimators=n_estimators)

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
    study_file = f'./output/xgboost_{args.dataset}.txt' if args.use_covs == 'False' \
        else f'./output/xgboost_covariates_{args.dataset}.txt'
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
        key = "xgboost_covariates" if args.use_covs == 'True' else "xgboost"
        assert formatter.params[key] is not None, "No saved hyperparameters found for this model"
        best_params = formatter.params[key]
    
    # select best hyperparameters
    in_len = best_params['in_len']
    out_len = formatter.params['length_pred']
    stride = out_len // 2
    lr = best_params['lr']
    subsample = best_params['subsample']
    min_child_weight = best_params['min_child_weight']
    colsample_bytree = best_params['colsample_bytree']
    max_depth = best_params['max_depth']
    gamma = best_params['gamma']
    alpha = best_params['alpha']
    lambda_ = best_params['lambda_']
    n_estimators = best_params['n_estimators']
    lags_past_covariates, lags_future_covariates = set_lags(in_len, args)

    # Set model seed
    model_seeds = list(range(10, 20))
    id_errors_model = {key: [] for key in reductions if key is not None}
    ood_errors_model = {key: [] for key in reductions if key is not None}
    id_likelihoods_model = []; ood_likelihoods_model = []
    id_cal_errors_model = []; ood_cal_errors_model = []
    for model_seed in model_seeds:
        # Backtest on the test set
        seeds = list(range(1, 3))
        id_errors_cv = {key: [] for key in reductions if key is not None}
        ood_errors_cv = {key: [] for key in reductions if key is not None}
        id_likelihoods_cv = []; ood_likelihoods_cv = []
        id_cal_errors_cv = []; ood_cal_errors_cv = []
        for seed in seeds:
            formatter, series, scalers = reshuffle_data(formatter=formatter, 
                                                        seed=seed, 
                                                        use_covs=True if args.use_covs == 'True' else False,
                                                        cov_type='mixed',
                                                        use_static_covs=True)
            # build the model
            model = models.XGBModel(lags=in_len, 
                                    lags_past_covariates = lags_past_covariates,
                                    lags_future_covariates = lags_future_covariates,
                                    learning_rate=lr,
                                    subsample=subsample,
                                    min_child_weight=min_child_weight,
                                    colsample_bytree=colsample_bytree,
                                    max_depth=max_depth,
                                    gamma=gamma,
                                    reg_alpha=alpha,
                                    reg_lambda=lambda_,
                                    n_estimators=n_estimators,
                                    random_state=model_seed)
            # train the model
            model.fit(series['train']['target'],
                      past_covariates=series['train']['dynamic'],
                      future_covariates=series['train']['future'])

            # backtest on the test set
            forecasts = model.historical_forecasts(series['test']['target'],
                                                   past_covariates=series['test']['dynamic'],
                                                   future_covariates=series['test']['future'],
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
            
             # backtest on the ood test set
            forecasts = model.historical_forecasts(series['test_ood']['target'],
                                                   past_covariates=series['test_ood']['dynamic'],
                                                   future_covariates=series['test_ood']['future'],
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
                        f.write(f"\t\tModel Seed: {model_seed} Seed: {seed} ID {reduction} of (MSE, MAE): {id_errors_sample_red}\n")
                        f.write(f"\t\tModel Seed: {model_seed} Seed: {seed} OOD {reduction} of (MSE, MAE) stats: {ood_errors_sample_red}\n")
                # save
                id_likelihoods_cv.append(id_likelihood_sample)
                ood_likelihoods_cv.append(ood_likelihood_sample)
                id_cal_errors_cv.append(id_cal_errors_sample)
                ood_cal_errors_cv.append(ood_cal_errors_sample)
                # print
                f.write(f"\t\tModel Seed: {model_seed} Seed: {seed} ID likelihoods: {id_likelihood_sample}\n")
                f.write(f"\t\tModel Seed: {model_seed} Seed: {seed} OOD likelihoods: {ood_likelihood_sample}\n")
                f.write(f"\t\tModel Seed: {model_seed} Seed: {seed} ID calibration errors: {id_cal_errors_sample}\n")
                f.write(f"\t\tModel Seed: {model_seed} Seed: {seed} OOD calibration errors: {ood_cal_errors_sample}\n")

        # compute, save, and print results
        with open(study_file, "a") as f:
            for reduction in reductions:
                if reduction is not None:
                    # compute
                    id_errors_cv[reduction] = np.vstack(id_errors_cv[reduction])
                    ood_errors_cv[reduction] = np.vstack(ood_errors_cv[reduction])
                    id_errors_cv[reduction] = np.mean(id_errors_cv[reduction], axis=0)
                    ood_errors_cv[reduction] = np.mean(ood_errors_cv[reduction], axis=0)
                    # save
                    id_errors_model[reduction].append(id_errors_cv[reduction])
                    ood_errors_model[reduction].append(ood_errors_cv[reduction])
                    # print
                    f.write(f"\tModel Seed: {model_seed} ID {reduction} of (MSE, MAE): {id_errors_cv[reduction]}\n")
                    f.write(f"\tModel Seed: {model_seed} OOD {reduction} of (MSE, MAE): {ood_errors_cv[reduction]}\n")
            # compute
            id_likelihoods_cv = np.mean(id_likelihoods_cv)
            ood_likelihoods_cv = np.mean(ood_likelihoods_cv)
            id_cal_errors_cv = np.vstack(id_cal_errors_cv)
            ood_cal_errors_cv = np.vstack(ood_cal_errors_cv)
            id_cal_errors_cv = np.mean(id_cal_errors_cv, axis=0)
            ood_cal_errors_cv = np.mean(ood_cal_errors_cv, axis=0)
            # save
            id_likelihoods_model.append(id_likelihoods_cv)
            ood_likelihoods_model.append(ood_likelihoods_cv)
            id_cal_errors_model.append(id_cal_errors_cv)
            ood_cal_errors_model.append(ood_cal_errors_cv)
            # print
            f.write(f"\tModel Seed: {model_seed} ID likelihoods: {id_likelihoods_cv}\n")
            f.write(f"\tModel Seed: {model_seed} OOD likelihoods: {ood_likelihoods_cv}\n")
            f.write(f"\tModel Seed: {model_seed} ID calibration errors: {id_cal_errors_cv}\n")
            f.write(f"\tModel Seed: {model_seed} OOD calibration errors: {ood_cal_errors_cv}\n")
                
    # compute, save, and print results
    with open(study_file, "a") as f:
        for reduction in reductions:
            if reduction is not None:
                # compute mean and std
                id_errors_model[reduction] = np.vstack(id_errors_model[reduction])
                ood_errors_model[reduction] = np.vstack(ood_errors_model[reduction])
                id_mean = np.mean(id_errors_model[reduction], axis=0)
                ood_mean = np.mean(ood_errors_model[reduction], axis=0)
                id_std = np.std(id_errors_model[reduction], axis=0)
                ood_std = np.std(ood_errors_model[reduction], axis=0)
                # print
                f.write(f"ID {reduction} of (MSE, MAE): {id_mean.tolist()} +- {id_std.tolist()}\n")
                f.write(f"OOD {reduction} of (MSE, MAE): {ood_mean.tolist()} +- {ood_std.tolist()}\n")
        # compute mean and std of likelihoods
        id_mean = np.mean(id_likelihoods_model)
        ood_mean = np.mean(ood_likelihoods_model)
        id_std = np.std(id_likelihoods_model)
        ood_std = np.std(ood_likelihoods_model)
        # print
        f.write(f"ID likelihoods: {id_mean} +- {id_std}\n")
        f.write(f"OOD likelihoods: {ood_mean} +- {ood_std}\n")
        # compute mean and std of calibration errors
        id_cal_errors_model = np.vstack(id_cal_errors_model)
        ood_cal_errors_model = np.vstack(ood_cal_errors_model)
        id_means = np.mean(id_cal_errors_model, axis=0)
        ood_means = np.mean(ood_cal_errors_model, axis=0)
        id_stds = np.std(id_cal_errors_model, axis=0)
        ood_stds = np.std(ood_cal_errors_model, axis=0)
        # print
        f.write(f"ID calibration errors: {id_means.tolist()} +- {id_stds.tolist()}\n")
        f.write(f"OOD calibration errors: {ood_means.tolist()} +- {ood_stds.tolist()}\n")

