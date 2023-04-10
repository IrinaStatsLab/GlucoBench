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
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# import data formatter
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data_formatter.base import *
from utils.darts_processing import load_data, reshuffle_data
from utils.darts_evaluation import rescale_and_test
from utils.darts_training import *
from utils.darts_dataset import SamplingDatasetMixed, SamplingDatasetInferenceMixed

# define objective function
def objective(trial):
    # set parameters
    out_len = formatter.params['length_pred']
    model_name = f'tensorboard_tft_{args.dataset}' if args.use_covs == 'False' \
        else f'tensorboard_tft_covariates_{args.dataset}'
    work_dir = os.path.join(os.path.dirname(__file__), '../output')
    # suggest hyperparameters: input size
    in_len = trial.suggest_int("in_len", 96, formatter.params['max_length_input'], step=12)
    max_samples_per_ts = trial.suggest_int("max_samples_per_ts", 50, 200, step=50)
    if max_samples_per_ts < 100:
        max_samples_per_ts = None # unlimited
    # suggest hyperparameters: model
    hidden_size = trial.suggest_int("hidden_size", 16, 256, step=16)
    num_attention_heads = trial.suggest_int("num_attention_heads", 1, 4, step=1)
    dropout = trial.suggest_float("dropout", 0.1, 0.3)
    # suggest hyperparameters: training
    lr = trial.suggest_float("lr", 1e-4, 1e-2)
    batch_size = trial.suggest_int("batch_size", 32, 64, step=16)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.01, 1)
    # model callbacks
    el_stopper = EarlyStopping(monitor="val_loss", patience=10, min_delta=0.02, mode='min') 
    loss_logger = LossLogger()
    pruner = PyTorchLightningPruningCallback(trial, monitor="val_loss")
    pl_trainer_kwargs = {"accelerator": "gpu", 
                         "devices": [2], 
                         "callbacks": [el_stopper, loss_logger, pruner],
                         "gradient_clip_val": max_grad_norm,}
    
    # create datasets
    train_dataset = SamplingDatasetMixed(target_series=series['train']['target'],
                                         past_covariates=series['train']['dynamic'],
                                         future_covariates=series['train']['future'],
                                         max_samples_per_ts=max_samples_per_ts,
                                         input_chunk_length=in_len,
                                         output_chunk_length=out_len,
                                         use_static_covariates=True)
    val_dataset = SamplingDatasetMixed(target_series=series['val']['target'],
                                      past_covariates=series['val']['dynamic'],
                                      future_covariates=series['val']['future'],
                                      max_samples_per_ts=max_samples_per_ts,
                                      input_chunk_length=in_len,
                                      output_chunk_length=out_len,
                                      use_static_covariates=True)
    
    # build the TFTModel model
    model = models.TFTModel(input_chunk_length = in_len, 
                            output_chunk_length = out_len, 
                            hidden_size = hidden_size,
                            lstm_layers = 1,
                            num_attention_heads = num_attention_heads,
                            full_attention = False,
                            dropout = dropout,
                            hidden_continuous_size = 8,
                            add_relative_index = True,
                            model_name = model_name,
                            work_dir = work_dir,
                            log_tensorboard = True,
                            pl_trainer_kwargs = pl_trainer_kwargs,
                            batch_size = batch_size,
                            optimizer_kwargs = {'lr': lr},
                            save_checkpoints = True,
                            force_reset=True)
    # train the model
    model.fit_from_dataset(train_dataset, val_dataset, verbose=False)
    model.load_from_checkpoint(model_name, work_dir = work_dir)

    # backtest on the validation set
    errors = model.backtest(series['val']['target'],
                            future_covariates=series['val']['future'],
                            past_covariates=series['val']['dynamic'],
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
    study_file = f'./output/tft_{args.dataset}.txt' if args.use_covs == 'False' \
        else f'./output/tft_covariates_{args.dataset}.txt'
    if not os.path.exists(study_file):
        with open(study_file, "w") as f:
            f.write(f"Optimization started at {datetime.datetime.now()}\n")
    formatter, series, scalers = load_data(study_file=study_file, 
                                           dataset=args.dataset,
                                           use_covs=True if args.use_covs == 'True' else False,
                                           cov_type='mixed',
                                           use_static_covs=True if args.use_covs == 'True' else False)
    
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
        key = "tft_covariates" if args.use_covs == 'True' else "tft"
        assert formatter.params[key] is not None, "No saved hyperparameters found for this model"
        best_params = formatter.params[key]

    # set parameters
    out_len = formatter.params['length_pred']
    stride = out_len // 2
    model_name = f'tensorboard_tft_{args.dataset}' if args.use_covs == 'False' \
        else f'tensorboard_tft_covariates_{args.dataset}'
    work_dir = os.path.join(os.path.dirname(__file__), '../output')
    # suggest hyperparameters: input size
    in_len = best_params["in_len"]
    max_samples_per_ts = best_params["max_samples_per_ts"]
    if max_samples_per_ts < 100:
        max_samples_per_ts = None # unlimited
    # suggest hyperparameters: model
    hidden_size = best_params["hidden_size"]
    num_attention_heads = best_params["num_attention_heads"]
    dropout = best_params["dropout"]
    # suggest hyperparameters: training
    lr = best_params["lr"]
    batch_size = best_params["batch_size"]
    max_grad_norm = best_params["max_grad_norm"]

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
                                                        use_static_covs=True if args.use_covs == 'True' else False)
            # model callbacks
            el_stopper = EarlyStopping(monitor="val_loss", patience=10, min_delta=0.001, mode='min') 
            loss_logger = LossLogger()
            pl_trainer_kwargs = {"accelerator": "gpu", 
                                    "devices": [2], 
                                    "callbacks": [el_stopper, loss_logger],
                                    "gradient_clip_val": max_grad_norm,}
            # create datasets
            train_dataset = SamplingDatasetMixed(target_series=series['train']['target'],
                                                past_covariates=series['train']['dynamic'],
                                                future_covariates=series['train']['future'],
                                                max_samples_per_ts=max_samples_per_ts,
                                                input_chunk_length=in_len,
                                                output_chunk_length=out_len,
                                                use_static_covariates=True)
            val_dataset = SamplingDatasetMixed(target_series=series['val']['target'],
                                            past_covariates=series['val']['dynamic'],
                                            future_covariates=series['val']['future'],
                                            max_samples_per_ts=max_samples_per_ts,
                                            input_chunk_length=in_len,
                                            output_chunk_length=out_len,
                                            use_static_covariates=True)
            test_dataset = SamplingDatasetInferenceMixed(target_series = series['test']['target'],
                                                        past_covariates = series['test']['dynamic'],
                                                        future_covariates = series['test']['future'],
                                                        n=out_len,
                                                        input_chunk_length = in_len,
                                                        output_chunk_length = out_len,
                                                        use_static_covariates = True,
                                                        max_samples_per_ts = None)
            test_ood_dataset = SamplingDatasetInferenceMixed(target_series = series['test_ood']['target'],
                                                            past_covariates = series['test_ood']['dynamic'],
                                                            future_covariates = series['test_ood']['future'],
                                                            n = out_len,
                                                            input_chunk_length = in_len,
                                                            output_chunk_length = out_len,
                                                            use_static_covariates = True,
                                                            max_samples_per_ts = None,)
            # build the model
            model = models.TFTModel(input_chunk_length = in_len, 
                                    output_chunk_length = out_len, 
                                    hidden_size = hidden_size,
                                    lstm_layers = 1,
                                    num_attention_heads = num_attention_heads,
                                    full_attention = False,
                                    dropout = dropout,
                                    hidden_continuous_size = 8,
                                    add_relative_index = True,
                                    model_name = model_name,
                                    work_dir = work_dir,
                                    log_tensorboard = True,
                                    pl_trainer_kwargs = pl_trainer_kwargs,
                                    batch_size = batch_size,
                                    optimizer_kwargs = {'lr': lr},
                                    save_checkpoints = True,
                                    force_reset=True)
            # train the model
            model.fit_from_dataset(train_dataset, val_dataset, verbose=False)
            model.load_from_checkpoint(model_name, work_dir = work_dir)

            # backtest on the test set
            forecasts = model.predict_from_dataset(n=out_len, 
                                                   input_series_dataset=test_dataset, 
                                                   num_samples=20,
                                                   verbose=False)
            trues = [test_dataset.evalsample(i) for i in range(len(test_dataset))]
            id_errors_sample, \
                id_likelihood_sample, \
                    id_cal_errors_sample = rescale_and_test(trues,
                                                            forecasts,  
                                                            [metrics.mse, metrics.mae],
                                                            scalers['target'],
                                                            likelihood="Quantile")
            # backtest on the ood test set
            forecasts = model.predict_from_dataset(n=out_len, 
                                                   input_series_dataset=test_ood_dataset, 
                                                   num_samples=20,
                                                   verbose=False)
            trues = [test_ood_dataset.evalsample(i) for i in range(len(test_ood_dataset))]
            ood_errors_sample, \
                ood_likelihood_sample, \
                    ood_cal_errors_sample = rescale_and_test(trues,
                                                            forecasts,  
                                                            [metrics.mse, metrics.mae],
                                                            scalers['target'],
                                                            likelihood="Quantile")
            
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

