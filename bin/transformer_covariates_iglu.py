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
from torch.optim.lr_scheduler import StepLR
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import statsforecast as sf

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
    
    # attach observed covariate series to known input series
    # there are no dynamic nor future covariates
    # for i in range(len(series['train']['future'])):
    #     series['train']['future'][i] = series['train']['future'][i].concatenate(series['train']['dynamic'][i], axis=1)
    # for i in range(len(series['val']['future'])):
    #     series['val']['future'][i] = series['val']['future'][i].concatenate(series['val']['dynamic'][i], axis=1)
    # for i in range(len(series['test']['future'])):
    #     series['test']['future'][i] = series['test']['future'][i].concatenate(series['test']['dynamic'][i], axis=1)
    # for i in range(len(series['test_ood']['future'])):
    #     series['test_ood']['future'][i] = series['test_ood']['future'][i].concatenate(series['test_ood']['dynamic'][i], axis=1)
    
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

    # attach observed covariate series to known input series
    # for i in range(len(series['train']['future'])):
    #     series['train']['future'][i] = series['train']['future'][i].concatenate(series['train']['dynamic'][i], axis=1)
    # for i in range(len(series['val']['future'])):
    #     series['val']['future'][i] = series['val']['future'][i].concatenate(series['val']['dynamic'][i], axis=1)
    # for i in range(len(series['test']['future'])):
    #     series['test']['future'][i] = series['test']['future'][i].concatenate(series['test']['dynamic'][i], axis=1)
    # for i in range(len(series['test_ood']['future'])):
    #     series['test_ood']['future'][i] = series['test_ood']['future'][i].concatenate(series['test_ood']['dynamic'][i], axis=1)
    
    
    return formatter, series, scalers

# define objective function
def objective(trial):
    # set parameters
    out_len = formatter.params['length_pred']
    model_name = f'tensorboard_transformer_covariates_iglu'
    work_dir = './GitHub/GluNet/output'
    
    # suggest hyperparameters: input size
    in_len = trial.suggest_int("in_len", 96, formatter.params['max_length_input'], step=12)
    max_samples_per_ts = trial.suggest_int("max_samples_per_ts", 50, 200, step=50)
    if max_samples_per_ts < 100:
        max_samples_per_ts = None # unlimited
    # suggest hyperparameters: model
    d_model = trial.suggest_int("d_model", 32, 128, step=32)
    n_heads = trial.suggest_int("n_heads", 2, 4, step=2)
    num_encoder_layers = trial.suggest_int("num_encoder_layers", 1, 4, step=1)
    num_decoder_layers = trial.suggest_int("num_decoder_layers", 1, 4, step=1)
    dim_feedforward = trial.suggest_int("dim_feedforward", 32, 512, step=32)
    dropout = trial.suggest_uniform("dropout", 0, 0.2)
    # suggest hyperparameters: training
    lr = trial.suggest_uniform("lr", 1e-4, 1e-3)
    batch_size = trial.suggest_int("batch_size", 32, 64, step=16)
    lr_epochs = trial.suggest_int("lr_epochs", 2, 20, step=2)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.1, 1)
    # model callbacks
    el_stopper = EarlyStopping(monitor="val_loss", patience=10, min_delta=0.001, mode='min') 
    loss_logger = utils.LossLogger()
    pruner = utils.PyTorchLightningPruningCallback(trial, monitor="val_loss")
    pl_trainer_kwargs = {"accelerator": "gpu", "devices": [0], "callbacks": [el_stopper, loss_logger, pruner], "gradient_clip_val": max_grad_norm}
    # optimizer scheduler
    scheduler_kwargs = {'step_size': lr_epochs, 'gamma': 0.5}
    
    # build the TransformerModel model
    model = models.TransformerModel(input_chunk_length=in_len,
                                    output_chunk_length=out_len, 
                                    d_model=d_model, 
                                    nhead=n_heads, 
                                    num_encoder_layers=num_encoder_layers, 
                                    num_decoder_layers=num_decoder_layers, 
                                    dim_feedforward=dim_feedforward, 
                                    dropout=dropout,
                                    log_tensorboard = True,
                                    pl_trainer_kwargs = pl_trainer_kwargs,
                                    batch_size = batch_size,
                                    optimizer_kwargs = {'lr': lr},
                                    lr_scheduler_cls = StepLR,
                                    lr_scheduler_kwargs = scheduler_kwargs,
                                    save_checkpoints = True,
                                    model_name = model_name,
                                    work_dir = work_dir,
                                    force_reset = True,)

    # train the model
    model.fit(series=series['train']['target'],
              # past_covariates=series['train']['future'], # no future covariates
              val_series=series['val']['target'],
              # val_past_covariates=series['val']['future'], # potential bug?
              max_samples_per_ts=max_samples_per_ts,
              verbose=False,)
    model.load_from_checkpoint(model_name, work_dir=work_dir)

    # backtest on the validation set
    errors = model.backtest(series['val']['target'],
                            # past_covariates=series['val']['future'],
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
    study_file = './GitHub/GluNet/output/transformer_covariates_iglu.txt'

    # check that file exists otherwise create it
    with open(study_file, "a+") as f:
        # write current date and time
        f.write(f"Optimization started at {datetime.datetime.now()}")

    # load data
    formatter, series, scalers = load_data(study_file=study_file)
    study = optuna.create_study(direction="minimize")
    print_call = partial(utils.print_callback, study_file=study_file)
    study.optimize(objective, n_trials=100, 
                   callbacks=[print_call], 
                   catch=(RuntimeError, KeyError))

    # Select best hyperparameters 
    best_params = study.best_trial.params
    # set parameters
    out_len = formatter.params['length_pred']
    stride = out_len // 2
    model_name = f'tensorboard_transformer_covariates_iglu'
    work_dir = os.path.join(os.path.dirname(__file__), '../output')
    print("work directory: ", work_dir)

    # suggest hyperparameters: input size
    in_len = best_params["in_len"]
    max_samples_per_ts = best_params["max_samples_per_ts"]
    if max_samples_per_ts < 100:
        max_samples_per_ts = None # unlimited
    # suggest hyperparameters: model
    d_model = best_params["d_model"]
    n_heads = best_params["n_heads"]
    num_encoder_layers = best_params["num_encoder_layers"]
    num_decoder_layers = best_params["num_decoder_layers"]
    dim_feedforward = best_params["dim_feedforward"]
    dropout = best_params["dropout"]
    # suggest hyperparameters: training
    lr = best_params["lr"]
    batch_size = best_params["batch_size"]
    lr_epochs = best_params["lr_epochs"]
    max_grad_norm = best_params["max_grad_norm"]
    scheduler_kwargs = {'step_size': lr_epochs, 'gamma': 0.5}

    # Set model seed
    model_seeds = list(range(10, 20))
    id_model_results = {'mean': [], 'std': [], 'quantile25': [], 'quantile75': [], 'median': [], 'min': [], 'max': []}
    ood_model_results = {'mean': [], 'std': [], 'quantile25': [], 'quantile75': [], 'median': [], 'min': [], 'max': []}
    for model_seed in model_seeds:
        # Backtest on the test set
        seeds = list(range(1, 3))
        id_errors_stats = {'mean': [], 'std': [], 'quantile25': [], 'quantile75': [], 'median': [], 'min': [], 'max': []}
        ood_errors_stats = {'mean': [], 'std': [], 'quantile25': [], 'quantile75': [], 'median': [], 'min': [], 'max': []}
        for seed in seeds:
            formatter, series, scalers = reshuffle_data(formatter, seed)
            # model callbacks
            el_stopper = EarlyStopping(monitor="val_loss", patience=10, min_delta=0.001, mode='min') 
            loss_logger = utils.LossLogger()
            pl_trainer_kwargs = {"accelerator": "gpu", "devices": [0], "callbacks": [el_stopper, loss_logger], "gradient_clip_val": max_grad_norm}
            # build the model
            model = models.TransformerModel(input_chunk_length=in_len,
                                            output_chunk_length=out_len, 
                                            d_model=d_model, 
                                            nhead=n_heads, 
                                            num_encoder_layers=num_encoder_layers, 
                                            num_decoder_layers=num_decoder_layers, 
                                            dim_feedforward=dim_feedforward, 
                                            dropout=dropout,
                                            log_tensorboard = True,
                                            pl_trainer_kwargs = pl_trainer_kwargs,
                                            batch_size = batch_size,
                                            optimizer_kwargs = {'lr': lr},
                                            lr_scheduler_cls = StepLR,
                                            lr_scheduler_kwargs = scheduler_kwargs,
                                            save_checkpoints = True,
                                            model_name = model_name,
                                            work_dir = work_dir,
                                            force_reset = True,)

            # train the model
            model.fit(series=series['train']['target'],
                   #  past_covariates=series['train']['future'],
                    val_series=series['val']['target'],
                   #  val_past_covariates=series['val']['future'],
                    max_samples_per_ts=max_samples_per_ts,
                    verbose=False,)
            model.load_from_checkpoint(model_name, work_dir = work_dir)

            # backtest on the test set
            forecasts = model.historical_forecasts(series['test']['target'],
                                                  #  past_covariates=series['test']['future'],
                                                    forecast_horizon=out_len, 
                                                    stride=stride,
                                                    retrain=False,
                                                    verbose=False,
                                                    last_points_only=False,
                                                    start=formatter.params["max_length_input"])
            id_errors_sample = utils.rescale_and_backtest(series['test']['target'],
                                        forecasts,  
                                        [metrics.mse, metrics.mae],
                                        scalers['target'],
                                        reduction=None)
            id_errors_sample = np.vstack(id_errors_sample)
            id_error_stats_sample = utils.compute_error_statistics(id_errors_sample)
            for key in id_errors_stats.keys():
                id_errors_stats[key].append(id_error_stats_sample[key])
                
            with open(study_file, "a") as f:
                f.write(f"\t\tModel Seed: {model_seed} Seed: {seed} ID errors (MSE, MAE) stats: {id_error_stats_sample}\n")

            # backtest on the ood test set
            forecasts = model.historical_forecasts(series['test_ood']['target'],
                                                  #  past_covariates=series['test_ood']['future'],
                                                    forecast_horizon=out_len, 
                                                    stride=stride,
                                                    retrain=False,
                                                    verbose=False,
                                                    last_points_only=False,
                                                    start=formatter.params["max_length_input"])
            ood_errors_sample = utils.rescale_and_backtest(series['test_ood']['target'],
                                        forecasts,  
                                        [metrics.mse, metrics.mae],
                                        scalers['target'],
                                        reduction=None)
            ood_errors_sample = np.vstack(ood_errors_sample)
            ood_errors_stats_sample = utils.compute_error_statistics(ood_errors_sample)
            for key in ood_errors_stats.keys():
                ood_errors_stats[key].append(ood_errors_stats_sample[key])
            with open(study_file, "a") as f:
                f.write(f"\t\tModel Seed: {model_seed} Seed: {seed} OOD errors (MSE, MAE) stats: {ood_errors_stats_sample}\n")

        # report estimation error for each statistic
        with open(study_file, "a") as f:
            for key in id_errors_stats.keys():
                id_errors_stats[key] = np.mean(id_errors_stats[key], axis=0)
                id_model_results[key].append(id_errors_stats[key])
            f.write(f"\tModel seed: {model_seed} RS ID (MSE, MAE) errors stats: {id_errors_stats}\n")
            
            for key in ood_errors_stats.keys():
                ood_errors_stats[key] = np.mean(ood_errors_stats[key], axis=0)
                ood_model_results[key].append(ood_errors_stats[key])
            f.write(f"\tModel seed: {model_seed} RS OOD (MSE, MAE) errors stats: {ood_errors_stats}\n")
                
    for key in id_model_results.keys():
        errors = np.vstack(id_model_results[key])
        errors_stats = utils.compute_error_statistics(errors)
        with open(study_file, "a") as f:
            f.write(f"Key: {key} RS ID (MSE, MAE) stats: {errors_stats}\n")
    for key in ood_model_results.keys():
        errors = np.vstack(ood_model_results[key])
        errors_stats = utils.compute_error_statistics(errors)
        with open(study_file, "a") as f:
            f.write(f"Key: {key} RS OOD (MSE, MAE) stats: {errors_stats}\n")