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
from torch.optim.lr_scheduler import StepLR
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# import data formatter
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data_formatter.base import *
from utils.darts_processing import load_data, reshuffle_data
from utils.darts_evaluation import rescale_and_test
from utils.darts_training import *
from utils.darts_dataset import SamplingDatasetPast, SamplingDatasetInferencePast

# define objective function
def objective(trial):
    # set parameters
    out_len = formatter.params['length_pred']
    model_name = f'tensorboard_transformer_{args.dataset}' if args.use_covs == 'False' \
        else f'tensorboard_transformer_covariates_{args.dataset}'
    work_dir = os.path.join(os.path.dirname(__file__), '../output')
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
    loss_logger = LossLogger()
    pruner = PyTorchLightningPruningCallback(trial, monitor="val_loss")
    pl_trainer_kwargs = {"accelerator": "gpu", "devices": [0], "callbacks": [el_stopper, loss_logger, pruner], "gradient_clip_val": max_grad_norm}
    # optimizer scheduler
    scheduler_kwargs = {'step_size': lr_epochs, 'gamma': 0.5}

    # create dataset
    train_dataset = SamplingDatasetPast(target_series=series['train']['target'],
                                        covariates=series['train']['dynamic'],
                                        max_samples_per_ts=max_samples_per_ts,
                                        input_chunk_length=in_len,
                                        output_chunk_length=out_len,
                                        use_static_covariates=False)
    val_dataset = SamplingDatasetPast(target_series=series['val']['target'],
                                      covariates=series['val']['dynamic'],
                                      max_samples_per_ts=max_samples_per_ts,
                                      input_chunk_length=in_len,
                                      output_chunk_length=out_len,
                                      use_static_covariates=False)
    
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
    model.fit_from_dataset(train_dataset, val_dataset, verbose=False)
    model.load_from_checkpoint(model_name, work_dir=work_dir)

    # backtest on the validation set
    errors = model.backtest(series['val']['target'],
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
    study_file = f'./output/transformer_{args.dataset}.txt' if args.use_covs == 'False' \
        else f'./output/transformer_covariates_{args.dataset}.txt'
    if not os.path.exists(study_file):
        with open(study_file, "w") as f:
            f.write(f"Optimization started at {datetime.datetime.now()}\n")
    formatter, series, scalers = load_data(study_file=study_file, 
                                           dataset=args.dataset,
                                           use_covs=True if args.use_covs == 'True' else False,
                                           cov_type='past',
                                           use_static_covs=False,)
    
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
        key = "transformer_covariates" if args.use_covs == 'True' else "transformer"
        assert formatter.params[key] is not None, "No saved hyperparameters found for this model"
        best_params = formatter.params[key]

    # set parameters
    out_len = formatter.params['length_pred']
    stride = out_len // 2
    model_name = f'tensorboard_transformer_{args.dataset}' if args.use_covs == 'False' \
        else f'tensorboard_transformer_covariates_{args.dataset}'
    work_dir = os.path.join(os.path.dirname(__file__), '../output')
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
                                                        cov_type='past',
                                                        use_static_covs=False,)
            # model callbacks
            el_stopper = EarlyStopping(monitor="val_loss", patience=10, min_delta=0.001, mode='min') 
            loss_logger = LossLogger()
            pl_trainer_kwargs = {"accelerator": "gpu", "devices": [0], "callbacks": [el_stopper, loss_logger], "gradient_clip_val": max_grad_norm}
            # create datasets
            train_dataset = SamplingDatasetPast(target_series=series['train']['target'],
                                                covariates=series['train']['dynamic'],
                                                max_samples_per_ts=max_samples_per_ts,
                                                input_chunk_length=in_len,
                                                output_chunk_length=out_len,
                                                use_static_covariates=False)
            val_dataset = SamplingDatasetPast(target_series=series['val']['target'],
                                            covariates=series['val']['dynamic'],
                                            max_samples_per_ts=max_samples_per_ts,
                                            input_chunk_length=in_len,
                                            output_chunk_length=out_len,
                                            use_static_covariates=False)
            test_dataset = SamplingDatasetInferencePast(target_series=series['test']['target'],
                                                        covariates=series['test']['dynamic'],
                                                        n=out_len,
                                                        input_chunk_length=in_len,
                                                        output_chunk_length=out_len,
                                                        use_static_covariates=False,
                                                        max_samples_per_ts = None)
            test_ood_dataset = SamplingDatasetInferencePast(target_series=series['test_ood']['target'],
                                                            covariates=series['test_ood']['dynamic'],
                                                            n=out_len,
                                                            input_chunk_length=in_len,
                                                            output_chunk_length=out_len,
                                                            use_static_covariates=False,
                                                            max_samples_per_ts = None)

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
            model.fit_from_dataset(train_dataset, val_dataset, verbose=False)
            model.load_from_checkpoint(model_name, work_dir = work_dir)

            # backtest on the test set
            forecasts = model.predict_from_dataset(n=out_len, 
                                                   input_series_dataset=test_dataset,
                                                   verbose=False)
            trues = [test_dataset.evalsample(i) for i in range(len(test_dataset))]
            id_errors_sample, \
                id_likelihood_sample, \
                    id_cal_errors_sample = rescale_and_test(trues,
                                                            forecasts,  
                                                            [metrics.mse, metrics.mae],
                                                            scalers['target'])
            # backtest on the ood test set
            forecasts = model.predict_from_dataset(n=out_len, 
                                                   input_series_dataset=test_ood_dataset,
                                                   verbose=False)
            trues = [test_ood_dataset.evalsample(i) for i in range(len(test_ood_dataset))]
            ood_errors_sample, \
                ood_likelihood_sample, \
                    ood_cal_errors_sample = rescale_and_test(trues,
                                                            forecasts,  
                                                            [metrics.mse, metrics.mae],
                                                            scalers['target'])
            
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

