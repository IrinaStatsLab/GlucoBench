from typing import List, Dict
import sys
import os
import yaml
import datetime
import argparse

from darts import models
from darts import metrics
from darts import TimeSeries
from torch.optim.lr_scheduler import StepLR
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# Import data formatter
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data_formatter.base import *
from utils.darts_processing import load_data, reshuffle_data
from utils.darts_evaluation import rescale_and_test
from utils.darts_training import *
from utils.darts_dataset import SamplingDatasetPast, SamplingDatasetInferencePast

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='weinstock')
parser.add_argument('--use_covs', type=str, default='False')
parser.add_argument('--reduction1', type=str, default='mean')
parser.add_argument('--reduction2', type=str, default='median')
parser.add_argument('--reduction3', type=str, default=None)
args = parser.parse_args()

reductions = [args.reduction1, args.reduction2, args.reduction3]

if __name__ == '__main__':
    # Load data
    study_file = f'./output/transformer_{args.dataset}.txt' if args.use_covs == 'False' \
        else f'./output/transformer_covariates_{args.dataset}.txt'
    
    if not os.path.exists(study_file):
        with open(study_file, "w") as f:
            f.write(f"Training and testing started at {datetime.datetime.now()}\n")
    
    formatter, series, scalers = load_data(study_file=study_file, 
                                           dataset=args.dataset,
                                           use_covs=True if args.use_covs == 'True' else False,
                                           cov_type='past',
                                           use_static_covs=False,)
    
    # Set fixed parameters
    out_len = formatter.params['length_pred']
    in_len = 96  # Fixed input length
    d_model = 64  # Fixed model dimension
    n_heads = 2  # Fixed number of attention heads
    num_encoder_layers = 2  # Fixed number of encoder layers
    num_decoder_layers = 2  # Fixed number of decoder layers
    dim_feedforward = 128  # Fixed dimension of the feedforward network
    dropout = 0.1  # Fixed dropout rate
    lr = 1e-4  # Fixed learning rate
    batch_size = 32  # Fixed batch size
    lr_epochs = 10  # Fixed learning rate scheduler step size
    max_grad_norm = 0.5  # Fixed maximum gradient norm
    scheduler_kwargs = {'step_size': lr_epochs, 'gamma': 0.5}

    model_name = f'tensorboard_transformer_{args.dataset}' if args.use_covs == 'False' \
        else f'tensorboard_transformer_covariates_{args.dataset}'
    work_dir = os.path.join(os.path.dirname(__file__), '../output')

    # Model callbacks
    el_stopper = EarlyStopping(monitor="val_loss", patience=10, min_delta=0.001, mode='min') 
    loss_logger = LossLogger()
    pl_trainer_kwargs = {"accelerator": "gpu", "devices": [0], "callbacks": [el_stopper, loss_logger], "gradient_clip_val": max_grad_norm}

    # Create datasets
    train_dataset = SamplingDatasetPast(target_series=series['train']['target'],
                                        covariates=series['train']['dynamic'],
                                        max_samples_per_ts=None,
                                        input_chunk_length=in_len,
                                        output_chunk_length=out_len,
                                        use_static_covariates=False)
    val_dataset = SamplingDatasetPast(target_series=series['val']['target'],
                                      covariates=series['val']['dynamic'],
                                      max_samples_per_ts=None,
                                      input_chunk_length=in_len,
                                      output_chunk_length=out_len,
                                      use_static_covariates=False)
    test_dataset = SamplingDatasetInferencePast(target_series=series['test']['target'],
                                                covariates=series['test']['dynamic'],
                                                n=out_len,
                                                input_chunk_length=in_len,
                                                output_chunk_length=out_len,
                                                use_static_covariates=False,
                                                max_samples_per_ts=None)
    test_ood_dataset = SamplingDatasetInferencePast(target_series=series['test_ood']['target'],
                                                    covariates=series['test_ood']['dynamic'],
                                                    n=out_len,
                                                    input_chunk_length=in_len,
                                                    output_chunk_length=out_len,
                                                    use_static_covariates=False,
                                                    max_samples_per_ts=None)

    # Build the TransformerModel model
    model = models.TransformerModel(input_chunk_length=in_len,
                                    output_chunk_length=out_len, 
                                    d_model=d_model, 
                                    nhead=n_heads, 
                                    num_encoder_layers=num_encoder_layers, 
                                    num_decoder_layers=num_decoder_layers, 
                                    dim_feedforward=dim_feedforward, 
                                    dropout=dropout,
                                    log_tensorboard=True,
                                    pl_trainer_kwargs=pl_trainer_kwargs,
                                    batch_size=batch_size,
                                    optimizer_kwargs={'lr': lr},
                                    lr_scheduler_cls=StepLR,
                                    lr_scheduler_kwargs=scheduler_kwargs,
                                    save_checkpoints=True,
                                    model_name=model_name,
                                    work_dir=work_dir,
                                    force_reset=True)

    # Train the model
    model.fit_from_dataset(train_dataset, val_dataset, verbose=False)
    model.load_from_checkpoint(model_name, work_dir=work_dir)

    # Backtest on the test set
    forecasts = model.predict_from_dataset(n=out_len, 
                                           input_series_dataset=test_dataset,
                                           verbose=False)
    trues = [test_dataset.evalsample(i) for i in range(len(test_dataset))]
    id_errors_sample, id_likelihood_sample, id_cal_errors_sample = rescale_and_test(trues,
                                                                                    forecasts,  
                                                                                    [metrics.mse, metrics.mae],
                                                                                    scalers['target'])
    
    # Backtest on the OOD test set
    forecasts = model.predict_from_dataset(n=out_len, 
                                           input_series_dataset=test_ood_dataset,
                                           verbose=False)
    trues = [test_ood_dataset.evalsample(i) for i in range(len(test_ood_dataset))]
    ood_errors_sample, ood_likelihood_sample, ood_cal_errors_sample = rescale_and_test(trues,
                                                                                       forecasts,  
                                                                                       [metrics.mse, metrics.mae],
                                                                                       scalers['target'])
    
    # Compute, save, and print results
    with open(study_file, "a") as f:
        for reduction in reductions:
            if reduction is not None:
                # Compute
                reduction_f = getattr(np, reduction)
                id_errors_sample_red = reduction_f(id_errors_sample, axis=0)
                ood_errors_sample_red = reduction_f(ood_errors_sample, axis=0)
                # Print
                f.write(f"ID {reduction} of (MSE, MAE): {id_errors_sample_red}\n")
                f.write(f"OOD {reduction} of (MSE, MAE): {ood_errors_sample_red}\n")

        # Likelihoods and Calibration Errors
        f.write(f"ID likelihoods: {id_likelihood_sample}\n")
        f.write(f"OOD likelihoods: {ood_likelihood_sample}\n")
        f.write(f"ID calibration errors: {id_cal_errors_sample}\n")
        f.write(f"OOD calibration errors: {ood_cal_errors_sample}\n")
