import os
import sys
import argparse
import datetime

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

# Load model
from latent_ode.trainer_glunet import LatentODEWrapper
from latent_ode.eval_glunet import test

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
# Utils for darts
from utils.darts_dataset import SamplingDatasetDual, SamplingDatasetInferenceDual
from utils.darts_processing import load_data

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='weinstock')
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--reduction1', type=str, default='mean')
parser.add_argument('--reduction2', type=str, default='median')
parser.add_argument('--reduction3', type=str, default=None)
args = parser.parse_args()

reductions = [args.reduction1, args.reduction2, args.reduction3]

if __name__ == '__main__':
    # Define device
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    study_file = f'./output/latentode_{args.dataset}.txt'
    if not os.path.exists(study_file):
        with open(study_file, "w") as f:
            f.write(f"Training and testing started at {datetime.datetime.now()}\n")
    
    formatter, series, scalers = load_data(seed=0, 
                                           study_file=study_file, 
                                           dataset=args.dataset,
                                           use_covs=True, 
                                           cov_type='dual',
                                           use_static_covs=True)
    
    # Set fixed parameters
    out_len = formatter.params['length_pred']
    in_len = 96  # Fixed input length
    max_samples_per_ts = None  # Unlimited samples per time series
    latents = 20  # Fixed latent dimensions
    rec_dims = 40  # Fixed reconstruction dimensions
    rec_layers = 3  # Fixed number of reconstruction layers
    gen_layers = 3  # Fixed number of generation layers
    units = 150  # Fixed number of units
    gru_units = 150  # Fixed number of GRU units
    model_path = os.path.join(os.path.dirname(__file__),
                              f'../output/tensorboard_latentode_{args.dataset}/model.ckpt')

    writer = SummaryWriter(os.path.join(os.path.dirname(__file__), 
                           f'../output/tensorboard_latentode_{args.dataset}/run'))

    # Create datasets
    dataset_train = SamplingDatasetDual(series['train']['target'],
                                        series['train']['future'],
                                        output_chunk_length=out_len,
                                        input_chunk_length=in_len,
                                        use_static_covariates=True,
                                        max_samples_per_ts=max_samples_per_ts)
    dataset_val = SamplingDatasetDual(series['val']['target'],
                                      series['val']['future'],   
                                      output_chunk_length=out_len,
                                      input_chunk_length=in_len,
                                      use_static_covariates=True)
    dataset_test = SamplingDatasetInferenceDual(target_series=series['test']['target'],
                                                covariates=series['test']['future'],
                                                input_chunk_length=in_len,
                                                output_chunk_length=out_len,
                                                use_static_covariates=True,
                                                array_output_only=True)
    dataset_test_ood = SamplingDatasetInferenceDual(target_series=series['test_ood']['target'],
                                                    covariates=series['test_ood']['future'],
                                                    input_chunk_length=in_len,
                                                    output_chunk_length=out_len,
                                                    use_static_covariates=True,
                                                    array_output_only=True)

    # Build the LatentODE model
    model = LatentODEWrapper(device=device,
                             latents=latents,
                             rec_dims=rec_dims,
                             rec_layers=rec_layers,
                             gen_layers=gen_layers,
                             units=units,
                             gru_units=gru_units)

    # Train the model
    model.fit(dataset_train,
              dataset_val,
              learning_rate=1e-3,
              batch_size=32,
              epochs=100,
              num_samples=1,
              device=device,
              model_path=model_path,
              trial=None,
              logger=writer)

    # Backtest on the test set
    predictions = model.predict(dataset_test,
                                batch_size=32,
                                num_samples=10,
                                device=device)
    trues = np.array([dataset_test.evalsample(i).values() for i in range(len(dataset_test))])
    trues = (trues - scalers['target'].min_) / scalers['target'].scale_
    predictions = (predictions - scalers['target'].min_) / scalers['target'].scale_
    obsrv_std = 0.01 / scalers['target'].scale_
    id_errors_sample, id_likelihood_sample, id_cal_errors_sample = test(trues, predictions, obsrv_std)

    # Backtest on the OOD test set
    predictions = model.predict(dataset_test_ood,
                                batch_size=32,
                                num_samples=10,
                                device=device)
    trues = np.array([dataset_test_ood.evalsample(i).values() for i in range(len(dataset_test_ood))])
    trues = (trues - scalers['target'].min_) / scalers['target'].scale_
    predictions = (predictions - scalers['target'].min_) / scalers['target'].scale_
    obsrv_std = 0.01 / scalers['target'].scale_
    ood_errors_sample, ood_likelihood_sample, ood_cal_errors_sample = test(trues, predictions, obsrv_std)
    
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
