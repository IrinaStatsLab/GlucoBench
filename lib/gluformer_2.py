import sys
import os
import yaml
import datetime
import argparse
from functools import partial

import torch
from torch.utils.tensorboard import SummaryWriter

# Import data formatter
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data_formatter.base import *
from lib.gluformer.model import Gluformer
from lib.gluformer.utils.evaluation import test
from utils.darts_processing import load_data, reshuffle_data
from utils.darts_dataset import SamplingDatasetDual, SamplingDatasetInferenceDual

# Define function for setting lags
def set_lags(in_len, args):
    lags_past_covariates = None
    lags_future_covariates = None
    if args.use_covs == 'True':
        if series['train']['future'] is not None:
            lags_past_covariates = in_len
        if series['train']['static'] is not None:
            lags_future_covariates = (in_len, formatter.params['length_pred'])
    return lags_past_covariates, lags_future_covariates

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
    study_file = f'./output/gluformer_{args.dataset}.txt'
    if not os.path.exists(study_file):
        with open(study_file, "w") as f:
            f.write(f"Training and testing started at {datetime.datetime.now()}\n")
    
    formatter, series, scalers = load_data(seed=0, 
                                           study_file=study_file, 
                                           dataset=args.dataset,
                                           use_covs=True, 
                                           cov_type='dual',
                                           use_static_covs=True)
    
    # Set model parameters directly
    in_len = 96  # Fixed input length
    label_len = in_len // 3
    out_len = formatter.params['length_pred']
    max_samples_per_ts = 200  # Fixed max samples per time series
    d_model = 512  # Fixed model dimension
    n_heads = 8  # Fixed number of attention heads
    d_fcn = 1024  # Fixed dimension of FCN
    num_enc_layers = 2  # Fixed number of encoder layers
    num_dec_layers = 2  # Fixed number of decoder layers
    
    num_dynamic_features = series['train']['future'][-1].n_components
    num_static_features = series['train']['static'][-1].n_components
    model_path = os.path.join(os.path.dirname(__file__),
                              f'../output/tensorboard_gluformer_{args.dataset}/model.pt')

    # Set model seed
    writer = SummaryWriter(os.path.join(os.path.dirname(__file__), 
                           f'../output/tensorboard_gluformer_{args.dataset}/run'))
    
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

    # Build the Gluformer model
    model = Gluformer(d_model=d_model, 
                      n_heads=n_heads, 
                      d_fcn=d_fcn, 
                      r_drop=0.2, 
                      activ='relu', 
                      num_enc_layers=num_enc_layers, 
                      num_dec_layers=num_dec_layers,
                      distil=True, 
                      len_seq=in_len,
                      label_len=label_len,
                      len_pred=out_len,
                      num_dynamic_features=num_dynamic_features,
                      num_static_features=num_static_features)

    # Train the model
    model.fit(dataset_train,
              dataset_val,
              learning_rate=1e-4,
              batch_size=32,
              epochs=100,
              num_samples=1,
              device=device,
              model_path=model_path,
              trial=None,
              logger=writer)

    # Backtest on the test set
    predictions, logvar = model.predict(dataset_test,
                                        batch_size=32,
                                        num_samples=3,
                                        device=device)
    trues = np.array([dataset_test.evalsample(i).values() for i in range(len(dataset_test))])
    trues = (trues - scalers['target'].min_) / scalers['target'].scale_
    predictions = (predictions - scalers['target'].min_) / scalers['target'].scale_
    var = np.exp(logvar) / scalers['target'].scale_**2
    id_errors_sample, id_likelihood_sample, id_cal_errors_sample = test(trues, predictions, var=var)

    # Backtest on the OOD test set
    predictions, logvar = model.predict(dataset_test_ood,
                                        batch_size=32,
                                        num_samples=3,
                                        device=device)
    trues = np.array([dataset_test_ood.evalsample(i).values() for i in range(len(dataset_test_ood))])
    trues = (trues - scalers['target'].min_) / scalers['target'].scale_
    predictions = (predictions - scalers['target'].min_) / scalers['target'].scale_
    var = np.exp(logvar) / scalers['target'].scale_**2
    ood_errors_sample, ood_likelihood_sample, ood_cal_errors_sample = test(trues, predictions, var=var)

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
