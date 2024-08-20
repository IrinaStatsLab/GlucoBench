import sys
import os
import yaml
import datetime
from functools import partial
from pathlib import Path
from typing import Optional

import torch
from torch.utils.tensorboard import SummaryWriter
import typer
import numpy as np

# Import data formatter
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data_formatter.base import *
from lib.gluformer.model import Gluformer
from lib.gluformer.utils.evaluation import test
from utils.darts_processing import load_data, reshuffle_data
from utils.darts_dataset import SamplingDatasetDual, SamplingDatasetInferenceDual

def main(dataset: str = 'livia_mini',
         gpu_id: int = 0,
         reduction1: str = 'mean',
         reduction2: str = 'median',
         reduction3: Optional[str] = None,
         num_samples: int = 1,
         epochs: int = 100,
         n_heads: int = 12,
         batch_size: int = 320,
         activ: str = "gelu"):

    reductions = [reduction1, reduction2, reduction3]

    torch.set_float32_matmul_precision('medium') #to make things a bit faster

    # Define device
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

    # Load data
    study_file = Path(f'./output/gluformer_{dataset}.txt')
    if not study_file.exists():
        with study_file.open("w") as f:
            f.write(f"Training and testing started at {datetime.datetime.now()}\n")

    formatter, series, scalers = load_data(seed=0,
                                           study_file=str(study_file),
                                           dataset=dataset,
                                           use_covs=True,
                                           cov_type='dual',
                                           use_static_covs=True)

    # Set model parameters directly
    in_len = 96  # Fixed input length
    label_len = in_len // 3
    out_len = formatter.params['length_pred']
    max_samples_per_ts = 200  # Fixed max samples per time series
    d_model = 512  # Fixed model dimension
    d_fcn = 1024  # Fixed dimension of FCN
    num_enc_layers = 2  # Fixed number of encoder layers
    num_dec_layers = 2  # Fixed number of decoder layers

    num_dynamic_features = series['train']['future'][-1].n_components
    num_static_features = series['train']['static'][-1].n_components
    model_path = Path(__file__).parent / f'../output/tensorboard_gluformer_{dataset}/model.pt'

    # Set model seed
    writer = SummaryWriter(Path(__file__).parent / f'../output/tensorboard_gluformer_{dataset}/run')

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
                      activ=activ,
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
              batch_size=batch_size,
              epochs=epochs,
              num_samples=num_samples,
              device=device,
              model_path=model_path,
              trial=None,
              logger=writer)

    # Backtest on the test set
    predictions, logvar = model.predict(dataset_test,
                                        batch_size=batch_size,
                                        num_samples=num_samples,
                                        device=device)
    trues = np.array([dataset_test.evalsample(i).values() for i in range(len(dataset_test))])
    trues = (trues - scalers['target'].min_) / scalers['target'].scale_
    predictions = (predictions - scalers['target'].min_) / scalers['target'].scale_
    var = np.exp(logvar) / scalers['target'].scale_**2
    id_errors_sample, id_likelihood_sample, id_cal_errors_sample = test(trues, predictions, var=var)

    # Backtest on the OOD test set
    predictions, logvar = model.predict(dataset_test_ood,
                                        batch_size=batch_size,
                                        num_samples=num_samples,
                                        device=device)
    trues = np.array([dataset_test_ood.evalsample(i).values() for i in range(len(dataset_test_ood))])
    trues = (trues - scalers['target'].min_) / scalers['target'].scale_
    predictions = (predictions - scalers['target'].min_) / scalers['target'].scale_
    var = np.exp(logvar) / scalers['target'].scale_**2
    ood_errors_sample, ood_likelihood_sample, ood_cal_errors_sample = test(trues, predictions, var=var)


    # CREATE THE MODEL OUTPUT FOLDER
    modelsp = Path("output") / "models"
    modelsp.mkdir(exist_ok=True)
    dataset_models = modelsp / dataset
    dataset_models.mkdir(exist_ok=True)
    metricsp = dataset_models / "metrics.csv"
    metricsp.touch(exist_ok=True)

    # Compute, save, and print results
    with study_file.open("a") as f:
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

        model_path = dataset_models / f"gluformer_{num_samples}samples_{epochs}epochs_{n_heads}heads_{batch_size}batch_{activ}activation_{dataset}.pth"
        torch.save(model, str(model_path))

    # Write metrics if the file is not empty
    if metricsp.stat().st_size == 0:
        with metricsp.open("a") as f:
            f.write(f"model,ID RMSE/MAE,OOD RMSE/MAE\n")
    with metricsp.open("a") as f:
        f.write(f"{model_path},{id_errors_sample_red},{ood_errors_sample_red}\n")

if __name__ == '__main__':
    typer.run(main)