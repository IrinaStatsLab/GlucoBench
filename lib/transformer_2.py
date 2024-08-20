import datetime
import os
import sys

import typer

# Import data formatter
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import torch
from pathlib import Path
from utils.darts_evaluation import rescale_and_test
from torch.optim.lr_scheduler import StepLR
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from utils.darts_processing import load_data
from utils.darts_training import *
from utils.darts_dataset import SamplingDatasetPast, SamplingDatasetInferencePast

# CLI setup using Typer
cli = typer.Typer()

@cli.command()
def main(
        dataset: str = typer.Option('weinstock', help="Dataset to use"),
        use_covs: bool = typer.Option(False, help="Whether to use covariates"),
        reduction1: Optional[str] = typer.Option('mean', help="First reduction method"),
        reduction2: Optional[str] = typer.Option('median', help="Second reduction method"),
        reduction3: Optional[str] = typer.Option(None, help="Third reduction method"),
        batch_size: int = typer.Option(32, help="Batch size for training"),
        activation: str = typer.Option('gelu', help="Activation function for the Transformer model"),
        num_samples: int = typer.Option(3, help="Number of samples for prediction"),
        device: str = typer.Option('cuda', help="Device to use for prediction"),
        epochs: int = typer.Option(100, help="Number of training epochs")
):

    reductions = [reduction1, reduction2, reduction3]
    study_file = Path(f'./output/transformer_{dataset}.txt') if not use_covs \
        else Path(f'./output/transformer_covariates_{dataset}.txt')

    if not study_file.exists():
        with study_file.open("w") as f:
            f.write(f"Training and testing started at {datetime.datetime.now()}\n")

    formatter, series, scalers = load_data(study_file=study_file,
                                           dataset=dataset,
                                           use_covs=use_covs,
                                           cov_type='past',
                                           use_static_covs=False)

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
    lr_epochs = 10  # Fixed learning rate scheduler step size
    max_grad_norm = 0.5  # Fixed maximum gradient norm
    scheduler_kwargs = {'step_size': lr_epochs, 'gamma': 0.5}

    model_name = f'tensorboard_transformer_{dataset}' if not use_covs \
        else f'tensorboard_transformer_covariates_{dataset}'
    work_dir = Path(os.path.dirname(__file__)) / '../output'

    # CREATE THE MODEL OUTPUT FOLDER
    modelsp = Path("output") / "models"
    modelsp.mkdir(exist_ok=True)
    dataset_models = modelsp / dataset
    dataset_models.mkdir(exist_ok=True)
    metricsp = dataset_models / "metrics.csv"
    metricsp.touch(exist_ok=True)

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
                                    activation=activation,
                                    log_tensorboard=True,
                                    pl_trainer_kwargs=pl_trainer_kwargs,
                                    batch_size=batch_size,
                                    optimizer_kwargs={'lr': lr},
                                    lr_scheduler_cls=StepLR,
                                    lr_scheduler_kwargs=scheduler_kwargs,
                                    save_checkpoints=True,
                                    model_name=model_name,
                                    work_dir=str(work_dir),
                                    force_reset=True)

    torch.set_float32_matmul_precision('medium') #to make things a bit faster


    # Train the model
    model.fit_from_dataset(train_dataset, val_dataset, verbose=False)
    model.load_from_checkpoint(model_name, work_dir=str(work_dir))

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

        model_path = dataset_models / f"transformer_{num_samples}samples_{epochs}epochs_{n_heads}heads_{batch_size}batch_{activation}activation_{dataset}.pth"
        torch.save(model, str(model_path))

    # Write metrics if the file is not empty
    if metricsp.stat().st_size == 0:
        with metricsp.open("a") as f:
            f.write(f"model,ID RMSE/MAE,OOD RMSE/MAE\n")
    with metricsp.open("a") as f:
        f.write(f"{model_path},{id_errors_sample_red},{ood_errors_sample_red}\n")

if __name__ == "__main__":
    cli()
