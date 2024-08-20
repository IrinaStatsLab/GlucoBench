import sys
import os
import pickle
import gzip
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats

import darts
from darts import metrics

from lib.gluformer.model import *
from lib.latent_ode.trainer_glunet import *
from utils.darts_processing import *
from utils.darts_dataset import *

# Ensure directories for saving results exist
os.makedirs('./paper_results/data', exist_ok=True)
os.makedirs('./paper_results/plots', exist_ok=True)

# MODELS: TRANSFORMER, NHiTS, TFT, XGBOOST, LINEAR REGRESSION

# model params
model_params = {
    'transformer': {'darts': models.TransformerModel, 'darts_data': SamplingDatasetInferencePast, 'use_covs': False, 'use_static_covs': False, 'cov_type': 'past'},
    'nhits': {'darts': models.NHiTSModel, 'darts_data': SamplingDatasetInferencePast, 'use_covs': False, 'use_static_covs': False, 'cov_type': 'past'},
    'tft': {'darts': models.TFTModel, 'darts_data': SamplingDatasetInferenceMixed, 'use_covs': False, 'use_static_covs': True, 'cov_type': 'mixed'},
    'xgboost': {'darts': models.XGBModel, 'use_covs': False, 'use_static_covs': False, 'cov_type': 'past'},
    'linreg': {'darts': models.LinearRegressionModel, 'use_covs': False, 'use_static_covs': False, 'cov_type': 'past'}
}

# data sets
datasets = ['weinstock', 'dubosson', 'colas', 'iglu', 'hall']
save_trues = {}
save_forecasts = {}
save_inputs = {}

# iterate through models and datasets
for model_name in model_params.keys():
    for dataset in datasets:
        print(f'Testing {model_name} for {dataset}')
        formatter, series, scalers = load_data(seed=0, study_file=None, dataset=dataset, 
                                               use_covs=model_params[model_name]['use_covs'], 
                                               use_static_covs=model_params[model_name]['use_static_covs'],
                                               cov_type=model_params[model_name]['cov_type'])
        # load model or refit model
        if model_name in ['tft', 'transformer', 'nhits']:
            # load model
            model = model_params[model_name]['darts'](input_chunk_length=formatter.params[model_name]['in_len'],
                                                      output_chunk_length=formatter.params['length_pred'])
            model = model.load_from_checkpoint(f'tensorboard_{model_name}_{dataset}', work_dir='./output', best=True)
            # define dataset for inference
            test_dataset = model_params[model_name]['darts_data'](target_series=series['test']['target'],
                                                                  n=formatter.params['length_pred'],
                                                                  input_chunk_length=formatter.params[model_name]['in_len'],
                                                                  output_chunk_length=formatter.params['length_pred'],
                                                                  use_static_covariates=model_params[model_name]['use_static_covs'],
                                                                  max_samples_per_ts=None)
            # get predictions
            forecasts = model.predict_from_dataset(n=formatter.params['length_pred'], 
                                                   input_series_dataset=test_dataset,
                                                   verbose=True,
                                                   num_samples=20 if model_name == 'tft' else 1)
            forecasts = scalers['target'].inverse_transform(forecasts)
            save_forecasts[f'{model_name}_{dataset}'] = forecasts
            # get true values
            save_trues[f'{model_name}_{dataset}'] = [test_dataset.evalsample(i) for i in range(len(test_dataset))]
            save_trues[f'{model_name}_{dataset}'] = scalers['target'].inverse_transform(save_trues[f'{model_name}_{dataset}'])
            # get inputs
            inputs = [test_dataset[i][0] for i in range(len(test_dataset))]
            save_inputs[f'{model_name}_{dataset}'] = (np.array(inputs) - scalers['target'].min_) / scalers['target'].scale_

        elif model_name == 'xgboost':
            # load and fit model
            model = model_params[model_name]['darts'](lags=formatter.params[model_name]['in_len'], 
                                                      learning_rate=formatter.params[model_name]['lr'],
                                                      subsample=formatter.params[model_name]['subsample'],
                                                      min_child_weight=formatter.params[model_name]['min_child_weight'],
                                                      colsample_bytree=formatter.params[model_name]['colsample_bytree'],
                                                      max_depth=formatter.params[model_name]['max_depth'],
                                                      gamma=formatter.params[model_name]['gamma'],
                                                      reg_alpha=formatter.params[model_name]['alpha'],
                                                      reg_lambda=formatter.params[model_name]['lambda_'],
                                                      n_estimators=formatter.params[model_name]['n_estimators'],
                                                      random_state=0)
            model.fit(series['train']['target'])
            # get predictions
            forecasts = model.historical_forecasts(series['test']['target'],
                                                   forecast_horizon=formatter.params['length_pred'],
                                                   stride=1,
                                                   retrain=False,
                                                   verbose=True,
                                                   last_points_only=False)
            forecasts = [scalers['target'].inverse_transform(forecast) for forecast in forecasts]
            save_forecasts[f'{model_name}_{dataset}'] = forecasts
            # get true values
            save_trues[f'{model_name}_{dataset}'] = scalers['target'].inverse_transform(series['test']['target'])

        elif model_name == 'linreg':
            # load and fit model
            model = models.LinearRegressionModel(lags=formatter.params[model_name]['in_len'],
                                                 output_chunk_length=formatter.params['length_pred'])
            model.fit(series['train']['target'])
            # get predictions
            forecasts = model.historical_forecasts(series['test']['target'],
                                                   forecast_horizon=formatter.params['length_pred'], 
                                                   stride=1,
                                                   retrain=False,
                                                   verbose=False,
                                                   last_points_only=False)
            forecasts = [scalers['target'].inverse_transform(forecast) for forecast in forecasts]
            save_forecasts[f'{model_name}_{dataset}'] = forecasts
            # get true values
            save_trues[f'{model_name}_{dataset}'] = scalers['target'].inverse_transform(series['test']['target'])

# MODELS: LATENT ODE and GLUFORMER
device = 'cuda'
for dataset in datasets:
    print(f'Testing {dataset}')
    formatter, series, scalers = load_data(seed=0, study_file=None, dataset=dataset, use_covs=True, use_static_covs=True)
    # define dataset for inference: gluformer
    dataset_test_glufo = SamplingDatasetInferenceDual(target_series=series['test']['target'],
                                                      covariates=series['test']['future'],
                                                      input_chunk_length=formatter.params['gluformer']['in_len'],
                                                      output_chunk_length=formatter.params['length_pred'],
                                                      use_static_covariates=True,
                                                      array_output_only=True)
    # define dataset for inference: latent ode
    dataset_test_latod = SamplingDatasetInferenceDual(target_series=series['test']['target'],
                                                      covariates=series['test']['future'],
                                                      input_chunk_length=formatter.params['latentode']['in_len'],
                                                      output_chunk_length=formatter.params['length_pred'],
                                                      use_static_covariates=True,
                                                      array_output_only=True)
    # load model: gluformer
    num_dynamic_features = series['train']['future'][-1].n_components
    num_static_features = series['train']['static'][-1].n_components
    glufo = Gluformer(d_model=formatter.params['gluformer']['d_model'],
                      n_heads=formatter.params['gluformer']['n_heads'],
                      d_fcn=formatter.params['gluformer']['d_fcn'],
                      r_drop=0.2, 
                      activ='relu', 
                      num_enc_layers=formatter.params['gluformer']['num_enc_layers'], 
                      num_dec_layers=formatter.params['gluformer']['num_dec_layers'],
                      distil=True, 
                      len_seq=formatter.params['gluformer']['in_len'],
                      label_len=formatter.params['gluformer']['in_len'] // 3,
                      len_pred=formatter.params['length_pred'],
                      num_dynamic_features=num_dynamic_features,
                      num_static_features=num_static_features)
    glufo.to(device)
    glufo.load_state_dict(torch.load(f'./output/tensorboard_gluformer_{dataset}/model.pt', map_location=torch.device(device)))

    # load model: latent ode
    latod = LatentODEWrapper(device=device,
                             latents=formatter.params['latentode']['latents'],
                             rec_dims=formatter.params['latentode']['rec_dims'],
                             rec_layers=formatter.params['latentode']['rec_layers'],
                             gen_layers=formatter.params['latentode']['gen_layers'],
                             units=formatter.params['latentode']['units'],
                             gru_units=formatter.params['latentode']['gru_units'])
    latod.load(f'./output/tensorboard_latentode_{dataset}/model.ckpt', device)

    # get predictions: gluformer
    print('Gluformer')
    forecasts, _ = glufo.predict(dataset_test_glufo,
                                 batch_size=8,
                                 num_samples=10,
                                 device=device,
                                 use_tqdm=True)
    forecasts = scalers['target'].inverse_transform(forecasts)
    trues = [dataset_test_glufo.evalsample(i) for i in range(len(dataset_test_glufo))]
    trues = scalers['target'].inverse_transform(trues)
    inputs = [dataset_test_glufo[i][0] for i in range(len(dataset_test_glufo))]
    inputs = (np.array(inputs) - scalers['target'].min_) / scalers['target'].scale_
    save_forecasts[f'gluformer_{dataset}'] = forecasts
    save_trues[f'gluformer_{dataset}'] = trues
    save_inputs[f'gluformer_{dataset}'] = inputs

    # get predictions: latent ode
    print('Latent ODE')
    forecasts = latod.predict(dataset_test_latod,
                              batch_size=32,
                              num_samples=20,
                              device=device,
                              use_tqdm=True)
    forecasts = scalers['target'].inverse_transform(forecasts)
    trues = [dataset_test_latod.evalsample(i) for i in range(len(dataset_test_latod))]
    trues = scalers['target'].inverse_transform(trues)
    inputs = [dataset_test_latod[i][0] for i in range(len(dataset_test_latod))]
    inputs = (np.array(inputs) - scalers['target'].min_) / scalers['target'].scale_
    save_forecasts[f'latentode_{dataset}'] = forecasts
    save_trues[f'latentode_{dataset}'] = trues
    save_inputs[f'latentode_{dataset}'] = inputs

# save forecasts
with gzip.open('./paper_results/data/compressed_forecasts.pkl', 'wb') as file:
    pickle.dump(save_forecasts, file)

# save true values
with gzip.open('./paper_results/data/compressed_trues.pkl', 'wb') as file:
    pickle.dump(save_trues, file)

# save inputs
with gzip.open('./paper_results/data/compressed_inputs.pkl', 'wb') as file:
    pickle.dump(save_inputs, file)

# Load the saved forecasts, trues, and inputs for further analysis or plotting

# define the color gradient
colors = ['#00264c', '#0a2c62', '#14437f', '#1f5a9d', '#2973bb', '#358ad9', '#4d9af4', '#7bb7ff', '#add5ff', '#e6f3ff']
cmap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)

# set matplotlib theme to seaborn whitegrid
sns.set_theme(style="whitegrid")

# load forecasts
with gzip.open('./paper_results/data/compressed_forecasts.pkl', 'rb') as file:
    save_forecasts = pickle.load(file)
save_forecasts['linreg_weinstock'] = [item for sublist in save_forecasts['linreg_weinstock'] for item in sublist]

# load true values
with gzip.open('./paper_results/data/compressed_trues.pkl', 'rb') as file:
    save_trues = pickle.load(file)

# load inputs
with gzip.open('./paper_results/data/compressed_inputs.pkl', 'rb') as file:
    save_inputs = pickle.load(file)

# plot forecasts on Weinstock for all models
models = ['linreg', 'latentode', 'transformer', 'gluformer', 'tft']
models = [f'{name}_weinstock' for name in models]
offsets = {
    'gluformer_weinstock': 0, 
    'latentode_weinstock': 144-48, 
    'tft_weinstock': 144-132, 
    'linreg_weinstock': 144-84, 
    'transformer_weinstock': 144-96
}
sidx = [50, 75, 100, 150, 175]

fig, axs = plt.subplots(len(models), 5, figsize=(20, 20))

inputs = save_inputs['tft_weinstock'].squeeze()
trues = save_trues['tft_weinstock']
trues = np.array([trues[i].values() for i in range(len(trues))]).squeeze()

for i, model in enumerate(models):
    forecasts = save_forecasts[model]
    if model not in ['latentode_weinstock', 'gluformer_weinstock']:
        forecasts = np.array([forecasts[i].all_values() for i in range(len(forecasts))])
    if 'gluformer' in model:
        # generate samples from predictive distribution
        samples = np.random.normal(loc=forecasts[..., None],
                                   scale=1,
                                   size=(forecasts.shape[0], 
                                         forecasts.shape[1], 
                                         forecasts.shape[2],
                                         30))
        samples = samples.reshape(samples.shape[0], samples.shape[1], -1)
    for j in range(5):
        # put vertical line at 0
        axs[i, j].axvline(x=0, color='black', linestyle='--')
        # plot inputs + trues
        axs[i, j].plot(np.arange(-12, 12), np.concatenate([inputs[sidx[j] + offsets['tft_weinstock'], -12:], 
                                                           trues[sidx[j] + offsets['tft_weinstock'], :]]))
        # plot forecasts
        if 'tft' in model:
            forecast = forecasts[sidx[j] + offsets[model]]
            lower_quantile = np.quantile(forecast, 0.05, axis=-1)
            upper_quantile = np.quantile(forecast, 0.95, axis=-1)
            median = np.quantile(forecast, 0.5, axis=-1)
            axs[i, j].fill_between(np.arange(12),
                                   lower_quantile[:, 0],
                                   upper_quantile[:, 0],
                                   alpha=0.7,
                                   edgecolor='black',
                                   color=cmap(0.5))
            axs[i, j].plot(np.arange(12), median[:, 0], color='red', marker='o')
        if 'linreg' in model:
            forecast = forecasts[sidx[j] + offsets[model]]
            axs[i, j].plot(np.arange(12), forecast[:, 0], color='red', marker='o')
        if 'transformer' in model:
            forecast = forecasts[sidx[j] + offsets[model]]
            axs[i, j].plot(np.arange(12), forecast[:, 0], color='red', marker='o')
        if 'latentode' in model:
            forecast = forecasts[:, sidx[j] + offsets[model], :, 0]
            median = np.quantile(forecast, 0.5, axis=0)
            axs[i, j].plot(np.arange(12), median, color='red', marker='o')
        if 'gluformer' in model:
            ind = sidx[j] + offsets[model]
            # plot predictive distribution
            for point in range(samples.shape[1]):
                kde = stats.gaussian_kde(samples[ind, point, :])
                maxi, mini = 1.2 * np.max(samples[ind, point, :]), 0.8 * np.min(samples[ind, point, :])
                y_grid = np.linspace(mini, maxi, 200)
                x = kde(y_grid)
                axs[i, j].fill_betweenx(y_grid, x1=point, x2=point - x * 15, 
                                        alpha=0.7, 
                                        edgecolor='black',
                                        color=cmap(point / samples.shape[1]))
            # plot median
            forecast = samples[ind, :, :]
            median = np.quantile(forecast, 0.5, axis=-1)
            axs[i, j].plot(np.arange(12), median, color='red', marker='o')
        # for last row only, xlabel = Time (in 5 minute intervals)
        if i == len(models) - 1:
            axs[i, j].set_xlabel('Time (in 5 minute intervals)')
        # for first column only, ylabel = model name in upper case letters \n Glucose (mg/dL)
        if j == 0:
            axs[i, j].set_ylabel(model.split('_')[0].upper() + '\nGlucose (mg/dL)')

for ax in axs.flatten():
    for item in [ax.xaxis.label, ax.yaxis.label, ax.title]:
        item.set_fontsize(16)
    for item in ax.get_xticklabels() + ax.get_yticklabels():
        item.set_fontsize(16)
    if ax.get_legend() is not None:
        for item in ax.get_legend().get_texts():
            item.set_fontsize(20)

# save figure
plt.tight_layout()
plt.savefig('paper_results/plots/figure6.pdf', dpi=300, bbox_inches='tight')
