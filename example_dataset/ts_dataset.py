import torch
from torch import from_numpy

import pandas as pd
import numpy as np
import click
from os import path
import itertools as it
from random import triangular
from scipy.stats import kendalltau

from dataset.base import InputTypes
from torch.utils.data import Dataset
import dataset.utils as utils


class TSDataset(Dataset):
    ## Mostly adapted from original TFT Github, dataset
    def __init__(self, conf, data_formatter):

        self.params = conf.all_params
        
        # load the data
        self.csv = utils.data_csv_path(conf.ds_name)
        self.data = pd.read_csv(self.csv, index_col=0, na_filter=False)
        self.train_set, self.valid_set, self.test_set = data_formatter.split_data(self.data)
        self.params['column_definition'] = data_formatter.get_column_definition()
        
        # split by columns
        self.label = None
        self.inputs = None
        self.outputs = None
        self.time = None
        self.identifiers = None

        # set up the integral transform (enforce Gaussian marginals)
        if self.params['integral_transform']:
            # fit the integral transform on training data
            self.train()
            self.transform_cdf = utils.IntegralTransform(self.outputs)

    def train(self):
        self.label = 'train'
        max_samples = self.params['train_samples']
        if path.exists(utils.csv_path_to_folder(self.csv) + "processed_traindata.npz"):
            f = np.load(utils.csv_path_to_folder(self.csv) + "processed_traindata.npz", allow_pickle=True)
            self.inputs, self.outputs, self.time, self.identifiers = f[f.files[0]], f[f.files[1]], f[f.files[2]], f[f.files[3]]
        else:
            self.preprocess(self.train_set, max_samples)
            np.savez(utils.csv_path_to_folder(self.csv) + "processed_traindata.npz", self.inputs, self.outputs,
                     self.time,
                     self.identifiers)

    def test(self):
        self.label = 'test'
        max_samples = self.params['test_samples']
        if path.exists(utils.csv_path_to_folder(self.csv) + "processed_testdata.npz"):
            f = np.load(utils.csv_path_to_folder(self.csv) + "processed_testdata.npz", allow_pickle=True)
            self.inputs, self.outputs, self.time, self.identifiers = f[f.files[0]], f[f.files[1]], f[f.files[2]], f[f.files[3]]
        else:
            self.preprocess(self.test_set, max_samples)
            np.savez(utils.csv_path_to_folder(self.csv) + "processed_testdata.npz", self.inputs, self.outputs, 
                    self.time, 
                    self.identifiers)

    def val(self):
        self.label = 'val'
        max_samples = self.params['val_samples']
        if path.exists(utils.csv_path_to_folder(self.csv) + "processed_validdata.npz"):
            f = np.load(utils.csv_path_to_folder(self.csv) + "processed_validdata.npz", allow_pickle=True)
            self.inputs, self.outputs, self.time, self.identifiers = f[f.files[0]], f[f.files[1]], f[f.files[2]], f[f.files[3]]
        else:
            self.preprocess(self.valid_set, max_samples)
            np.savez(utils.csv_path_to_folder(self.csv) + "processed_validdata.npz", self.inputs, self.outputs,
                     self.time,
                     self.identifiers)

    def preprocess(self, data, max_samples):
        time_steps = int(self.params['total_time_steps'])
        input_size = int(self.params['input_size'])
        output_size = int(self.params['output_size'])
        column_definition = self.params['column_definition']

        id_col = self._get_single_col_by_type(InputTypes.ID)
        time_col = self._get_single_col_by_type(InputTypes.TIME)

        data.sort_values(by=[id_col, time_col], inplace=True)
        print('Getting valid sampling locations.')
        valid_sampling_locations = []
        split_data_map = {}
        for identifier, df in data.groupby(id_col):
            # print('Getting locations for {}'.format(identifier))
            num_entries = len(df)
            if num_entries >= time_steps:
                valid_sampling_locations += [
                    (identifier, time_steps + i)
                    for i in range(num_entries - time_steps + 1)
                ]
            split_data_map[identifier] = df

        self.inputs = np.zeros((max_samples, time_steps, input_size))
        self.outputs = np.zeros((max_samples, time_steps, output_size))
        self.time = np.empty((max_samples, time_steps, 1), dtype=object)
        self.identifiers = np.empty((max_samples, time_steps, 1), dtype=object)
        print('# available segments={}'.format(len(valid_sampling_locations)))

        if max_samples > 0 and len(valid_sampling_locations) > max_samples:
            print('Extracting {} samples...'.format(max_samples))
            ranges = [
                valid_sampling_locations[i] for i in np.random.choice(
                    len(valid_sampling_locations), max_samples, replace=False)
            ]
        else:
            print('Max samples={} exceeds # available segments={}'.format(
                max_samples, len(valid_sampling_locations)))
            ranges = valid_sampling_locations

        id_col = self._get_single_col_by_type(InputTypes.ID)
        time_col = self._get_single_col_by_type(InputTypes.TIME)
        target_col = self._get_single_col_by_type(InputTypes.TARGET)
        input_cols = [
            tup[0]
            for tup in column_definition
            if tup[2] not in {InputTypes.ID, InputTypes.TIME}
        ]

        for i, tup in enumerate(ranges):
            if ((i + 1) % 1000) == 0:
                print(i + 1, 'of', max_samples, 'samples done...')
            identifier, start_idx = tup
            sliced = split_data_map[identifier].iloc[start_idx - time_steps:start_idx]

            self.inputs[i, :, :] = sliced[input_cols]
            self.outputs[i, :, :] = sliced[[target_col]]
            self.time[i, :, 0] = sliced[time_col]
            self.identifiers[i, :, 0] = sliced[id_col]

    def estimate_corr(self):
        if self.label != 'train':
            raise ValueError('Can only estimate correlations for train data.')
        if path.exists(utils.csv_path_to_folder(self.csv) + "est_corrmatrix.npz"):
            f = np.load(utils.csv_path_to_folder(self.csv) + "est_corrmatrix.npz", allow_pickle=True)
            est_corr = f[f.files[0]]
            print('Loaded estimated correlation matrix...')
            return torch.tensor(est_corr)
        else:
            print('Estimating correlation matrix...')
            est_corr = utils.compute_corr(self.outputs)
            print('Estimated correlation matrix.')
            np.savez(utils.csv_path_to_folder(self.csv) + "est_corrmatrix.npz", est_corr)
            return torch.tensor(est_corr)

    def __getitem__(self, index):

        num_encoder_steps = int(self.params['num_encoder_steps'])
        x = self.outputs[index, :num_encoder_steps, 0].astype(np.float32)
        y = self.outputs[index, num_encoder_steps:, 0].astype(np.float32)
        id = self.identifiers[index].tolist()

        return x, y, id

    def __len__(self):
        return self.inputs.shape[0]

    def _get_single_col_by_type(self, input_type):
        """Returns name of single column for input type."""
        return utils.get_single_col_by_input_type(input_type, self.params['column_definition'])