from torch import from_numpy
import pandas as pd
import data_formatters.utils as utils
from data_formatters.base import InputTypes
from torch.utils.data import Dataset
import numpy as np
import click
from os import path

class TSDataset(Dataset):
    def __init__(self, cnf, data_formatter, split):
        '''Initialize the dataset.
        
        Args:
        data: pandas dataframe
        cnf: config file
        data_formatter: data formatter
        '''

        # store data and params
        self.params = cnf.all_params
        self.data = None
        if split == 'train':
            self.data = data_formatter.data.iloc[data_formatter.train_idx, :].copy()
        elif split == 'val':
            self.data = data_formatter.data.iloc[data_formatter.val_idx, :].copy()
        elif split == 'test':
            self.data = data_formatter.data.iloc[data_formatter.test_idx, :].copy()

        # load parameters from data formatter
        # 1. input_size: number of features
        # 2. output_size: number of targets
        # 3. input_cols: list of feature names
        # 4. target_col: name of target column
        # 5. time_col: name of time column
        # 6. id_col: name of identifier column
        self.column_definition = data_formatter.get_column_definition()
        self.input_size = data_formatter.get_input_size()
        self.output_size = data_formatter.get_output_size()
        self.id_col = data_formatter.get_cols_by_input_type(InputTypes.ID)[0]
        self.time_col = data_formatter.get_cols_by_input_type(InputTypes.TIME)[0]
        self.target_col = data_formatter.get_cols_by_input_type(InputTypes.TARGET)
        self.input_cols = data_formatter.get_cols_except_input_types({InputTypes.ID, InputTypes.TIME})
        
        # process data
        self.inputs = None
        self.outputs = None
        self.time = None
        self.id = None
        self.process(self.data, self.params['max_samples'])
    
    def process(self, data, max_samples):
        data.sort_values(by=[self.id_col, self.time_col], inplace=True)
        print('Getting valid sampling locations.')
        valid_sampling_locations = []
        split_data_map = {}
        for id, df in data.groupby(self.id_col):
            num_entries = len(df)
            if num_entries >= self.params['total_time_steps']:
                valid_sampling_locations += [
                    (id, self.params['total_time_steps'] + i)
                    for i in range(num_entries - self.params['total_time_steps'] + 1)
                ]
            split_data_map[id] = df
        # determine number of samples to extract
        print('# available segments={}'.format(len(valid_sampling_locations)))
        if max_samples > 0 and len(valid_sampling_locations) >= max_samples:
            print('Extracting {} samples out of {}'.format(
                max_samples, len(valid_sampling_locations)))
            ranges = [
                valid_sampling_locations[i] for i in np.random.choice(
                    len(valid_sampling_locations), max_samples, replace=False)
            ]
        else:
            print('Extracting all available segments.')
            max_samples = len(valid_sampling_locations)
            ranges = valid_sampling_locations

        # set variables to store extracted samples
        # TODO: inputs and outputs should be numeric types -> need to set up encoders
        self.inputs = np.empty((max_samples, self.params['total_time_steps'], self.input_size), dtype=object)
        self.outputs = np.empty((max_samples, self.params['total_time_steps'], self.output_size), dtype=object)
        self.time = np.empty((max_samples, self.params['total_time_steps'], 1), dtype=object)
        self.id = np.empty((max_samples, self.params['total_time_steps'], 1), dtype=object)

        # extract samples
        for i, tup in enumerate(ranges):
            if ((i + 1) % 1000) == 0:
                print(i + 1, 'of', max_samples, 'samples done...')
            id, start_idx = tup
            sliced = split_data_map[id].iloc[start_idx - self.params['total_time_steps']:start_idx]
            # TODO: pad out unknown in the future values from the inputs (OBSERVED + TARGET)
            self.inputs[i, :, :] = sliced[self.input_cols]
            self.outputs[i, :, :] = sliced[self.target_col]
            self.time[i, :, 0] = sliced[self.time_col]
            self.id[i, :, 0] = sliced[self.id_col]

    def __getitem__(self, index):
        # pad inputs 
        s = {
            'inputs': self.inputs[index],
            'outputs': self.outputs[index],
            'time': self.time[index].tolist(),
            'id': self.id[index].tolist()
        }

        return s

    def __len__(self):
        return self.inputs.shape[0]