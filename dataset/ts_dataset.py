from torch import from_numpy
import pandas as pd
import data_formatters.utils as utils
from data_formatters.base import InputTypes
from torch.utils.data import Dataset
import numpy as np
import click
from os import path

class TSDataset(Dataset):
    def __init__(self, cnf, data_formatter):

        self.params = cnf.all_params
        self.data = pd.read_csv(self.params['data_csv_path'], index_col=0, na_filter=False)

        # TODO: define splitter for train/val/test
        # self.train_set, self.valid_set, self.test_set = data_formatter.split_data(self.data)
        self.params['column_definition'] = data_formatter.get_column_definition()
        self.params['input_size'] = data_formatter.get_input_size()
        self.params['output_size'] = data_formatter.get_output_size()
        
        # process data
        self.inputs = None
        self.outputs = None
        self.time = None
        self.identifiers = None
        self.process(self.data, self.params['max_samples'])

    
    def process(self, data, max_samples):
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
            num_entries = len(df)
            if num_entries >= time_steps:
                valid_sampling_locations += [
                    (identifier, time_steps + i)
                    for i in range(num_entries - time_steps + 1)
                ]
            split_data_map[identifier] = df

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
        self.inputs = np.empty((max_samples, time_steps, input_size), dtype=object)
        self.outputs = np.empty((max_samples, time_steps, output_size), dtype=object)
        self.time = np.empty((max_samples, time_steps, 1), dtype=object)
        self.identifiers = np.empty((max_samples, time_steps, 1), dtype=object)

        # extract column types
        id_col = self._get_single_col_by_type(InputTypes.ID)
        time_col = self._get_single_col_by_type(InputTypes.TIME)
        target_col = self._get_single_col_by_type(InputTypes.TARGET)
        input_cols = [
            tup[0]
            for tup in column_definition
            if tup[2] not in {InputTypes.ID, InputTypes.TIME}
        ]

        # extract samples
        for i, tup in enumerate(ranges):
            if ((i + 1) % 1000) == 0:
                print(i + 1, 'of', max_samples, 'samples done...')
            identifier, start_idx = tup
            sliced = split_data_map[identifier].iloc[start_idx - time_steps:start_idx]
            # TODO: pad out unknown in the future values from the inputs (OBSERVED + TARGET)
            self.inputs[i, :, :] = sliced[input_cols]
            self.outputs[i, :, :] = sliced[[target_col]]
            self.time[i, :, 0] = sliced[time_col]
            self.identifiers[i, :, 0] = sliced[id_col]

    def __getitem__(self, index):
        # pad inputs 
        s = {
            'inputs': self.inputs[index],
            'outputs': self.outputs[index],
            'time': self.time[index].tolist(),
            'identifier': self.identifiers[index].tolist()
        }

        return s

    def __len__(self):
        return self.inputs.shape[0]

    def _get_single_col_by_type(self, input_type):
        """Returns name of single column for input type."""
        return utils.get_single_col_by_input_type(input_type, self.params['column_definition'])