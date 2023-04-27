import sys
import os
import yaml
import random
from typing import Any, BinaryIO, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np 
from scipy import stats
import pandas as pd
import darts

from darts import models
from darts import metrics
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from pytorch_lightning.callbacks import Callback

# for darts dataset
from darts.logging import get_logger, raise_if_not

from darts.utils.data.training_dataset import PastCovariatesTrainingDataset, \
                                              DualCovariatesTrainingDataset, \
                                              MixedCovariatesTrainingDataset
from darts.utils.data.inference_dataset import PastCovariatesInferenceDataset, \
                                                DualCovariatesInferenceDataset, \
                                                MixedCovariatesInferenceDataset
from darts.utils.data.utils import CovariateType

# import data formatter
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from data_formatter.base import *

def get_valid_sampling_locations(target_series: Union[TimeSeries, Sequence[TimeSeries]],
                                 output_chunk_length: int = 12,
                                 input_chunk_length: int = 12,
                                 random_state: Optional[int] = 0,
                                 max_samples_per_ts: Optional[int] = None):
    """
    Get valid sampling indices data for the model.

    Parameters
    ----------
    target_series
        The target time series.
    output_chunk_length
        The length of the output chunk.
    input_chunk_length
        The length of the input chunk.
    use_static_covariates
        Whether to use static covariates.
    max_samples_per_ts
        The maximum number of samples per time series.
    """
    random.seed(random_state)
    valid_sampling_locations = {}
    total_length = input_chunk_length + output_chunk_length
    for id, series in enumerate(target_series):
        num_entries = len(series)
        if num_entries >= total_length:
            valid_sampling_locations[id] = [i for i in range(num_entries - total_length + 1)]
    if max_samples_per_ts is not None:
        updated_sampling_locations = {}
        for id, locations in valid_sampling_locations.items():
            if len(locations) > max_samples_per_ts:
                updated_sampling_locations[id] = random.sample(locations, max_samples_per_ts)
            else:
                updated_sampling_locations[id] = locations
        valid_sampling_locations = updated_sampling_locations
            
    return valid_sampling_locations

class SamplingDatasetPast(PastCovariatesTrainingDataset):
    def __init__(
        self,
        target_series: Union[TimeSeries, Sequence[TimeSeries]],
        covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        output_chunk_length: int = 12,
        input_chunk_length: int = 12,
        use_static_covariates: bool = True,
        random_state: Optional[int] = 0,
        max_samples_per_ts: Optional[int] = None,
    ) -> None:
        """
        Parameters
        ----------
        target_series
            One or a sequence of target `TimeSeries`.
        covariates:
            Optionally, one or a sequence of `TimeSeries` containing past-observed covariates. If this parameter is set,
            the provided sequence must have the same length as that of `target_series`. Moreover, all
            covariates in the sequence must have a time span large enough to contain all the required slices.
            The joint slicing of the target and covariates is relying on the time axes of both series.
        output_chunk_length
            The length of the "output" series emitted by the model
        input_chunk_length
            The length of the "input" series fed to the model
        use_static_covariates
            Whether to use/include static covariate data from input series.
        random_state
            The random state to use for sampling.
        max_samples_per_ts
            The maximum number of samples to be drawn from each time series. If None, all samples will be drawn.
        """
        super().__init__()

        self.target_series = (
            [target_series] if isinstance(target_series, TimeSeries) else target_series
        )
        self.covariates = (
            [covariates] if isinstance(covariates, TimeSeries) else covariates
        )

        # checks
        raise_if_not(
            covariates is None or len(self.target_series) == len(self.covariates),
            "The provided sequence of target series must have the same length as "
            "the provided sequence of covariate series.",
        )

        # get valid sampling locations
        self.valid_sampling_locations = get_valid_sampling_locations(target_series,
                                                                     output_chunk_length,
                                                                     input_chunk_length,
                                                                     random_state,
                                                                     max_samples_per_ts)
        
        # set parameters
        self.output_chunk_length = output_chunk_length
        self.input_chunk_length = input_chunk_length
        self.total_length = input_chunk_length + output_chunk_length
        self.total_number_samples = sum([len(v) for v in self.valid_sampling_locations.values()])
        self.use_static_covariates = use_static_covariates

    def __len__(self):
        """
        Returns the total number of possible (input, target) splits.
        """
        return self.total_number_samples

    def __getitem__(
        self, idx: int
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
        # get idx of target series
        target_idx = 0
        while idx >= len(self.valid_sampling_locations[target_idx]):
            idx -= len(self.valid_sampling_locations[target_idx])
            target_idx += 1
        # get sampling location within the target series
        sampling_location = self.valid_sampling_locations[target_idx][idx]
        # get target series
        target_series = self.target_series[target_idx].values()
        past_target_series = target_series[sampling_location : sampling_location + self.input_chunk_length]
        future_target_series = target_series[sampling_location + self.input_chunk_length : sampling_location + self.total_length]
        # get covariates
        if self.covariates is not None:
            covariates = self.covariates[target_idx].values()
            covariates = covariates[sampling_location : sampling_location + self.input_chunk_length]
        else:
            covariates = None
        # get static covariates
        if self.use_static_covariates:
            static_covariates = self.target_series[target_idx].static_covariates_values(copy=True)
        else:
            static_covariates = None
        return (past_target_series, 
                covariates, 
                static_covariates, 
                future_target_series)
    
class SamplingDatasetDual(DualCovariatesTrainingDataset):
    def __init__(
        self,
        target_series: Union[TimeSeries, Sequence[TimeSeries]],
        covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        output_chunk_length: int = 12,
        input_chunk_length: int = 12,
        use_static_covariates: bool = True,
        random_state: Optional[int] = 0,
        max_samples_per_ts: Optional[int] = None,
    ) -> None:
        """
        Parameters
        ----------
        target_series
            One or a sequence of target `TimeSeries`.
        covariates:
            Optionally, one or a sequence of `TimeSeries` containing future-known covariates. If this parameter is set,
            the provided sequence must have the same length as that of `target_series`. Moreover, all
            covariates in the sequence must have a time span large enough to contain all the required slices.
            The joint slicing of the target and covariates is relying on the time axes of both series.
        output_chunk_length
            The length of the "output" series emitted by the model
        input_chunk_length
            The length of the "input" series fed to the model
        use_static_covariates
            Whether to use/include static covariate data from input series.
        random_state
            The random state to use for sampling.
        max_samples_per_ts
            The maximum number of samples to be drawn from each time series. If None, all samples will be drawn.
        """
        super().__init__()

        self.target_series = (
            [target_series] if isinstance(target_series, TimeSeries) else target_series
        )
        self.covariates = (
            [covariates] if isinstance(covariates, TimeSeries) else covariates
        )

        # checks
        raise_if_not(
            covariates is None or len(self.target_series) == len(self.covariates),
            "The provided sequence of target series must have the same length as "
            "the provided sequence of covariate series.",
        )

        # get valid sampling locations
        self.valid_sampling_locations = get_valid_sampling_locations(target_series,
                                                                     output_chunk_length,
                                                                     input_chunk_length,
                                                                     random_state,
                                                                     max_samples_per_ts,)
        
        # set parameters
        self.output_chunk_length = output_chunk_length
        self.input_chunk_length = input_chunk_length
        self.total_length = input_chunk_length + output_chunk_length
        self.total_number_samples = sum([len(v) for v in self.valid_sampling_locations.values()])
        self.use_static_covariates = use_static_covariates

    def __len__(self):
        """
        Returns the total number of possible (input, target) splits.
        """
        return self.total_number_samples

    def __getitem__(
        self, idx: int
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
        # get idx of target series
        target_idx = 0
        while idx >= len(self.valid_sampling_locations[target_idx]):
            idx -= len(self.valid_sampling_locations[target_idx])
            target_idx += 1
        # get sampling location within the target series
        sampling_location = self.valid_sampling_locations[target_idx][idx]
        # get target series
        target_series = self.target_series[target_idx].values()
        past_target_series = target_series[sampling_location : sampling_location + self.input_chunk_length]
        future_target_series = target_series[sampling_location + self.input_chunk_length : sampling_location + self.total_length]
        # get covariates
        if self.covariates is not None:
            covariates = self.covariates[target_idx].values()
            past_covariates = covariates[sampling_location : sampling_location + self.input_chunk_length]
            future_covariates = covariates[sampling_location + self.input_chunk_length : sampling_location + self.total_length]
        else:
            past_covariates = None
            future_covariates = None
        # get static covariates
        if self.use_static_covariates:
            static_covariates = self.target_series[target_idx].static_covariates_values(copy=True)
        else:
            static_covariates = None
        return (past_target_series, 
                past_covariates, 
                future_covariates, 
                static_covariates, 
                future_target_series)
    
class SamplingDatasetMixed(MixedCovariatesTrainingDataset):
    def __init__(
        self,
        target_series: Union[TimeSeries, Sequence[TimeSeries]],
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        output_chunk_length: int = 12,
        input_chunk_length: int = 12,
        use_static_covariates: bool = True,
        random_state: Optional[int] = 0,
        max_samples_per_ts: Optional[int] = None,
    ) -> None:
        """
        Parameters
        ----------
        target_series
            One or a sequence of target `TimeSeries`.
        past_covariates
            Optionally, one or a sequence of `TimeSeries` containing past-observed covariates. If this parameter is set,
            the provided sequence must have the same length as that of `target_series`. Moreover, all
            covariates in the sequence must have a time span large enough to contain all the required slices.
            The joint slicing of the target and covariates is relying on the time axes of both series.
        future_covariates
            Optionally, one or a sequence of `TimeSeries` containing future-known covariates. This has to follow
            the same constraints as `past_covariates`.
        output_chunk_length
            The length of the "output" series emitted by the model
        input_chunk_length
            The length of the "input" series fed to the model
        use_static_covariates
            Whether to use/include static covariate data from input series.
        random_state
            The random state to use for sampling.
        max_samples_per_ts
            The maximum number of samples to be drawn from each time series. If None, all samples will be drawn.
        """
        super().__init__()

        self.target_series = (
            [target_series] if isinstance(target_series, TimeSeries) else target_series
        )
        self.past_covariates = (
            [past_covariates] if isinstance(past_covariates, TimeSeries) else past_covariates
        )
        self.future_covariates = (
            [future_covariates] if isinstance(future_covariates, TimeSeries) else future_covariates
        )

        # checks
        raise_if_not(
            future_covariates is None or len(self.target_series) == len(self.future_covariates),
            "The provided sequence of target series must have the same length as "
            "the provided sequence of covariate series.",
        )
        raise_if_not(
            past_covariates is None or len(self.target_series) == len(self.past_covariates),
            "The provided sequence of target series must have the same length as "
            "the provided sequence of covariate series.",
        )

        # get valid sampling locations
        self.valid_sampling_locations = get_valid_sampling_locations(target_series,
                                                                     output_chunk_length,
                                                                     input_chunk_length,
                                                                     random_state,
                                                                     max_samples_per_ts,)
        
        # set parameters
        self.output_chunk_length = output_chunk_length
        self.input_chunk_length = input_chunk_length
        self.total_length = input_chunk_length + output_chunk_length
        self.total_number_samples = sum([len(v) for v in self.valid_sampling_locations.values()])
        self.use_static_covariates = use_static_covariates

    def __len__(self):
        """
        Returns the total number of possible (input, target) splits.
        """
        return self.total_number_samples

    def __getitem__(
        self, idx: int
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], np.ndarray]:
        # get idx of target series
        target_idx = 0
        while idx >= len(self.valid_sampling_locations[target_idx]):
            idx -= len(self.valid_sampling_locations[target_idx])
            target_idx += 1
        # get sampling location within the target series
        sampling_location = self.valid_sampling_locations[target_idx][idx]
        # get target series
        target_series = self.target_series[target_idx].values()
        past_target_series = target_series[sampling_location : sampling_location + self.input_chunk_length]
        future_target_series = target_series[sampling_location + self.input_chunk_length : sampling_location + self.total_length]
        # get past covariates
        if self.past_covariates is not None:
            past_covariates = self.past_covariates[target_idx].values()
            past_covariates = past_covariates[sampling_location : sampling_location + self.input_chunk_length]
        else:
            past_covariates = None
        # get future covariates
        if self.future_covariates is not None:
            future_covariates = self.future_covariates[target_idx].values()
            historic_future_covariates = future_covariates[sampling_location : sampling_location + self.input_chunk_length]
            future_covariates = future_covariates[sampling_location + self.input_chunk_length : sampling_location + self.total_length]
        else:
            future_covariates = None
            historic_future_covariates = None
        # get static covariates
        if self.use_static_covariates:
            static_covariates = self.target_series[target_idx].static_covariates_values(copy=True)
        else:
            static_covariates = None
        return (past_target_series,
                past_covariates,
                historic_future_covariates,
                future_covariates,
                static_covariates,
                future_target_series,)

class SamplingDatasetInferenceMixed(MixedCovariatesInferenceDataset):
    def __init__(
        self,
        target_series: Union[TimeSeries, Sequence[TimeSeries]],
        past_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        future_covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        n: int = 1,
        input_chunk_length: int = 12,
        output_chunk_length: int = 1,
        use_static_covariates: bool = True,
        random_state: Optional[int] = 0,
        max_samples_per_ts: Optional[int] = None,
        array_output_only: bool = False,
    ):
        """
        Parameters
        ----------
        target_series
            One or a sequence of target `TimeSeries`.
        past_covariates
            Optionally, one or a sequence of `TimeSeries` containing past-observed covariates. If this parameter is set,
            the provided sequence must have the same length as that of `target_series`. Moreover, all
            covariates in the sequence must have a time span large enough to contain all the required slices.
            The joint slicing of the target and covariates is relying on the time axes of both series.
        future_covariates
            Optionally, one or a sequence of `TimeSeries` containing future-known covariates. This has to follow
            the same constraints as `past_covariates`.
        n
            Number of predictions into the future, could be greater than the output chunk length, in which case, the model
            will be called autorregressively.
        output_chunk_length
            The length of the "output" series emitted by the model
        input_chunk_length
            The length of the "input" series fed to the model
        use_static_covariates
            Whether to use/include static covariate data from input series.
        random_state
            The random state to use for sampling.
        max_samples_per_ts
            The maximum number of samples to be drawn from each time series. If None, all samples will be drawn.
        array_output_only
            Whether __getitem__ returns only the arrays or adds the full `TimeSeries` object to the output tuple
            This may cause problems with the torch collate and loader functions but works for Darts.
        """
        super().__init__(target_series = target_series,
                         past_covariates = past_covariates,
                         future_covariates = future_covariates,
                         n = n,
                         input_chunk_length = input_chunk_length,
                         output_chunk_length = output_chunk_length,)

        self.target_series = (
            [target_series] if isinstance(target_series, TimeSeries) else target_series
        )
        self.past_covariates = (
            [past_covariates] if isinstance(past_covariates, TimeSeries) else past_covariates
        )
        self.future_covariates = (
            [future_covariates] if isinstance(future_covariates, TimeSeries) else future_covariates
        )

        # checks
        raise_if_not(
            future_covariates is None or len(self.target_series) == len(self.future_covariates),
            "The provided sequence of target series must have the same length as "
            "the provided sequence of covariate series.",
        )
        raise_if_not(
            past_covariates is None or len(self.target_series) == len(self.past_covariates),
            "The provided sequence of target series must have the same length as "
            "the provided sequence of covariate series.",
        )

        # get valid sampling locations
        self.valid_sampling_locations = get_valid_sampling_locations(target_series,
                                                                     output_chunk_length,
                                                                     input_chunk_length,
                                                                     random_state,
                                                                     max_samples_per_ts,)
        
        # set parameters
        self.output_chunk_length = output_chunk_length
        self.input_chunk_length = input_chunk_length
        self.total_length = input_chunk_length + output_chunk_length
        self.total_number_samples = sum([len(v) for v in self.valid_sampling_locations.values()])
        self.use_static_covariates = use_static_covariates
        self.array_output_only = array_output_only

    def __len__(self):
        """
        Returns the total number of possible (input, target) splits.
        """
        return self.total_number_samples

    def __getitem__(
        self, idx: int
    ) -> Tuple[np.ndarray, 
               Optional[np.ndarray], 
               Optional[np.ndarray], 
               Optional[np.ndarray],
               Optional[np.ndarray],
               Optional[np.ndarray],
               TimeSeries]:
        # get idx of target series
        target_idx = 0
        while idx >= len(self.valid_sampling_locations[target_idx]):
            idx -= len(self.valid_sampling_locations[target_idx])
            target_idx += 1
        # get sampling location within the target series
        sampling_location = self.valid_sampling_locations[target_idx][idx]
        # get target series
        target_series = self.target_series[target_idx]
        past_target_series_with_time = target_series[sampling_location : sampling_location + self.input_chunk_length]
        target_series = self.target_series[target_idx].values()
        past_target_series = target_series[sampling_location : sampling_location + self.input_chunk_length]
        # get past covariates
        if self.past_covariates is not None:
            past_covariates = self.past_covariates[target_idx].values()
            past_covariates = past_covariates[sampling_location : sampling_location + self.input_chunk_length]
            future_past_covariates = past_covariates[sampling_location + self.input_chunk_length : sampling_location + self.total_length]
        else:
            past_covariates = None
            future_past_covariates = None
        # get future covariates
        if self.future_covariates is not None:
            future_covariates = self.future_covariates[target_idx].values()
            historic_future_covariates = future_covariates[sampling_location : sampling_location + self.input_chunk_length]
            future_covariates = future_covariates[sampling_location + self.input_chunk_length : sampling_location + self.total_length]
        else:
            future_covariates = None
            historic_future_covariates = None
        # get static covariates
        if self.use_static_covariates:
            static_covariates = self.target_series[target_idx].static_covariates_values(copy=True)
        else:
            static_covariates = None
        # whether to remove Timeseries and None and return only arrays   
        if self.array_output_only:
            output = []
            for element in [past_target_series,
                            past_covariates,
                            historic_future_covariates,
                            future_covariates,
                            future_past_covariates,
                            static_covariates]:
                if element is not None:
                    output.append(element)
            return tuple(output)
        else:
            return (past_target_series,
                    past_covariates,
                    historic_future_covariates,
                    future_covariates,
                    future_past_covariates,
                    static_covariates,
                    past_target_series_with_time)

    def evalsample(
            self, idx: int
        ) -> TimeSeries:
        """
        Returns the future target series at the given index.
        """
        # get idx of target series
        target_idx = 0
        while idx >= len(self.valid_sampling_locations[target_idx]):
            idx -= len(self.valid_sampling_locations[target_idx])
            target_idx += 1
        # get sampling location within the target series
        sampling_location = self.valid_sampling_locations[target_idx][idx]
        # get target series
        target_series = self.target_series[target_idx][sampling_location + self.input_chunk_length : sampling_location + self.total_length]

        return target_series

class SamplingDatasetInferencePast(PastCovariatesInferenceDataset):
    def __init__(
        self,
        target_series: Union[TimeSeries, Sequence[TimeSeries]],
        covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        n: int = 1,
        input_chunk_length: int = 12,
        output_chunk_length: int = 1,
        use_static_covariates: bool = True,
        random_state: Optional[int] = 0,
        max_samples_per_ts: Optional[int] = None,
        array_output_only: bool = False,
    ):
        """
        Parameters
        ----------
        target_series
            One or a sequence of target `TimeSeries`.
        past_covariates
            Optionally, one or a sequence of `TimeSeries` containing past-observed covariates. If this parameter is set,
            the provided sequence must have the same length as that of `target_series`. Moreover, all
            covariates in the sequence must have a time span large enough to contain all the required slices.
            The joint slicing of the target and covariates is relying on the time axes of both series.
        n
            Number of predictions into the future, could be greater than the output chunk length, in which case, the model
            will be called autorregressively.
        output_chunk_length
            The length of the "output" series emitted by the model
        input_chunk_length
            The length of the "input" series fed to the model
        use_static_covariates
            Whether to use/include static covariate data from input series.
        random_state
            The random state to use for sampling.
        max_samples_per_ts
            The maximum number of samples to be drawn from each time series. If None, all samples will be drawn.
        array_output_only
            Whether __getitem__ returns only the arrays or adds the full `TimeSeries` object to the output tuple
            This may cause problems with the torch collate and loader functions but works for Darts.
        """
        super().__init__(target_series = target_series,
                         covariates = covariates,
                         n = n,
                         input_chunk_length = input_chunk_length,
                         output_chunk_length = output_chunk_length,)

        self.target_series = (
            [target_series] if isinstance(target_series, TimeSeries) else target_series
        )
        self.covariates = (
            [covariates] if isinstance(covariates, TimeSeries) else covariates
        )

        raise_if_not(
            covariates is None or len(self.target_series) == len(self.covariates),
            "The provided sequence of target series must have the same length as "
            "the provided sequence of covariate series.",
        )

        # get valid sampling locations
        self.valid_sampling_locations = get_valid_sampling_locations(target_series,
                                                                     output_chunk_length,
                                                                     input_chunk_length,
                                                                     random_state,
                                                                     max_samples_per_ts,)
        
        # set parameters
        self.output_chunk_length = output_chunk_length
        self.input_chunk_length = input_chunk_length
        self.total_length = input_chunk_length + output_chunk_length
        self.total_number_samples = sum([len(v) for v in self.valid_sampling_locations.values()])
        self.use_static_covariates = use_static_covariates
        self.array_output_only = array_output_only

    def __len__(self):
        """
        Returns the total number of possible (input, target) splits.
        """
        return self.total_number_samples

    def __getitem__(
        self, idx: int
    ) -> Tuple[
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        TimeSeries,
        ]:
        # get idx of target series
        target_idx = 0
        while idx >= len(self.valid_sampling_locations[target_idx]):
            idx -= len(self.valid_sampling_locations[target_idx])
            target_idx += 1
        # get sampling location within the target series
        sampling_location = self.valid_sampling_locations[target_idx][idx]
        # get target series
        target_series = self.target_series[target_idx]
        past_target_series_with_time = target_series[sampling_location : sampling_location + self.input_chunk_length]
        target_series = self.target_series[target_idx].values()
        past_target_series = target_series[sampling_location : sampling_location + self.input_chunk_length]
        # get past covariates
        if self.covariates is not None:
            past_covariates = self.covariates[target_idx].values()
            past_covariates = past_covariates[sampling_location : sampling_location + self.input_chunk_length]
            future_past_covariates = past_covariates[sampling_location + self.input_chunk_length : sampling_location + self.total_length]
        else:
            past_covariates = None
            future_past_covariates = None
        # get static covariates
        if self.use_static_covariates:
            static_covariates = self.target_series[target_idx].static_covariates_values(copy=True)
        else:
            static_covariates = None
        # return arrays or arrays with TimeSeries
        if self.array_output_only:
            output = []
            for element in (past_target_series,
                            past_covariates,
                            future_past_covariates,
                            static_covariates):
                if element is not None:
                    output.append(element)
            return tuple(output)
        else:
            return (past_target_series,
                    past_covariates,
                    future_past_covariates,
                    static_covariates,
                    past_target_series_with_time)

    def evalsample(
            self, idx: int
        ) -> TimeSeries:
        """
        Returns the future target series at the given index.
        """
        # get idx of target series
        target_idx = 0
        while idx >= len(self.valid_sampling_locations[target_idx]):
            idx -= len(self.valid_sampling_locations[target_idx])
            target_idx += 1
        # get sampling location within the target series
        sampling_location = self.valid_sampling_locations[target_idx][idx]
        # get target series
        target_series = self.target_series[target_idx][sampling_location + self.input_chunk_length : sampling_location + self.total_length]

        return target_series

class SamplingDatasetInferenceDual(DualCovariatesInferenceDataset):
    def __init__(
        self,
        target_series: Union[TimeSeries, Sequence[TimeSeries]],
        covariates: Optional[Union[TimeSeries, Sequence[TimeSeries]]] = None,
        n: int = 1,
        input_chunk_length: int = 12,
        output_chunk_length: int = 1,
        use_static_covariates: bool = True,
        random_state: Optional[int] = 0,
        max_samples_per_ts: Optional[int] = None,
        array_output_only: bool = False,
    ):
        """
        Parameters
        ----------
        target_series
            One or a sequence of target `TimeSeries`.
        covariates
            Optionally, some future-known covariates that are used for predictions. This argument is required
            if the model was trained with future-known covariates.
        n
            Number of predictions into the future, could be greater than the output chunk length, in which case, the model
            will be called autorregressively.
        output_chunk_length
            The length of the "output" series emitted by the model
        input_chunk_length
            The length of the "input" series fed to the model
        use_static_covariates
            Whether to use/include static covariate data from input series.
        random_state
            The random state to use for sampling.
        max_samples_per_ts
            The maximum number of samples to be drawn from each time series. If None, all samples will be drawn.
        array_output_only
            Whether __getitem__ returns only the arrays or adds the full `TimeSeries` object to the output tuple
            This may cause problems with the torch collate and loader functions but works for Darts.
        """
        super().__init__(target_series = target_series,
                         covariates = covariates,
                         n = n,
                         input_chunk_length = input_chunk_length,
                         output_chunk_length = output_chunk_length,)

        self.target_series = (
            [target_series] if isinstance(target_series, TimeSeries) else target_series
        )
        self.covariates = (
            [covariates] if isinstance(covariates, TimeSeries) else covariates
        )

        raise_if_not(
            covariates is None or len(self.target_series) == len(self.covariates),
            "The provided sequence of target series must have the same length as "
            "the provided sequence of covariate series.",
        )

        # get valid sampling locations
        self.valid_sampling_locations = get_valid_sampling_locations(target_series,
                                                                     output_chunk_length,
                                                                     input_chunk_length,
                                                                     random_state,
                                                                     max_samples_per_ts,)
        
        # set parameters
        self.output_chunk_length = output_chunk_length
        self.input_chunk_length = input_chunk_length
        self.total_length = input_chunk_length + output_chunk_length
        self.total_number_samples = sum([len(v) for v in self.valid_sampling_locations.values()])
        self.use_static_covariates = use_static_covariates
        self.array_output_only = array_output_only

    def __len__(self):
        """
        Returns the total number of possible (input, target) splits.
        """
        return self.total_number_samples

    def __getitem__(
        self, idx: int
    ) -> Tuple[
        np.ndarray,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        TimeSeries,
        ]:
        # get idx of target series
        target_idx = 0
        while idx >= len(self.valid_sampling_locations[target_idx]):
            idx -= len(self.valid_sampling_locations[target_idx])
            target_idx += 1
        # get sampling location within the target series
        sampling_location = self.valid_sampling_locations[target_idx][idx]
        # get target series
        target_series = self.target_series[target_idx]
        past_target_series_with_time = target_series[sampling_location : sampling_location + self.input_chunk_length]
        target_series = self.target_series[target_idx].values()
        past_target_series = target_series[sampling_location : sampling_location + self.input_chunk_length]
        # get past covariates
        if self.covariates is not None:
            future_covariates = self.covariates[target_idx].values()
            historic_future_covariates = future_covariates[sampling_location : sampling_location + self.input_chunk_length]
            future_covariates = future_covariates[sampling_location + self.input_chunk_length : sampling_location + self.total_length]
        else:
            historic_future_covariates = None
            future_covariates = None
        # get static covariates
        if self.use_static_covariates:
            static_covariates = self.target_series[target_idx].static_covariates_values(copy=True)
        else:
            static_covariates = None
        # return arrays or arrays with TimeSeries
        if self.array_output_only:
            output  = []
            for element in (past_target_series,
                            historic_future_covariates,
                            future_covariates,
                            static_covariates):
                if element is not None:
                    output.append(element)
            return tuple(output)
        else:
            return (past_target_series,
                    historic_future_covariates,
                    future_covariates,
                    static_covariates,
                    past_target_series_with_time)

    def evalsample(
            self, idx: int
        ) -> TimeSeries:
        """
        Returns the future target series at the given index.
        """
        # get idx of target series
        target_idx = 0
        while idx >= len(self.valid_sampling_locations[target_idx]):
            idx -= len(self.valid_sampling_locations[target_idx])
            target_idx += 1
        # get sampling location within the target series
        sampling_location = self.valid_sampling_locations[target_idx][idx]
        # get target series
        target_series = self.target_series[target_idx][sampling_location + self.input_chunk_length : sampling_location + self.total_length]

        return target_series
        