import sys
import os
import yaml
from typing import Any, BinaryIO, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import darts

from darts import models
from darts import metrics
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from pytorch_lightning.callbacks import Callback

# for optuna callback
import warnings
import optuna
from optuna.storages._cached_storage import _CachedStorage
from optuna.storages._rdb.storage import RDBStorage
# Define key names of `Trial.system_attrs`.
_PRUNED_KEY = "ddp_pl:pruned"
_EPOCH_KEY = "ddp_pl:epoch"
with optuna._imports.try_import() as _imports:
    import pytorch_lightning as pl
    from pytorch_lightning import LightningModule
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import Callback
if not _imports.is_successful():
    Callback = object  # type: ignore  # NOQA
    LightningModule = object  # type: ignore  # NOQA
    Trainer = object  # type: ignore  # NOQA


def compute_error_statistics(errors: np.ndarray, 
                             ) -> Dict[str, List[float]]:
    """
    Compute min, 25%, 50%, 75%, max, mean, std of the errors.

    Parameters
    ----------
    errors
        The errors arranged as n-by-p, where n is the number of samples and p is the number of metrics.

    Returns
    -------
    Dict[str, float]
        The error statistics for each metric.
    """
    error_statistics = {'min': [], 'max': [], 'mean': [], 'std': [], 'quantile25': [], 'median': [], 'quantile75': []}
    error_statistics['min'].append(np.min(errors, axis=0))
    error_statistics['max'].append(np.max(errors, axis=0))
    error_statistics['mean'].append(np.mean(errors, axis=0))
    error_statistics['std'].append(np.std(errors, axis=0))
    error_statistics['quantile25'].append(np.percentile(errors, 25, axis=0))
    error_statistics['median'].append(np.percentile(errors, 50, axis=0))
    error_statistics['quantile75'].append(np.percentile(errors, 75, axis=0))
    return error_statistics

class LossLogger(Callback):
    def __init__(self):
        self.train_loss = []
        self.val_loss = []

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.train_loss.append(float(trainer.callback_metrics["train_loss"]))

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.val_loss.append(float(trainer.callback_metrics["val_loss"]))

def print_callback(study, trial, study_file=None):
    # write output to a file
    with open(study_file, "a") as f:
        f.write(f"Current value: {trial.value}, Current params: {trial.params}\n")
        f.write(f"Best value: {study.best_value}, Best params: {study.best_trial.params}\n")

def make_series(data: Dict[str, pd.DataFrame],
                time_col: str,
                group_col: str,
                value_cols: Dict[str, List[str]],
                include_sid: bool = False,
                ) -> Dict[str, darts.TimeSeries]:
    """
    Make TimeSeries from data.
    data: dict of train, val, test dataframes
    time_col: name of time column
    group_col: name of group column
    value_cols: list of value columns
    include_sid: whether to include segment id as static covariate

    Returns: dict of TimeSeries
    """
    series = {i: {j: None for j in value_cols} for i in data.keys()}
    scalers = {}
    for key, df in data.items():
        for name, cols in value_cols.items():
            series[key][name] = TimeSeries.from_group_dataframe(df = df,
                                                                group_cols = group_col,
                                                                time_col = time_col,
                                                                value_cols = cols) if cols is not None else None
            if series[key][name] is not None and include_sid is False:
                for i in range(len(series[key][name])):
                    series[key][name][i] = series[key][name][i].with_static_covariates(None)
            if cols is not None: 
                if key == 'train':
                    scalers[name] = Scaler()
                    series[key][name] = scalers[name].fit_transform(series[key][name])
                else:
                    series[key][name] = scalers[name].transform(series[key][name])
            else:
                scalers[name] = None
    return series, scalers

def early_stopping_check(study, 
                         trial, 
                         study_file, 
                         early_stopping_rounds=10):
      """
      Early stopping callback for Optuna.
      This function checks the current trial number and the best trial number.
      """
      current_trial_number = trial.number
      best_trial_number = study.best_trial.number
      should_stop = (current_trial_number - best_trial_number) >= early_stopping_rounds
      if should_stop:
          with open(study_file, 'a') as f:
              f.write('\nEarly stopping at trial {} (best trial: {})'.format(current_trial_number, best_trial_number))
          study.stop()

def rescale_and_backtest(series: Union[TimeSeries, Sequence[TimeSeries]],
                         forecasts: Union[TimeSeries, Sequence[TimeSeries]], 
                         metric: Union[
                                    Callable[[TimeSeries, TimeSeries], float],
                                    List[Callable[[TimeSeries, TimeSeries], float]],
                                ], 
                         scaler: Callable[[TimeSeries], TimeSeries] = None,
                         reduction: Union[Callable[[np.ndarray], float], None] = np.mean,
                        ):
    """
    Backtest the forecasts on the series.

    Parameters
    ----------
    series
        The target time series.
    forecasts
        The forecasts.
    scaler
        The scaler used to scale the series.
    metric
        The metric to use for backtesting.
    reduction
        The reduction to apply to the metric.

    Returns
    -------
    float or List[float] or List[List[float]]
        The (sequence of) error score on a series, or list of list containing error scores for each
        provided series and each sample.
    """
    series = [series] if isinstance(series, TimeSeries) else series
    if len(series) == 1:
        forecasts = [forecasts]
    if not isinstance(metric, list):
        metric = [metric]

    # reverse scaling, forecasts and true values, compute errors
    backtest_list = []
    for idx, target_ts in enumerate(series):
        if scaler is not None:
            target_ts = scaler.inverse_transform(target_ts)
            predicted_ts = [scaler.inverse_transform(f) for f in forecasts[idx]]
        errors = [
            [metric_f(target_ts, f) for metric_f in metric]
            if len(metric) > 1
            else metric[0](target_ts, f)
            for f in predicted_ts
        ]
        if reduction is None:
            backtest_list.append(np.array(errors))
        else:
            backtest_list.append(reduction(np.array(errors), axis=0))
    return backtest_list if len(backtest_list) > 1 else backtest_list[0]


class PyTorchLightningPruningCallback(Callback):
    """PyTorch Lightning callback to prune unpromising trials.
    See `the example <https://github.com/optuna/optuna-examples/blob/
    main/pytorch/pytorch_lightning_simple.py>`__
    if you want to add a pruning callback which observes accuracy.
    Args:
        trial:
            A :class:`~optuna.trial.Trial` corresponding to the current evaluation of the
            objective function.
        monitor:
            An evaluation metric for pruning, e.g., ``val_loss`` or
            ``val_acc``. The metrics are obtained from the returned dictionaries from e.g.
            ``pytorch_lightning.LightningModule.training_step`` or
            ``pytorch_lightning.LightningModule.validation_epoch_end`` and the names thus depend on
            how this dictionary is formatted.
    """

    def __init__(self, trial: optuna.trial.Trial, monitor: str) -> None:
        super().__init__()

        self._trial = trial
        self.monitor = monitor

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        # When the trainer calls `on_validation_end` for sanity check,
        # do not call `trial.report` to avoid calling `trial.report` multiple times
        # at epoch 0. The related page is
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/1391.
        if trainer.sanity_checking:
            return

        epoch = pl_module.current_epoch

        current_score = trainer.callback_metrics.get(self.monitor)
        if current_score is None:
            message = (
                "The metric '{}' is not in the evaluation logs for pruning. "
                "Please make sure you set the correct metric name.".format(self.monitor)
            )
            warnings.warn(message)
            return

        self._trial.report(current_score, step=epoch)
        if self._trial.should_prune():
            message = "Trial was pruned at epoch {}.".format(epoch)
            raise optuna.TrialPruned(message)