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
from pytorch_lightning.callbacks import Callback
from darts.logging import get_logger, raise_if_not

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

def print_callback(study, trial, study_file=None):
    # write output to a file
    with open(study_file, "a") as f:
        f.write(f"Current value: {trial.value}, Current params: {trial.params}\n")
        f.write(f"Best value: {study.best_value}, Best params: {study.best_trial.params}\n")

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

class LossLogger(Callback):
    def __init__(self):
        self.train_loss = []
        self.val_loss = []

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.train_loss.append(float(trainer.callback_metrics["train_loss"]))

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.val_loss.append(float(trainer.callback_metrics["val_loss"]))

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