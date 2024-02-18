import os
import re
import numpy as np
from typing import Sequence, Dict

def avg_results(model_names: str, 
                model_names_with_covs: str = None,
                time_steps: int = 12)->Sequence[Dict[str, Dict[str, np.array]]]:
    """
    Function to load final model results: averaged across random seeds / folds
    ----------
    model_names: str
        path to models' results file
    model_name_with_covs: str
        path to models' results file with covariates
    time_steps: int
        number of time steps that were predicted
    NOTE: model_names and model_names_with_covs should be in the same order

    Output
    ------
    Computes the following set of dictionaries:
    dict1:
        Dictionary of MSE / MAE values for ID / OOD sets with and without covariates
        key1: id / od, key2: covs / no_covs
    dict2:
        Dictionary of likelihood / calibration values for ID / OOD sets with and without covariates
        key1: id / od, key2: covs / no_covs
    """
    def parser(model_names):
        arr_id_errors = np.full((len(model_names), 2), np.nan)
        arr_ood_errors = arr_id_errors.copy()
        arr_id_likelihoods = arr_id_errors.copy()
        arr_ood_likelihoods = arr_id_errors.copy()
        arr_id_errors_std = np.full((len(model_names), 2, 2), np.nan)
        arr_ood_errors_std = arr_id_errors_std.copy()
        arr_id_likelihoods_std = arr_id_errors_std.copy()
        arr_ood_likelihoods_std = arr_id_errors_std.copy()
        for model_name in model_names:
            if not os.path.isfile(model_name):
                    continue
            with open(model_name, 'r') as f:
                for line in f:
                    if line.startswith('ID median of (MSE, MAE):'):
                        id_mse_mae = re.findall(r'\d+\.\d+(?:e-\d+)?', line)
                        arr_id_errors[model_names.index(model_name), 0] = float(id_mse_mae[0])
                        arr_id_errors[model_names.index(model_name), 1] = float(id_mse_mae[1])
                        if len(id_mse_mae) > 2:
                            arr_id_errors_std[model_names.index(model_name), 0, 0] = float(id_mse_mae[2])
                            arr_id_errors_std[model_names.index(model_name), 0, 1] = float(id_mse_mae[3])
                        if len(id_mse_mae) > 4:
                            arr_id_errors_std[model_names.index(model_name), 1, 0] = float(id_mse_mae[4])
                            arr_id_errors_std[model_names.index(model_name), 1, 1] = float(id_mse_mae[5])
                    elif line.startswith('OOD median of (MSE, MAE):'):
                        ood_mse_mae = re.findall(r'\d+\.\d+(?:e-\d+)?', line)
                        arr_ood_errors[model_names.index(model_name), 0] = float(ood_mse_mae[0])
                        arr_ood_errors[model_names.index(model_name), 1] = float(ood_mse_mae[1])
                        if len(ood_mse_mae) > 2:
                            arr_ood_errors_std[model_names.index(model_name), 0, 0] = float(ood_mse_mae[2])
                            arr_ood_errors_std[model_names.index(model_name), 0, 1] = float(ood_mse_mae[3])
                        if len(ood_mse_mae) > 4:
                            arr_ood_errors_std[model_names.index(model_name), 1, 0] = float(ood_mse_mae[4])
                            arr_ood_errors_std[model_names.index(model_name), 1, 1] = float(ood_mse_mae[5])
                    elif line.startswith('ID likelihoods:'):
                        id_likelihoods = re.findall(r'-?\d+\.\d+(?:e-\d+)?', line)
                        arr_id_likelihoods[model_names.index(model_name), 0] = float(id_likelihoods[0])
                        if len(id_likelihoods) > 1:
                            arr_id_likelihoods_std[model_names.index(model_name), 0, 0] = float(id_likelihoods[1])
                        if len(id_likelihoods) > 2:
                            arr_id_likelihoods_std[model_names.index(model_name), 1, 0] = float(id_likelihoods[2])
                    elif line.startswith('OOD likelihoods:'):
                        ood_likelihoods = re.findall(r'-?\d+\.\d+(?:e-\d+)?', line)
                        arr_ood_likelihoods[model_names.index(model_name), 0] = float(ood_likelihoods[0])
                        if len(ood_likelihoods) > 1:
                            arr_ood_likelihoods_std[model_names.index(model_name), 0, 0] = float(ood_likelihoods[1])
                        if len(ood_likelihoods) > 2:
                            arr_ood_likelihoods_std[model_names.index(model_name), 1, 0] = float(ood_likelihoods[2])
                    elif line.startswith('ID calibration errors:'):
                        id_calib = re.findall(r'-?\d+\.\d+(?:e-\d+)?', line)
                        arr_id_likelihoods[model_names.index(model_name), 1] = np.mean([float(x) for x in id_calib[:time_steps]])
                        if len(id_calib) > time_steps:
                            arr_id_likelihoods_std[model_names.index(model_name), 0, 1] = np.mean([float(x) for x in id_calib[time_steps:]])
                        if len(id_calib) > 2*time_steps:
                            arr_id_likelihoods_std[model_names.index(model_name), 1, 1] = np.mean([float(x) for x in id_calib[2*time_steps:]])
                    elif line.startswith('OOD calibration errors:'):
                        ood_calib = re.findall(r'-?\d+\.\d+(?:e-\d+)?', line)
                        arr_ood_likelihoods[model_names.index(model_name), 1] = np.mean([float(x) for x in ood_calib[:time_steps]])
                        if len(ood_calib) > time_steps:
                            arr_ood_likelihoods_std[model_names.index(model_name), 0, 1] = np.mean([float(x) for x in ood_calib[time_steps:]])
                        if len(ood_calib) > 2*time_steps:
                            arr_ood_likelihoods_std[model_names.index(model_name), 1, 1] = np.mean([float(x) for x in ood_calib[2*time_steps:]])
        return (arr_id_errors, arr_ood_errors, arr_id_likelihoods, arr_ood_likelihoods), \
                (arr_id_errors_std, arr_ood_errors_std, arr_id_likelihoods_std, arr_ood_likelihoods_std)

    dict_errors, dict_errors_std = {}, {}
    dict_likelihoods, dict_likelihoods_std = {}, {}
    error, error_std = parser(model_names)
    dict_errors['id'] = {'no_covs': error[0]}
    dict_errors['ood'] = {'no_covs': error[1]}
    dict_likelihoods['id'] = {'no_covs': error[2]}
    dict_likelihoods['ood'] = {'no_covs': error[3]}
    dict_errors_std['id'] = {'no_covs': error_std[0]}
    dict_errors_std['ood'] = {'no_covs': error_std[1]}
    dict_likelihoods_std['id'] = {'no_covs': error_std[2]}
    dict_likelihoods_std['ood'] = {'no_covs': error_std[3]}

    if model_names_with_covs is not None:
        error, error_std = parser(model_names_with_covs)
        dict_errors['id']['covs'] = error[0]
        dict_errors['ood']['covs'] = error[1]
        dict_likelihoods['id']['covs'] = error[2]
        dict_likelihoods['ood']['covs'] = error[3]
        dict_errors_std['id']['covs'] = error_std[0]
        dict_errors_std['ood']['covs'] = error_std[1]
        dict_likelihoods_std['id']['covs'] = error_std[2]
        dict_likelihoods_std['ood']['covs'] = error_std[3]
    
    return (dict_errors, dict_likelihoods), (dict_errors_std, dict_likelihoods_std)