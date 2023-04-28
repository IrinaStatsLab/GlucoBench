# GlucoBench

The official implemntation of the paper "GlucoBench: Curated List of Continuous Glucose Monitoring datasets with Prediction Benchmarks."
If you found our work interesting and plan to re-use the code, please cite us as:
```
@article{
  author  = {Renat Sergazinov and Valeriya Rogovchenko and Elizabeth Chun and Nathaniel Fernandes and Irina Gaynanova},
  title   = {GlucoBench: Curated List of Continuous Glucose Monitoring datasets with Prediction Benchmarks},
  journal = {arXiv}
  year    = {2023},
}
```

## Dependencies

We recommend to setup clean Python enviroment with [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html "conda-env") and install all dependenices by running `conda env export --no-builds > environment.yml`. 

To run [Latent ODE](https://github.com/YuliaRubanova/latent_ode) model, install [`torchdiffeq`](https://github.com/rtqichen/torchdiffeq).

## Code organization

The code is organized as follows:

- bin: training commands for all models
- config: configuration files for all datasets
- data_formatter
    - base.py: performs **all** pre-processing for all CGM datasets
- exploratory_analysis: notebooks with processing steps for pulling the data and converting to `.csv` files
- lib
    - gluformer: model implementation
    - latent_ode: model implementation
    - *.py: hyper-paraemter tuning, training, validation, and testing scripts
- output: hyper-parameter optimization and testing logs
- paper_results: code for producing tables and plots, found in the paper
- raw_data: web-pulled CGM data (processed using `exploratory_analysis`)
- utils: helper functions for model training and testing

## How to reproduce results?

## How to work with the repository?



