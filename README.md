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

# Dependencies

We recommend to setup clean Python enviroment with [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html "conda-env") and install all dependenices by running `conda env export --no-builds > environment.yml`. 

To run [Latent ODE](https://github.com/YuliaRubanova/latent_ode) model, install [`torchdiffeq`](https://github.com/rtqichen/torchdiffeq).

# Code organization

The code is organized as follows:

- `bin/`: training commands for all models
- `config/`: configuration files for all datasets
- `data_formatter/`
    - base.py: performs **all** pre-processing for all CGM datasets
- `exploratory_analysis/`: notebooks with processing steps for pulling the data and converting to `.csv` files
- `lib/`
    - `gluformer/`: model implementation
    - `latent_ode/`: model implementation
    - `*.py`: hyper-paraemter tuning, training, validation, and testing scripts
- `output/`: hyper-parameter optimization and testing logs
- `paper_results/`: code for producing tables and plots, found in the paper
- `utils/`: helper functions for model training and testing
- `raw_data.zip`: web-pulled CGM data (processed using `exploratory_analysis`)
- `environment.yml`: conda environment file

# Data

The datasets are distributed according to the following licences and can be downloaded from the following links outlined in the table below.

| Dataset | License | Number of patients | CGM Frequency |
| ------- | ------- | ------------------ | ------------- |
| [Colas](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0225817#sec018) | [Creative Commons 4.0](https://creativecommons.org/licenses/by/3.0/us/) | 208 | 5 minutes |
| [Dubosson](https://doi.org/10.5281/zenodo.1421615) | [Creative Commons 4.0](https://creativecommons.org/licenses/by-sa/4.0/legalcode) | 9 | 5 minutes |
| [Hall](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.2005143#pbio.2005143.s010) | [Creative Commons 4.0](https://creativecommons.org/licenses/by/4.0/) | 57 | 5 minutes |
| [Broll](https://github.com/irinagain/iglu) | [GPL-2](https://www.r-project.org/Licenses/GPL-2) | 5 | 5 minutes |
| [Weinstock](https://public.jaeb.org/dataset/537) | [Creative Commons 4.0](https://creativecommons.org/licenses/by-sa/4.0/legalcode) | 200 | 5 minutes |

To process the data, follow the instructions in the `exploratory_analysis/` folder. Processed datasets should be saved in the `raw_data/` folder. We provide examples in the `raw_data.zip` file.

# How to reproduce results?

# How to work with the repository?



