from distutils.command.register import register
from typing import Optional, List

import typer
from pathlib import Path

from darts import TimeSeries
from darts.models import LinearRegressionModel
from sympy import pprint
import pandas as pd
app = typer.Typer()



def get_static_covariate_fields(model: LinearRegressionModel) -> Optional[List[str]]:
    """
    Returns the names of the static covariate fields that the model was trained with.

    Parameters:
    model (LinearRegressionModel): The trained LinearRegressionModel from which to extract static covariate fields.

    Returns:
    Optional[List[str]]: A list of static covariate field names if they exist, otherwise None.
    """
    # General check across common attributes where static covariates might be stored
    possible_attributes = [
        "_static_covariates",        # Possible private attribute
        "static_covariates",         # Possible public attribute
        "_static_covariate_columns", # Attribute that might hold covariate column names
        "static_covariate_columns"   # Public version of the above
    ]

    for attr in possible_attributes:
        if hasattr(model, attr):
            covariates = getattr(model, attr)
            if covariates is not None:
                if isinstance(covariates, (list, pd.Index)):  # Assuming covariates are stored as a list or pandas Index
                    return list(covariates)
                elif isinstance(covariates, pd.DataFrame):
                    return list(covariates.columns)

    # Check if there are other attributes or logs to look into
    print("Model attributes and their types:")
    for attr_name, attr_value in model.__dict__.items():
        print(f"{attr_name}: {type(attr_value)}")

    return None


@app.command()
def linear(model_path: Path = Path("output/models/livia/linear_livia_10_pkl"), file: Optional[Path] = Path("raw_data/livia_mini.csv"), n: int = 10):
    model = LinearRegressionModel.load(model_path)
    pprint(get_static_covariate_fields(model))

    data = pd.read_csv(file, sep=",")
    data['time'] = pd.to_datetime(data['time'])

     # Extract the 'id' column as a static covariate
    static_covariates = data[['id']].iloc[0]

    # Choose a target frequency, e.g., 5 minutes ('5T')
    target_frequency = '5T'
    data_resampled = data.set_index('time').resample(target_frequency).ffill().reset_index()

   

    # Convert the resampled DataFrame to a TimeSeries object
    series = TimeSeries.from_dataframe(data_resampled, time_col='time',value_cols= 'gl',static_covariates=static_covariates)

    # Predict future glucose values
    predictions = model.predict(n=n,  verbose=True, series=series)
    print(f"loaded {model_path}")
    #pprint(predictions)


if __name__ == "__main__":
    app()