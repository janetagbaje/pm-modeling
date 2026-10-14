import os
import pickle
import polars as pl
import pandas as pd
import numpy as np
import data_scalers as ds
from sklearn.pipeline import Pipeline


import argparse

def get_default_normalizer():
    """
    Creates and returns a default sklearn-like Pipeline with specified scalers.

    Returns:
        sklearn.pipeline.
        Pipeline: Pipeline of custom normalization transformers from data_scaler script.
    """
    land_cover_values = [41, 22, 0, 31, 42, 12, 90, 23, 11, 43, 82, 81, 71, 24, 21, 95, 52]
    normalizer = Pipeline( [('time', ds.MinMaxScaler('time')), 
                            ('lat_lon', ds.LatLonScaler('lat','lon')), 
                            ('elevation_lb', ds.LowerThreshold('elevation', 0)),
                            ('elevation_scale', ds.MinMaxScaler('elevation')),
                            ('conc_lb', ds.LowerThreshold('conc', 0.005)),
                            ('conc_ub', ds.UpperThreshold('conc', 100)),
                            ('conc', ds.LogarithmicScaler('conc', base=10)),
                            ('min_air_temperature', ds.StandardScaler('min_air_temperature')),
                            ('max_air_temperature', ds.StandardScaler('max_air_temperature')),
                            ('min_relative_humidity', ds.StandardScaler('min_relative_humidity')),
                            ('max_relative_humidity', ds.StandardScaler('max_relative_humidity')),
                            ('land_cover', ds.CategoricalScaler('land_cover', land_cover_values)),
                            ('wind_speed', ds.StandardScaler('wind_speed')),
                            ('precipitation_amount', ds.StandardScaler('precipitation_amount')),
                            ('wind_direction', ds.StandardScaler('wind_direction')),
                            ('day_population', ds.StandardScaler('day_population')),
                            ('night_population', ds.StandardScaler('night_population'))]
    )
    return normalizer

gridmet_columns = [
            "min_air_temperature", "max_air_temperature",
            "min_relative_humidity", "max_relative_humidity", 
            "wind_speed", "precipitation_amount", "wind_direction"
        ]

def normalize_lags(data, columns_to_lag):
    """
    Normalize the lagged feature columns.

    Args:
        data (pd.DataFrame): Input dataframe with lag columns as lists.
        columns_to_lag (list of str): Column names that have corresponding lag columns to normalize.

    Returns:
        pd.DataFrame: Data with normalized lag features.
    """
    data_normalized = data.copy()
    
    # Generate lagged column names
    lagged_columns = [f"lags_{col}" for col in columns_to_lag] + ['lags_pm_prior']
    
    for col in lagged_columns:
        if col not in data_normalized.columns:
            continue
            
        # Convert list to numpy array for processing
        data_arrays = np.array(data_normalized[col].tolist())
        
        # Create a DataFrame with the flattened array
        data_flat_arrays = pd.DataFrame(data_arrays.reshape(-1, 1), columns=[col])

        #Apply appropriate scaling
        if col == 'lags_pm_prior':
            data_flat_arrays[data_flat_arrays <= 0] = 1e-8
            scaler = ds.LogarithmicScaler(col, base=10)
        else:
            base_col = col.replace('lags_', '')
            if base_col in gridmet_columns:
                scaler = ds.StandardScaler(col)
        
        # Transform and reshape
        data_flat_arrays = scaler.fit_transform(data_flat_arrays)
        normalized = data_flat_arrays[col].values.reshape(data_arrays.shape)
        data_normalized[col] = normalized.tolist()
            
    return data_normalized


def get_normalized_data(data, normalizer_path, drop_null = True, drop_zero_pm = True):
    """
    Applies feature normalization to the input data. Loads a previously saved normalizer if needed and path provided.

    Args:
        data (pd.DataFrame): Input data.
        normalizer_path (str or None): If provided, path to a pickled normalizer to load.
        drop_null (bool): Placeholder for future null-handling.
        drop_zero_pm (bool): Placeholder for filtering low PM.

    Returns:
        Tuple[pd.DataFrame, Pipeline]: Tuple of normalized data and the fitted or loaded pipeline.
    """
    if normalizer_path is None:
        normalizer = get_default_normalizer()
        data = normalizer.fit_transform(data)

        # Normalize the lag columns
        data = normalize_lags(data, gridmet_columns)

    else:
        with open(normalizer_path,'rb') as f:
            normalizer = pickle.load(f)
        data = normalizer.transform(data)
    return data, normalizer

def main(args):
    """
    Loads data, applies normalization, and saves both data and normalizer.

    Args:
        args (argparse.Namespace): `data_path` and `normalizer_path (optional)`.
    """
    # load the data
    data = pd.read_parquet(args.data_path)

    # transform it
    data, normalizer = get_normalized_data(data, args.normalizer_path)

    # save data
    save_name = args.data_path[:-8] + '_normalized.parquet'
    data.to_parquet(save_name)
    print(f'data saved to {save_name}')

    # save normalizer
    normalizer_name = os.path.join(os.path.dirname(save_name), 'normalizer.pickle')
    with open(normalizer_name,'wb') as f:
        pickle.dump(normalizer, f)
    print(f'normalizer saved to {normalizer_name}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process some paths.")
    
    # Add arguments for data path and normalizer path
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--normalizer_path', type=str, required=False, default=None, help='Path to the normalizer')

    # Parse the arguments
    args = parser.parse_args()
    main(args)
