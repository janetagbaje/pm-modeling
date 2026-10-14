import os
import polars as pl
from datetime import datetime, timedelta
from prefect import flow, task
from prefect.artifacts import create_markdown_artifact
import argparse
import json

from data.pm25_cher import download_pmdata_from_database as pm
from data.gridmet_cher import download_gridmetpm25_from_database as gm
from data.hexes_centres import download_hexes_data_from_database as hm

def convert_date_to_time(date):
    """Convert date string to a datetime ordinal representation."""
    return datetime.strptime(date, "%Y-%m-%d").toordinal()

@task
def load_and_preprocess_pm_data(file_path, year=2013, columns_to_lag=None, max_lags=15):
    """
    Load, process, and merge PM2.5 and GridMET data with lag features and save as Parquet.

    Args:
        data_dir (str): Base path to data directory.
        year (int): Year specified.
        save_name (str): Name for the saved Parquet file.
        max_lags (int): Number of lag values to compute per feature.

    Returns:
        None: Saves the merged DataFrame to a Parquet file for all year.
    """
    
    data = pl.read_csv(file_path).rename(str.lower)

    # Aggregate data by site, date, and location for colocated sites
    data = data.group_by(["site", "date", "lat", "lon"]).agg([
        pl.mean("conc").alias("conc"),
        pl.mean("land_cover").alias("land_cover"),
        pl.mean("elevation").alias("elevation"),
        pl.mean("day_population").alias("day_population"),
        pl.mean("night_population").alias("night_population"),
        pl.first("h3_08").alias("h3_08")
    ])


    # Convert date to ordinal format
    data = data.with_columns([
        pl.col("date").map_elements(convert_date_to_time, return_dtype=pl.Int64).alias("date")
    ])

    # Group by site
    unique_sites = data["site"].unique().to_list()
    pm_data_list = []

    print(f"Found {len(unique_sites)} unique sites. Processing data site by site...")

    # Process each site independently
    for i, site in enumerate(unique_sites):
        print(f"Processing site {i + 1}/{len(unique_sites)}: {site}")
        site_data = data.filter(pl.col("site") == site).sort("date")
        processed_site_data = process_site_data(site_data, columns_to_lag, max_lags)
        pm_data_list.append(processed_site_data)

    # Concatenate processed data for all sites
    pm_data = pl.concat(pm_data_list, how="vertical")
    print("Completed processing for all sites.")

    return pm_data

def process_site_data(site_data, columns_to_lag, max_lags):
    """
    Process data for a single site, getting prior lags for specified columns.
    
    Args:
        site_data (polars.DataFrame): Data for a single site.
        columns_to_lag (list of str): List of column names to calculate lags for.
        max_lags (int): Maximum number of previous values to retrieve for lags.
    
    Returns:
        polars.DataFrame: Processed site data with lagged values for specified columns.
    """
    data_dicts = site_data.to_dicts()
    processed_rows = []

    for row in data_dicts:
        processed_row = row.copy()

        # Calculate lags for each lag column
        for col in columns_to_lag:
            processed_row[f"lags_{col}"] = get_prior_lags(row, site_data, col, max_lags)
        processed_rows.append(processed_row)

    return pl.DataFrame(processed_rows)


def get_prior_lags(row, site_data, column, max_lags):
    """
    Get previous values for a specific column within the past 15 days.
    
    Args:
        row (dict): Current row of data.
        site_data (polars.DataFrame): Data for the entire site.
        column (str): Name of the column to calculate lags for.
        max_lags (int): Maximum number of previous values to retrieve.
    
    Returns:
        list: List of prior values for the specified column.
    """
    # Current row date
    end_date = datetime.fromordinal(row["date"])
    start_date = end_date - timedelta(days=15)

    # Filter site data for the past 10 days
    prior_rows = site_data.filter(
        (pl.col("date") < row["date"]) &  
        (pl.col("date") >= start_date.toordinal())  
    ).sort("date")

    # Extract values for the specified column
    prior_values = prior_rows[column].to_list()[::-1][:max_lags]

    # Pad with the current row's value if fewer than max_lags
    while len(prior_values) < max_lags:
        prior_values.append(row[column])

    return prior_values

@task
def load_and_preprocess_gridmet_data(file_path):
    """
    Load and preprocess GridMET data, adding prior lags for specified columns.
    
    Args:
        file_path (str): Path to the GridMET data file.
        year (int): Year to filter the data for.
        columns_to_lag (list of str): List of column names to calculate lags for.
        max_lags (int): Maximum number of previous values to retrieve for lags.
    
    Returns:
        polars.DataFrame: Preprocessed GridMET data with lags added.
    """

    data = pl.read_csv(file_path)

    # Convert column names to lowercase
    data = data.rename({col: col.lower() for col in data.columns})
    
    data = data.with_columns([
        pl.col("pm_date").map_elements(convert_date_to_time, return_dtype=pl.Int64).alias("date")
    ])

    data = data.rename({
        "min_air_temperature_value": "min_air_temperature",
        "max_air_temperature_value": "max_air_temperature",
        "min_relative_humidity_value": "min_relative_humidity",
        "max_relative_humidity_value": "max_relative_humidity",
        "wind_speed_value": "wind_speed",
        "precipitation_amount_value": "precipitation_amount",
        "wind_direction_value": "wind_direction",
        "pm_site": "site",
        "pm_lat": "lat",
        "pm_lon": "lon"
    })

    data = data.drop_nulls()

    data = data.drop(["lat", "lon"])

    data = data.group_by(["site", "date"]).agg([
        pl.mean("min_air_temperature"),
        pl.mean("max_air_temperature"),
        pl.mean("min_relative_humidity"),
        pl.mean("max_relative_humidity"),
        pl.mean("wind_speed"),
        pl.mean("precipitation_amount"),
        pl.mean("wind_direction"),
        pl.first("id").alias("id")
    ])

    print(f"GridMET Data shape after loading and processing: {data.shape}")
    return data

@task
def merge(pm_data, gridmet_data):
    """Merge PM and GridMET data and normalize."""
    data = pm_data.join(gridmet_data, on=["site", "date"], how="inner")
    print(f"Merged Data shape after joining PM and GridMET data: {data.shape}")

    data = data.drop_nulls()

    data = data.with_columns([
        pl.lit("train").alias("split")
    ])

    return data

@task
def load_and_process_hexes_centres(file_path, ordinal_date):
    """
    Load hexes centre data. Adds inference date and test split.

    Args:
        file_path (str): Path to hexes data file.
        ordinal_date (int): Date to assign to all rows (in ordinal format).

    Returns:
        polars.DataFrame: Prepared hexes centre dataset for inference.
    """
    data = pl.read_csv(file_path)

    # Convert column names to lowercase
    data = data.rename({col: col.lower() for col in data.columns})

    # Rename lat, lon, and elevation to Lat, Lon, Elevation for consistency
    data = data.rename({
        "min_air_temperature_value": "min_air_temperature",
        "max_air_temperature_value": "max_air_temperature",
        "min_relative_humidity_value": "min_relative_humidity",
        "max_relative_humidity_value": "max_relative_humidity",
        "wind_speed_value": "wind_speed",
        "precipitation_amount_value": "precipitation_amount",
        "wind_direction_value": "wind_direction"
    })

    data = data.drop_nulls()

    data = data.with_columns([
        pl.lit("test").alias("split"),
        pl.lit(ordinal_date).alias("date").cast(pl.Int64)  # Ensure the date is cast to Int64
    ])

    print(f"Hexes Centres Data shape after loading and processing: {data.shape}")
    return data

@task
def select_year_day(data, date):
    """
    Filter data to only include rows from the specified year.

    Args:
        data (pl.DataFrame): Input data.
        year (int): Year to filter by.
        date_key (str): Column containing date strings.

    Returns:
        pl.DataFrame: Filtered data.
    """
    ordinal_date = convert_date_to_time(date)
    data = data.filter(pl.col("date") == ordinal_date)  # Correct filter usage
    return data

@task
def concatenate_and_save(pm_data, hexes_data, save_path):
    """
    Concatenate PM+GridMET data and hexes data into one inference dataset.

    Ensures:
        - Column alignment
        - Type consistency
        - 'time' column added
        - Save as Parquet

    Args:
        pm_data (pl.DataFrame): Processed sensor data (split=train).
        hexes_data (pl.DataFrame): Processed hex grid data (split=test).
        save_path (str): Path to save the final dataset.

    Returns:
        polars.DataFrame: Final concatenated dataset.
    """

    # Get the set of all columns in both DataFrames
    pm_columns = set(pm_data.columns)
    hexes_columns = set(hexes_data.columns)

    # Add missing columns to hexes_data with None values and cast to match types
    missing_in_hexes = pm_columns - hexes_columns
    for col in missing_in_hexes:
        col_dtype = pm_data[col].dtype
        hexes_data = hexes_data.with_columns([pl.lit(None).cast(col_dtype).alias(col)])

    # Add missing columns to pm_data with None values and cast to match types
    missing_in_pm = hexes_columns - pm_columns
    for col in missing_in_pm:
        col_dtype = hexes_data[col].dtype
        pm_data = pm_data.with_columns([pl.lit(None).cast(col_dtype).alias(col)])

    # Ensure both DataFrames have the same column order
    all_columns = sorted(list(pm_columns.union(hexes_columns)))
    pm_data = pm_data.select(all_columns)
    hexes_data = hexes_data.select(all_columns)

    # Standardize numeric column types to Float64 to avoid type conflicts
    for col in all_columns:
        if pm_data[col].dtype in [pl.Int64, pl.Float64] or hexes_data[col].dtype in [pl.Int64, pl.Float64]:
            pm_data = pm_data.with_columns([pm_data[col].cast(pl.Float64)])
            hexes_data = hexes_data.with_columns([hexes_data[col].cast(pl.Float64)])

    # Concatenate the two DataFrames
    combined_data = pl.concat([pm_data, hexes_data], how="vertical")

    # Rename 'date' to 'time' in the final combined DataFrame
    combined_data = combined_data.rename({"date": "time"})

    print(f"Concatenated Data shape: {combined_data.shape}")

    # Save the concatenated data as a Parquet file
    combined_data.write_parquet(save_path)

    return combined_data


@flow
def process_and_save_data(data_dir, date, inference_save_name="inference_dataset.parquet", max_lags=15, chunk_size=100000):

    """
    Orchestrates full data pipeline:
        - Downloads PM2.5, GridMET, and Hex data if not available
        - Processes & merges them
        - Filters by date
        - Concatenates with hexes data
        - Saves

    Args:
        data_dir (str): Base directory containing raw and processed folders.
        date (str): Date string to filter for inference (format: 'YYYY-MM-DD').
        inference_save_name (str): Name for the saved inference dataset.
        max_lags (int): Number of lag values to compute.
        chunk_size (int): Placeholder for batching if needed.

    Returns:
        None
    """
    print("Starting the data processing flow...")
    pm_file_path = os.path.join(data_dir, "raw", "pmdata.csv")
    gridmet_file_path = os.path.join(data_dir, "raw", "gridmetpm25.csv")
    hexes_centres_file_path = os.path.join(data_dir, "raw", "conus_hexes_data.csv")

    if not os.path.isfile(pm_file_path):
        print("PM data not found, downloading...")
        pm_file_path = pm(data_dir)
        if pm_file_path is None or not os.path.isfile(pm_file_path):
            raise ValueError(f"Download failed or file not found at path: {pm_file_path}")
    else:
        print("PM data already exists, skipping download.")

    if not os.path.isfile(gridmet_file_path):
        print("GridMET data not found, downloading...")
        gridmet_file_path = gm(data_dir)
        if gridmet_file_path is None or not os.path.isfile(gridmet_file_path):
            raise ValueError(f"Download failed or file not found at path: {gridmet_file_path}")
    else:
        print("GridMET data already exists, skipping download.")

    if not os.path.isfile(hexes_centres_file_path):
        print("HEXES data not found, downloading...")
        hexes_centres_file_path = hm(data_dir)
        if hexes_centres_file_path is None or not os.path.isfile(hexes_centres_file_path):
            raise ValueError(f"Download failed or file not found at path: {hexes_centres_file_path}")
    else:
        print("HEXES data already exists, skipping download.")

    pm_data = load_and_preprocess_pm_data(pm_file_path,  columns_to_lag=["conc"], max_lags=max_lags)
    gridmet_data = load_and_preprocess_gridmet_data(gridmet_file_path)

    merged_data = merge(pm_data, gridmet_data)
    merged_data = select_year_day(merged_data, date)

    # Convert the selected date to an ordinal for the hexes data
    ordinal_date = convert_date_to_time(date)
    hexes_centres_data = load_and_process_hexes_centres(hexes_centres_file_path, ordinal_date)

    save_path = os.path.join(data_dir, 'processed')
    os.makedirs(save_path, exist_ok=True)

    inference_save_path = os.path.join(save_path, inference_save_name)
    concatenated_data = concatenate_and_save(merged_data, hexes_centres_data, inference_save_path)
    create_markdown_artifact(
        key="inference-data-save-summary",
        markdown=f"## Inference Dataset successfully saved\n\nFile Path: {inference_save_path}\n\n### Preview\n\n{concatenated_data.head().to_pandas().to_markdown()}"
    )
    print(f"Inference Dataset successfully saved to {inference_save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--date',
        type=str,
        help='The date to process (YYYY-MM-DD).',
        default='2013-01-01'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        help='Path to the data.',
        default='../data'
    )

    # Parse the arguments
    args = parser.parse_args()
    process_and_save_data(data_dir=args.data_dir, date=args.date, inference_save_name="inference_dataset.parquet")
