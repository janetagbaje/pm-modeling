import os
import h3
import argparse
import polars as pl
import numpy as np
import pandas as pd
from prefect import flow, task
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from prefect.artifacts import create_markdown_artifact
from data.pm25_cher import download_pmdata_from_database as pm
from data.gridmet_cher import download_gridmetpm25_from_database as gm


def convert_date_to_time(date):
    """
    Convert a string-formatted date to an integer ordinal format.

    Args:
        date (str): Date string in 'YYYY-MM-DD' format.

    Returns:
        int: Ordinal representation of the date.
    """
    """Convert date string to a datetime ordinal representation."""
    return datetime.strptime(date, "%Y-%m-%d").toordinal()

# Create h3 index from h3_08
def create_h3_from_h3_08(h3_08, target_resolution=3):
    """
    Convert an H3 index from resolution 8 to a coarser resolution (level = 3).

    Args:
        h3_08 (str): H3 index at resolution 8.
        target_resolution (int): Coarser H3 resolution (default is 3).

    Returns:
        str: H3 index at the new resolution.
    """
    """convert level 8 to target resolution"""
    return h3.h3_to_parent(h3_08, target_resolution)

@task
def load_pm_data(file_path, year=2013, columns_to_lag=None, max_lags=15):
    """
    Load and preprocess PM2.5 data per site.

    Args:
        file_path (str): Path to the CSV file containing PM2.5 data.
        year (int): Year to filter data by.
        columns_to_lag (list): Column specified for lag values.
        max_lags (int): Number of lag values to generate.

    Returns:
        polars.DataFrame: Preprocessed data with lag features per site.
    """
    """Load PM data and process lat, lon, elevation, processing by unique site."""
    
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

    # Filter by year 
    if year is not None:
        data = select_year(data, year, date_key="date")

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


def get_prior_lags(row, site_data, column, max_lags, interpolate='linear'):
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
    days = 15
    end_date = datetime.fromordinal(row["date"][0])
    start_date = end_date - timedelta(days=days)

    # Filter site data for the past 15 days
    prior_rows = site_data.filter(
        (pl.col("date") < row["date"][0]) &  # Dates before the current row
        (pl.col("date") >= start_date.toordinal())  # Dates within the 15-day window
    ).sort("date")

    if interpolate == 'linear':
        dates = prior_rows["date"].to_numpy()
        dates = dates-dates[0]
        values = prior_rows[column].to_numpy()
        prior_values = list(np.interp(np.arange(days), dates, values))

    # Pad with the current row's value if fewer than max_lags
    elif interpolate == 'pad':
        prior_values = prior_rows[column].to_list()[::-1][:max_lags]
        while len(prior_values) < max_lags:
            prior_values.append(row[column])

    return prior_values


@task
def load_gridmet_data(file_path, year, columns_to_lag, max_lags=15):
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
    print("Loading GridMET data...")
    data = pl.read_csv(file_path).rename(str.lower)

    # Filter by year if provided
    if year is not None:
        data = select_year(data, year, date_key="pm_date")

    # Convert date to ordinal format
    data = data.with_columns([
        pl.col("pm_date").map_elements(convert_date_to_time, return_dtype=pl.Int64).alias("date")
    ])

    # Rename columns to match pmdata columns
    data = data.rename({
        "pm_site": "site",
        "pm_lat": "lat",
        "pm_lon": "lon"
    })

    # Group by Site, Date, Lat, Lon and average the relevant columns
    gridmet_columns = [
        "min_air_temperature_value", "max_air_temperature_value",
        "min_relative_humidity_value", "max_relative_humidity_value",
        "wind_speed_value", "precipitation_amount_value", "wind_direction_value"
    ]

    for col in gridmet_columns:
        data = data.with_columns([
            pl.col(col)
        ])

    # Drop gridmet rows with NaNs
    data = data.drop_nulls()

    data = data.group_by(["site", "date", "lat", "lon"]).agg([
        pl.mean("min_air_temperature_value").alias("min_air_temperature"),
        pl.mean("max_air_temperature_value").alias("max_air_temperature"),
        pl.mean("min_relative_humidity_value").alias("min_relative_humidity"),
        pl.mean("max_relative_humidity_value").alias("max_relative_humidity"),
        pl.mean("wind_speed_value").alias("wind_speed"),
        pl.mean("precipitation_amount_value").alias("precipitation_amount"),
        pl.mean("wind_direction_value").alias("wind_direction"),
        pl.first("id").alias("id")  # Include id from gridmet data
    ])

    # Add prior lags for specified columns
    columns_to_lag = ["min_air_temperature", "max_air_temperature", "min_relative_humidity",
                      "max_relative_humidity", "wind_speed", "precipitation_amount", "wind_direction"]
    unique_sites = data["site"].unique().to_list()
    processed_data_list = []

    print(f"Found {len(unique_sites)} unique sites in GridMET data. Processing site by site...")
    for i, site in enumerate(unique_sites):
        print(f"Processing site {i + 1}/{len(unique_sites)}: {site}")
        site_data = data.filter(pl.col("site") == site).sort("date")
        processed_site_data = process_site_data(site_data, columns_to_lag, max_lags)
        processed_data_list.append(processed_site_data)

    # Concatenate processed data for all sites
    processed_data = pl.concat(processed_data_list, how="vertical")
    print("Completed processing for all sites in GridMET data.")

    return processed_data

@task
def split_data(data: pl.DataFrame):
    """
    Split dataset into train, validation, and test by site.

    Args:
        data (pl.DataFrame): Full dataset.

    Returns:
        pl.DataFrame: Dataset with new 'split' column.
    """
    sites = data["site"].unique()
    train_sites, test_sites = train_test_split(sites, test_size=0.1, random_state=42)
    train_sites, val_sites = train_test_split(train_sites, test_size=0.1, random_state=42)

    def assign_split(site):
        if site in test_sites:
            return "test"
        elif site in val_sites:
            return "val"
        else:
            return "train"

    # Assign split labels to each row
    split_data = data.with_columns([
        pl.col("site").map_elements(assign_split).alias("split")
    ])

    # Shuffle the final data
    split_data = split_data.sample(fraction=1.0, with_replacement=False, seed=42)

    return split_data


@task
def select_year(data, year, date_key):
    """
    Filter data to only include rows from the specified year.

    Args:
        data (pl.DataFrame): Input data.
        year (int): Year to filter by.
        date_key (str): Column containing date strings.

    Returns:
        pl.DataFrame: Filtered data.
    """

    data = data.with_columns(pl.col(date_key).str.strptime(pl.date,"%Y-%m-%d").dt.year().alias("year"))
    data = data.filter(pl.col("year") == year)
    data = data.drop("year")
    return data


@flow
def merge_and_save_data(data_dir, year, save_name="merged_raw_pm25_gridmet.parquet", max_lags=15):
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
    
    print("Starting the data processing flow...")
    pm_file_path = os.path.join(data_dir, "raw", "pmdata.csv")
    gridmet_file_path = os.path.join(data_dir, "raw", "gridmetpm25.csv")

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

    # Define the gridmet columns
    gridmet_columns = [
        "min_air_temperature", "max_air_temperature", "min_relative_humidity",
        "max_relative_humidity", "wind_speed", "precipitation_amount", "wind_direction"
    ]

    pm_data = load_pm_data(pm_file_path, year=year, columns_to_lag=["conc"], max_lags=max_lags)
    gridmet_data = load_gridmet_data(gridmet_file_path, year=year, columns_to_lag=gridmet_columns, max_lags=max_lags)


    # Ensuring to avoid column conflicts by dropping duplicates
    gridmet_data = gridmet_data.drop(["lat", "lon"])

    merged_data = pm_data.join(gridmet_data, on=["date", "site"], how="inner")

    final_data = merged_data.with_columns([
        pl.col("h3_08").map_elements(lambda h: create_h3_from_h3_08(h)).alias("h3")
    ])

    # Define the static columns
    static_columns = [
        "date", "lat", "lon", "elevation", "conc", "h3_08", "h3", "site", "id",
        "land_cover", "day_population", "night_population"
    ]

    # Define columns for which lags were added
    columns_to_lag = ["conc"] + gridmet_columns

    # Generate lagged column names
    lagged_columns = [f"lags_{col}" for col in columns_to_lag]

    # Combine all columns for the final data
    final_columns = static_columns + gridmet_columns + lagged_columns

    # Select the final set of columns
    final_data = final_data.select(final_columns)


    final_data = split_data(final_data)

    final_data = final_data.rename({"date": "time"})

    save_path = os.path.join(data_dir, 'processed')
    os.makedirs(save_path, exist_ok=True)

    final_data.write_parquet(os.path.join(save_path, save_name))

    create_markdown_artifact(
        key="data-save-summary",
        markdown=f"## Data successfully saved\n\nFile Path: `{os.path.join(save_path, save_name)}`\n\n### Preview\n\n{final_data.head().to_pandas().to_markdown()}"
    )
    print(f"Data successfully saved to {os.path.join(save_path, save_name)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--year',
        type=int,
        help='The year to process.',
        default=2013
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        help='path to the data.',
        default='../data'
    )

    # Parse the arguments
    args = parser.parse_args()
    if args.year == 0:
        args.year = None
    merge_and_save_data(data_dir=args.data_dir, year = args.year, save_name=f"processed_pm25_gridmet_{args.year}.parquet" )
