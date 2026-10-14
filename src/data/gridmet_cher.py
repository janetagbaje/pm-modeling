import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from prefect import flow, task

load_dotenv()

@task
def download_gridmetpm25_from_database(data_dir):
    """
    Downloads daily GridMet weather data 
    from a PostgreSQL database and saves it as a CSV file.

    Args:
        data_dir (str): The base directory where the 'raw' subdirectory will be created 
                        and the output CSV file will be saved.

    Returns:
        str: Full path to the saved CSV file ('gridmetpm25.csv').
    """
    # Connect to the database with database credentials:

    username = os.getenv("DB_USERNAME")
    password = os.getenv("DB_PASSWORD")
    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT")
    database = os.getenv("DB_NAME")

    engine = create_engine(f'postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}')
    
    df = pd.read_sql("SELECT * FROM <schema>.<table_name>", con=engine)

    # Write data to CSV file with column names
    path_to_raw = os.path.join(data_dir, 'raw')
    os.makedirs(path_to_raw, exist_ok=True)
          
    gridmet_file_path = os.path.join(path_to_raw, 'gridmetpm25.csv')
    df.to_csv(gridmet_file_path, index=False)
    return gridmet_file_path

@flow
def gridmet_flow(data_dir):
    """
    Prefect flow to download GridMet data and return the saved file path.

    Args:
        data_dir (str): Base directory for storing the output data.

    Returns:
        str: Path to the saved CSV file.
    """
    return download_gridmetpm25_from_database(data_dir)

if __name__ == "__main__":
    data_dir = os.getenv("DATA_DIR", "../../data")
    gridmet_flow(data_dir)
