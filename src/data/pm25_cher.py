import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from prefect import flow, task

load_dotenv()

@task
def download_pmdata_from_database(data_dir):
    """
    Download daily EPA pm2.5 data and other features from a PostgreSQL database and saves it as a CSV file.

    Args:
        data_dir (str): The base directory where the 'raw' subdirectory will be created 
                        and the output CSV file will be saved.

    Returns:
        str: Full path to the saved CSV file ('pmdata.csv').
    """

    # Connect to the database with database credentials:
    username = os.getenv("DB_USERNAME")
    password = os.getenv("DB_PASSWORD")
    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT")
    database = os.getenv("DB_NAME")

    engine = create_engine(f'postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}')

    df = pd.read_sql("SELECT * FROM <schema>.<table_name>",con=engine)

    # Write data to CSV file with column names
    path_to_raw = os.path.join(data_dir,'raw')
    os.makedirs(path_to_raw, exist_ok=True)
          
    pmdata_file_path = os.path.join(path_to_raw, 'pmdata.csv')
    df.to_csv(pmdata_file_path, index=False)
    return pmdata_file_path

@flow
def pmdata_flow(data_dir):
    """
    Prefect flow to download pm2.5 data and return the saved file path.

    Args:
        data_dir (str): Base directory for storing the output data.

    Returns:
        str: Path to the saved CSV file.
    """
    return download_pmdata_from_database(data_dir)

if __name__ == "__main__":
    data_dir = os.getenv("DATA_DIR", "../../data")
    pmdata_flow(data_dir)
