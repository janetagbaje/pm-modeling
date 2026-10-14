import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()

"""
Join landcover values to PM2.5 data using H3 indices.
"""

# Connect to the database with database credentials:
username = os.getenv("DB_USERNAME")
password = os.getenv("DB_PASSWORD")
host = os.getenv("DB_HOST")
port = os.getenv("DB_PORT")
database = os.getenv("DB_NAME")

engine = create_engine(f'postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}')

# Load pmdata 
data_dir = os.getenv("DATA_DIR", "../../data")
pmdata = pd.read_csv(
		os.path.join(data_dir, "raw", "pmdata.csv")
)

land_cover_data = pd.read_sql("SELECT h3_index, land_cover FROM <schema>.<table_name>", con=engine)

# Drop duplicates from land_cover_data based on h3_index
land_cover_data = land_cover_data.drop_duplicates(subset='h3_index')

# Merge pmdata with landcover values
merged_data = pd.merge(pmdata, land_cover_data, left_on='h3_index', right_on='h3_index', how='left')

# Ensure the number of rows in merged_data is the same as pmdata
if len(merged_data) != len(pmdata):
    print(f"Row count mismatch: pmdata has {len(pmdata)} rows, but merged data has {len(merged_data)} rows.")

# Save the result 
output_file = os.path.join(data_dir, "raw", "pmdata_landcover.csv")

merged_data.to_csv(output_fle, index=False)

print(merged_data.head())
