import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine


load_dotenv()

"""
Join LandScan day and night population values to PM2.5 data using H3 indices.
"""

# Connect to the database with database credentials:
username = os.getenv("DB_USERNAME")
password = os.getenv("DB_PASSWORD")
host = os.getenv("DB_HOST")
port = os.getenv("DB_PORT")
database = os.getenv("DB_NAME")

    
# Create engine 
engine = create_engine(f'postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}')
    

# Load pmdata
data_dir = os.getenv("DATA_DIR", "../../data")

pmdata = pd.read_csv(data_dir,  "raw", "pmdata.csv")
 
# Query day and night population data 
day_population = pd.read_sql("SELECT h3_index, population FROM <schema>.<day_population_table>", con=engine)
night_population = pd.read_sql("SELECT h3_index, population FROM <schema>.<night_population_table>", con=engine)
    
# Remove duplicates in day and night population data based on h3_index
day_population = day_population.drop_duplicates(subset='h3_index')
night_population = night_population.drop_duplicates(subset='h3_index')
    
# Rename the columns in day and night population data
day_population = day_population.rename(columns={'population': 'day_population'})
night_population_pd = night_population.rename(columns={'population': 'night_population'})
    
# Merge 
merged_data = pmdata.join(
    day_population,
    left_on="h3_08",
    right_on="h3_index",
    how="left"
    )
    
merged_data = merged_data.join(
    night_population,
    left_on="h3_08",
    right_on="h3_index",
    how="left"
    )
    
# Drop the redundant h3_index column if it exists
if "h3_index" in merged_data.columns:
    merged_data = merged_data.drop("h3_index")
    
output_file = os.path.join(
    data_dir,
    "raw",
    "pmdata_landscan.csv"
)

# Save  
merged_data.write_csv(output_file, index=False)
    
print(f"Merged data shape: {merged_data.shape}")
print(merged_data.head())
