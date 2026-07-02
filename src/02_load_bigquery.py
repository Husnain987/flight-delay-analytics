"""
02_load_bigquery.py
Loads the cleaned 2024 flight data from Parquet into BigQuery.
Auth: Application Default Credentials (gcloud auth application-default login).
"""

import pandas as pd
import pandas_gbq

# --- Configuration ---
PROJECT_ID = "project-9688db14-3c65-402e-b92"
DATASET = "flights"
TABLE = "flights_2024"
DESTINATION = f"{DATASET}.{TABLE}"
PARQUET_PATH = "data/processed/flights_2024.parquet"

# --- Load the cleaned data from disk ---
print(f"Reading {PARQUET_PATH} ...")
df = pd.read_parquet(PARQUET_PATH)
print(f"Loaded {len(df):,} rows and {len(df.columns)} columns from Parquet.")

# --- Push to BigQuery ---
print(f"Uploading to {PROJECT_ID}.{DESTINATION} ...")
pandas_gbq.to_gbq(
    df,
    destination_table=DESTINATION,
    project_id=PROJECT_ID,
    if_exists="replace",
    progress_bar=True,
)

print("Done. Table is live in BigQuery.") 