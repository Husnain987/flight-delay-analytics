"""
01_clean.py
Reads all raw BTS monthly CSVs, cleans and standardizes them,
derives analytical fields, and writes a single combined Parquet file.
"""

import pandas as pd
import glob
import os

RAW_DIR = "data/raw"
OUT_PATH = "data/processed/flights_2024.parquet"

# Columns we actually want, mapped to clean snake_case names
COLUMN_MAP = {
    "FL_DATE": "flight_date",
    "OP_UNIQUE_CARRIER": "airline",
    "OP_CARRIER_FL_NUM": "flight_number",
    "ORIGIN": "origin",
    "DEST": "dest",
    "CRS_DEP_TIME": "scheduled_dep_time",
    "DEP_TIME": "actual_dep_time",
    "DEP_DELAY": "dep_delay",
    "DEP_DELAY_NEW": "dep_delay_min",
    "CRS_ARR_TIME": "scheduled_arr_time",
    "ARR_TIME": "actual_arr_time",
    "ARR_DELAY": "arr_delay",
    "ARR_DELAY_NEW": "arr_delay_min",
    "CANCELLED": "cancelled",
    "CANCELLATION_CODE": "cancellation_code",
    "DIVERTED": "diverted",
    "AIR_TIME": "air_time",
    "DISTANCE": "distance",
    "CARRIER_DELAY": "carrier_delay",
    "WEATHER_DELAY": "weather_delay",
    "NAS_DELAY": "nas_delay",
    "SECURITY_DELAY": "security_delay",
    "LATE_AIRCRAFT_DELAY": "late_aircraft_delay",
}


def load_and_clean():
    csv_files = glob.glob(os.path.join(RAW_DIR, "*.csv"))
    print(f"Found {len(csv_files)} CSV files to process.")

    frames = []
    for f in csv_files:
        print(f"  Reading {os.path.basename(f)} ...")
        # BTS files have a trailing comma -> phantom 'Unnamed' column.
        # usecols with our known columns sidesteps that entirely.
        df = pd.read_csv(
            f,
            usecols=lambda c: c in COLUMN_MAP,
            low_memory=False,
        )
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    print(f"Combined raw rows: {len(df):,}")

    # Rename to clean column names
    df = df.rename(columns=COLUMN_MAP)

    # --- Type fixes ---
    df["flight_date"] = pd.to_datetime(df["flight_date"])
    df["cancelled"] = df["cancelled"].astype(bool)
    df["diverted"] = df["diverted"].astype(bool)

    # Delay-cause columns are only populated for delayed flights; fill NaN with 0
    cause_cols = [
        "carrier_delay", "weather_delay", "nas_delay",
        "security_delay", "late_aircraft_delay",
    ]
    df[cause_cols] = df[cause_cols].fillna(0)

    # --- Derived analytical fields ---
    df["route"] = df["origin"] + "-" + df["dest"]
    df["day_of_week"] = df["flight_date"].dt.day_name()
    df["month"] = df["flight_date"].dt.month
    df["is_weekend"] = df["flight_date"].dt.dayofweek >= 5

    # A flight is "delayed" if it arrived 15+ min late (BTS standard)
    df["is_delayed"] = df["arr_delay_min"] >= 15

    # Delay severity buckets for easy dashboard filtering
    df["delay_bucket"] = pd.cut(
        df["arr_delay_min"],
        bins=[-float("inf"), 0, 15, 60, 180, float("inf")],
        labels=["on_time_or_early", "minor", "moderate", "severe", "extreme"],
    )

    print(f"Final shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


def main():
    os.makedirs("data/processed", exist_ok=True)
    df = load_and_clean()
    df.to_parquet(OUT_PATH, index=False)
    size_mb = os.path.getsize(OUT_PATH) / 1_000_000
    print(f"\nWrote {OUT_PATH} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()