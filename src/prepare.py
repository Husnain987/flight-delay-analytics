"""
prepare.py

Builds the model-ready table used for the statistical-modeling extension of
this project (OLS / logistic regression on arrival delay).

Run (with venv activated):
    python src/prepare.py

Reads:  data/processed/flights_2024.parquet
Writes: data/model/model_table.parquet
"""

import os
import pandas as pd

SRC_PATH = "data/processed/flights_2024.parquet"
OUT_DIR = "data/model"
OUT_PATH = os.path.join(OUT_DIR, "model_table.parquet")

# I only load the columns I actually need. In particular I deliberately don't
# load any of the "leakage" columns from the source file:
#   dep_delay, dep_delay_min, actual_dep_time, actual_arr_time, air_time,
#   arr_delay_min, delay_bucket, cancellation_code, carrier_delay,
#   weather_delay, nas_delay, security_delay, late_aircraft_delay
# Every one of those is only known AFTER a flight has departed or landed.
# A model is meant to predict delay risk using information available BEFORE
# departure (schedule, route, calendar). Including any post-departure column
# would let the model "see the answer" (e.g. arr_delay_min is a clipped copy
# of the target itself, and *_delay reason columns only exist once a delay
# has already happened) -- this is target leakage, not a real feature.
COLUMNS = [
    "flight_date",
    "airline",
    "origin",
    "dest",
    "scheduled_dep_time",
    "arr_delay",
    "cancelled",
    "diverted",
    "distance",
    "month",
    "is_delayed",
]

# 2024 US federal holidays (actual observed calendar dates), used below to
# build a holiday_flag feature (+/- 3 days around each).
HOLIDAYS_2024 = {
    "New Year's Day": "2024-01-01",
    "MLK Day": "2024-01-15",
    "Memorial Day": "2024-05-27",
    "July 4th": "2024-07-04",
    "Labor Day": "2024-09-02",
    "Thanksgiving": "2024-11-28",
    "Christmas": "2024-12-25",
}

DOW_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

KEEP_COLUMNS = [
    "flight_date", "airline", "origin", "dest",
    "dep_hour", "dow", "month", "holiday_flag",
    "distance", "arr_delay", "is_delayed",
]


def main():
    print("=" * 70)
    print("SECTION 1: LOAD")
    print("=" * 70)
    df = pd.read_parquet(SRC_PATH, columns=COLUMNS)
    print(f"Rows loaded: {len(df):,}")

    print()
    print("=" * 70)
    print("SECTION 2: DROP CANCELLED / DIVERTED FLIGHTS")
    print("=" * 70)
    # A cancelled or diverted flight never produces a real scheduled arrival,
    # so arr_delay is null for exactly these rows (I confirmed beforehand:
    # 113,814 null arr_delay rows == 96,315 cancelled + 17,499 diverted, with
    # zero overlap between the two groups). Since the modeling target here is
    # arrival delay / is_delayed, a row with no target is not a usable
    # training example. I drop these rows rather than impute a delay value,
    # because fabricating a delay for a flight that never landed would be
    # inventing the label, not measuring it.
    n_cancelled = int(df["cancelled"].sum())
    df = df.loc[~df["cancelled"]].copy()
    print(f"Removed cancelled == True: {n_cancelled:,} rows")

    n_diverted = int(df["diverted"].sum())
    df = df.loc[~df["diverted"]].copy()
    print(f"Removed diverted == True: {n_diverted:,} rows")

    print(f"Rows remaining: {len(df):,}")

    # Sanity check on the assumption above: if it's true that null arr_delay
    # occurs only for cancelled/diverted flights, there should be nothing left
    # to fail on here. If this assert fires, my assumptions about the data
    # were wrong and the script should stop rather than silently drop/impute
    # more rows.
    assert df["arr_delay"].isna().sum() == 0, (
        "arr_delay still has nulls after removing cancelled/diverted rows -- "
        "null arr_delay is NOT exactly cancelled+diverted as assumed."
    )
    print("Assert passed: zero arr_delay nulls remain after filtering.")

    print()
    print("=" * 70)
    print("SECTION 3: DERIVE FEATURES")
    print("=" * 70)

    # --- dep_hour ---
    # scheduled_dep_time is an integer in HHMM form (e.g. 1015 -> 10:15am).
    # Integer-dividing by 100 drops the minutes and keeps the hour, e.g.
    # 1015 // 100 == 10.
    #
    # DATA NOTE: I didn't expect this one -- exactly one row uses the
    # schedule-industry convention of "2400" to mean midnight, instead of
    # "0000". 2400 // 100 == 24, which falls outside the expected 0..23 hour
    # range. Rather than silently truncating every row with modulo (which
    # would hide any *other* out-of-range values too), I special-case only
    # this one confirmed value, print that I did it, and then assert the
    # full 0..23 range afterwards so any unexpected value still fails loudly.
    n_2400 = int((df["scheduled_dep_time"] == 2400).sum())
    if n_2400:
        print(f"NOTE: {n_2400} row(s) have scheduled_dep_time == 2400 "
              f"(midnight, schedule-convention edge case) -- remapping to hour 0.")
        df.loc[df["scheduled_dep_time"] == 2400, "scheduled_dep_time"] = 0

    df["dep_hour"] = df["scheduled_dep_time"] // 100
    assert df["dep_hour"].between(0, 23).all(), (
        "dep_hour outside 0..23 -- scheduled_dep_time not in the expected HHMM range"
    )
    print("Derived dep_hour = scheduled_dep_time // 100; assert 0..23 passed.")

    # --- dow (day of week) ---
    # I derive this straight from flight_date rather than trusting any
    # pre-existing day-of-week column, so I know exactly how it was computed.
    # Stored as an ORDERED categorical (Mon..Sun) so that later modeling code
    # (e.g. a weekday effect in an OLS) treats it as a proper ordered factor
    # instead of an arbitrary/alphabetically-sorted set of strings.
    dow_raw = df["flight_date"].dt.day_name()
    df["dow"] = pd.Categorical(dow_raw, categories=DOW_ORDER, ordered=True)
    assert df["dow"].isna().sum() == 0, "day-name derivation produced unexpected values"
    print("Derived dow from flight_date as an ordered categorical (Mon..Sun).")

    # --- month ---
    # I derive month from flight_date and cross-check it against the month
    # column that already exists in the source file. This is a consistency
    # check on my ETL, not a new feature -- if the two disagree, something
    # upstream is wrong and I want to know immediately rather than silently
    # picking one.
    derived_month = df["flight_date"].dt.month.astype(df["month"].dtype)
    assert (derived_month == df["month"]).all(), (
        "flight_date-derived month does not match the existing month column"
    )
    print("Assert passed: existing month column matches flight_date-derived month.")
    # (df["month"] is kept as-is; derived_month was only for validation.)

    # --- holiday_flag ---
    # True if flight_date falls within +/- 3 days (inclusive) of any of the
    # seven major 2024 US holidays. This captures the elevated-traffic /
    # elevated-delay window around holidays (both the holiday itself and the
    # travel days immediately surrounding it), rather than only the single
    # calendar date.
    holiday_dates = pd.to_datetime(list(HOLIDAYS_2024.values()))
    window = pd.Timedelta(days=3)
    holiday_flag = pd.Series(False, index=df.index)
    for h in holiday_dates:
        holiday_flag |= df["flight_date"].between(h - window, h + window)
    df["holiday_flag"] = holiday_flag
    print(f"Derived holiday_flag: {int(holiday_flag.sum()):,} rows flagged "
          f"({holiday_flag.mean():.2%} of remaining rows).")

    print()
    print("=" * 70)
    print("SECTION 4: FINALIZE")
    print("=" * 70)
    df = df[KEEP_COLUMNS]

    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)

    print(f"Wrote {OUT_PATH}")
    print(f"Final shape: {df.shape}")
    print("Final dtypes:")
    print(df.dtypes)


if __name__ == "__main__":
    main()
