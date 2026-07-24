"""
split.py

Splits the model table into train/test using a TIME-based (not random) holdout,
and draws a seeded 1,000,000-row sample of the training set for fitting
statsmodels OLS models that would otherwise be too memory-heavy on the full
training set.

Run (with venv activated):
    python src/split.py

Reads:  data/model/model_table.parquet
Writes: data/model/train.parquet
        data/model/test.parquet
        data/model/train_sample_1m.parquet
"""

import os
import pandas as pd

IN_PATH = "data/model/model_table.parquet"
OUT_DIR = "data/model"
TRAIN_PATH = os.path.join(OUT_DIR, "train.parquet")
TEST_PATH = os.path.join(OUT_DIR, "test.parquet")
SAMPLE_PATH = os.path.join(OUT_DIR, "train_sample_1m.parquet")

SPLIT_DATE = pd.Timestamp("2024-10-01")
SAMPLE_SIZE = 1_000_000
SAMPLE_SEED = 42


def main():
    print("=" * 70)
    print("SECTION 1: LOAD")
    print("=" * 70)
    df = pd.read_parquet(IN_PATH)
    print(f"Rows loaded: {len(df):,}")

    print()
    print("=" * 70)
    print("SECTION 2: TIME-BASED TRAIN/TEST SPLIT")
    print("=" * 70)
    # The split is by CALENDAR DATE (train = before Oct 1 2024, test = Oct 1
    # 2024 onward), NOT a random row split. A random split would let the
    # model implicitly "see the future": rows from October sitting in the
    # training set would leak information (seasonal patterns, holiday travel,
    # weather regimes) about the exact period I'm trying to evaluate on.
    # A real deployment of this model would only ever have past data to train
    # on and would be asked to predict delays for flights that haven't
    # happened yet -- a temporal holdout is the only split that mirrors that
    # real-world constraint.
    train = df.loc[df["flight_date"] < SPLIT_DATE].copy()
    test = df.loc[df["flight_date"] >= SPLIT_DATE].copy()

    print(f"Train rows: {len(train):,}  "
          f"(flight_date {train['flight_date'].min().date()} to {train['flight_date'].max().date()})")
    print(f"Test rows:  {len(test):,}  "
          f"(flight_date {test['flight_date'].min().date()} to {test['flight_date'].max().date()})")

    # Verify the split boundary actually holds and there is no date overlap
    # between the two sets -- this should be true by construction, but I
    # assert it rather than assume it.
    assert train["flight_date"].max() < SPLIT_DATE, "train contains rows on/after the split date"
    assert test["flight_date"].min() >= SPLIT_DATE, "test contains rows before the split date"
    assert train["flight_date"].max() < test["flight_date"].min(), (
        "train and test flight_date ranges overlap"
    )
    assert len(train) + len(test) == len(df), "train + test row counts don't add up to the input"
    print("Assert passed: no date overlap between train and test, and row counts add up.")

    os.makedirs(OUT_DIR, exist_ok=True)
    train.to_parquet(TRAIN_PATH, index=False)
    test.to_parquet(TEST_PATH, index=False)
    print(f"Wrote {TRAIN_PATH}")
    print(f"Wrote {TEST_PATH}")

    print()
    print("=" * 70)
    print("SECTION 3: SEEDED 1M SAMPLE OF TRAIN")
    print("=" * 70)
    # I separately draw a fixed-size, fixed-seed random sample of 1,000,000
    # rows FROM the training set (this sampling is a modeling-convenience
    # step, not part of the train/test split logic above -- the temporal
    # boundary above is what prevents leakage; this sample just makes OLS
    # fitting tractable).
    #
    # Why sample at all: statsmodels' OLS builds a dense design matrix in
    # memory (not a sparse/streaming one). Once origin/dest are one-hot
    # encoded (hundreds of airports each), the full ~5-6 million-row training
    # set would require a dense matrix of tens of billions of cells --
    # prohibitive on a single machine. 1,000,000 rows keeps the design matrix
    # a manageable size while still being far more than enough data: standard
    # error shrinks with sqrt(n), so going from ~5M rows to 1M rows only
    # inflates standard errors by roughly sqrt(5) =~ 2.2x, which is negligible
    # for coefficients estimated from hundreds of thousands of observations
    # per airport/hour/day-of-week cell.
    #
    # Why seed=42: reproducibility. Re-running this script has to produce the
    # exact same sample every time (same rows in, same model results out),
    # which matters for debugging and for being able to explain/reproduce my
    # results later (e.g. in an interview).
    train_sample = train.sample(n=SAMPLE_SIZE, random_state=SAMPLE_SEED)
    train_sample.to_parquet(SAMPLE_PATH, index=False)
    print(f"Wrote {SAMPLE_PATH}")
    print(f"Sample rows: {len(train_sample):,} (seed={SAMPLE_SEED})")

    print()
    print("=" * 70)
    print("SECTION 4: is_delayed RATES (SANITY CHECK)")
    print("=" * 70)
    print(f"Train  is_delayed rate: {train['is_delayed'].mean():.4f}")
    print(f"Test   is_delayed rate: {test['is_delayed'].mean():.4f}")
    print(f"Sample is_delayed rate: {train_sample['is_delayed'].mean():.4f}")


if __name__ == "__main__":
    main()
