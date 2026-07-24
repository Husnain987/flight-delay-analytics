"""
01_eda.py

Exploratory data analysis for the flight-delay statistical modeling extension.

TRAIN-ONLY: every section below reads exclusively from data/model/train.parquet.
data/model/test.parquet is never opened in this script. The test set exists to
give an unbiased estimate of how a finished model performs on unseen data; if
I let choices made here (which features to bucket, which transform to apply,
how to handle outliers) be informed by looking at test-set patterns, I'd be
leaking test information into modeling decisions before the model is even
fit -- the resulting "held-out" evaluation would no longer be held out.

Run (with venv activated):
    python analysis/01_eda.py

Reads:  data/model/train.parquet
Writes: reports/figures/01_arr_delay_distribution.png
        reports/figures/02_log_transform_motivation.png
        reports/figures/03_delay_rate_by_hour.png
        reports/figures/04_delay_rate_by_airline.png
        reports/figures/05_delay_rate_by_month.png
"""

import os

import matplotlib
matplotlib.use("Agg")  # headless backend -- this script never opens a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TRAIN_PATH = "data/model/train.parquet"
FIG_DIR = "reports/figures"

# --- chart styling: keeps every figure in this file looking the same ---
COLOR_MAIN = "#2a78d6"     # sequential blue, mid step
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID_COLOR = "#e1e0d9"
AXIS_COLOR = "#c3c2b7"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS_COLOR,
    "axes.labelcolor": INK_SECONDARY,
    "axes.titlecolor": INK_PRIMARY,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "grid.color": GRID_COLOR,
    "text.color": INK_PRIMARY,
    "font.family": "sans-serif",
})


def section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def summarize_by(df, group_col, label, sort_by="index", ascending=True):
    """
    For a grouping column, compute: row count, mean(is_delayed), and
    mean(arr_delay) restricted to arr_delay > 0 (i.e. average lateness among
    flights that actually ran late -- mixing in early/on-time flights here
    would understate how bad a "late" flight typically is).

    This is the core table behind Section 3: it's what tells me which hours/
    days/months/airlines/airports actually move the delay rate, which in turn
    is what decides which categorical features are worth including (and how
    to bucket high-cardinality ones like origin/dest) in the downstream model.
    """
    grp = df.groupby(group_col, observed=True)
    n = grp.size()
    delay_rate = grp["is_delayed"].mean()

    late_only = df.loc[df["arr_delay"] > 0]
    mean_late_delay = late_only.groupby(group_col, observed=True)["arr_delay"].mean()

    table = pd.DataFrame({
        "n": n,
        "delay_rate": delay_rate,
        "mean_arr_delay_if_late": mean_late_delay,
    })

    if sort_by == "value":
        table = table.sort_values("delay_rate", ascending=ascending)
    elif sort_by == "volume":
        table = table.sort_values("n", ascending=ascending)
    else:
        table = table.sort_index()

    print(f"--- Delay rate by {label} (n={len(df):,} rows) ---")
    print(table.to_string(float_format=lambda x: f"{x:,.3f}"))
    print()
    return table


def bar_chart(table, title, xlabel, ylabel, path, rotate=0):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(table.index.astype(str), table["delay_rate"], color=COLOR_MAIN, width=0.7)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", linewidth=0.8)
    ax.set_axisbelow(True)
    if rotate:
        plt.setp(ax.get_xticklabels(), rotation=rotate, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    df = pd.read_parquet(TRAIN_PATH)

    # ------------------------------------------------------------------
    section("SECTION 1: SCHEMA AND COVERAGE")
    # ------------------------------------------------------------------
    # First checkpoint before any analysis: confirm the file I'm about to
    # explore actually matches what the prepare/split steps promised.
    print(f"Shape: {df.shape}")
    print("Dtypes:")
    print(df.dtypes)
    print(f"Date range: {df['flight_date'].min().date()} to {df['flight_date'].max().date()}")
    print(f"Unique airlines: {df['airline'].nunique()}")
    print(f"Unique origin airports: {df['origin'].nunique()}")
    print(f"Unique dest airports: {df['dest'].nunique()}")

    # ------------------------------------------------------------------
    section("SECTION 2: TARGET DISTRIBUTION (arr_delay)")
    # ------------------------------------------------------------------
    # arr_delay is the continuous target for an OLS model; is_delayed (>=15
    # min) is the binary target for a classification model. Understanding
    # the shape of arr_delay -- how skewed it is, how much mass sits below
    # zero -- directly informs whether OLS on the raw value is even
    # appropriate, or whether a transform / a different population (e.g.
    # "model lateness only among late flights") is needed.
    print("describe():")
    print(df["arr_delay"].describe())

    pct_levels = [1, 5, 25, 50, 75, 95, 99, 99.9]
    pcts = df["arr_delay"].quantile([p / 100 for p in pct_levels])
    print()
    print("Percentiles:")
    for p, v in zip(pct_levels, pcts):
        print(f"  p{p:>5}: {v:>10.2f} min")

    n = len(df)
    pct_early = (df["arr_delay"] < 0).mean() * 100
    pct_ontime = (df["arr_delay"] == 0).mean() * 100
    pct_late = (df["arr_delay"] > 0).mean() * 100
    pct_delayed15 = df["is_delayed"].mean() * 100
    print()
    print(f"% early   (arr_delay <  0): {pct_early:6.2f}%")
    print(f"% on time (arr_delay == 0): {pct_ontime:6.2f}%")
    print(f"% late    (arr_delay >  0): {pct_late:6.2f}%")
    print(f"% delayed 15+ (is_delayed): {pct_delayed15:6.2f}%")

    # Two views of the same variable, same color (it's one entity): the raw
    # distribution (to see the full tail of extreme delays) and a zoomed
    # view over the range where almost all of the mass actually sits, which
    # is the range that matters for everyday model interpretation.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(df["arr_delay"], bins=100, color=COLOR_MAIN)
    axes[0].set_title("arr_delay -- full range")
    axes[0].set_xlabel("arr_delay (minutes)")
    axes[0].set_ylabel("flights")
    axes[0].grid(axis="y", linewidth=0.8)
    axes[0].set_axisbelow(True)

    zoomed = df.loc[df["arr_delay"].between(-60, 180), "arr_delay"]
    axes[1].hist(zoomed, bins=48, color=COLOR_MAIN)
    axes[1].set_title("arr_delay -- zoomed to [-60, 180] min")
    axes[1].set_xlabel("arr_delay (minutes)")
    axes[1].set_ylabel("flights")
    axes[1].grid(axis="y", linewidth=0.8)
    axes[1].set_axisbelow(True)

    fig.tight_layout()
    fig_path = os.path.join(FIG_DIR, "01_arr_delay_distribution.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Saved {fig_path}")

    # The OLS-on-lateness model (predicting how late a flight runs, given
    # that it runs late) is fit only on the arr_delay > 0 population -- early
    # and on-time flights aren't "less late," they're a qualitatively
    # different outcome (nothing to explain). This count is exactly the
    # training population size for that model.
    n_positive_delay = int((df["arr_delay"] > 0).sum())
    print()
    print(f"Rows with arr_delay > 0 (OLS modeling population): {n_positive_delay:,}")

    # arr_delay > 0 is heavily right-skewed (most late flights are only a
    # few minutes late, a few are hours late) -- OLS assumes roughly
    # normal, homoscedastic residuals, which a raw right-skewed target
    # violates badly. log(arr_delay) compresses the long right tail and is
    # the standard fix; this figure is the evidence for making that call.
    positive = df.loc[df["arr_delay"] > 0, "arr_delay"]
    log_positive = np.log(positive)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(positive, bins=100, color=COLOR_MAIN)
    axes[0].set_title("Raw arr_delay (arr_delay > 0 subset)")
    axes[0].set_xlabel("arr_delay (minutes)")
    axes[0].set_ylabel("flights")
    axes[0].grid(axis="y", linewidth=0.8)
    axes[0].set_axisbelow(True)

    axes[1].hist(log_positive, bins=100, color=COLOR_MAIN)
    axes[1].set_title("log(arr_delay) (arr_delay > 0 subset)")
    axes[1].set_xlabel("log(arr_delay)")
    axes[1].set_ylabel("flights")
    axes[1].grid(axis="y", linewidth=0.8)
    axes[1].set_axisbelow(True)

    fig.tight_layout()
    fig_path = os.path.join(FIG_DIR, "02_log_transform_motivation.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Saved {fig_path}")

    # ------------------------------------------------------------------
    section("SECTION 3: DELAY RATE BY FEATURE")
    # ------------------------------------------------------------------
    # For each candidate feature, I want two numbers side by side: how often
    # flights in this bucket are delayed at all (delay_rate), and, among the
    # ones that are late, how late (mean_arr_delay_if_late). A feature can
    # move one of these without moving the other (e.g. a route that's rarely
    # late but brutal when it is) -- both matter for feature selection.
    dep_hour_table = summarize_by(df, "dep_hour", "dep_hour")
    dow_table = summarize_by(df, "dow", "dow")
    month_table = summarize_by(df, "month", "month")
    airline_table = summarize_by(df, "airline", "airline", sort_by="value", ascending=False)
    holiday_table = summarize_by(df, "holiday_flag", "holiday_flag")

    top15_origin = df["origin"].value_counts().head(15).index
    origin_top15_table = summarize_by(
        df.loc[df["origin"].isin(top15_origin)], "origin", "origin (top 15 by volume)",
        sort_by="volume", ascending=False,
    )

    bar_chart(
        dep_hour_table, "Delay rate by scheduled departure hour",
        "dep_hour (0-23)", "P(is_delayed)",
        os.path.join(FIG_DIR, "03_delay_rate_by_hour.png"),
    )
    bar_chart(
        airline_table, "Delay rate by airline",
        "airline", "P(is_delayed)",
        os.path.join(FIG_DIR, "04_delay_rate_by_airline.png"),
        rotate=45,
    )
    bar_chart(
        month_table, "Delay rate by month",
        "month (1-12)", "P(is_delayed)",
        os.path.join(FIG_DIR, "05_delay_rate_by_month.png"),
    )

    # ------------------------------------------------------------------
    section("SECTION 4: AIRPORT COVERAGE")
    # ------------------------------------------------------------------
    # origin/dest have hundreds of distinct airports; one-hot-encoding all of
    # them blows up the design matrix for very little benefit, since traffic
    # is extremely concentrated in a handful of hubs. This section quantifies
    # exactly how concentrated it is, to justify keeping only the top-K
    # airports as their own dummy variables and folding the long tail into a
    # single "OTHER" bucket.
    for col, label in [("origin", "origin"), ("dest", "dest")]:
        counts = df[col].value_counts()
        total = len(df)
        print(f"--- {label} cumulative coverage (unique {label} = {counts.shape[0]}) ---")
        for top_n in [10, 20, 30, 50, 100]:
            covered = int(counts.iloc[:top_n].sum())
            pct = covered / total * 100
            print(f"  top {top_n:>3} {label}: {pct:6.2f}% of flights ({covered:,} / {total:,})")
        print()

    # ------------------------------------------------------------------
    section("SECTION 5: DISTANCE")
    # ------------------------------------------------------------------
    # Longer flights have more time in the air to make up a delay (or more
    # opportunity to accumulate one), so distance is a plausible predictor.
    # I check both a linear correlation and a coarser quintile breakdown,
    # since a weak Pearson correlation can still hide a non-linear pattern
    # (e.g. very short and very long flights both worse than the middle).
    corr_arr_delay = df["distance"].corr(df["arr_delay"])
    corr_is_delayed = df["distance"].corr(df["is_delayed"].astype(int))
    print(f"corr(distance, arr_delay):  {corr_arr_delay:.4f}")
    print(f"corr(distance, is_delayed): {corr_is_delayed:.4f}")
    print("(Expected weak -- distance is not a strong direct driver of delay.)")

    quintiles = pd.qcut(df["distance"], 5, labels=["Q1 (shortest)", "Q2", "Q3", "Q4", "Q5 (longest)"])
    quintile_table = df.groupby(quintiles, observed=True).agg(
        n=("is_delayed", "size"),
        delay_rate=("is_delayed", "mean"),
        mean_arr_delay=("arr_delay", "mean"),
        distance_min=("distance", "min"),
        distance_max=("distance", "max"),
    )
    print()
    print("Delay rate by distance quintile:")
    print(quintile_table.to_string(float_format=lambda x: f"{x:,.3f}"))

    # ------------------------------------------------------------------
    section("SECTION 6: CLASS BALANCE SUMMARY")
    # ------------------------------------------------------------------
    # For a binary classifier on is_delayed, the positive rate IS the
    # no-skill baseline for precision-recall AUC (a classifier that predicts
    # "delayed" completely at random, at the base rate, scores a PR-AUC
    # equal to the positive rate). Any model I build has to clear this
    # number to be worth anything -- ROC-AUC's no-skill baseline is always
    # 0.5 regardless of class balance, which is why PR-AUC is the more
    # informative metric on an imbalanced target like this one.
    rate = df["is_delayed"].mean()
    n_pos = int(df["is_delayed"].sum())
    n_neg = len(df) - n_pos
    print(f"is_delayed rate:         {rate:.4f}")
    print(f"positive (delayed) count: {n_pos:,}")
    print(f"negative (on-time) count: {n_neg:,}")
    print(f"No-skill PR-AUC baseline (= positive rate): {rate:.4f}")


if __name__ == "__main__":
    main()
