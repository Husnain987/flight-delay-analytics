"""
03_logistic_delay_risk.py

Question this script answers: AT BOOKING TIME, can I flag a flight as likely
to arrive 15+ minutes late (is_delayed)?

"At booking time" is the framing that decides every feature choice below. A
person buying a ticket knows: which airline, which airport pair, what time of
day and day of week it departs, whether it's near a holiday, and how far it
is. They do NOT know anything that only becomes true after the schedule is
built or the flight actually happens.

ALLOWED FEATURES (all knowable at booking time):
    airline, origin, dest, dep_hour, dow, holiday_flag, distance

EXCLUDED, AND WHY (leakage relative to the booking-time framing):
    - month            : the train/test split is temporal (train = months
                          1-9, test = months 10-12, ZERO overlap). A month
                          dummy fit on train is undefined for any test row --
                          this isn't a booking-time argument, it's simply
                          impossible to use correctly across this split.
    - arr_delay         : this is the raw quantity is_delayed is thresholded
                          from (arr_delay >= 15). Using it as a feature would
                          be using the answer to predict the answer.
    - is_delayed itself : the target.
    - anything from the "actuals" family (dep_delay, actual_dep_time,
      actual_arr_time, air_time, the *_delay reason columns, cancellation
      info) : these were already excluded from data/model/model_table.parquet
      by src/prepare.py because they only exist after a flight has departed
      or landed. They are not booking-time knowable regardless of the
      train/test split, so they're excluded here for the same reason even
      though they aren't physically present in this file.

Run (with venv activated):
    python analysis/03_logistic_delay_risk.py

Reads:  data/model/train.parquet        (full train, sklearn model)
        data/model/train_sample_1m.parquet (statsmodels inference model)
        data/model/test.parquet          (evaluation only)
Writes: reports/figures/09_calibration_curve.png
        reports/figures/09b_calibration_before_after_recalibration.png
        reports/figures/10_cost_threshold.png
"""

import os

import matplotlib
matplotlib.use("Agg")  # headless backend -- never opens a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

TRAIN_PATH = "data/model/train.parquet"
TRAIN_SAMPLE_PATH = "data/model/train_sample_1m.parquet"
TEST_PATH = "data/model/test.parquet"
FIG_DIR = "reports/figures"

TOP_N_AIRPORTS = 50
REF_AIRLINE = "WN"     # explicit reference: highest-volume carrier
REF_AIRPORT = "ATL"    # explicit reference: highest-volume airport

FALSE_ALARM_COST = 2.0     # $ cost of proactively notifying a passenger who was NOT actually delayed
MISSED_DELAY_COST = 25.0   # $ cost of failing to flag a flight that WAS actually delayed

FEATURE_COLS = ["dep_hour_bin", "dow", "airline", "origin_top50", "dest_top50",
                 "log_distance", "holiday_flag"]

# --- chart styling: same look as my other analysis scripts ---
COLOR_MAIN = "#2a78d6"
COLOR_ALT = "#eb6834"   # second series in a shared figure (e.g. before/after)
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


def bin_dep_hour(dep_hour):
    """
    Same bucketing I used in 02_linear_delay_magnitude.py: hours 0-4 are too
    sparse individually for a stable coefficient, so I pool them into "red_eye".
    """
    bins = [-1, 4, 7, 11, 15, 19, 23]
    labels = ["red_eye", "early_morning", "morning", "midday", "evening", "night"]
    return pd.cut(dep_hour, bins=bins, labels=labels).astype(str)


def engineer_features(df, origin_top_set, dest_top_set):
    df = df.copy()
    df["dep_hour_bin"] = bin_dep_hour(df["dep_hour"])
    df["dow"] = df["dow"].astype(str)
    df["airline"] = df["airline"].astype(str)
    # origin_top50/dest_top50: the top-N set was computed from TRAIN volume
    # only (see SECTION 2) and is applied unchanged here to whatever
    # DataFrame is passed in (train, train_sample, or test) -- test airport
    # volume never influences which airports get their own dummy.
    df["origin_top50"] = np.where(df["origin"].isin(origin_top_set), df["origin"], "OTHER")
    df["dest_top50"] = np.where(df["dest"].isin(dest_top_set), df["dest"], "OTHER")
    df["log_distance"] = np.log(df["distance"])
    return df


def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    section("SECTION 1: LOAD")
    # ------------------------------------------------------------------
    train = pd.read_parquet(TRAIN_PATH)
    train_sample = pd.read_parquet(TRAIN_SAMPLE_PATH)
    test = pd.read_parquet(TEST_PATH)
    print(f"train (full):      {len(train):,} rows")
    print(f"train_sample_1m:   {len(train_sample):,} rows")
    print(f"test:              {len(test):,} rows")

    assert train["flight_date"].min() >= pd.Timestamp("2024-01-01")
    assert train["flight_date"].max() < pd.Timestamp("2024-10-01")
    assert test["flight_date"].min() >= pd.Timestamp("2024-10-01")

    # ------------------------------------------------------------------
    section("SECTION 2: FEATURE ENGINEERING (fit on TRAIN, applied to test)")
    # ------------------------------------------------------------------
    origin_top_set = set(train["origin"].value_counts().head(TOP_N_AIRPORTS).index)
    dest_top_set = set(train["dest"].value_counts().head(TOP_N_AIRPORTS).index)
    assert REF_AIRPORT in origin_top_set, f"{REF_AIRPORT} not in origin top-{TOP_N_AIRPORTS}"
    assert REF_AIRPORT in dest_top_set, f"{REF_AIRPORT} not in dest top-{TOP_N_AIRPORTS}"
    assert REF_AIRLINE in train["airline"].unique(), f"{REF_AIRLINE} not in train airlines"

    train = engineer_features(train, origin_top_set, dest_top_set)
    train_sample = engineer_features(train_sample, origin_top_set, dest_top_set)
    test = engineer_features(test, origin_top_set, dest_top_set)

    for label, df_ in [("train", train), ("test", test)]:
        for col in ["origin_top50", "dest_top50"]:
            other_pct = (df_[col] == "OTHER").mean() * 100
            print(f"{col} OTHER share in {label}: {other_pct:.2f}%")

    # Explicit reference levels: I put the reference value FIRST in each
    # categories list, then tell OneHotEncoder(drop='first') to drop the
    # first category for each column. I do this explicitly rather than
    # leaving it to the library default (which would drop alphabetically-
    # first -- I hit exactly this trap in 02_linear_delay_magnitude.py,
    # where 'ATL' silently became the reference by alphabetical accident
    # rather than by volume).
    dep_hour_bin_categories = ["early_morning", "red_eye", "morning", "midday", "evening", "night"]
    dow_categories = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    airline_counts = train["airline"].value_counts()
    airline_categories = [REF_AIRLINE] + [a for a in airline_counts.index if a != REF_AIRLINE]

    origin_categories = [REF_AIRPORT] + sorted(origin_top_set - {REF_AIRPORT}) + ["OTHER"]
    dest_categories = [REF_AIRPORT] + sorted(dest_top_set - {REF_AIRPORT}) + ["OTHER"]

    print(f"Reference levels -- dep_hour_bin: '{dep_hour_bin_categories[0]}', "
          f"dow: '{dow_categories[0]}', airline: '{REF_AIRLINE}', "
          f"origin_top50: '{REF_AIRPORT}', dest_top50: '{REF_AIRPORT}'")

    ohe_categories = [dep_hour_bin_categories, dow_categories, airline_categories,
                       origin_categories, dest_categories]

    # ------------------------------------------------------------------
    section("SECTION 3: BASELINES (before any model)")
    # ------------------------------------------------------------------
    test_positive_rate = test["is_delayed"].mean()
    print(f"Test positive rate (no-skill PR-AUC floor): {test_positive_rate:.4f}")

    # B0: majority-class classifier. This is the "accuracy looks great, model
    # is useless" cautionary baseline -- with an imbalanced target, always
    # predicting the majority class racks up high accuracy for zero
    # discrimination ability, which is exactly why accuracy is the wrong
    # headline metric here.
    b0_pred = np.zeros(len(test), dtype=bool)
    b0_accuracy = accuracy_score(test["is_delayed"], b0_pred)
    print(f"B0 (always predict 'not delayed') accuracy: {b0_accuracy:.4f} "
          f"-- high accuracy, ZERO ability to identify a single delayed flight.")

    # B1: single-rule classifier straight from the EDA (delay rate rises
    # ~monotonically through the afternoon/evening). Sets a floor any
    # trained model needs to clear to justify its complexity.
    b1_pred = (test["dep_hour"] >= 16).values
    b1_precision = precision_score(test["is_delayed"], b1_pred)
    b1_recall = recall_score(test["is_delayed"], b1_pred)
    b1_f1 = f1_score(test["is_delayed"], b1_pred)
    print(f"B1 (flag if dep_hour >= 16) on test: "
          f"precision={b1_precision:.4f}, recall={b1_recall:.4f}, f1={b1_f1:.4f}")

    # ------------------------------------------------------------------
    section("SECTION 4: SKLEARN LOGISTIC REGRESSION (deployable model)")
    # ------------------------------------------------------------------
    # I fit two separate models in this script, on purpose:
    #   - sklearn, fit on FIT (train up to Aug 31): this is the model
    #     that would actually be deployed to score flights at scale. sklearn
    #     handles a sparse one-hot design over millions of rows comfortably,
    #     which is exactly the situation I avoided in
    #     02_linear_delay_magnitude.py by sampling down to 1M rows for
    #     statsmodels.
    #   - statsmodels, fit on the 1M-row sample (SECTION 5 below): this is
    #     purely for the coefficient table -- confidence intervals and
    #     p-values that sklearn's LogisticRegression does not compute at all.
    #     I don't need the inference machinery to be exact at this scale;
    #     1M is already the documented "negligible SE inflation" compromise
    #     from split.py.
    #
    # I carve FIT/VALIDATION out of train here, before fitting anything:
    #   - FIT   = flight_date < Sep 1 2024 -- what the model actually trains on.
    #   - VALIDATION = Sep 1-30 2024 -- held out from FIT, used in SECTION 8/8B
    #     to pick the cost-based decision threshold.
    # I need this split because the threshold-selection step (SECTION 8) has
    # to be scored with genuinely unseen predictions. If I fit on the full
    # train set and then swept thresholds on a Sep slice the model had
    # already seen during training, the model's predictions on that slice
    # would look better than they really are -- the same reason a training-
    # set accuracy number always looks rosier than a held-out one. Keeping
    # VALIDATION out of FIT is what makes the September cost numbers honest.
    fit_cutoff = pd.Timestamp("2024-09-01")
    val_end = pd.Timestamp("2024-09-30")
    fit_mask = train["flight_date"] < fit_cutoff
    val_mask = (train["flight_date"] >= fit_cutoff) & (train["flight_date"] <= val_end)
    assert (fit_mask | val_mask).all(), "some train row falls outside both FIT and VALIDATION"
    assert not (fit_mask & val_mask).any(), "FIT and VALIDATION overlap"
    fit_df = train.loc[fit_mask].copy()
    val_df = train.loc[val_mask].copy()
    print(f"FIT rows (flight_date < {fit_cutoff.date()}): {len(fit_df):,}")
    print(f"FIT is_delayed rate: {fit_df['is_delayed'].mean():.4f}")
    print(f"VALIDATION rows (Sep 1-30): {len(val_df):,}")
    print(f"VALIDATION is_delayed rate: {val_df['is_delayed'].mean():.4f}")

    preprocessor = ColumnTransformer(transformers=[
        ("cat", OneHotEncoder(categories=ohe_categories, drop="first", handle_unknown="ignore"),
         ["dep_hour_bin", "dow", "airline", "origin_top50", "dest_top50"]),
        ("num", "passthrough", ["log_distance", "holiday_flag"]),
    ])

    logreg = LogisticRegression(max_iter=1000, solver="lbfgs")
    pipeline = Pipeline([("preprocess", preprocessor), ("logreg", logreg)])
    pipeline.fit(fit_df[FEATURE_COLS], fit_df["is_delayed"])

    n_iter = int(np.max(logreg.n_iter_))
    converged = n_iter < logreg.max_iter
    print(f"sklearn LogisticRegression (solver=lbfgs): n_iter={n_iter}, converged={converged}")
    if not converged:
        print("lbfgs did not converge within max_iter -- refitting with solver='saga'.")
        logreg = LogisticRegression(max_iter=1000, solver="saga")
        pipeline = Pipeline([("preprocess", preprocessor), ("logreg", logreg)])
        pipeline.fit(fit_df[FEATURE_COLS], fit_df["is_delayed"])
        n_iter = int(np.max(logreg.n_iter_))
        converged = n_iter < logreg.max_iter
        print(f"sklearn LogisticRegression (solver=saga): n_iter={n_iter}, converged={converged}")

    y_prob_test = pipeline.predict_proba(test[FEATURE_COLS])[:, 1]

    # ------------------------------------------------------------------
    section("SECTION 5: STATSMODELS LOGIT (inference / confidence intervals)")
    # ------------------------------------------------------------------
    train_sample["dep_hour_bin"] = pd.Categorical(train_sample["dep_hour_bin"], categories=dep_hour_bin_categories)
    train_sample["dow"] = pd.Categorical(train_sample["dow"], categories=dow_categories)
    train_sample["airline"] = pd.Categorical(train_sample["airline"], categories=airline_categories)
    train_sample["origin_top50"] = pd.Categorical(train_sample["origin_top50"], categories=origin_categories)
    train_sample["dest_top50"] = pd.Categorical(train_sample["dest_top50"], categories=dest_categories)
    # patsy encodes a bool endog as a two-column dummy rather than a single
    # 0/1 target -- cast to int explicitly so smf.logit gets a plain numeric
    # response.
    train_sample["is_delayed"] = train_sample["is_delayed"].astype(int)

    logit_formula = (
        "is_delayed ~ C(dep_hour_bin) + C(dow) + holiday_flag "
        "+ C(airline) + C(origin_top50) + C(dest_top50) + log_distance"
    )
    logit_model = smf.logit(logit_formula, data=train_sample).fit(disp=0, maxiter=100)
    print(f"statsmodels Logit formula: {logit_formula}")
    print(f"Converged: {logit_model.mle_retvals['converged']}, "
          f"iterations: {logit_model.mle_retvals['iterations']}")
    print(f"n: {int(logit_model.nobs):,}, pseudo R-squared: {logit_model.prsquared:.4f}")

    # ------------------------------------------------------------------
    section("SECTION 6: DISCRIMINATION -- ROC-AUC and PR-AUC")
    # ------------------------------------------------------------------
    # PR-AUC leads because the target is imbalanced (test positive rate
    # 16.31%) and ROC-AUC's no-skill baseline is always 0.5 regardless of
    # class balance -- it can look deceptively good on an imbalanced problem.
    # PR-AUC's no-skill baseline IS the positive rate, so it's directly
    # comparable to how hard this specific problem is.
    roc_auc = roc_auc_score(test["is_delayed"], y_prob_test)
    pr_auc = average_precision_score(test["is_delayed"], y_prob_test)
    print(f"PR-AUC:  {pr_auc:.4f}  (no-skill floor = {test_positive_rate:.4f}, "
          f"lift = {pr_auc / test_positive_rate:.2f}x)")
    print(f"ROC-AUC: {roc_auc:.4f}")

    # ------------------------------------------------------------------
    section("SECTION 7: CALIBRATION")
    # ------------------------------------------------------------------
    # This is the section that matters most given the verified base-rate
    # shift (train 22.35% vs test 16.31% is_delayed). A model trained on
    # train's base rate will encode that base rate into its intercept, and
    # will systematically OVER-predict probability of delay on test, where
    # delays were genuinely less common. Discrimination (which flights rank
    # higher-risk than others) can be fine even while the absolute
    # probabilities are miscalibrated -- these are different failure modes,
    # and only calibration diagnostics catch the second one.
    prob_true, prob_pred = calibration_curve(test["is_delayed"], y_prob_test, n_bins=10)
    brier_before_all = brier_score_loss(test["is_delayed"], y_prob_test)
    mean_pred_prob = y_prob_test.mean()
    print(f"Brier score (full test, pre-recalibration): {brier_before_all:.5f}")
    print(f"Mean predicted probability (full test): {mean_pred_prob:.4f}")
    print(f"Actual test positive rate:               {test_positive_rate:.4f}")
    print(f"Gap (mean predicted - actual):            {mean_pred_prob - test_positive_rate:+.4f} "
          f"-- this gap is the base-rate shift showing up as miscalibration.")

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], color=INK_MUTED, linewidth=1.5, linestyle="--", label="perfect calibration")
    ax.plot(prob_pred, prob_true, color=COLOR_MAIN, marker="o", markersize=5, linewidth=2,
            label="model (full test)")
    ax.set_title("Calibration curve: predicted vs. observed delay rate (test, 10 bins)")
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed is_delayed rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    ax.grid(linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "09_calibration_curve.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")

    # Intercept-only recalibration: freeze every trained coefficient, and
    # re-solve ONLY the intercept using a small slice of genuinely fresh
    # data (the first 2 weeks of test). This simulates the realistic
    # deployment scenario -- you don't get to retrain on the future, but you
    # typically DO get a couple of weeks of live outcomes before you need the
    # model to be trustworthy for the rest of the period, and re-fitting a
    # single intercept from that trickle of fresh data is cheap and doesn't
    # require touching the feature coefficients at all.
    oct_start = pd.Timestamp("2024-10-01")
    recal_end = pd.Timestamp("2024-10-14")
    remaining_start = pd.Timestamp("2024-10-15")
    recal_mask = (test["flight_date"] >= oct_start) & (test["flight_date"] <= recal_end)
    remaining_mask = test["flight_date"] >= remaining_start
    assert (recal_mask | remaining_mask).all(), "some test row falls outside both recalibration slices"
    assert not (recal_mask & remaining_mask).any(), "recalibration slices overlap"
    print(f"Recalibration slice (Oct 1-14): {recal_mask.sum():,} rows")
    print(f"Remaining test slice (Oct 15+): {remaining_mask.sum():,} rows")

    raw_logodds_test = pipeline.decision_function(test[FEATURE_COLS])
    original_intercept = logreg.intercept_[0]
    offset_test = raw_logodds_test - original_intercept  # feature contribution only, no intercept

    offset_recal = offset_test[recal_mask.values]
    y_recal = test.loc[recal_mask, "is_delayed"].astype(int).values
    recal_fit = sm.Logit(y_recal, np.ones((len(y_recal), 1)), offset=offset_recal).fit(disp=0)
    new_intercept = recal_fit.params[0]
    print(f"Original intercept: {original_intercept:.4f}  ->  recalibrated intercept: {new_intercept:.4f}")

    offset_remaining = offset_test[remaining_mask.values]
    y_remaining = test.loc[remaining_mask, "is_delayed"].astype(int).values
    prob_remaining_before = 1 / (1 + np.exp(-(offset_remaining + original_intercept)))
    prob_remaining_after = 1 / (1 + np.exp(-(offset_remaining + new_intercept)))

    brier_remaining_before = brier_score_loss(y_remaining, prob_remaining_before)
    brier_remaining_after = brier_score_loss(y_remaining, prob_remaining_after)
    print(f"Brier score (Oct 15+ only) BEFORE recalibration: {brier_remaining_before:.5f}")
    print(f"Brier score (Oct 15+ only) AFTER  recalibration: {brier_remaining_after:.5f}")
    improved = brier_remaining_after < brier_remaining_before
    print(f"Recalibration {'IMPROVED' if improved else 'DID NOT improve'} Brier score on held-out remainder.")

    prob_true_before, prob_pred_before = calibration_curve(y_remaining, prob_remaining_before, n_bins=10)
    prob_true_after, prob_pred_after = calibration_curve(y_remaining, prob_remaining_after, n_bins=10)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], color=INK_MUTED, linewidth=1.5, linestyle="--", label="perfect calibration")
    ax.plot(prob_pred_before, prob_true_before, color=COLOR_MAIN, marker="o", markersize=5,
            linewidth=2, label="before recalibration")
    ax.plot(prob_pred_after, prob_true_after, color=COLOR_ALT, marker="s", markersize=5,
            linewidth=2, label="after recalibration")
    ax.set_title("Calibration before/after intercept recalibration (test, Oct 15 onward)")
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed is_delayed rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    ax.grid(linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "09b_calibration_before_after_recalibration.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")

    # ------------------------------------------------------------------
    section("SECTION 8: COST-BASED DECISION THRESHOLD")
    # ------------------------------------------------------------------
    # 0.5 is not a decision threshold grounded in anything -- it's just the
    # point where the model itself is indifferent. The actual operational
    # question is "at what predicted probability does proactively notifying
    # a passenger become worth it?", which depends on the asymmetric costs
    # of the two error types (a false alarm costs $2; a missed real delay
    # costs $25), not on model probability alone.
    #
    # I choose the threshold on VALIDATION (Sep 1-30) rather than on test,
    # because test exists to give an unbiased read of how the FINAL,
    # already-chosen configuration performs -- if I tuned the threshold by
    # looking at test, the test-set cost number would no longer be a fair
    # evaluation. VALIDATION is also genuinely unseen by the model itself
    # (SECTION 4 fits on FIT = flight_date < Sep 1, and holds VALIDATION out
    # of that fit) -- if I'd fit on the full train set and then swept
    # thresholds on a Sep slice the model already trained on, the cost
    # numbers below would be optimistic for the same reason a training-set
    # accuracy number always looks better than a held-out one.
    print(f"Using VALIDATION slice (Sep 1-30, held out of FIT): {len(val_df):,} rows")

    sep_prob = pipeline.predict_proba(val_df[FEATURE_COLS])[:, 1]
    sep_actual = val_df["is_delayed"].values

    thresholds = np.round(np.arange(0.05, 0.951, 0.01), 2)
    costs_per_1000 = []
    for t in thresholds:
        pred = sep_prob >= t
        fp = np.sum(pred & ~sep_actual)
        fn = np.sum((~pred) & sep_actual)
        cost = (fp * FALSE_ALARM_COST + fn * MISSED_DELAY_COST) / len(sep_actual) * 1000
        costs_per_1000.append(cost)
    costs_per_1000 = np.array(costs_per_1000)

    best_idx = int(np.argmin(costs_per_1000))
    best_threshold = thresholds[best_idx]
    best_cost = costs_per_1000[best_idx]
    print(f"Chosen threshold (argmin expected cost, Sep validation slice): {best_threshold:.2f}")
    print(f"Expected cost per 1000 flights at chosen threshold (Sep slice): ${best_cost:.2f}")

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(thresholds, costs_per_1000, color=COLOR_MAIN, linewidth=2)
    ax.axvline(best_threshold, color=INK_MUTED, linewidth=1.5, linestyle="--")
    ax.scatter([best_threshold], [best_cost], color=COLOR_ALT, s=60, zorder=5,
               label=f"chosen threshold = {best_threshold:.2f}")
    ax.set_title("Expected cost per 1000 flights vs. decision threshold\n"
                 "(validation slice: train, Sep 1-30 2024)")
    ax.set_xlabel("probability threshold")
    ax.set_ylabel("expected cost per 1000 flights ($)")
    ax.legend(frameon=False)
    ax.grid(linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "10_cost_threshold.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")

    def threshold_report(threshold, label):
        pred = y_prob_test >= threshold
        precision = precision_score(test["is_delayed"], pred, zero_division=0)
        recall = recall_score(test["is_delayed"], pred, zero_division=0)
        f1 = f1_score(test["is_delayed"], pred, zero_division=0)
        fp = np.sum(pred & ~test["is_delayed"].values)
        fn = np.sum((~pred) & test["is_delayed"].values)
        cost = (fp * FALSE_ALARM_COST + fn * MISSED_DELAY_COST) / len(test) * 1000
        print(f"[{label}] threshold={threshold:.2f}: precision={precision:.4f}, "
              f"recall={recall:.4f}, f1={f1:.4f}, cost/1000={cost:.2f}")
        return cost

    print()
    print("Evaluated on TEST:")
    cost_chosen = threshold_report(best_threshold, "cost-optimal")
    cost_default = threshold_report(0.5, "default 0.5")
    print(f"Cost-optimal threshold saves ${cost_default - cost_chosen:.2f} per 1000 flights "
          f"vs. the default 0.5 threshold, on test.")

    # ------------------------------------------------------------------
    section("SECTION 8B: THRESHOLD SENSITIVITY TO THE COST RATIO")
    # ------------------------------------------------------------------
    # SECTION 8 picked threshold 0.10 at one specific cost pair ($2 false
    # alarm / $25 miss). That threshold flags 91.8% of test flights -- close
    # enough to a trivial "notify every passenger" policy that I want to
    # check how much of that decision is coming from the MODEL versus from
    # the cost assumptions themselves. So I re-run the identical threshold-
    # selection procedure (same Sep 1-30 train validation slice, same sweep)
    # across several plausible cost pairs, so the chosen threshold and
    # flagged share can be read as a function of the cost ratio rather than
    # of one arbitrarily chosen pair.
    #
    # Note: the argmin threshold is mathematically a function of the RATIO
    # cost_fn/cost_fp only (minimizing cost_fp*FP + cost_fn*FN over threshold
    # is the same problem as minimizing FP + ratio*FN) -- so two pairs with
    # the same ratio, e.g. (5,10) and (2,4), will pick the identical
    # threshold even though their absolute dollar costs differ. I include
    # both below specifically to make that visible rather than assumed.
    cost_scenarios = [(2, 25), (2, 10), (2, 6), (5, 10), (10, 25), (2, 4)]

    sensitivity_rows = []
    for cost_fp, cost_fn in cost_scenarios:
        # Re-cost the SAME Sep 1-30 sweep (sep_prob/sep_actual already
        # computed above) rather than re-scoring the validation slice for
        # every scenario.
        scenario_costs = []
        for t in thresholds:
            pred = sep_prob >= t
            fp = np.sum(pred & ~sep_actual)
            fn = np.sum((~pred) & sep_actual)
            cost = (fp * cost_fp + fn * cost_fn) / len(sep_actual) * 1000
            scenario_costs.append(cost)
        scenario_threshold = thresholds[int(np.argmin(scenario_costs))]

        # Evaluate the chosen threshold on TEST (never on the slice it was
        # chosen from), same discipline as SECTION 8.
        pred_test = y_prob_test >= scenario_threshold
        precision = precision_score(test["is_delayed"], pred_test, zero_division=0)
        recall = recall_score(test["is_delayed"], pred_test, zero_division=0)
        flagged_share = pred_test.mean()
        fp_test = np.sum(pred_test & ~test["is_delayed"].values)
        fn_test = np.sum((~pred_test) & test["is_delayed"].values)
        model_cost = (fp_test * cost_fp + fn_test * cost_fn) / len(test) * 1000

        # Flag-everything policy at this same cost pair: every flight is a
        # "positive" prediction, so there are zero missed delays (FN=0) and
        # every genuinely on-time flight becomes a false alarm (FP = all
        # actual negatives). This is the "model adds nothing" floor to beat.
        n_actual_negative = int((~test["is_delayed"]).sum())
        flag_all_cost = (n_actual_negative * cost_fp) / len(test) * 1000

        savings_vs_flag_all = flag_all_cost - model_cost
        savings_pct_of_flag_all = savings_vs_flag_all / flag_all_cost * 100

        sensitivity_rows.append({
            "cost_fp": cost_fp,
            "cost_fn": cost_fn,
            "cost_ratio_fn_fp": cost_fn / cost_fp,
            "threshold": scenario_threshold,
            "precision": precision,
            "recall": recall,
            "flagged_share": flagged_share,
            "model_cost_per_1000": model_cost,
            "flag_all_cost_per_1000": flag_all_cost,
            "savings_vs_flag_all": savings_vs_flag_all,
            "savings_pct_of_flag_all": savings_pct_of_flag_all,
        })

    sensitivity_table = pd.DataFrame(sensitivity_rows)
    print("Threshold sensitivity across cost scenarios (all evaluated on TEST):")
    print(sensitivity_table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print()
    print("Interpretation:")
    sorted_by_ratio = sensitivity_table.sort_values("cost_ratio_fn_fp", ascending=False)
    below_half = sorted_by_ratio.loc[sorted_by_ratio["flagged_share"] < 0.5]
    if len(below_half):
        row = below_half.iloc[0]
        print(f"  - Flagged share first drops below 50% at cost ratio {row['cost_ratio_fn_fp']:.2f} "
              f"(fp=${row['cost_fp']:.0f}, fn=${row['cost_fn']:.0f}): threshold={row['threshold']:.2f}, "
              f"flagged={row['flagged_share']:.1%}. Above this ratio, the chosen threshold is a "
              f"near-'flag everything' corner solution driven by the cost assumption, not by the "
              f"model's discrimination.")
    else:
        print("  - None of the tested cost ratios drop flagged share below 50% -- every scenario "
              "tried still resolves to a near-'flag everything' policy; a materially lower ratio "
              "than tested here would be needed to see the model actually discriminate.")

    meaningful = sensitivity_table.loc[sensitivity_table["savings_pct_of_flag_all"] >= 10].sort_values(
        "cost_ratio_fn_fp"
    )
    if len(meaningful):
        row = meaningful.iloc[0]
        print(f"  - Model beats flag-everything by >=10% starting at cost ratio {row['cost_ratio_fn_fp']:.2f} "
              f"and below (fp=${row['cost_fp']:.0f}, fn=${row['cost_fn']:.0f}): "
              f"savings={row['savings_pct_of_flag_all']:.1f}% of the flag-everything cost.")
    else:
        print("  - No tested cost ratio shows the model beating flag-everything by a meaningful "
              "(>=10%) margin -- across every scenario tried, most of the cost reduction comes from "
              "the cost assumption pushing toward a low threshold, not from the model's ranking ability.")

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(sensitivity_table["cost_ratio_fn_fp"], sensitivity_table["flagged_share"],
               color=COLOR_MAIN, s=70, zorder=5)
    # Several scenarios cluster at nearby or identical ratios (e.g. (5,10)
    # and (2,4) share ratio 2.0 exactly, and (10,25) at 2.5 sits right next
    # to it) -- stack labels vertically within any such cluster, and print
    # the actual fp/fn pair (not just the ratio), instead of letting the
    # text overlap illegibly. Note: no literal "$" in the label text --
    # matplotlib treats a pair of "$" as mathtext markup, which would
    # otherwise silently mangle the numbers into italic math formatting.
    plot_order = sensitivity_table.sort_values("cost_ratio_fn_fp").reset_index(drop=True)
    CLUSTER_GAP = 0.6  # ratio units within which labels are treated as "the same neighborhood"
    stack_level = 0
    prev_ratio = None
    for _, row in plot_order.iterrows():
        ratio = row["cost_ratio_fn_fp"]
        stack_level = stack_level + 1 if (prev_ratio is not None and abs(ratio - prev_ratio) < CLUSTER_GAP) else 0
        prev_ratio = ratio
        label = f"t={row['threshold']:.2f} (fp={row['cost_fp']:.0f}, fn={row['cost_fn']:.0f})"
        ax.annotate(label, (ratio, row["flagged_share"]),
                    textcoords="offset points", xytext=(8, 6 + stack_level * 13),
                    fontsize=8, color=INK_SECONDARY)
    ax.axhline(0.5, color=INK_MUTED, linewidth=1.2, linestyle="--", label="50% of test flights flagged")
    ax.set_title("Flagged share of test flights vs. cost ratio (miss cost / false-alarm cost)")
    ax.set_xlabel("cost ratio = cost_fn / cost_fp")
    ax.set_ylabel("flagged share of test flights")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False)
    ax.grid(linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "11_threshold_sensitivity.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")

    # ------------------------------------------------------------------
    section("SECTION 9: ODDS RATIO TABLE (statsmodels Logit)")
    # ------------------------------------------------------------------
    coefs = logit_model.params.drop("Intercept")
    top15_terms = coefs.abs().sort_values(ascending=False).head(15).index
    conf_int = logit_model.conf_int()
    odds_table = pd.DataFrame({
        "coef": logit_model.params[top15_terms],
        "odds_ratio": np.exp(logit_model.params[top15_terms]),
        "or_ci_low": np.exp(conf_int.loc[top15_terms, 0]),
        "or_ci_high": np.exp(conf_int.loc[top15_terms, 1]),
        "p_value": logit_model.pvalues[top15_terms],
    })
    print("Top 15 features by |coefficient| (odds ratios, log-odds scale coefficients underneath):")
    print(odds_table.to_string(float_format=lambda x: f"{x:.4f}"))

    def describe_term(term, odds_ratio):
        """Generic plain-English phrasing for any patsy dummy term, so I
        always have 3 things to say regardless of which specific features
        happen to land in the top 15 for a given fit."""
        direction = "higher" if odds_ratio > 1 else "lower"
        pct = abs(odds_ratio - 1) * 100
        if "dep_hour_bin" in term:
            level = term.split("[T.")[1].rstrip("]")
            return (f"a '{level}' departure has about {pct:.0f}% {direction} odds of a 15+ minute "
                    f"delay than an early-morning (5-7am) departure, all else equal.")
        if "dow" in term:
            level = term.split("[T.")[1].rstrip("]")
            return f"a {level} departure has about {pct:.0f}% {direction} odds of delay than Monday, all else equal."
        if "airline" in term:
            level = term.split("[T.")[1].rstrip("]")
            return (f"carrier {level} has about {pct:.0f}% {direction} odds of a 15+ minute delay "
                    f"than {REF_AIRLINE} (the reference, highest-volume carrier), all else equal.")
        if "origin_top50" in term:
            level = term.split("[T.")[1].rstrip("]")
            return f"departing FROM {level} carries about {pct:.0f}% {direction} odds of delay than {REF_AIRPORT} (reference), all else equal."
        if "dest_top50" in term:
            level = term.split("[T.")[1].rstrip("]")
            return f"flying TO {level} carries about {pct:.0f}% {direction} odds of delay than {REF_AIRPORT} (reference), all else equal."
        if term == "holiday_flag":
            return f"flights within 3 days of a major holiday have about {pct:.0f}% {direction} odds of delay than non-holiday flights."
        if term == "log_distance":
            return f"a 1% increase in distance is associated with roughly a {pct:.2f}% {direction} shift in delay odds (elasticity)."
        return f"{term} has odds ratio {odds_ratio:.2f}."

    # Pick 3 terms from the actual top-15 list, spread across different
    # feature families where possible, so the interpretation is illustrative
    # rather than three variants of the same feature type.
    seen_families = set()
    picks = []
    for term in top15_terms:
        family = term.split("[T.")[0] if "[T." in term else term
        if family not in seen_families:
            picks.append(term)
            seen_families.add(family)
        if len(picks) == 3:
            break
    if len(picks) < 3:
        picks = list(top15_terms[:3])

    print()
    print("Plain-English interpretation of 3 headline odds ratios:")
    for term in picks:
        odds_ratio = odds_table.loc[term, "odds_ratio"]
        print(f"  - {term} (OR={odds_ratio:.2f}): {describe_term(term, odds_ratio)}")


if __name__ == "__main__":
    main()
