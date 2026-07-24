"""
02_linear_delay_magnitude.py

Question this script answers: among flights that arrive late, what drives HOW
LATE they run? This is a magnitude question (regression), distinct from a
"will it be late at all" classification question -- so the modeling
population here is restricted to arr_delay > 0 throughout.

Fitting data:    data/model/train_sample_1m.parquet
                 (statsmodels OLS builds a DENSE design matrix; the full ~5.2M
                 row train set was already judged memory-prohibitive for that
                 in split.py, hence the pre-drawn, seeded 1M sample.)
Evaluation data: data/model/test.parquet -- read ONLY to score fitted models,
                 never to fit anything or to choose features/bins/reference
                 levels. Any choice made using test-set values (e.g. which
                 airports are "top 50") would leak information into a set
                 that's supposed to give an honest, unbiased performance
                 estimate.

Run (with venv activated):
    python analysis/02_linear_delay_magnitude.py

Reads:  data/model/train_sample_1m.parquet
        data/model/test.parquet
Writes: reports/figures/06_residuals_vs_fitted.png
        reports/figures/07_qq_residuals.png
        reports/figures/08_qq_raw_vs_log.png
"""

import os

import matplotlib
matplotlib.use("Agg")  # headless backend -- never opens a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import patsy
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor

TRAIN_PATH = "data/model/train_sample_1m.parquet"
TEST_PATH = "data/model/test.parquet"
FIG_DIR = "reports/figures"

DIAG_SAMPLE_N = 50_000
DIAG_SEED = 42
TOP_N_AIRPORTS = 50

# --- chart styling: same look as 01_eda.py ---
COLOR_MAIN = "#2a78d6"
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
    Collapse dep_hour (0-23) into 6 buckets. Hours 0-4 individually have only
    379-9,198 rows in the full train set (I saw this in the EDA) -- far too
    few to give a stable per-hour coefficient, so I pool them into one
    "red_eye" level. The rest are grouped into recognizable travel-day
    segments rather than kept as 24 separate dummies, which would mostly
    just re-fit noise for the sparse hours anyway.
    """
    bins = [-1, 4, 7, 11, 15, 19, 23]
    labels = ["red_eye", "early_morning", "morning", "midday", "evening", "night"]
    return pd.cut(dep_hour, bins=bins, labels=labels)


def find_param(index, *needles):
    """I need this because patsy names each dummy column in its own format
    (like "C(airline)[T.F9]"), and I don't want to hardcode that exact string
    everywhere I need to look up one coefficient -- this just finds the one
    param whose name contains every substring I give it."""
    matches = [p for p in index if all(needle in p for needle in needles)]
    assert len(matches) == 1, f"expected exactly 1 param match for {needles}, got {matches}"
    return matches[0]


def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    section("SECTION 1: LOAD AND RESTRICT TO arr_delay > 0")
    # ------------------------------------------------------------------
    # "How late, given late" is only defined for flights that actually ran
    # late. Early/on-time flights aren't "mildly late" -- they're a different
    # outcome with nothing to explain on this scale, so they're excluded
    # rather than coded as delay = 0 (which would corrupt the log transform
    # below and pull the regression toward a population it isn't modeling).
    train_full = pd.read_parquet(TRAIN_PATH)
    test_full = pd.read_parquet(TEST_PATH)
    print(f"train_sample_1m rows loaded: {len(train_full):,}")
    print(f"test rows loaded:            {len(test_full):,}")

    train = train_full.loc[train_full["arr_delay"] > 0].copy()
    test = test_full.loc[test_full["arr_delay"] > 0].copy()
    print(f"train rows with arr_delay > 0: {len(train):,}")
    print(f"test  rows with arr_delay > 0: {len(test):,}")

    assert (train["arr_delay"] > 0).all()
    assert (test["arr_delay"] > 0).all()

    # log() requires strictly positive input -- guaranteed by the filter
    # above, but asserted explicitly rather than trusted silently.
    train["log_delay"] = np.log(train["arr_delay"])
    test["log_delay"] = np.log(test["arr_delay"])
    assert np.isfinite(train["log_delay"]).all(), "log(arr_delay) produced non-finite values in train"
    assert np.isfinite(test["log_delay"]).all(), "log(arr_delay) produced non-finite values in test"
    print("log_delay = log(arr_delay) computed for train and test; asserted finite.")

    # ------------------------------------------------------------------
    section("SECTION 2: FEATURE ENGINEERING (fit on train, applied to test)")
    # ------------------------------------------------------------------
    # month is deliberately NOT used as a feature anywhere below. The
    # train/test split is temporal: train covers months 1-9, test covers
    # 10-12, with zero overlap. A month dummy fit on train would have no
    # coefficient at all for October/November/December -- statsmodels would
    # either error or silently give nonsense when asked to score test rows
    # in months it never saw. Any seasonal effect month would have captured
    # is instead partially absorbed by holiday_flag and (for the schedule-only
    # model) simply left unmodeled.
    for label, df_ in [("train", train), ("test", test)]:
        df_["dep_hour_bin"] = bin_dep_hour(df_["dep_hour"])
        assert df_["dep_hour_bin"].isna().sum() == 0, f"dep_hour_bin has nulls in {label}"
    # early_morning is the reference level (lowest observed delay rate in the
    # EDA, and a natural "baseline travel time" to compare every other bucket
    # against) -- putting it first in the categories list makes patsy use it
    # as the omitted reference in C(dep_hour_bin).
    dep_hour_bin_categories = ["early_morning", "red_eye", "morning", "midday", "evening", "night"]
    train["dep_hour_bin"] = pd.Categorical(train["dep_hour_bin"], categories=dep_hour_bin_categories)
    test["dep_hour_bin"] = pd.Categorical(test["dep_hour_bin"], categories=dep_hour_bin_categories)
    print(f"dep_hour_bin built; reference level = '{dep_hour_bin_categories[0]}'")

    # dow is already an ordered Mon..Sun categorical from prepare.py, so
    # Monday (first category) is already the reference level -- nothing to
    # change here.
    assert list(train["dow"].cat.categories)[0] == "Monday"
    print("dow kept as-is; reference level = 'Monday' (already first category).")

    # origin/dest have 345 unique values each. One-hot-encoding all of them
    # is exactly what the EDA's airport-coverage section argued against: the
    # top 50 airports by volume already cover ~79% of flights, so everything
    # outside that set is folded into a single "OTHER" dummy. Critically, the
    # top-50 SET is chosen using TRAIN volume only, then applied unchanged to
    # test -- if I let test volume influence which airports get their own
    # dummy, I'd be leaking test-set structure into a modeling decision.
    for col, top_col in [("origin", "origin_top50"), ("dest", "dest_top50")]:
        top_set = set(train[col].value_counts().head(TOP_N_AIRPORTS).index)
        train[top_col] = np.where(train[col].isin(top_set), train[col], "OTHER")
        test[top_col] = np.where(test[col].isin(top_set), test[col], "OTHER")
        train_other_pct = (train[top_col] == "OTHER").mean() * 100
        test_other_pct = (test[top_col] == "OTHER").mean() * 100
        print(f"{top_col}: {len(top_set)} airports kept; "
              f"OTHER = {train_other_pct:.2f}% of train, {test_other_pct:.2f}% of test")
        # I didn't set an explicit reference level for this feature (unlike
        # dep_hour_bin/dow/airline above), so patsy falls back to its default
        # for a plain string column: alphabetically-first level. I print it
        # here so the implicit reference is visible rather than something I
        # discover later by noticing a missing coefficient.
        implicit_reference = sorted(train[top_col].unique())[0]
        print(f"  (implicit reference level, alphabetically first: '{implicit_reference}')")

    # airline: keep all 15 carriers as their own dummy (low cardinality, no
    # bucketing needed). Reference = the highest-volume carrier in train, so
    # every other airline's coefficient reads as "delay difference vs. the
    # airline flying the most routes in this population."
    airline_counts = train["airline"].value_counts()
    ref_airline = airline_counts.idxmax()
    airline_categories = [ref_airline] + [a for a in airline_counts.index if a != ref_airline]
    train["airline"] = pd.Categorical(train["airline"], categories=airline_categories)
    test["airline"] = pd.Categorical(test["airline"], categories=airline_categories)
    print(f"airline reference level = '{ref_airline}' (highest volume in train, "
          f"{int(airline_counts.max()):,} rows)")

    # log_distance: the EDA showed distance is essentially uncorrelated with
    # delay (r = -0.003) -- included here mainly to confirm that finding
    # holds up once schedule/airline/airport effects are controlled for, and
    # log-scaled to match the log-scaled target (coefficient reads as an
    # elasticity: % change in delay per % change in distance).
    assert (train["distance"] > 0).all() and (test["distance"] > 0).all()
    train["log_distance"] = np.log(train["distance"])
    test["log_distance"] = np.log(test["distance"])

    # holiday_flag is used as-is (already boolean).

    # ------------------------------------------------------------------
    section("SECTION 3: BASELINES (evaluated on TEST, before any model)")
    # ------------------------------------------------------------------
    # Baselines set the bar a "real" model has to clear. If M1/M2 can't beat
    # even the airline-mean baseline, the extra complexity isn't earning its
    # keep.
    def evaluate(name, pred_log, actual_log, actual_minutes):
        resid_log = actual_log - pred_log
        rmse_log = float(np.sqrt(np.mean(resid_log ** 2)))
        mae_log = float(np.mean(np.abs(resid_log)))
        # Naive back-transform via exp(). Note this is a biased (systematically
        # low) estimator of the conditional MEAN of arr_delay because of
        # Jensen's inequality (E[exp(X)] >= exp(E[X]) for non-degenerate X) --
        # a production model would apply a smearing correction. I report it
        # here only to put "how late" back into an interpretable unit (minutes),
        # not as a bias-corrected point forecast.
        pred_minutes = np.exp(pred_log)
        resid_minutes = actual_minutes - pred_minutes
        rmse_minutes = float(np.sqrt(np.mean(resid_minutes ** 2)))
        mae_minutes = float(np.mean(np.abs(resid_minutes)))
        return {
            "model": name,
            "rmse_log": rmse_log,
            "mae_log": mae_log,
            "rmse_minutes": rmse_minutes,
            "mae_minutes": mae_minutes,
        }

    results = []

    # B0: predict the train mean of log_delay for every test row.
    b0_value = train["log_delay"].mean()
    b0_pred = np.full(len(test), b0_value)
    print(f"B0 (train mean log_delay): {b0_value:.4f}")
    results.append(evaluate("B0: train mean", b0_pred, test["log_delay"].values, test["arr_delay"].values))

    # B1: predict the train per-airline mean of log_delay.
    b1_lookup = train.groupby("airline", observed=True)["log_delay"].mean()
    b1_pred = test["airline"].astype(str).map(b1_lookup).astype(float).values
    assert not np.isnan(b1_pred).any(), "test contains an airline with no train baseline mean"
    print("B1 (train per-airline mean log_delay):")
    print(b1_lookup.to_string(float_format=lambda x: f"{x:.4f}"))
    results.append(evaluate("B1: per-airline mean", b1_pred, test["log_delay"].values, test["arr_delay"].values))

    # ------------------------------------------------------------------
    section("SECTION 4: MODELS (statsmodels OLS, formula API, HC1 robust SE)")
    # ------------------------------------------------------------------
    # I use HC1 robust standard errors throughout because OLS's default
    # (classical) SEs assume homoscedastic errors -- an assumption flight
    # delay data is expected to violate (see the Breusch-Pagan test below),
    # so I use a heteroscedasticity-consistent estimator up front rather
    # than fit first and patch later.
    m1_formula = "log_delay ~ C(dep_hour_bin) + C(dow) + holiday_flag"
    m1 = smf.ols(m1_formula, data=train).fit(cov_type="HC1")
    print(f"M1 formula: {m1_formula}")
    print(m1.summary())

    m2_formula = (
        "log_delay ~ C(dep_hour_bin) + C(dow) + holiday_flag "
        "+ C(airline) + C(origin_top50) + C(dest_top50) + log_distance"
    )
    m2 = smf.ols(m2_formula, data=train).fit(cov_type="HC1")
    print()
    print(f"M2 formula: {m2_formula}")
    top15 = m2.params.reindex(m2.params.abs().sort_values(ascending=False).index).head(15)
    print("M2 top 15 coefficients by |estimate|:")
    print(top15.to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"M2 R-squared: {m2.rsquared:.4f}")
    print(f"M2 n:         {int(m2.nobs):,}")
    print(f"M2 df_resid:  {m2.df_resid:.0f}")

    results.append(evaluate(
        "M1: schedule only", m1.predict(test), test["log_delay"].values, test["arr_delay"].values
    ))
    results.append(evaluate(
        "M2: + airline/airport/distance", m2.predict(test), test["log_delay"].values, test["arr_delay"].values
    ))

    # ------------------------------------------------------------------
    section("SECTION 5: DIAGNOSTICS (M2)")
    # ------------------------------------------------------------------
    # Plotting all ~380K fitted rows would produce an unreadable, overplotted
    # smear and a slow-to-render file for no extra information -- a seeded
    # 50K subsample of the same fitted values/residuals is visually
    # equivalent and fast.
    diag = pd.DataFrame({"fitted": m2.fittedvalues, "resid": m2.resid})
    diag_sample = diag.sample(n=DIAG_SAMPLE_N, random_state=DIAG_SEED)

    # 1. Residuals vs fitted -- checks for remaining non-linear structure and
    # for the fan/funnel shape that signals heteroscedasticity.
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(diag_sample["fitted"], diag_sample["resid"], color=COLOR_MAIN, s=4, alpha=0.3)
    ax.axhline(0, color=INK_MUTED, linewidth=1, linestyle="--")
    ax.set_title("M2 residuals vs. fitted values (log_delay scale)")
    ax.set_xlabel("fitted log_delay")
    ax.set_ylabel("residual")
    ax.grid(linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "06_residuals_vs_fitted.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")

    # 2. QQ plot of M2 residuals -- checks the normality assumption behind
    # OLS's standard (non-robust) inference; systematic curvature away from
    # the 45-degree line means the residual distribution has fatter/thinner
    # tails than a normal distribution.
    (osm, osr), (slope, intercept, _r) = stats.probplot(diag_sample["resid"], dist="norm")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(osm, osr, color=COLOR_MAIN, s=6, alpha=0.5)
    ax.plot(osm, slope * osm + intercept, color=INK_MUTED, linewidth=1.5, linestyle="--")
    ax.set_title("QQ plot: M2 residuals (log_delay model)")
    ax.set_xlabel("theoretical normal quantiles")
    ax.set_ylabel("sample quantiles (residuals)")
    ax.grid(linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "07_qq_residuals.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")

    # 3. Same QQ comparison, but the left panel refits the identical M2
    # right-hand side on RAW arr_delay instead of log_delay. This is the
    # direct visual case for the log transform: if raw-scale residuals show
    # much worse tail behavior than log-scale residuals, that's the
    # justification, not just an assertion.
    m2_raw_formula = m2_formula.replace("log_delay ~", "arr_delay ~")
    # Only residuals are needed here (not valid inference), so a plain fit()
    # is enough -- no need for the HC1 robust covariance step.
    m2_raw = smf.ols(m2_raw_formula, data=train).fit()
    raw_resid_sample = m2_raw.resid.reindex(diag_sample.index)

    (osm_raw, osr_raw), (slope_raw, intercept_raw, _r) = stats.probplot(raw_resid_sample, dist="norm")

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].scatter(osm_raw, osr_raw, color=COLOR_MAIN, s=6, alpha=0.5)
    axes[0].plot(osm_raw, slope_raw * osm_raw + intercept_raw, color=INK_MUTED, linewidth=1.5, linestyle="--")
    axes[0].set_title("Raw arr_delay model")
    axes[0].set_xlabel("theoretical normal quantiles")
    axes[0].set_ylabel("sample quantiles (residuals, minutes)")
    axes[0].grid(linewidth=0.8)
    axes[0].set_axisbelow(True)

    axes[1].scatter(osm, osr, color=COLOR_MAIN, s=6, alpha=0.5)
    axes[1].plot(osm, slope * osm + intercept, color=INK_MUTED, linewidth=1.5, linestyle="--")
    axes[1].set_title("log(arr_delay) model")
    axes[1].set_xlabel("theoretical normal quantiles")
    axes[1].set_ylabel("sample quantiles (residuals, log units)")
    axes[1].grid(linewidth=0.8)
    axes[1].set_axisbelow(True)

    fig.suptitle("QQ plot comparison: raw vs. log-transformed target, same M2 predictors")
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "08_qq_raw_vs_log.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")

    # 4. Breusch-Pagan test -- formal test for heteroscedasticity (whether
    # residual variance depends on the predictors), complementing the visual
    # check in figure 06.
    bp_lm, bp_lm_pvalue, bp_fvalue, bp_fpvalue = het_breuschpagan(m2.resid, m2.model.exog)
    print()
    print(f"Breusch-Pagan LM statistic: {bp_lm:.2f}")
    print(f"Breusch-Pagan LM p-value:   {bp_lm_pvalue:.3g}")
    # With n in the hundreds of thousands, the Breusch-Pagan test has enormous
    # power -- it will detect and reject the null of homoscedasticity even for
    # very mild variance-vs-predictor dependence that has almost no practical
    # effect on the point estimates. The rejection here does NOT mean the
    # model is broken; it means classical (non-robust) OLS standard errors
    # would understate/overstate some coefficients' true sampling variability.
    # That is exactly why every model in this script is fit with cov_type=
    # 'HC1' (heteroscedasticity-consistent) robust standard errors up front,
    # rather than treating this as a surprise to fix after the fact.
    print("Interpretation: rejection of homoscedasticity (expected at this n) is why "
          "HC1 robust SEs are used for every model in this script instead of classical OLS SEs.")

    # 5. VIF (variance inflation factor) for the numeric / low-cardinality
    # terms only. origin_top50/dest_top50 each contribute ~50 dummy columns;
    # VIF among a large dummy block from the same categorical variable is
    # mechanically inflated by the coding itself and isn't informative, and
    # computing it for ~100 columns is slow for no benefit, so I skip it
    # here.
    vif_formula = "log_delay ~ C(dep_hour_bin) + C(dow) + holiday_flag + C(airline) + log_distance"
    _y_vif, x_vif = patsy.dmatrices(vif_formula, data=train, return_type="dataframe")
    vif_table = pd.DataFrame({
        "term": x_vif.columns,
        "VIF": [variance_inflation_factor(x_vif.values, i) for i in range(x_vif.shape[1])],
    })
    print()
    print("VIF (numeric / low-cardinality terms only; origin/dest dummies skipped):")
    print(vif_table.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    high_vif = vif_table.loc[(vif_table["VIF"] > 10) & (vif_table["term"] != "Intercept")]
    if len(high_vif):
        print("Terms with VIF > 10 (excluding Intercept):")
        print(high_vif.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    else:
        print("No terms with VIF > 10 (excluding Intercept).")

    # ------------------------------------------------------------------
    section("SECTION 6: OUTPUT DELIVERABLE")
    # ------------------------------------------------------------------
    comparison = pd.DataFrame(results).set_index("model")
    print("Baseline vs. model comparison (evaluated on TEST):")
    print(comparison.to_string(float_format=lambda x: f"{x:.4f}"))

    # ~10 headline M2 coefficients, chosen to span every feature family in
    # the model (schedule, calendar, airline, airport, distance) so the
    # table tells a complete story rather than an arbitrary top-N.
    headline_terms = [
        find_param(m2.params.index, "holiday_flag"),
        "log_distance",
        find_param(m2.params.index, "dep_hour_bin", "night"),
        find_param(m2.params.index, "dep_hour_bin", "midday"),
        find_param(m2.params.index, "dow", "Friday"),
        find_param(m2.params.index, "airline", "F9"),
        find_param(m2.params.index, "airline", "HA"),
        find_param(m2.params.index, "origin_top50", "ORD"),
        find_param(m2.params.index, "origin_top50", "DEN"),
        find_param(m2.params.index, "origin_top50", "DFW"),
    ]
    conf_int = m2.conf_int()
    coef_rows = []
    for term in headline_terms:
        est = m2.params[term]
        coef_rows.append({
            "term": term,
            "coef": est,
            "robust_se": m2.bse[term],
            "ci_low": conf_int.loc[term, 0],
            "ci_high": conf_int.loc[term, 1],
            "p_value": m2.pvalues[term],
            # exp(coef)-1: for a dummy, this is "delay is X% higher/lower than
            # the reference level, holding everything else fixed." For
            # log_distance (a log-scaled continuous term) it instead reads as
            # an elasticity: "a 1% increase in distance is associated with an
            # X% change in delay" -- not a level comparison to a reference.
            "approx_pct_change": (np.exp(est) - 1) * 100,
        })
    coef_table = pd.DataFrame(coef_rows).set_index("term")
    print()
    print("Headline M2 coefficients (log_delay scale; approx_pct_change is exp(coef)-1, "
          "elasticity for log_distance, level-shift-vs-reference for dummies):")
    print(coef_table.to_string(float_format=lambda x: f"{x:.4f}"))

    print()
    print(f"M1 R-squared: {m1.rsquared:.4f}")
    print(f"M2 R-squared: {m2.rsquared:.4f}")
    # Flight arrival delay is driven heavily by day-of-operations factors this
    # dataset doesn't contain (weather, air-traffic-control programs,
    # mechanical issues, upstream aircraft rotation) -- schedule/airline/
    # airport/calendar features were never expected to explain most of the
    # variance in HOW LATE a late flight runs. An R-squared in the 0.05-0.15
    # range is a realistic, honest result for this kind of model, not a sign
    # something is wrong; a much higher R-squared here would be more likely
    # to indicate leakage than a better model.
    print("Note: R-squared in the 0.05-0.15 range is expected and honest for flight-delay "
          "magnitude models -- most of the variance is driven by day-of factors "
          "(weather, ATC, mechanical, aircraft rotation) that aren't in this dataset.")


if __name__ == "__main__":
    main()
