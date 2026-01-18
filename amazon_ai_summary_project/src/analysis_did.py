"""
Difference-in-Differences and Event Study analysis module.

Implements DiD around Amazon AI review summary rollout (2023-08-14)
and event study with parallel trends testing.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

from linearmodels.panel import PanelOLS
from scipy import stats

from . import config
from .analysis_fe import prepare_panel_for_regression


def run_did_regression(
    panel_df: pd.DataFrame,
    outcome: str,
    treatment_col: str = "treated",
    post_col: str = "post",
    interaction_col: str = "treated_post",
    entity_effects: bool = True,
    time_effects: bool = True,
    cluster_entity: bool = True,
    additional_controls: List[str] = None,
) -> Dict:
    """
    Run Difference-in-Differences regression.

    Y_it = delta * treated_post + [controls] + entity_FE + time_FE + error_it

    Note: Main effects (treated, post) are absorbed by entity and time FEs.

    Parameters
    ----------
    panel_df : pd.DataFrame
        Panel-indexed DataFrame
    outcome : str
        Dependent variable
    treatment_col : str
        Treatment indicator column
    post_col : str
        Post-period indicator column
    interaction_col : str
        Interaction term column (treated * post)
    entity_effects : bool
        Include entity fixed effects
    time_effects : bool
        Include time fixed effects
    cluster_entity : bool
        Cluster SEs by entity
    additional_controls : list, optional
        Additional control variables

    Returns
    -------
    dict
        Regression results
    """
    # Build regressor list
    regressors = [interaction_col]

    if additional_controls:
        regressors.extend([c for c in additional_controls if c in panel_df.columns])

    # Prepare data
    cols_needed = [outcome] + regressors
    data = panel_df[cols_needed].dropna()

    if len(data) == 0:
        print(f"  Warning: No valid observations for {outcome}")
        return None

    # Check variation in interaction term
    n_treated_post = (data[interaction_col] == 1).sum()
    n_treated_pre = ((panel_df.index.get_level_values(0).isin(data.index.get_level_values(0))) &
                     (panel_df[post_col] == 0) & (panel_df[treatment_col] == 1)).sum()

    print(f"\n  Running DiD for: {outcome}")
    print(f"  Treated*Post obs: {n_treated_post:,}")
    print(f"  Total obs: {len(data):,}")

    # Build formula
    formula = f"{outcome} ~ " + " + ".join(regressors)
    if entity_effects:
        formula += " + EntityEffects"
    if time_effects:
        formula += " + TimeEffects"

    try:
        model = PanelOLS.from_formula(
            formula,
            data=data,
            drop_absorbed=True,
        )

        if cluster_entity:
            results = model.fit(cov_type="clustered", cluster_entity=True)
        else:
            results = model.fit()

        # Extract DiD estimate
        did_coef = results.params[interaction_col]
        did_se = results.std_errors[interaction_col]
        did_pval = results.pvalues[interaction_col]
        did_tstat = results.tstats[interaction_col]

        result_dict = {
            "outcome": outcome,
            "did_coef": did_coef,
            "did_se": did_se,
            "did_pval": did_pval,
            "did_tstat": did_tstat,
            "n_obs": results.nobs,
            "n_entities": results.entity_info["total"],
            "n_time": results.time_info["total"],
            "r2_within": results.rsquared_within,
            "model_results": results,
        }

        # Significance level
        if did_pval < 0.01:
            result_dict["significance"] = "***"
        elif did_pval < 0.05:
            result_dict["significance"] = "**"
        elif did_pval < 0.1:
            result_dict["significance"] = "*"
        else:
            result_dict["significance"] = ""

        print(f"  DiD estimate: {did_coef:.4f} ({did_se:.4f}){result_dict['significance']}")

        return result_dict

    except Exception as e:
        print(f"  Error fitting DiD model: {e}")
        return None


def run_all_did_regressions(
    panel_df: pd.DataFrame,
    outcomes: List[str],
    entity_col: str = "parent_asin",
    time_col: str = "week_start",
) -> Tuple[pd.DataFrame, Dict]:
    """
    Run DiD regressions for all specified outcomes.

    Parameters
    ----------
    panel_df : pd.DataFrame
        Raw panel data
    outcomes : list
        Outcome variable names
    entity_col : str
        Entity identifier column
    time_col : str
        Time identifier column

    Returns
    -------
    tuple
        (results_df, full_results_dict)
    """
    print("=" * 70)
    print("Running Difference-in-Differences Analysis")
    print(f"Treatment: rating_number >= {config.TREATMENT_THRESHOLD}")
    print(f"Cutoff date: {config.AI_ROLLOUT_DATE}")
    print("=" * 70)

    # Prepare panel
    indexed_panel = prepare_panel_for_regression(panel_df, entity_col, time_col)

    all_results = []
    full_results = {}

    for outcome in outcomes:
        if outcome not in indexed_panel.columns:
            print(f"\n  Skipping {outcome}: not in panel")
            continue

        result = run_did_regression(
            indexed_panel,
            outcome=outcome,
        )

        if result:
            all_results.append(result)
            full_results[outcome] = result

    # Create summary DataFrame
    if all_results:
        results_df = pd.DataFrame([
            {
                "outcome": r["outcome"],
                "did_coef": r["did_coef"],
                "did_se": r["did_se"],
                "did_pval": r["did_pval"],
                "did_tstat": r["did_tstat"],
                "significance": r["significance"],
                "n_obs": r["n_obs"],
                "r2_within": r["r2_within"],
            }
            for r in all_results
        ])
    else:
        results_df = pd.DataFrame()

    return results_df, full_results


def create_event_time_dummies(
    panel_df: pd.DataFrame,
    event_time_col: str = "event_time",
    window: int = 8,
    omit_period: int = -1,
    treatment_col: str = "treated",
    min_cell_size: int = 30,
) -> Tuple[pd.DataFrame, List[str], List[int]]:
    """
    Create event-time dummy variables for event study.

    Parameters
    ----------
    panel_df : pd.DataFrame
        Panel data
    event_time_col : str
        Column with event time (weeks relative to cutoff)
    window : int
        Event window (+/- weeks from cutoff)
    omit_period : int
        Period to omit as baseline (typically -1)
    treatment_col : str
        Treatment indicator column
    min_cell_size : int
        Minimum observations per cell (treatment x time)

    Returns
    -------
    tuple
        (modified_df, dummy_names, valid_periods)
    """
    df = panel_df.copy()

    # Define event time range
    min_et = -window
    max_et = window

    # Filter to window
    df = df[(df[event_time_col] >= min_et) & (df[event_time_col] <= max_et)].copy()

    # Check cell sizes
    valid_periods = []
    cell_counts = df.groupby([treatment_col, event_time_col]).size().unstack(fill_value=0)

    print(f"\n  Event time cell sizes (min required: {min_cell_size}):")

    for et in range(min_et, max_et + 1):
        if et == omit_period:
            continue

        if et in cell_counts.columns:
            treated_count = cell_counts.loc[1, et] if 1 in cell_counts.index else 0
            control_count = cell_counts.loc[0, et] if 0 in cell_counts.index else 0

            if treated_count >= min_cell_size and control_count >= min_cell_size:
                valid_periods.append(et)
            else:
                print(f"    Dropping ET={et}: treated={treated_count}, control={control_count}")

    print(f"  Valid periods: {valid_periods}")

    # Create dummies for valid periods only
    dummy_names = []

    for et in valid_periods:
        # Use 'm' prefix for negative event times to avoid formula parsing issues
        if et < 0:
            dummy_name = f"ET_m{abs(et)}"
        else:
            dummy_name = f"ET_p{et}"
        # Interaction: treated * event_time_dummy
        df[dummy_name] = ((df[event_time_col] == et) & (df[treatment_col] == 1)).astype(int)
        dummy_names.append(dummy_name)

    return df, dummy_names, valid_periods


def run_event_study(
    panel_df: pd.DataFrame,
    outcome: str,
    event_time_col: str = "event_time",
    treatment_col: str = "treated",
    window: int = None,
    omit_period: int = None,
    min_cell_size: int = None,
    entity_effects: bool = True,
    time_effects: bool = True,
    cluster_entity: bool = True,
) -> Dict:
    """
    Run event study regression with parallel trends testing.

    Y_it = sum_{k != omit} gamma_k * (treated_i * ET_k) + entity_FE + time_FE + error_it

    Parameters
    ----------
    panel_df : pd.DataFrame
        Panel-indexed DataFrame
    outcome : str
        Dependent variable
    event_time_col : str
        Event time column
    treatment_col : str
        Treatment indicator column
    window : int
        Event window (+/- weeks)
    omit_period : int
        Omitted period (baseline)
    min_cell_size : int
        Minimum cell size for inclusion
    entity_effects : bool
        Include entity FE
    time_effects : bool
        Include time FE
    cluster_entity : bool
        Cluster SEs by entity

    Returns
    -------
    dict
        Event study results including coefficients and parallel trends test
    """
    window = window or config.EVENT_STUDY_WINDOW
    omit_period = omit_period if omit_period is not None else config.OMIT_PERIOD
    min_cell_size = min_cell_size or config.MIN_CELL_SIZE

    print(f"\n  Running event study for: {outcome}")
    print(f"  Window: +/- {window} weeks, Omitted period: {omit_period}")

    # Create event time dummies
    df_es, dummy_names, valid_periods = create_event_time_dummies(
        panel_df,
        event_time_col=event_time_col,
        window=window,
        omit_period=omit_period,
        treatment_col=treatment_col,
        min_cell_size=min_cell_size,
    )

    if len(dummy_names) == 0:
        print("  Warning: No valid event time periods")
        return None

    # Prepare data
    cols_needed = [outcome] + dummy_names + [treatment_col, event_time_col]
    df_es = df_es[cols_needed].dropna()

    if len(df_es) == 0:
        print(f"  Warning: No valid observations for {outcome}")
        return None

    print(f"  Observations: {len(df_es):,}")

    # Build formula
    formula = f"{outcome} ~ " + " + ".join(dummy_names)
    if entity_effects:
        formula += " + EntityEffects"
    if time_effects:
        formula += " + TimeEffects"

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            model = PanelOLS.from_formula(
                formula,
                data=df_es,
                drop_absorbed=True,
            )

            if cluster_entity:
                results = model.fit(cov_type="clustered", cluster_entity=True, low_memory=True)
            else:
                results = model.fit(low_memory=True)

        # Extract coefficients for each event time
        coefficients = []
        for et, dummy in zip(valid_periods, dummy_names):
            if dummy in results.params.index:
                coefficients.append({
                    "event_time": et,
                    "dummy_name": dummy,
                    "coef": results.params[dummy],
                    "se": results.std_errors[dummy],
                    "pval": results.pvalues[dummy],
                    "ci_lower": results.params[dummy] - 1.96 * results.std_errors[dummy],
                    "ci_upper": results.params[dummy] + 1.96 * results.std_errors[dummy],
                })

        coef_df = pd.DataFrame(coefficients)

        # Add omitted period (zero by construction)
        if omit_period < 0:
            omit_dummy_name = f"ET_m{abs(omit_period)}"
        else:
            omit_dummy_name = f"ET_p{omit_period}"
        omit_row = pd.DataFrame([{
            "event_time": omit_period,
            "dummy_name": omit_dummy_name,
            "coef": 0,
            "se": 0,
            "pval": np.nan,
            "ci_lower": 0,
            "ci_upper": 0,
        }])
        coef_df = pd.concat([coef_df, omit_row], ignore_index=True)
        coef_df = coef_df.sort_values("event_time").reset_index(drop=True)

        # Parallel trends test: joint test that pre-period coefficients = 0
        pre_period_dummies = [d for et, d in zip(valid_periods, dummy_names) if et < omit_period]

        parallel_trends_test = None
        if len(pre_period_dummies) >= 1:
            parallel_trends_test = run_parallel_trends_test(results, pre_period_dummies)

        result_dict = {
            "outcome": outcome,
            "coefficients": coef_df,
            "valid_periods": valid_periods,
            "omit_period": omit_period,
            "n_obs": results.nobs,
            "r2_within": results.rsquared_within,
            "model_results": results,
            "parallel_trends_test": parallel_trends_test,
        }

        if parallel_trends_test:
            print(f"  Parallel trends test: F={parallel_trends_test['f_stat']:.2f}, "
                  f"p={parallel_trends_test['p_value']:.4f}")

        return result_dict

    except Exception as e:
        print(f"  Error fitting event study model: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_parallel_trends_test(
    model_results,
    pre_period_dummies: List[str]
) -> Dict:
    """
    Run joint Wald test for parallel trends (pre-period coefficients = 0).

    Parameters
    ----------
    model_results : PanelOLSResults
        Fitted model results
    pre_period_dummies : list
        Names of pre-period dummy variables

    Returns
    -------
    dict
        Test results (F-statistic, p-value, etc.)
    """
    if len(pre_period_dummies) == 0:
        return None

    try:
        # Build restriction matrix for H0: all pre-period coeffs = 0
        # Get parameter names that are in the model
        available_dummies = [d for d in pre_period_dummies if d in model_results.params.index]

        if len(available_dummies) == 0:
            return None

        # Manual Wald test using coefficient estimates and variance-covariance
        params = model_results.params[available_dummies].values
        vcov = model_results.cov[available_dummies].loc[available_dummies].values

        # Wald statistic: beta' * inv(V) * beta
        try:
            vcov_inv = np.linalg.inv(vcov)
            wald_stat = params @ vcov_inv @ params
            df = len(available_dummies)
            p_value = 1 - stats.chi2.cdf(wald_stat, df)

            # Also compute F-statistic version
            f_stat = wald_stat / df
            f_pvalue = 1 - stats.f.cdf(f_stat, df, model_results.df_resid)

            return {
                "wald_stat": wald_stat,
                "chi2_pvalue": p_value,
                "f_stat": f_stat,
                "p_value": f_pvalue,
                "df": df,
                "df_resid": model_results.df_resid,
                "tested_dummies": available_dummies,
                "null_rejected": f_pvalue < 0.05,
            }
        except np.linalg.LinAlgError:
            print("  Warning: Singular covariance matrix in Wald test")
            return None

    except Exception as e:
        print(f"  Warning: Parallel trends test failed: {e}")
        return None


def run_all_event_studies(
    panel_df: pd.DataFrame,
    outcomes: List[str],
    entity_col: str = "parent_asin",
    time_col: str = "week_start",
    window: int = None,
) -> Tuple[Dict[str, pd.DataFrame], Dict]:
    """
    Run event studies for all specified outcomes.

    Parameters
    ----------
    panel_df : pd.DataFrame
        Raw panel data
    outcomes : list
        Outcome variable names
    entity_col : str
        Entity identifier column
    time_col : str
        Time identifier column
    window : int
        Event window

    Returns
    -------
    tuple
        (dict of coefficient DataFrames, full results dict)
    """
    print("=" * 70)
    print("Running Event Study Analysis")
    print(f"Treatment: rating_number >= {config.TREATMENT_THRESHOLD}")
    print(f"Cutoff date: {config.AI_ROLLOUT_DATE}")
    print("=" * 70)

    # Prepare panel
    indexed_panel = prepare_panel_for_regression(panel_df, entity_col, time_col)

    coef_dfs = {}
    full_results = {}

    for outcome in outcomes:
        if outcome not in indexed_panel.columns:
            print(f"\n  Skipping {outcome}: not in panel")
            continue

        result = run_event_study(
            indexed_panel,
            outcome=outcome,
            window=window,
        )

        if result and result["coefficients"] is not None:
            coef_dfs[outcome] = result["coefficients"]
            full_results[outcome] = result

    return coef_dfs, full_results


def format_did_table(did_results_df: pd.DataFrame) -> str:
    """Format DiD results as readable table."""
    lines = []
    lines.append("=" * 80)
    lines.append("Difference-in-Differences Results")
    lines.append(f"Treatment: rating_number >= {config.TREATMENT_THRESHOLD}")
    lines.append(f"Post period: week_start >= {config.AI_ROLLOUT_DATE}")
    lines.append("=" * 80)

    if did_results_df.empty:
        lines.append("No results available")
        return "\n".join(lines)

    lines.append(f"{'Outcome':<20} {'DiD Coef':>12} {'SE':>10} {'t-stat':>10} {'p-value':>10}")
    lines.append("-" * 80)

    for _, row in did_results_df.iterrows():
        sig = row.get("significance", "")
        lines.append(
            f"{row['outcome']:<20} "
            f"{row['did_coef']:>12.4f} "
            f"{row['did_se']:>10.4f} "
            f"{row['did_tstat']:>10.2f} "
            f"{row['did_pval']:>10.4f}{sig}"
        )

    lines.append("=" * 80)
    lines.append("Significance: *** p<0.01, ** p<0.05, * p<0.1")
    lines.append("Note: Main effects absorbed by entity and time fixed effects")

    return "\n".join(lines)


def format_parallel_trends_summary(full_results: Dict) -> str:
    """Format parallel trends test results."""
    lines = []
    lines.append("\nParallel Trends Tests (H0: all pre-period coefficients = 0)")
    lines.append("-" * 60)
    lines.append(f"{'Outcome':<20} {'F-stat':>10} {'p-value':>10} {'Reject H0':>12}")
    lines.append("-" * 60)

    for outcome, result in full_results.items():
        pt_test = result.get("parallel_trends_test")
        if pt_test:
            reject = "Yes" if pt_test["null_rejected"] else "No"
            lines.append(
                f"{outcome:<20} "
                f"{pt_test['f_stat']:>10.2f} "
                f"{pt_test['p_value']:>10.4f} "
                f"{reject:>12}"
            )
        else:
            lines.append(f"{outcome:<20} {'N/A':>10} {'N/A':>10} {'N/A':>12}")

    lines.append("-" * 60)
    lines.append("Rejection at 5% level indicates potential violation of parallel trends")

    return "\n".join(lines)


if __name__ == "__main__":
    print("DiD Analysis module loaded. Run from notebook or script.")
