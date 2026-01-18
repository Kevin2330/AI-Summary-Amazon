"""
Fixed effects regression analysis module.

Implements two-way fixed effects panel regressions using linearmodels.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from linearmodels.panel import PanelOLS
from statsmodels.stats.outliers_influence import variance_inflation_factor

from . import config


def prepare_panel_for_regression(
    panel_df: pd.DataFrame,
    entity_col: str = "parent_asin",
    time_col: str = "week_start"
) -> pd.DataFrame:
    """
    Prepare panel DataFrame for linearmodels regression.

    Sets MultiIndex with entity and time dimensions.

    Parameters
    ----------
    panel_df : pd.DataFrame
        Raw panel data
    entity_col : str
        Column name for entity (product) identifier
    time_col : str
        Column name for time identifier

    Returns
    -------
    pd.DataFrame
        Panel-indexed DataFrame
    """
    df = panel_df.copy()

    # Ensure time column is datetime
    df[time_col] = pd.to_datetime(df[time_col])

    # Set MultiIndex
    df = df.set_index([entity_col, time_col])

    # Sort index
    df = df.sort_index()

    return df


def compute_correlation_matrix(
    panel_df: pd.DataFrame,
    columns: List[str]
) -> pd.DataFrame:
    """
    Compute correlation matrix for specified columns.

    Parameters
    ----------
    panel_df : pd.DataFrame
        Panel data
    columns : list
        Columns to include in correlation

    Returns
    -------
    pd.DataFrame
        Correlation matrix
    """
    available_cols = [c for c in columns if c in panel_df.columns]
    return panel_df[available_cols].corr()


def compute_vif(
    panel_df: pd.DataFrame,
    columns: List[str]
) -> pd.DataFrame:
    """
    Compute Variance Inflation Factors for multicollinearity assessment.

    Parameters
    ----------
    panel_df : pd.DataFrame
        Panel data
    columns : list
        Feature columns to check

    Returns
    -------
    pd.DataFrame
        VIF values for each column
    """
    available_cols = [c for c in columns if c in panel_df.columns]
    X = panel_df[available_cols].dropna()

    if len(X) == 0 or len(available_cols) < 2:
        return pd.DataFrame({"variable": available_cols, "VIF": [np.nan] * len(available_cols)})

    # Add constant for VIF calculation
    X_with_const = X.copy()
    X_with_const["const"] = 1

    vif_data = []
    for i, col in enumerate(available_cols):
        try:
            vif = variance_inflation_factor(X_with_const.values, i)
            vif_data.append({"variable": col, "VIF": vif})
        except Exception as e:
            print(f"  Warning: VIF calculation failed for {col}: {e}")
            vif_data.append({"variable": col, "VIF": np.nan})

    return pd.DataFrame(vif_data)


def run_fe_regression(
    panel_df: pd.DataFrame,
    outcome: str,
    features: List[str],
    entity_effects: bool = True,
    time_effects: bool = True,
    cluster_entity: bool = True,
    drop_absorbed: bool = True,
) -> Dict:
    """
    Run two-way fixed effects regression.

    Y_it = beta' * X_it + entity_FE + time_FE + error_it

    Parameters
    ----------
    panel_df : pd.DataFrame
        Panel-indexed DataFrame (MultiIndex: entity, time)
    outcome : str
        Dependent variable column name
    features : list
        Independent variable column names
    entity_effects : bool
        Include entity fixed effects
    time_effects : bool
        Include time fixed effects
    cluster_entity : bool
        Cluster standard errors by entity
    drop_absorbed : bool
        Drop absorbed variables

    Returns
    -------
    dict
        Regression results including coefficients, SEs, p-values
    """
    # Prepare data
    available_features = [f for f in features if f in panel_df.columns]
    cols_needed = [outcome] + available_features

    # Drop rows with missing values
    data = panel_df[cols_needed].dropna()

    if len(data) == 0:
        print(f"  Warning: No valid observations for {outcome}")
        return None

    # Build formula
    formula = f"{outcome} ~ " + " + ".join(available_features)

    if entity_effects:
        formula += " + EntityEffects"
    if time_effects:
        formula += " + TimeEffects"

    print(f"\n  Running: {formula}")
    print(f"  Observations: {len(data):,}")

    try:
        # Fit model
        model = PanelOLS.from_formula(
            formula,
            data=data,
            drop_absorbed=drop_absorbed,
        )

        if cluster_entity:
            results = model.fit(cov_type="clustered", cluster_entity=True)
        else:
            results = model.fit()

        # Extract results
        result_dict = {
            "outcome": outcome,
            "n_obs": results.nobs,
            "n_entities": results.entity_info["total"],
            "n_time": results.time_info["total"],
            "r2_within": results.rsquared_within,
            "r2_between": results.rsquared_between,
            "r2_overall": results.rsquared_overall,
        }

        # Coefficients
        for feat in available_features:
            if feat in results.params.index:
                result_dict[f"coef_{feat}"] = results.params[feat]
                result_dict[f"se_{feat}"] = results.std_errors[feat]
                result_dict[f"pval_{feat}"] = results.pvalues[feat]
                result_dict[f"tstat_{feat}"] = results.tstats[feat]

        result_dict["model_results"] = results

        return result_dict

    except Exception as e:
        print(f"  Error fitting model: {e}")
        return None


def run_all_fe_regressions(
    panel_df: pd.DataFrame,
    outcomes: List[str],
    features: List[str],
    entity_col: str = "parent_asin",
    time_col: str = "week_start",
) -> Tuple[pd.DataFrame, Dict]:
    """
    Run FE regressions for all specified outcomes.

    Parameters
    ----------
    panel_df : pd.DataFrame
        Raw panel data
    outcomes : list
        Outcome variable names
    features : list
        Feature variable names
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
    print("Running Two-Way Fixed Effects Regressions")
    print("=" * 70)

    # Prepare panel
    indexed_panel = prepare_panel_for_regression(panel_df, entity_col, time_col)

    all_results = []
    full_results = {}

    for outcome in outcomes:
        if outcome not in indexed_panel.columns:
            print(f"\n  Skipping {outcome}: not in panel")
            continue

        result = run_fe_regression(
            indexed_panel,
            outcome=outcome,
            features=features,
            entity_effects=True,
            time_effects=True,
            cluster_entity=True,
        )

        if result:
            all_results.append(result)
            full_results[outcome] = result

    # Create summary DataFrame
    if all_results:
        # Extract key statistics
        summary_data = []
        for r in all_results:
            row = {
                "outcome": r["outcome"],
                "n_obs": r["n_obs"],
                "r2_within": r["r2_within"],
            }
            for feat in features:
                if f"coef_{feat}" in r:
                    row[f"{feat}_coef"] = r[f"coef_{feat}"]
                    row[f"{feat}_se"] = r[f"se_{feat}"]
                    row[f"{feat}_pval"] = r[f"pval_{feat}"]
            summary_data.append(row)

        results_df = pd.DataFrame(summary_data)
    else:
        results_df = pd.DataFrame()

    print("\n" + "=" * 70)
    print("FE Regression Summary")
    print("=" * 70)
    print(f"  Completed {len(all_results)} regressions")

    return results_df, full_results


def format_regression_table(
    results_df: pd.DataFrame,
    features: List[str],
    decimal_places: int = 4
) -> str:
    """
    Format regression results as a readable table.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results from run_all_fe_regressions
    features : list
        Feature names
    decimal_places : int
        Number of decimal places

    Returns
    -------
    str
        Formatted table string
    """
    lines = []
    lines.append("=" * 80)
    lines.append("Two-Way Fixed Effects Regression Results")
    lines.append("Entity FE: Yes | Time FE: Yes | Clustered SE: Entity")
    lines.append("=" * 80)

    if results_df.empty:
        lines.append("No results available")
        return "\n".join(lines)

    # Header
    header = f"{'Variable':<20}"
    for _, row in results_df.iterrows():
        header += f" {row['outcome']:>15}"
    lines.append(header)
    lines.append("-" * 80)

    # Coefficients and SEs for each feature
    for feat in features:
        coef_key = f"{feat}_coef"
        se_key = f"{feat}_se"
        pval_key = f"{feat}_pval"

        if coef_key not in results_df.columns:
            continue

        # Coefficient row
        coef_line = f"{feat:<20}"
        for _, row in results_df.iterrows():
            coef = row.get(coef_key, np.nan)
            pval = row.get(pval_key, 1.0)

            # Add significance stars
            stars = ""
            if pval < 0.01:
                stars = "***"
            elif pval < 0.05:
                stars = "**"
            elif pval < 0.1:
                stars = "*"

            if pd.notna(coef):
                coef_line += f" {coef:>12.{decimal_places}f}{stars}"
            else:
                coef_line += f" {'':>15}"
        lines.append(coef_line)

        # SE row
        se_line = f"{'':20}"
        for _, row in results_df.iterrows():
            se = row.get(se_key, np.nan)
            if pd.notna(se):
                se_line += f" ({se:>.{decimal_places}f})"
            else:
                se_line += f" {'':>15}"
        lines.append(se_line)

    lines.append("-" * 80)

    # N and R2
    n_line = f"{'N':<20}"
    for _, row in results_df.iterrows():
        n_line += f" {int(row['n_obs']):>15,}"
    lines.append(n_line)

    r2_line = f"{'R2 (within)':<20}"
    for _, row in results_df.iterrows():
        r2_line += f" {row['r2_within']:>15.4f}"
    lines.append(r2_line)

    lines.append("=" * 80)
    lines.append("Significance: *** p<0.01, ** p<0.05, * p<0.1")

    return "\n".join(lines)


def save_fe_results(
    results_df: pd.DataFrame,
    full_results: Dict,
    output_path: Path,
    features: List[str]
) -> None:
    """
    Save FE regression results to CSV and print formatted table.

    Parameters
    ----------
    results_df : pd.DataFrame
        Summary results
    full_results : dict
        Full results with model objects
    output_path : Path
        Path to save CSV
    features : list
        Feature names
    """
    # Save CSV
    results_df.to_csv(output_path, index=False)
    print(f"  Saved results to {output_path}")

    # Print formatted table
    table_str = format_regression_table(results_df, features)
    print("\n" + table_str)


if __name__ == "__main__":
    # Test with sample data
    print("FE Analysis module loaded. Run from notebook or script.")
