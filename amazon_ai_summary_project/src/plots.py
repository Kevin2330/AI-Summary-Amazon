"""
Visualization module for Amazon review analysis.

Provides plotting functions for time series, correlations, and event studies.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from . import config

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("colorblind")


def setup_plot_style():
    """Configure matplotlib defaults for publication-quality figures."""
    plt.rcParams.update({
        'figure.figsize': (10, 6),
        'figure.dpi': 100,
        'savefig.dpi': 150,
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'lines.linewidth': 1.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


setup_plot_style()


def plot_timeseries(
    panel_df: pd.DataFrame,
    y_col: str,
    time_col: str = "week_start",
    title: str = None,
    ylabel: str = None,
    add_rollout_line: bool = True,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 6),
    agg_func: str = "mean",
) -> plt.Figure:
    """
    Plot time series of a variable aggregated across products.

    Parameters
    ----------
    panel_df : pd.DataFrame
        Panel data
    y_col : str
        Column to plot
    time_col : str
        Time column
    title : str
        Plot title
    ylabel : str
        Y-axis label
    add_rollout_line : bool
        Add vertical line at AI rollout date
    save_path : Path
        Path to save figure
    figsize : tuple
        Figure size
    agg_func : str
        Aggregation function ('mean', 'sum', 'median')

    Returns
    -------
    matplotlib.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Aggregate by time
    if agg_func == "mean":
        ts = panel_df.groupby(time_col)[y_col].mean()
    elif agg_func == "sum":
        ts = panel_df.groupby(time_col)[y_col].sum()
    elif agg_func == "median":
        ts = panel_df.groupby(time_col)[y_col].median()
    else:
        ts = panel_df.groupby(time_col)[y_col].mean()

    # Plot
    ax.plot(ts.index, ts.values, marker='o', markersize=4, linewidth=1.5)

    # Add rollout line
    if add_rollout_line:
        rollout_date = pd.Timestamp(config.AI_ROLLOUT_DATE)
        ax.axvline(x=rollout_date, color='red', linestyle='--', linewidth=1.5,
                   label=f'AI Summary Rollout ({config.AI_ROLLOUT_DATE})')

    # Formatting
    ax.set_xlabel("Week")
    ax.set_ylabel(ylabel or y_col)
    ax.set_title(title or f"Weekly {y_col}")

    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45, ha='right')

    ax.legend(loc='best')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"  Saved: {save_path}")

    return fig


def plot_timeseries_by_treatment(
    panel_df: pd.DataFrame,
    y_col: str,
    treatment_col: str = "treated",
    time_col: str = "week_start",
    title: str = None,
    ylabel: str = None,
    add_rollout_line: bool = True,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 6),
) -> plt.Figure:
    """
    Plot time series comparing treatment and control groups.

    Parameters
    ----------
    panel_df : pd.DataFrame
        Panel data
    y_col : str
        Column to plot
    treatment_col : str
        Treatment indicator column
    time_col : str
        Time column
    title : str
        Plot title
    ylabel : str
        Y-axis label
    add_rollout_line : bool
        Add vertical line at AI rollout
    save_path : Path
        Path to save figure
    figsize : tuple
        Figure size

    Returns
    -------
    matplotlib.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Aggregate by treatment group and time
    for treat_val, label, color in [(1, "Treated", "blue"), (0, "Control", "orange")]:
        subset = panel_df[panel_df[treatment_col] == treat_val]
        ts = subset.groupby(time_col)[y_col].mean()
        ax.plot(ts.index, ts.values, marker='o', markersize=3, linewidth=1.5,
                label=label, color=color)

    # Add rollout line
    if add_rollout_line:
        rollout_date = pd.Timestamp(config.AI_ROLLOUT_DATE)
        ax.axvline(x=rollout_date, color='red', linestyle='--', linewidth=1.5,
                   label=f'AI Summary Rollout')

    # Formatting
    ax.set_xlabel("Week")
    ax.set_ylabel(ylabel or y_col)
    ax.set_title(title or f"Weekly {y_col} by Treatment Group")

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45, ha='right')

    ax.legend(loc='best')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"  Saved: {save_path}")

    return fig


def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    title: str = "Correlation Matrix",
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 8),
    annot: bool = True,
) -> plt.Figure:
    """
    Plot correlation matrix as heatmap.

    Parameters
    ----------
    corr_matrix : pd.DataFrame
        Correlation matrix
    title : str
        Plot title
    save_path : Path
        Path to save figure
    figsize : tuple
        Figure size
    annot : bool
        Show correlation values

    Returns
    -------
    matplotlib.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    cmap = sns.diverging_palette(250, 10, as_cmap=True)

    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap=cmap,
        vmin=-1, vmax=1,
        center=0,
        annot=annot,
        fmt='.2f',
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )

    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"  Saved: {save_path}")

    return fig


def plot_event_study(
    coef_df: pd.DataFrame,
    outcome: str,
    title: str = None,
    ylabel: str = None,
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 6),
    omit_period: int = -1,
) -> plt.Figure:
    """
    Plot event study coefficients with confidence intervals.

    Parameters
    ----------
    coef_df : pd.DataFrame
        DataFrame with columns: event_time, coef, ci_lower, ci_upper
    outcome : str
        Outcome variable name
    title : str
        Plot title
    ylabel : str
        Y-axis label
    save_path : Path
        Path to save figure
    figsize : tuple
        Figure size
    omit_period : int
        Omitted period (shown with different marker)

    Returns
    -------
    matplotlib.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Sort by event time
    df = coef_df.sort_values("event_time").copy()

    # Get data
    event_times = df["event_time"].values
    coefs = df["coef"].values
    ci_lower = df["ci_lower"].values
    ci_upper = df["ci_upper"].values

    # Plot coefficients with CI
    ax.errorbar(
        event_times, coefs,
        yerr=[coefs - ci_lower, ci_upper - coefs],
        fmt='o', markersize=6, capsize=3, capthick=1.5,
        color='blue', ecolor='gray', linewidth=1.5,
    )

    # Connect points with line
    ax.plot(event_times, coefs, 'b-', alpha=0.5, linewidth=1)

    # Add horizontal line at zero
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

    # Add vertical line at omitted period / treatment
    ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5,
               label='AI Summary Rollout')

    # Shade pre-period
    ax.axvspan(ax.get_xlim()[0], 0, alpha=0.1, color='gray', label='Pre-period')

    # Formatting
    ax.set_xlabel("Weeks Relative to AI Summary Rollout")
    ax.set_ylabel(ylabel or f"Effect on {outcome}")
    ax.set_title(title or f"Event Study: {outcome}")

    # Set x-ticks at integers
    ax.set_xticks(event_times)

    ax.legend(loc='best')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"  Saved: {save_path}")

    return fig


def plot_multiple_event_studies(
    coef_dfs: Dict[str, pd.DataFrame],
    save_dir: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 6),
) -> Dict[str, plt.Figure]:
    """
    Plot event studies for multiple outcomes.

    Parameters
    ----------
    coef_dfs : dict
        Mapping of outcome names to coefficient DataFrames
    save_dir : Path
        Directory to save figures
    figsize : tuple
        Figure size

    Returns
    -------
    dict
        Mapping of outcome names to figures
    """
    figures = {}

    for outcome, coef_df in coef_dfs.items():
        save_path = None
        if save_dir:
            save_path = save_dir / f"event_study_{outcome}.png"

        fig = plot_event_study(
            coef_df,
            outcome=outcome,
            save_path=save_path,
            figsize=figsize,
        )
        figures[outcome] = fig

    return figures


def plot_vif_bars(
    vif_df: pd.DataFrame,
    title: str = "Variance Inflation Factors",
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 6),
    threshold: float = 5.0,
) -> plt.Figure:
    """
    Plot VIF values as horizontal bar chart.

    Parameters
    ----------
    vif_df : pd.DataFrame
        DataFrame with columns: variable, VIF
    title : str
        Plot title
    save_path : Path
        Path to save figure
    figsize : tuple
        Figure size
    threshold : float
        VIF threshold to highlight (typically 5 or 10)

    Returns
    -------
    matplotlib.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    df = vif_df.sort_values("VIF", ascending=True)
    colors = ['red' if v > threshold else 'steelblue' for v in df["VIF"]]

    bars = ax.barh(df["variable"], df["VIF"], color=colors)

    # Add threshold line
    ax.axvline(x=threshold, color='red', linestyle='--', linewidth=1.5,
               label=f'Threshold (VIF={threshold})')

    # Add value labels
    for bar, vif in zip(bars, df["VIF"]):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                f'{vif:.2f}', va='center', fontsize=9)

    ax.set_xlabel("VIF")
    ax.set_title(title)
    ax.legend(loc='lower right')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"  Saved: {save_path}")

    return fig


def plot_distribution_grid(
    panel_df: pd.DataFrame,
    columns: List[str],
    ncols: int = 3,
    save_path: Optional[Path] = None,
    figsize_per_plot: Tuple[int, int] = (4, 3),
) -> plt.Figure:
    """
    Plot distributions for multiple variables in a grid.

    Parameters
    ----------
    panel_df : pd.DataFrame
        Panel data
    columns : list
        Columns to plot
    ncols : int
        Number of columns in grid
    save_path : Path
        Path to save figure
    figsize_per_plot : tuple
        Size per subplot

    Returns
    -------
    matplotlib.Figure
    """
    n_plots = len(columns)
    nrows = (n_plots + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows)
    )

    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, col in enumerate(columns):
        if col in panel_df.columns:
            ax = axes[i]
            data = panel_df[col].dropna()

            # Use histogram for continuous, bar for binary
            if data.nunique() <= 5:
                data.value_counts().sort_index().plot(kind='bar', ax=ax, color='steelblue')
            else:
                ax.hist(data, bins=50, color='steelblue', edgecolor='black', alpha=0.7)

            ax.set_title(col)
            ax.set_xlabel('')

    # Hide empty subplots
    for i in range(n_plots, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"  Saved: {save_path}")

    return fig


def create_summary_dashboard(
    panel_df: pd.DataFrame,
    topic_cols: List[str],
    save_dir: Optional[Path] = None,
) -> Dict[str, plt.Figure]:
    """
    Create a dashboard of summary plots.

    Parameters
    ----------
    panel_df : pd.DataFrame
        Panel data
    topic_cols : list
        Topic share column names
    save_dir : Path
        Directory to save figures

    Returns
    -------
    dict
        Dictionary of figure names to figures
    """
    figures = {}

    # Time series: Review count
    fig = plot_timeseries_by_treatment(
        panel_df,
        y_col="ReviewCount",
        title="Weekly Review Count by Treatment Group",
        ylabel="Average Review Count",
        save_path=save_dir / "ts_review_count_by_treatment.png" if save_dir else None,
    )
    figures["ts_review_count"] = fig

    # Time series: Verified share
    fig = plot_timeseries_by_treatment(
        panel_df,
        y_col="VerifiedShare",
        title="Verified Purchase Share by Treatment Group",
        ylabel="Share of Verified Purchases",
        save_path=save_dir / "ts_verified_share_by_treatment.png" if save_dir else None,
    )
    figures["ts_verified_share"] = fig

    # Time series: Average length
    fig = plot_timeseries_by_treatment(
        panel_df,
        y_col="AvgLen",
        title="Average Review Length by Treatment Group",
        ylabel="Characters",
        save_path=save_dir / "ts_avg_len_by_treatment.png" if save_dir else None,
    )
    figures["ts_avg_len"] = fig

    # Topic share correlation
    if topic_cols:
        from .analysis_fe import compute_correlation_matrix
        corr = compute_correlation_matrix(panel_df, topic_cols)
        fig = plot_correlation_heatmap(
            corr,
            title="Topic Share Correlations",
            save_path=save_dir / "topic_correlations.png" if save_dir else None,
        )
        figures["topic_correlations"] = fig

    return figures


if __name__ == "__main__":
    print("Plots module loaded. Run from notebook or script.")
