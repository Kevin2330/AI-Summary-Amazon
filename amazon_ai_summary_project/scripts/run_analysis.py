#!/usr/bin/env python3
"""
Script to run full analysis pipeline: FE regressions, DiD, and event studies.

Usage:
    python scripts/run_analysis.py [--category CATEGORY]

Example:
    python scripts/run_analysis.py --category diapers
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import config
from src.io_utils import load_panel
from src.analysis_fe import (
    compute_correlation_matrix,
    compute_vif,
    run_all_fe_regressions,
    format_regression_table,
    save_fe_results,
)
from src.analysis_did import (
    run_all_did_regressions,
    run_all_event_studies,
    format_did_table,
    format_parallel_trends_summary,
)
from src.plots import (
    plot_timeseries_by_treatment,
    plot_correlation_heatmap,
    plot_multiple_event_studies,
    plot_vif_bars,
    create_summary_dashboard,
)


def generate_memo(
    category: str,
    panel_df,
    fe_results_df,
    did_results_df,
    es_full_results,
    output_path: Path,
) -> None:
    """Generate summary memo markdown file."""

    memo_lines = []
    memo_lines.append("# Amazon AI Review Summary Analysis - Results Memo")
    memo_lines.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    memo_lines.append(f"\n**Category:** {category}")
    memo_lines.append(f"\n**AI Rollout Date:** {config.AI_ROLLOUT_DATE}")
    memo_lines.append(f"\n**Treatment Definition:** rating_number >= {config.TREATMENT_THRESHOLD}")

    # Panel construction summary
    memo_lines.append("\n## 1. Panel Construction")
    n_products = panel_df["parent_asin"].nunique()
    n_weeks = panel_df["week_start"].nunique()
    date_min = panel_df["week_start"].min().strftime('%Y-%m-%d')
    date_max = panel_df["week_start"].max().strftime('%Y-%m-%d')

    memo_lines.append(f"\n- **Products:** {n_products:,}")
    memo_lines.append(f"- **Weeks:** {n_weeks:,}")
    memo_lines.append(f"- **Product-weeks:** {len(panel_df):,}")
    memo_lines.append(f"- **Date range:** {date_min} to {date_max}")

    pre_weeks = panel_df[panel_df["post"] == 0]["week_start"].nunique()
    post_weeks = panel_df[panel_df["post"] == 1]["week_start"].nunique()
    memo_lines.append(f"- **Pre-period weeks:** {pre_weeks}")
    memo_lines.append(f"- **Post-period weeks:** {post_weeks}")

    treated = panel_df[panel_df["treated"] == 1]["parent_asin"].nunique()
    control = panel_df[panel_df["treated"] == 0]["parent_asin"].nunique()
    memo_lines.append(f"- **Treated products:** {treated:,}")
    memo_lines.append(f"- **Control products:** {control:,}")

    # FE Regression summary
    memo_lines.append("\n## 2. Two-Way Fixed Effects Regressions")
    memo_lines.append("\nBaseline specification: Y_it = beta' * TopicShares_it + entity_FE + time_FE + error_it")
    memo_lines.append("\nClustered standard errors by entity (product).")

    if not fe_results_df.empty:
        topic_cols = [c for c in config.TOPIC_KEYWORDS.keys()]
        for _, row in fe_results_df.iterrows():
            memo_lines.append(f"\n### {row['outcome']}")
            memo_lines.append(f"- N: {int(row['n_obs']):,}")
            memo_lines.append(f"- R2 (within): {row['r2_within']:.4f}")
            for topic in topic_cols:
                coef_key = f"{topic}_coef"
                pval_key = f"{topic}_pval"
                if coef_key in row and not pd.isna(row[coef_key]):
                    sig = "***" if row[pval_key] < 0.01 else "**" if row[pval_key] < 0.05 else "*" if row[pval_key] < 0.1 else ""
                    memo_lines.append(f"  - {topic}: {row[coef_key]:.4f}{sig}")

    # DiD summary
    memo_lines.append("\n## 3. Difference-in-Differences Results")
    memo_lines.append("\nSpecification: Y_it = delta * (treated_i * post_t) + entity_FE + time_FE + error_it")

    if not did_results_df.empty:
        for _, row in did_results_df.iterrows():
            memo_lines.append(f"\n### {row['outcome']}")
            memo_lines.append(f"- DiD estimate: {row['did_coef']:.4f} (SE: {row['did_se']:.4f}){row['significance']}")
            memo_lines.append(f"- t-statistic: {row['did_tstat']:.2f}")
            memo_lines.append(f"- p-value: {row['did_pval']:.4f}")

    # Event study and parallel trends
    memo_lines.append("\n## 4. Event Study & Parallel Trends")
    memo_lines.append(f"\nEvent window: +/- {config.EVENT_STUDY_WINDOW} weeks")
    memo_lines.append(f"Omitted period: {config.OMIT_PERIOD} (week before rollout)")

    if es_full_results:
        for outcome, result in es_full_results.items():
            memo_lines.append(f"\n### {outcome}")
            pt_test = result.get("parallel_trends_test")
            if pt_test:
                memo_lines.append(f"- **Parallel trends test:** F={pt_test['f_stat']:.2f}, p={pt_test['p_value']:.4f}")
                if pt_test["null_rejected"]:
                    memo_lines.append("- **Warning:** Parallel trends assumption may be violated (p < 0.05)")
                else:
                    memo_lines.append("- Parallel trends assumption not rejected at 5% level")

    # Caveats
    memo_lines.append("\n## 5. Caveats and Limitations")
    memo_lines.append("""
- **Treatment assignment proxy:** We use a threshold on pre-period review count (rating_number)
  to proxy for products likely to display AI summaries. This is an imperfect proxy as we don't
  observe actual AI summary deployment.

- **Limited post-period:** The UCSD Amazon Reviews 2023 dataset ends around September 2023,
  providing only ~4-6 weeks of post-rollout data. Long-term effects cannot be assessed.

- **Category-specific results:** Results are limited to the diaper category and may not
  generalize to other product categories.

- **Selection effects:** Products with high review counts (treated group) may differ
  systematically from low-review products in unobservable ways.

- **Endogenous review behavior:** Review timing and content may be influenced by factors
  correlated with the treatment that we cannot control for.
""")

    # Write memo
    with open(output_path, 'w') as f:
        f.write("\n".join(memo_lines))

    print(f"\n  Saved memo to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run analysis on product-week panel"
    )
    parser.add_argument(
        "--category",
        type=str,
        default=config.ACTIVE_CATEGORY,
        help=f"Product category (default: {config.ACTIVE_CATEGORY})"
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip generating plots"
    )

    args = parser.parse_args()
    paths = config.get_output_paths(args.category)

    print("\n" + "=" * 70)
    print("Amazon AI Summary Analysis - Full Analysis Pipeline")
    print("=" * 70)
    print(f"Category: {args.category}")
    print(f"AI rollout date: {config.AI_ROLLOUT_DATE}")
    print(f"Treatment threshold: {config.TREATMENT_THRESHOLD}")
    print("=" * 70 + "\n")

    # Load panel
    panel_path = paths["panel_parquet"]
    if not panel_path.exists():
        print(f"Panel file not found: {panel_path}")
        print("Run scripts/run_build_panel.py first.")
        sys.exit(1)

    panel_df = load_panel(panel_path)

    # Get topic columns
    topic_cols = list(config.TOPIC_KEYWORDS.keys())
    available_topics = [c for c in topic_cols if c in panel_df.columns]

    # =========================================================================
    # Part 1: Correlation and VIF
    # =========================================================================
    print("\n" + "=" * 70)
    print("Part 1: Multicollinearity Assessment")
    print("=" * 70)

    if available_topics:
        corr_matrix = compute_correlation_matrix(panel_df, available_topics)
        corr_matrix.to_csv(paths["correlation_matrix"])
        print(f"\n  Correlation matrix saved to {paths['correlation_matrix']}")
        print("\n  Topic Share Correlations:")
        print(corr_matrix.round(3).to_string())

        vif_df = compute_vif(panel_df, available_topics)
        vif_df.to_csv(paths["vif_table"], index=False)
        print(f"\n  VIF table saved to {paths['vif_table']}")
        print("\n  Variance Inflation Factors:")
        print(vif_df.to_string(index=False))

        if not args.skip_plots:
            plot_correlation_heatmap(
                corr_matrix,
                title="Topic Share Correlations",
                save_path=paths["correlation_heatmap"],
            )
            plot_vif_bars(
                vif_df,
                save_path=config.FIGURES_DIR / f"vif_{args.category}.png",
            )

    # =========================================================================
    # Part 2: Two-Way FE Regressions
    # =========================================================================
    print("\n" + "=" * 70)
    print("Part 2: Two-Way Fixed Effects Regressions")
    print("=" * 70)

    fe_results_df, fe_full_results = run_all_fe_regressions(
        panel_df,
        outcomes=config.PRIMARY_OUTCOMES,
        features=available_topics,
    )

    if not fe_results_df.empty:
        save_fe_results(
            fe_results_df,
            fe_full_results,
            paths["fe_results"],
            available_topics,
        )

    # =========================================================================
    # Part 3: Difference-in-Differences
    # =========================================================================
    print("\n" + "=" * 70)
    print("Part 3: Difference-in-Differences")
    print("=" * 70)

    did_results_df, did_full_results = run_all_did_regressions(
        panel_df,
        outcomes=config.PRIMARY_OUTCOMES,
    )

    if not did_results_df.empty:
        did_results_df.to_csv(paths["did_results"], index=False)
        print(f"\n  DiD results saved to {paths['did_results']}")
        print("\n" + format_did_table(did_results_df))

    # =========================================================================
    # Part 4: Event Study
    # =========================================================================
    print("\n" + "=" * 70)
    print("Part 4: Event Study Analysis")
    print("=" * 70)

    es_coef_dfs, es_full_results = run_all_event_studies(
        panel_df,
        outcomes=config.PRIMARY_OUTCOMES,
    )

    if es_coef_dfs:
        # Save coefficient tables
        for outcome, coef_df in es_coef_dfs.items():
            es_path = config.TABLES_DIR / f"event_study_coefs_{outcome}_{args.category}.csv"
            coef_df.to_csv(es_path, index=False)
            print(f"  Saved {outcome} event study coefficients to {es_path}")

        # Print parallel trends summary
        print(format_parallel_trends_summary(es_full_results))

        # Plot event studies
        if not args.skip_plots:
            plot_multiple_event_studies(
                es_coef_dfs,
                save_dir=config.FIGURES_DIR,
            )

    # =========================================================================
    # Part 5: Summary Plots
    # =========================================================================
    if not args.skip_plots:
        print("\n" + "=" * 70)
        print("Part 5: Summary Visualizations")
        print("=" * 70)

        create_summary_dashboard(
            panel_df,
            topic_cols=available_topics,
            save_dir=config.FIGURES_DIR,
        )

    # =========================================================================
    # Part 6: Generate Memo
    # =========================================================================
    print("\n" + "=" * 70)
    print("Part 6: Generating Results Memo")
    print("=" * 70)

    # Need pandas for the memo function
    import pandas as pd

    generate_memo(
        category=args.category,
        panel_df=panel_df,
        fe_results_df=fe_results_df,
        did_results_df=did_results_df,
        es_full_results=es_full_results,
        output_path=paths["memo"],
    )

    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)
    print(f"\nOutputs saved to: {config.OUTPUT_DIR}")
    print(f"  - Tables: {config.TABLES_DIR}")
    print(f"  - Figures: {config.FIGURES_DIR}")
    print(f"  - Memo: {paths['memo']}")


if __name__ == "__main__":
    main()
