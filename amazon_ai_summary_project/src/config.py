"""
Configuration module for Amazon AI Summary Analysis project.

Edit paths and parameters here. All other modules import from this file.
"""

import os
from pathlib import Path
from datetime import datetime, date
from typing import Dict, List

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Base paths - modify these to match your local setup
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = Path(os.getenv("AMAZON_DATA_DIR", PROJECT_ROOT.parent))

# Input data paths - UPDATE THESE TO YOUR LOCAL PATHS
REVIEWS_PATH = DATA_DIR / "Baby_Products.jsonl"
META_PATH = DATA_DIR / "meta_Baby_Products.jsonl"

# Output paths
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"

# Ensure output directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# PRODUCT CATEGORY FILTERS
# =============================================================================

# Keywords for filtering products by title (case-insensitive regex match)
# Add more category groups as needed
KEYWORD_GROUPS: Dict[str, List[str]] = {
    "diapers": ["diaper"],
    # Future categories - uncomment and customize:
    # "formula": ["formula", "infant formula"],
    # "wipes": ["wipe", "wipes"],
    # "bottles": ["bottle", "bottles", "nipple"],
    # "strollers": ["stroller", "pram"],
}

# Active category for current analysis
ACTIVE_CATEGORY = "diapers"

# =============================================================================
# ATTRIBUTE TOPIC DICTIONARIES
# =============================================================================

# Each topic maps to a list of keywords for regex matching (case-insensitive)
# Modify these dictionaries to customize attribute detection
TOPIC_KEYWORDS: Dict[str, List[str]] = {
    "ValueShare": [
        "value", "price", "cheap", "worth", "money", "expensive",
        "affordable", "cost", "budget", "deal", "bargain", "overpriced"
    ],
    "EffectShare": [
        "effective", "works", "absorb", "leak", "hold", "performance",
        "absorbent", "absorption", "absorbency", "leaking", "leakage",
        "overnight", "protection", "dry", "wetness"
    ],
    "ComfortShare": [
        "soft", "comfortable", "gentle", "rash", "irritation", "skin",
        "sensitive", "diaper rash", "comfort", "breathable", "chemical"
    ],
    "FitShare": [
        "fit", "size", "tight", "loose", "snug", "waist", "leg",
        "sizing", "fits", "stretchy", "elastic", "band", "tabs"
    ],
    "DesignShare": [
        "cute", "design", "color", "pattern", "look", "style",
        "print", "character", "aesthetic", "adorable"
    ],
}

# =============================================================================
# TIME CONFIGURATION
# =============================================================================

# Amazon AI Review Summary rollout date
AI_ROLLOUT_DATE = date(2023, 8, 14)
AI_ROLLOUT_DATETIME = datetime(2023, 8, 14)

# Analysis time window (optional filtering)
# Set to None to use all available data
ANALYSIS_START_DATE = None  # e.g., date(2022, 1, 1)
ANALYSIS_END_DATE = None    # e.g., date(2023, 12, 31)

# Event study window (weeks before/after rollout)
EVENT_STUDY_WINDOW = 8  # +/- 8 weeks from rollout

# =============================================================================
# DiD CONFIGURATION
# =============================================================================

# Treatment definition: products with rating_number >= threshold are treated
# This proxies for products likely to display AI summaries
TREATMENT_THRESHOLD = 50  # Minimum pre-period review count for treatment

# Alternative treatment thresholds to test robustness
TREATMENT_THRESHOLDS_ROBUSTNESS = [25, 50, 100, 200]

# =============================================================================
# EVENT STUDY CONFIGURATION
# =============================================================================

# Minimum cell size for event study bins
# Bins with fewer than this many observations in either treatment group are dropped
MIN_CELL_SIZE = 30

# Omitted period (baseline) for event study
OMIT_PERIOD = -1

# =============================================================================
# OUTCOME VARIABLES
# =============================================================================

# Primary outcome variables for regressions
PRIMARY_OUTCOMES = [
    "logReviewCount",
    "VerifiedShare",
    "AvgHelpful",
    "AvgLen",
]

# All outcome variables computed in panel
ALL_OUTCOMES = [
    "ReviewCount",
    "UniqueReviewers",
    "AvgRating",
    "RatingDisp",
    "VerifiedShare",
    "AvgHelpful",
    "AvgLen",
    "ImageShare",
    "logReviewCount",
]

# =============================================================================
# PROCESSING CONFIGURATION
# =============================================================================

# Chunk size for streaming JSONL (number of lines)
CHUNK_SIZE = 10000

# Whether to save intermediate filtered reviews
SAVE_FILTERED_REVIEWS = True

# Random seed for reproducibility
RANDOM_SEED = 42

# =============================================================================
# OUTPUT FILE NAMES
# =============================================================================

def get_output_paths(category: str = None):
    """Get output file paths for a given category."""
    cat = category or ACTIVE_CATEGORY
    return {
        "panel_parquet": OUTPUT_DIR / f"panel_{cat}_week.parquet",
        "panel_csv": OUTPUT_DIR / f"panel_{cat}_week.csv",
        "filtered_reviews": OUTPUT_DIR / f"filtered_reviews_{cat}.parquet",
        "summary_stats": TABLES_DIR / f"summary_stats_{cat}.csv",
        "correlation_matrix": TABLES_DIR / f"correlation_matrix_{cat}.csv",
        "vif_table": TABLES_DIR / f"vif_{cat}.csv",
        "fe_results": TABLES_DIR / f"fe_regression_results_{cat}.csv",
        "did_results": TABLES_DIR / f"did_results_{cat}.csv",
        "event_study_results": TABLES_DIR / f"event_study_results_{cat}.csv",
        "memo": OUTPUT_DIR / f"memo_{cat}.md",
        # Figures
        "timeseries_review_count": FIGURES_DIR / f"timeseries_review_count_{cat}.png",
        "timeseries_verified_share": FIGURES_DIR / f"timeseries_verified_share_{cat}.png",
        "correlation_heatmap": FIGURES_DIR / f"correlation_heatmap_{cat}.png",
        "event_study_plot": FIGURES_DIR / f"event_study_{cat}.png",
    }


# =============================================================================
# VALIDATION
# =============================================================================

def validate_config():
    """Validate configuration and print warnings if needed."""
    warnings = []

    if not REVIEWS_PATH.exists():
        warnings.append(f"Reviews file not found: {REVIEWS_PATH}")
    if not META_PATH.exists():
        warnings.append(f"Meta file not found: {META_PATH}")

    if ACTIVE_CATEGORY not in KEYWORD_GROUPS:
        warnings.append(f"Active category '{ACTIVE_CATEGORY}' not in KEYWORD_GROUPS")

    if warnings:
        print("Configuration warnings:")
        for w in warnings:
            print(f"  - {w}")
        return False
    return True


if __name__ == "__main__":
    # Print config summary when run directly
    print("=" * 60)
    print("Amazon AI Summary Analysis - Configuration")
    print("=" * 60)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Reviews path: {REVIEWS_PATH}")
    print(f"Meta path: {META_PATH}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Active category: {ACTIVE_CATEGORY}")
    print(f"Keywords: {KEYWORD_GROUPS.get(ACTIVE_CATEGORY, [])}")
    print(f"AI rollout date: {AI_ROLLOUT_DATE}")
    print(f"Treatment threshold: {TREATMENT_THRESHOLD}")
    print(f"Topic attributes: {list(TOPIC_KEYWORDS.keys())}")
    print("=" * 60)
    validate_config()
