"""
Panel construction module for Amazon review data.

Builds product-week panel from raw JSONL review and metadata files.
Implements streaming processing for memory efficiency.
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from . import config
from .io_utils import (
    load_meta_jsonl,
    filter_products_by_keywords,
    stream_reviews_jsonl,
    timestamp_to_datetime,
    save_panel,
    print_data_summary,
)
from .text_utils import TopicExtractor, clean_text


def get_week_start_from_timestamp(ts_ms: int) -> datetime:
    """
    Convert timestamp (ms) to week start (Monday 00:00:00).

    Parameters
    ----------
    ts_ms : int
        Unix timestamp in milliseconds

    Returns
    -------
    datetime
        Monday of the week containing the timestamp
    """
    dt = timestamp_to_datetime(ts_ms)
    # Find Monday of this week
    days_since_monday = dt.weekday()
    monday = dt - timedelta(days=days_since_monday)
    return monday.replace(hour=0, minute=0, second=0, microsecond=0)


class PanelAggregator:
    """
    Incrementally aggregates review data to product-week level.

    Uses dictionaries to accumulate statistics without loading all reviews.

    Parameters
    ----------
    topic_names : list
        Names of topic attributes to track
    """

    def __init__(self, topic_names: List[str]):
        self.topic_names = topic_names

        # Aggregation dictionaries
        # Key: (parent_asin, week_start)
        self.review_count: Dict[Tuple, int] = defaultdict(int)
        self.user_ids: Dict[Tuple, set] = defaultdict(set)
        self.rating_sum: Dict[Tuple, float] = defaultdict(float)
        self.rating_sq_sum: Dict[Tuple, float] = defaultdict(float)
        self.verified_sum: Dict[Tuple, int] = defaultdict(int)
        self.helpful_sum: Dict[Tuple, int] = defaultdict(int)
        self.text_len_sum: Dict[Tuple, int] = defaultdict(int)
        self.image_sum: Dict[Tuple, int] = defaultdict(int)

        # Topic sums (for computing shares)
        self.topic_sums: Dict[str, Dict[Tuple, int]] = {
            name: defaultdict(int) for name in topic_names
        }

        self.n_processed = 0

    def add_review(
        self,
        parent_asin: str,
        week_start: datetime,
        rating: float,
        verified: int,
        helpful: int,
        text_len: int,
        has_image: int,
        user_id: str,
        topic_flags: Dict[str, int]
    ) -> None:
        """Add a single review to the aggregator."""
        key = (parent_asin, week_start)

        self.review_count[key] += 1
        self.user_ids[key].add(user_id)
        self.rating_sum[key] += rating
        self.rating_sq_sum[key] += rating ** 2
        self.verified_sum[key] += verified
        self.helpful_sum[key] += helpful
        self.text_len_sum[key] += text_len
        self.image_sum[key] += has_image

        for name in self.topic_names:
            self.topic_sums[name][key] += topic_flags.get(name, 0)

        self.n_processed += 1

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert aggregated data to DataFrame.

        Returns
        -------
        pd.DataFrame
            Product-week panel with computed statistics
        """
        print(f"Building panel from {len(self.review_count):,} product-weeks...")
        records = []

        for key in self.review_count.keys():
            parent_asin, week_start = key
            n = self.review_count[key]

            if n == 0:
                continue

            # Compute mean rating and dispersion
            mean_rating = self.rating_sum[key] / n
            # Variance = E[X^2] - E[X]^2
            if n > 1:
                variance = (self.rating_sq_sum[key] / n) - (mean_rating ** 2)
                # Correct for floating point errors
                variance = max(0, variance)
                rating_disp = np.sqrt(variance)
            else:
                rating_disp = 0.0

            record = {
                "parent_asin": parent_asin,
                "week_start": week_start,
                "ReviewCount": n,
                "UniqueReviewers": len(self.user_ids[key]),
                "AvgRating": mean_rating,
                "RatingDisp": rating_disp,
                "VerifiedShare": self.verified_sum[key] / n,
                "AvgHelpful": self.helpful_sum[key] / n,
                "AvgLen": self.text_len_sum[key] / n,
                "ImageShare": self.image_sum[key] / n,
            }

            # Add topic shares
            for name in self.topic_names:
                record[name] = self.topic_sums[name][key] / n

            records.append(record)

        df = pd.DataFrame(records)

        # Add log review count
        df["logReviewCount"] = np.log1p(df["ReviewCount"])

        # Sort and set index
        df = df.sort_values(["parent_asin", "week_start"])

        return df


def build_product_week_panel(
    reviews_path: Path,
    meta_path: Path,
    keywords: List[str],
    topic_keywords: Dict[str, List[str]],
    output_parquet: Optional[Path] = None,
    output_csv: Optional[Path] = None,
    save_filtered_reviews: bool = False,
    filtered_reviews_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Build product-week panel from raw JSONL files.

    Main pipeline function that:
    1. Loads metadata and filters by keywords
    2. Streams reviews, filtering by target products
    3. Extracts features and aggregates to product-week
    4. Merges product metadata
    5. Saves output

    Parameters
    ----------
    reviews_path : Path
        Path to reviews JSONL
    meta_path : Path
        Path to metadata JSONL
    keywords : list
        Keywords to filter products by title
    topic_keywords : dict
        Topic name -> keyword list mapping
    output_parquet : Path, optional
        Path to save Parquet output
    output_csv : Path, optional
        Path to save CSV output
    save_filtered_reviews : bool
        Whether to save filtered reviews to disk
    filtered_reviews_path : Path, optional
        Path for filtered reviews

    Returns
    -------
    pd.DataFrame
        Product-week panel
    """
    print("=" * 70)
    print("Building Product-Week Panel")
    print("=" * 70)

    # Step 1: Load metadata
    print("\n[Step 1/5] Loading product metadata...")
    meta_cols = [
        "parent_asin", "title", "store", "price",
        "average_rating", "rating_number", "categories", "main_category"
    ]
    meta_df = load_meta_jsonl(meta_path, columns=meta_cols)

    # Step 2: Filter products by keywords
    print("\n[Step 2/5] Filtering products by keywords...")
    target_asins = filter_products_by_keywords(meta_df, keywords)

    if len(target_asins) == 0:
        raise ValueError(f"No products found matching keywords: {keywords}")

    # Filter meta to target products
    meta_df = meta_df[meta_df["parent_asin"].isin(target_asins)].copy()
    print(f"  Kept {len(meta_df):,} products in metadata")

    # Step 3: Initialize topic extractor and aggregator
    print("\n[Step 3/5] Processing reviews...")
    topic_extractor = TopicExtractor(topic_keywords)
    aggregator = PanelAggregator(topic_extractor.get_topic_names())

    # Optional: collect filtered reviews for saving
    filtered_reviews = [] if save_filtered_reviews else None

    # Stream reviews and aggregate
    for record in stream_reviews_jsonl(reviews_path, target_asins):
        # Extract timestamp and week
        timestamp = record.get("timestamp")
        if timestamp is None:
            continue

        week_start = get_week_start_from_timestamp(timestamp)

        # Extract text features
        text = record.get("text", "") or ""
        title = record.get("title", "") or ""
        full_text = f"{title} {text}"
        cleaned_text = clean_text(full_text)
        text_len = len(cleaned_text)

        # Extract topic flags
        topic_flags = topic_extractor.extract_topic_flags(full_text)

        # Check for images
        images = record.get("images", []) or []
        has_image = 1 if len(images) > 0 else 0

        # Add to aggregator
        aggregator.add_review(
            parent_asin=record.get("parent_asin"),
            week_start=week_start,
            rating=record.get("rating", 0) or 0,
            verified=1 if record.get("verified_purchase") else 0,
            helpful=record.get("helpful_vote", 0) or 0,
            text_len=text_len,
            has_image=has_image,
            user_id=record.get("user_id", ""),
            topic_flags=topic_flags,
        )

        # Save filtered review if requested
        if filtered_reviews is not None:
            filtered_reviews.append({
                "parent_asin": record.get("parent_asin"),
                "asin": record.get("asin"),
                "user_id": record.get("user_id"),
                "timestamp": timestamp,
                "week_start": week_start,
                "rating": record.get("rating"),
                "helpful_vote": record.get("helpful_vote", 0),
                "verified_purchase": record.get("verified_purchase"),
                "text_len": text_len,
                "has_image": has_image,
                **topic_flags,
            })

    print(f"  Processed {aggregator.n_processed:,} reviews")

    # Save filtered reviews if requested
    if save_filtered_reviews and filtered_reviews:
        print(f"\n  Saving filtered reviews...")
        reviews_df = pd.DataFrame(filtered_reviews)
        if filtered_reviews_path:
            reviews_df.to_parquet(filtered_reviews_path)
            print(f"  Saved {len(reviews_df):,} reviews to {filtered_reviews_path}")

    # Step 4: Build panel DataFrame
    print("\n[Step 4/5] Aggregating to product-week panel...")
    panel_df = aggregator.to_dataframe()

    # Merge metadata
    print("\n  Merging product metadata...")
    panel_df = panel_df.merge(
        meta_df[["parent_asin", "title", "store", "price",
                 "average_rating", "rating_number"]],
        on="parent_asin",
        how="left"
    )

    # Add time-related variables
    panel_df["week_start"] = pd.to_datetime(panel_df["week_start"])
    panel_df["year"] = panel_df["week_start"].dt.year
    panel_df["month"] = panel_df["week_start"].dt.month
    panel_df["week_of_year"] = panel_df["week_start"].dt.isocalendar().week

    # Add treatment indicators
    ai_rollout = pd.Timestamp(config.AI_ROLLOUT_DATE)
    panel_df["post"] = (panel_df["week_start"] >= ai_rollout).astype(int)

    # Treatment based on rating_number threshold
    panel_df["treated"] = (
        panel_df["rating_number"].fillna(0) >= config.TREATMENT_THRESHOLD
    ).astype(int)
    panel_df["treated_post"] = panel_df["treated"] * panel_df["post"]

    # Event time (weeks relative to rollout)
    panel_df["event_time"] = (
        (panel_df["week_start"] - ai_rollout).dt.days / 7
    ).round().astype(int)

    # Step 5: Save outputs
    print("\n[Step 5/5] Saving outputs...")
    if output_parquet:
        save_panel(panel_df, output_parquet, output_csv)

    # Print summary
    print_panel_summary(panel_df)

    return panel_df


def print_panel_summary(panel_df: pd.DataFrame) -> None:
    """Print summary statistics for the panel."""
    print("\n" + "=" * 70)
    print("Panel Summary")
    print("=" * 70)

    n_products = panel_df["parent_asin"].nunique()
    n_weeks = panel_df["week_start"].nunique()
    n_rows = len(panel_df)

    print(f"  Number of products: {n_products:,}")
    print(f"  Number of weeks: {n_weeks:,}")
    print(f"  Number of rows (product-weeks): {n_rows:,}")

    print(f"\n  Date range: {panel_df['week_start'].min()} to {panel_df['week_start'].max()}")

    # Treatment summary
    ai_rollout = pd.Timestamp(config.AI_ROLLOUT_DATE)
    pre_weeks = panel_df[panel_df["week_start"] < ai_rollout]["week_start"].nunique()
    post_weeks = panel_df[panel_df["week_start"] >= ai_rollout]["week_start"].nunique()
    print(f"\n  Pre-period weeks: {pre_weeks}")
    print(f"  Post-period weeks: {post_weeks}")

    # Treatment group summary
    treated_products = panel_df[panel_df["treated"] == 1]["parent_asin"].nunique()
    control_products = panel_df[panel_df["treated"] == 0]["parent_asin"].nunique()
    print(f"\n  Treated products (rating_number >= {config.TREATMENT_THRESHOLD}): {treated_products:,}")
    print(f"  Control products (rating_number < {config.TREATMENT_THRESHOLD}): {control_products:,}")

    # Outcome summary
    print(f"\n  Review count: mean={panel_df['ReviewCount'].mean():.2f}, "
          f"median={panel_df['ReviewCount'].median():.0f}, "
          f"total={panel_df['ReviewCount'].sum():,}")

    print(f"  Avg rating: mean={panel_df['AvgRating'].mean():.2f}")
    print(f"  Verified share: mean={panel_df['VerifiedShare'].mean():.2%}")

    # Topic share summary
    topic_cols = [c for c in panel_df.columns if c.endswith("Share") and c != "VerifiedShare" and c != "ImageShare"]
    if topic_cols:
        print(f"\n  Topic shares (means):")
        for col in topic_cols:
            print(f"    {col}: {panel_df[col].mean():.2%}")

    print("=" * 70 + "\n")


def build_panel_from_config(category: str = None) -> pd.DataFrame:
    """
    Build panel using configuration settings.

    Parameters
    ----------
    category : str, optional
        Category to build panel for. Uses ACTIVE_CATEGORY if not specified.

    Returns
    -------
    pd.DataFrame
        Product-week panel
    """
    cat = category or config.ACTIVE_CATEGORY
    keywords = config.KEYWORD_GROUPS.get(cat, [])

    if not keywords:
        raise ValueError(f"No keywords defined for category: {cat}")

    paths = config.get_output_paths(cat)

    panel = build_product_week_panel(
        reviews_path=config.REVIEWS_PATH,
        meta_path=config.META_PATH,
        keywords=keywords,
        topic_keywords=config.TOPIC_KEYWORDS,
        output_parquet=paths["panel_parquet"],
        output_csv=paths["panel_csv"],
        save_filtered_reviews=config.SAVE_FILTERED_REVIEWS,
        filtered_reviews_path=paths["filtered_reviews"],
    )

    return panel


if __name__ == "__main__":
    # Run panel building when executed directly
    import sys

    category = sys.argv[1] if len(sys.argv) > 1 else config.ACTIVE_CATEGORY
    print(f"Building panel for category: {category}")

    if not config.validate_config():
        print("Configuration validation failed. Please check paths.")
        sys.exit(1)

    panel = build_panel_from_config(category)
    print("Panel building complete.")
