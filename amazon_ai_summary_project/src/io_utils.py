"""
I/O utilities for loading and streaming Amazon review data.

Implements memory-efficient streaming for large JSONL files.
"""

import json
import re
from pathlib import Path
from typing import Dict, Generator, List, Optional, Set, Callable
from datetime import datetime

import pandas as pd


def load_meta_jsonl(
    meta_path: Path,
    columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Load product metadata from JSONL file into DataFrame.

    Meta files are typically smaller, so we load fully into memory.

    Parameters
    ----------
    meta_path : Path
        Path to meta JSONL file
    columns : list, optional
        Columns to keep. If None, keeps all.

    Returns
    -------
    pd.DataFrame
        Product metadata indexed by parent_asin
    """
    print(f"Loading metadata from {meta_path}...")
    records = []

    with open(meta_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    record = json.loads(line)
                    records.append(record)
                except json.JSONDecodeError as e:
                    print(f"  Warning: JSON decode error on line {line_num}: {e}")
                    continue

            if line_num % 50000 == 0:
                print(f"  Processed {line_num:,} lines...")

    df = pd.DataFrame(records)
    print(f"  Loaded {len(df):,} products")

    # Filter columns if specified
    if columns is not None:
        available_cols = [c for c in columns if c in df.columns]
        df = df[available_cols]

    return df


def filter_products_by_keywords(
    meta_df: pd.DataFrame,
    keywords: List[str],
    title_col: str = "title"
) -> Set[str]:
    """
    Filter products by keyword match in title.

    Parameters
    ----------
    meta_df : pd.DataFrame
        Product metadata DataFrame
    keywords : list
        Keywords to match (case-insensitive regex)
    title_col : str
        Name of title column

    Returns
    -------
    set
        Set of parent_asin values for matching products
    """
    if title_col not in meta_df.columns:
        raise ValueError(f"Column '{title_col}' not found in metadata")

    # Build regex pattern for all keywords (case-insensitive)
    pattern = '|'.join([re.escape(kw) for kw in keywords])
    regex = re.compile(pattern, re.IGNORECASE)

    # Handle missing titles
    mask = meta_df[title_col].fillna("").apply(lambda x: bool(regex.search(x)))

    # Get parent_asin for matching products
    if "parent_asin" not in meta_df.columns:
        raise ValueError("Column 'parent_asin' not found in metadata")

    matching_asins = set(meta_df.loc[mask, "parent_asin"].dropna().unique())

    print(f"  Found {len(matching_asins):,} products matching keywords: {keywords}")

    return matching_asins


def stream_reviews_jsonl(
    reviews_path: Path,
    target_asins: Optional[Set[str]] = None,
    chunk_size: int = 10000,
    progress_interval: int = 100000
) -> Generator[Dict, None, None]:
    """
    Stream reviews from JSONL file, optionally filtering by parent_asin.

    Memory-efficient generator that yields one review dict at a time.

    Parameters
    ----------
    reviews_path : Path
        Path to reviews JSONL file
    target_asins : set, optional
        Set of parent_asin values to keep. If None, yields all.
    chunk_size : int
        Not used directly (for future batch processing)
    progress_interval : int
        Print progress every N lines

    Yields
    ------
    dict
        Review record
    """
    print(f"Streaming reviews from {reviews_path}...")
    total_read = 0
    kept = 0

    with open(reviews_path, 'r', encoding='utf-8') as f:
        for line in f:
            total_read += 1

            if line.strip():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Filter by parent_asin if target set provided
                if target_asins is not None:
                    parent_asin = record.get("parent_asin")
                    if parent_asin not in target_asins:
                        continue

                kept += 1
                yield record

            if total_read % progress_interval == 0:
                if target_asins:
                    print(f"  Read {total_read:,} lines, kept {kept:,} reviews...")
                else:
                    print(f"  Read {total_read:,} lines...")

    print(f"  Finished: read {total_read:,} total, kept {kept:,} reviews")


def stream_reviews_to_dataframe(
    reviews_path: Path,
    target_asins: Optional[Set[str]] = None,
    columns: Optional[List[str]] = None,
    max_rows: Optional[int] = None,
    transform_fn: Optional[Callable[[Dict], Dict]] = None
) -> pd.DataFrame:
    """
    Stream reviews into a DataFrame with optional filtering and transformation.

    Parameters
    ----------
    reviews_path : Path
        Path to reviews JSONL
    target_asins : set, optional
        Filter to these parent_asin values
    columns : list, optional
        Columns to keep
    max_rows : int, optional
        Maximum rows to return (for testing)
    transform_fn : callable, optional
        Function to transform each record dict

    Returns
    -------
    pd.DataFrame
        Reviews DataFrame
    """
    records = []

    for record in stream_reviews_jsonl(reviews_path, target_asins):
        if transform_fn:
            record = transform_fn(record)

        if columns:
            record = {k: v for k, v in record.items() if k in columns}

        records.append(record)

        if max_rows and len(records) >= max_rows:
            print(f"  Reached max_rows limit ({max_rows})")
            break

    return pd.DataFrame(records)


def timestamp_to_datetime(ts_ms: int) -> datetime:
    """Convert millisecond timestamp to datetime."""
    return datetime.fromtimestamp(ts_ms / 1000.0)


def timestamp_to_date(ts_ms: int) -> datetime:
    """Convert millisecond timestamp to date."""
    return timestamp_to_datetime(ts_ms).date()


def get_week_start(dt: datetime) -> datetime:
    """
    Get Monday of the week containing the given datetime.

    Parameters
    ----------
    dt : datetime
        Input datetime

    Returns
    -------
    datetime
        Monday 00:00:00 of that week
    """
    # Monday is weekday 0
    days_since_monday = dt.weekday()
    monday = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    monday = monday - pd.Timedelta(days=days_since_monday)
    return monday


def save_panel(
    panel_df: pd.DataFrame,
    parquet_path: Path,
    csv_path: Optional[Path] = None
) -> None:
    """
    Save panel DataFrame to Parquet and optionally CSV.

    Parameters
    ----------
    panel_df : pd.DataFrame
        Panel data
    parquet_path : Path
        Path for Parquet output
    csv_path : Path, optional
        Path for CSV output
    """
    print(f"Saving panel to {parquet_path}...")
    panel_df.to_parquet(parquet_path, index=True)

    if csv_path:
        print(f"Saving panel to {csv_path}...")
        panel_df.to_csv(csv_path, index=True)

    print(f"  Saved {len(panel_df):,} rows")


def load_panel(
    parquet_path: Path
) -> pd.DataFrame:
    """
    Load panel DataFrame from Parquet.

    Parameters
    ----------
    parquet_path : Path
        Path to Parquet file

    Returns
    -------
    pd.DataFrame
        Panel data
    """
    print(f"Loading panel from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    print(f"  Loaded {len(df):,} rows")
    return df


def print_data_summary(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """Print summary statistics for a DataFrame."""
    print(f"\n{'='*60}")
    print(f"Summary: {name}")
    print(f"{'='*60}")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")

    if "parent_asin" in df.columns or df.index.name == "parent_asin":
        n_products = df.index.get_level_values("parent_asin").nunique() if isinstance(df.index, pd.MultiIndex) else df["parent_asin"].nunique() if "parent_asin" in df.columns else len(df)
        print(f"  Unique products: {n_products:,}")

    if "week_start" in df.columns or (isinstance(df.index, pd.MultiIndex) and "week_start" in df.index.names):
        if isinstance(df.index, pd.MultiIndex):
            weeks = df.index.get_level_values("week_start")
        else:
            weeks = df["week_start"]
        print(f"  Unique weeks: {weeks.nunique():,}")
        print(f"  Date range: {weeks.min()} to {weeks.max()}")

    print(f"{'='*60}\n")
