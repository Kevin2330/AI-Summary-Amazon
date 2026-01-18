#!/usr/bin/env python3
"""
Script to build the product-week panel from raw JSONL data.

Usage:
    python scripts/run_build_panel.py [--category CATEGORY]

Example:
    python scripts/run_build_panel.py --category diapers
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import config
from src.build_panel import build_panel_from_config


def main():
    parser = argparse.ArgumentParser(
        description="Build product-week panel from Amazon review data"
    )
    parser.add_argument(
        "--category",
        type=str,
        default=config.ACTIVE_CATEGORY,
        help=f"Product category to process (default: {config.ACTIVE_CATEGORY})"
    )
    parser.add_argument(
        "--reviews-path",
        type=str,
        default=None,
        help="Override path to reviews JSONL"
    )
    parser.add_argument(
        "--meta-path",
        type=str,
        default=None,
        help="Override path to meta JSONL"
    )

    args = parser.parse_args()

    # Override paths if provided
    if args.reviews_path:
        config.REVIEWS_PATH = Path(args.reviews_path)
    if args.meta_path:
        config.META_PATH = Path(args.meta_path)

    # Validate configuration
    print("\n" + "=" * 70)
    print("Amazon AI Summary Analysis - Panel Builder")
    print("=" * 70)
    print(f"Category: {args.category}")
    print(f"Reviews path: {config.REVIEWS_PATH}")
    print(f"Meta path: {config.META_PATH}")
    print(f"Output directory: {config.OUTPUT_DIR}")
    print("=" * 70 + "\n")

    if not config.validate_config():
        print("\nConfiguration validation failed. Please check paths.")
        print("Set AMAZON_DATA_DIR environment variable or update config.py")
        sys.exit(1)

    # Build panel
    try:
        panel = build_panel_from_config(args.category)
        print("\nPanel building completed successfully!")

        paths = config.get_output_paths(args.category)
        print(f"\nOutput files:")
        print(f"  - {paths['panel_parquet']}")
        print(f"  - {paths['panel_csv']}")
        if config.SAVE_FILTERED_REVIEWS:
            print(f"  - {paths['filtered_reviews']}")

    except Exception as e:
        print(f"\nError building panel: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
