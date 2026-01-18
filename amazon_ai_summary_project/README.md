# Amazon AI Review Summary Analysis

Reproducible Python project for analyzing the impact of Amazon's AI-generated review summaries on product reviews, using the UCSD Amazon Reviews 2023 dataset.

## Overview

This project builds a product-week panel from Amazon review data and runs:
- Two-way fixed effects (FE) regressions
- Difference-in-Differences (DiD) analysis
- Event study with parallel trends testing

**Key Event Date:** August 14, 2023 - Amazon AI review summary rollout

## Project Structure

```
amazon_ai_summary_project/
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── config.py         # Configuration and parameters
│   ├── io_utils.py       # Data loading utilities
│   ├── text_utils.py     # Text processing
│   ├── build_panel.py    # Panel construction
│   ├── analysis_fe.py    # Fixed effects regressions
│   ├── analysis_did.py   # DiD and event study
│   └── plots.py          # Visualization
├── scripts/
│   ├── run_build_panel.py   # Build panel script
│   └── run_analysis.py      # Full analysis script
├── notebooks/
│   ├── 01_build_panel_diapers.ipynb
│   ├── 02_eda_and_fe.ipynb
│   └── 03_did_event_study.ipynb
└── outputs/               # Generated outputs
    ├── figures/
    └── tables/
```

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# or: venv\Scripts\activate  # On Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Setup

You need two JSONL files from the UCSD Amazon Reviews 2023 dataset:

1. **Reviews file:** `Baby_Products.jsonl` (review-level data)
2. **Metadata file:** `meta_Baby_Products.jsonl` (product-level data)

### Option A: Set environment variable
```bash
export AMAZON_DATA_DIR=/path/to/your/data
```

### Option B: Modify config.py
Edit `src/config.py` and update the paths:
```python
REVIEWS_PATH = Path("/your/path/to/Baby_Products.jsonl")
META_PATH = Path("/your/path/to/meta_Baby_Products.jsonl")
```

## Usage

### Quick Start (Scripts)

1. **Build the panel:**
```bash
python scripts/run_build_panel.py --category diapers
```

2. **Run full analysis:**
```bash
python scripts/run_analysis.py --category diapers
```

### Interactive Analysis (Notebooks)

1. Launch Jupyter:
```bash
jupyter notebook
```

2. Run notebooks in order:
   - `01_build_panel_diapers.ipynb` - Build product-week panel
   - `02_eda_and_fe.ipynb` - EDA and fixed effects
   - `03_did_event_study.ipynb` - DiD and event study

## Configuration

All parameters are centralized in `src/config.py`:

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `AI_ROLLOUT_DATE` | 2023-08-14 | Amazon AI summary rollout date |
| `TREATMENT_THRESHOLD` | 50 | Min rating_number for treatment |
| `EVENT_STUDY_WINDOW` | 8 | +/- weeks for event study |
| `MIN_CELL_SIZE` | 30 | Min observations per event bin |

### Topic Keywords

Customize topic detection by editing `TOPIC_KEYWORDS`:

```python
TOPIC_KEYWORDS = {
    "ValueShare": ["value", "price", "cheap", ...],
    "EffectShare": ["effective", "works", "absorb", ...],
    "ComfortShare": ["soft", "comfortable", ...],
    "FitShare": ["fit", "size", "tight", ...],
    "DesignShare": ["cute", "design", "color", ...],
}
```

### Adding New Categories

1. Add keywords to `KEYWORD_GROUPS`:
```python
KEYWORD_GROUPS = {
    "diapers": ["diaper"],
    "formula": ["formula", "infant formula"],  # Add new category
}
```

2. Set active category:
```python
ACTIVE_CATEGORY = "formula"
```

## Outputs

After running the analysis, outputs are saved to `outputs/`:

### Tables (`outputs/tables/`)
- `summary_stats_diapers.csv` - Summary statistics
- `correlation_matrix_diapers.csv` - Topic correlations
- `vif_diapers.csv` - Variance Inflation Factors
- `fe_regression_results_diapers.csv` - FE regression results
- `did_results_diapers.csv` - DiD estimates
- `event_study_coefs_*.csv` - Event study coefficients

### Figures (`outputs/figures/`)
- Time series plots
- Correlation heatmaps
- Event study plots

### Panel Data
- `panel_diapers_week.parquet` - Main panel (Parquet)
- `panel_diapers_week.csv` - Main panel (CSV)

### Memo
- `memo_diapers.md` - Summary of results

## Methodology

### Panel Construction

1. Filter products by keywords in title
2. Stream reviews (memory-efficient)
3. Extract topic mentions using regex
4. Aggregate to product-week level

### Outcome Variables

| Variable | Description |
|----------|-------------|
| `ReviewCount` | Number of reviews |
| `logReviewCount` | log(1 + ReviewCount) |
| `UniqueReviewers` | Unique user IDs |
| `AvgRating` | Mean rating |
| `RatingDisp` | Rating std deviation |
| `VerifiedShare` | % verified purchases |
| `AvgHelpful` | Mean helpful votes |
| `AvgLen` | Mean text length |
| `ImageShare` | % with images |

### Treatment Definition

Products are classified as "treated" based on eligibility proxy:
- `treated = 1` if `rating_number >= TREATMENT_THRESHOLD`
- `treated = 0` otherwise

This proxies for products likely to display AI summaries.

### Fixed Effects Specification

```
Y_it = β' * TopicShares_it + α_i + γ_t + ε_it
```
- Entity FE: `α_i` (product)
- Time FE: `γ_t` (week)
- Clustered SE by entity

### DiD Specification

```
Y_it = δ * (Treated_i × Post_t) + α_i + γ_t + ε_it
```
- `δ` = Average treatment effect on treated

### Event Study Specification

```
Y_it = Σ_k γ_k * (Treated_i × 1[t=k]) + α_i + γ_t + ε_it
```
- `k` = weeks relative to rollout
- `k = -1` omitted (baseline)
- Pre-period coefficients test parallel trends

## Caveats

1. **Treatment proxy:** We don't observe actual AI summary deployment
2. **Limited post-period:** Data ends ~September 2023 (~4-6 post weeks)
3. **Category-specific:** Results limited to diaper products
4. **Selection:** High-review products may differ systematically

## Troubleshooting

### Memory Issues
The streaming approach handles large files, but if you encounter memory issues:
- Reduce data by filtering dates in config
- Process in smaller chunks

### Collinearity Errors
If event study fails with absorbed effects:
- Increase `MIN_CELL_SIZE` in config
- Reduce `EVENT_STUDY_WINDOW`

### Missing Dependencies
```bash
pip install --upgrade linearmodels statsmodels
```

## Citation

Data source:
```
McAuley, J., et al. (2023). Amazon Reviews 2023.
https://amazon-reviews-2023.github.io/
```

## License

This project is for research purposes.
