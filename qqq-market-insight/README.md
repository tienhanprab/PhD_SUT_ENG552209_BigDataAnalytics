# Machine Learning Analysis of Technical Indicators and Market Trends in the Invesco QQQ ETF

An end-to-end university Big Data Analytics project that analyzes daily QQQ OHLCV data over the configured multiyear period and investigates which technical indicators are useful for classifying the next trading day's price direction.

This is an educational data-analysis project, not an automated trading system or investment recommendation.

## Research questions

1. What are the major price, return, volume, and volatility patterns of QQQ during the configured analysis period?
2. Which technical indicators are the most important features for classifying QQQ's next-day price direction?
3. Does the Random Forest classifier perform better than a simple baseline on unseen chronological data?

## Target definition

For trading day `t`:

```text
next_return = Close[t+1] / Close[t] - 1
target_up = 1 if Close[t+1] > Close[t]
target_up = 0 otherwise
```

The final observation has no next-day outcome and is removed only after the target is created. Features for day `t` may use only information available at or before the close of day `t`.

## Project structure

```text
qqq-market-insight/
├── README.md
├── config.json
├── requirements.txt
├── src/
│   ├── pipeline_qqq_5_year.py
│   ├── pipeline_qqq_10_year.py
├── data/
│   ├── raw/
|   |   ├── 5_year/
│   │   |    └── qqq_nasdaq_raw.json
|   |   |    └── qqq_nasdaq_raw.metadata.json
|   |   └── 10_year/
│   │         └──qqq_nasdaq_raw.json
|   |         └── qqq_nasdaq_raw.metadata.json
│   └── processed/
|   |   ├── 5_year/
│   │   |    └── qqq_features.csv
|   |   └── 10_year/
│   │         └──qqq_features.csv
├── outputs/
│   ├── figures/
|   |   ├── 5_year/
|   |   └── 10_year/
│   └── tables/
|   |   ├── 5_year/
|   |   └── 10_year/
├── deliverables/
│   ├── report/
│   └── presentation/
├── docs/
│   ├── Big_Data_Analytics_Project_Assignment.pdf
│   └── PROJECT_PLAN_AND_RUBRIC.md
└── prompts/
    └── ITERATIVE_ENGINEERING_LOOP.md
```

The structure separates source code, data, analytical outputs, and submission deliverables. Files under `data/raw/` are immutable source extracts and must never be edited manually.

## Data source

Historical daily OHLCV data for QQQ are collected from Nasdaq's public historical quote service through its underlying API endpoint. The project calls the backend API directly and does not scrape webpage HTML.

- Collection API: [Nasdaq QQQ historical quote API](https://api.nasdaq.com/api/quote/QQQ/historical)
- Human-readable reference: [Nasdaq QQQ Historical Data](https://www.nasdaq.com/market-activity/etf/qqq/historical)
- Request parameters: `assetclass`, `fromdate`, `todate`, and `limit`
- Analysis parameters: `config.json`
- Required dataset size: at least 500 original observations

The initial configuration uses QQQ (`asset_class: etf`) from `2021-01-01`
through `2026-09-16`, reserves the final 20% for testing, sets the reproducible
random state to `42`, and uses `30` permutation-importance repeats. Change
these values in `config.json` instead of hard-coding them in notebooks.

## Prerequisites

- Python 3.11 or later
- Internet access for the data-collection stage
- A terminal opened at the repository root

The commands below use macOS/Linux path syntax. On Windows, activate the environment with `.venv\Scripts\activate` and then use `python` in place of `.venv/bin/python`.

## Installation

```bash
cd qqq-market-insight
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Verify the setup:

```bash
python -m unittest discover -s tests -v
# Direct-file form (also supported):
python tests/test_pipeline.py -v
python -c "from src.pipeline import load_config; print(load_config())"
```

## Run the project

The intended end-to-end workflow is:

```bash
# 1. Download the source data and run the processing/modeling pipeline.
.venv/bin/python src/pipeline.py --download

# On later runs, reuse the immutable raw response.
.venv/bin/python src/pipeline.py

# 2. Build the analysis notebook.
.venv/bin/python src/build_notebook.py

# 3. Execute the notebook from a clean kernel.
.venv/bin/jupyter nbconvert \
  --execute \
  --to notebook \
  --inplace notebooks/qqq_feature_importance.ipynb

# 4. Run the automated checks.
.venv/bin/python -m unittest discover -s tests -v
```

`src/pipeline.py` runs the analytical stages in this fixed order:

```text
Download (when requested or raw data is absent)
  -> Parse and Clean
  -> Feature Engineering
  -> Exploratory Data Analysis
  -> Chronological Train/Test Split
  -> Model Training
  -> Model Evaluation
  -> Held-out Permutation Feature Importance
  -> Save Tables and Figures
```

An existing raw response is reused by default. To replace only the raw response
and metadata, run `.venv/bin/python src/pipeline.py --stage download --force`.
The legacy command `.venv/bin/python src/pipeline.py --download --force`
refreshes the raw artifacts and then continues through the full pipeline.

### Run the data steps one at a time

For studying the code, run `pipeline.py` directly and stop after each data
boundary. Each command starts a new Python process, so the feature step parses
the saved raw JSON again before it creates the processed dataset.

```bash
# 1. Download only. Omit --force when no raw files exist yet.
.venv/bin/python src/pipeline.py --stage download --force

# 2. Parse, clean, and print the quality audit; no processed file is written.
.venv/bin/python src/pipeline.py --stage parse

# 3. Parse again, engineer features, and save data/processed/qqq_features.csv.
.venv/bin/python src/pipeline.py --stage features

# 4. Later, run EDA, modeling, evaluation, and feature importance as well.
.venv/bin/python src/pipeline.py --stage all
```

The raw JSON is immutable by default: `--stage download` stops if either raw
artifact already exists. Use `--force` only when you intentionally want to
replace both the Nasdaq response and its provenance metadata. The `parse` and
`features` stages never access the network.

For the separate five-year experiment, the focused downloader stops before
parsing and modeling and preserves its own raw artifacts:

```bash
.venv/bin/python src/pipeline_qqq.py --force
```

It writes `data/raw/qqq_nasdaq_ohlcv_5_year_raw.json` and
`data/raw/qqq_nasdaq_ohlcv_5_year_raw.metadata.json`; it does not overwrite the
original `qqq_nasdaq_raw` dataset.

## Main functions in `src/pipeline.py`

The implementation follows the supplied example's eight main functions.
The notebook imports them from `src.pipeline`; helper functions handle
configuration, validation, provenance, and file writing.

| Function | Input | Responsibility and output |
|---|---|---|
| `download_data()` | `config.json`; optional `force=True` | Download QQQ from Nasdaq; save unchanged raw JSON and metadata; return `None` |
| `parse_raw()` | Saved raw JSON; optional file path | Parse and clean a sorted OHLCV DataFrame; attach cleaning counts in `attrs['quality']` |
| `rsi(series, window=14)` | Closing-price Series | Return a causal RSI Series with an explicit warm-up period |
| `add_features(df)` | Sorted OHLCV DataFrame | Return OHLCV, 21 indicators in five families, `next_return`, and `target_up` |
| `metric_row(name, y_true, pred, prob)` | Labels and upward-class probabilities | Return accuracy, balanced accuracy, precision, recall, F1, and ROC AUC |
| `build_models(X_train, y_train, X_test, y_test)` | Chronological train/test matrices | Fit the original fixed Majority baseline, Logistic regression, and Random forest; return `(fitted, metrics)` |
| `save_eda(df)` | Feature-engineered development data | Save six EDA chart types and descriptive/yearly summaries; return `None` |
| `run(download=False)` | Configuration and raw source | Orchestrate the full workflow, save predictions/evaluation/importance, and return a summary dictionary |

Example from a notebook whose setup adds the repository root to `sys.path`:

```python
from src.pipeline import parse_raw, add_features, run

ohlcv = parse_raw()
features = add_features(ohlcv)
summary = run()  # Run all analytical steps using the saved raw response.
```

`run()` checks that the raw checksum, symbol, asset class, and requested date
range agree with the metadata and current configuration. Cleaning removes exact
duplicates and audits invalid values; conflicting quotes for the same date fail
with a clear error. If cleaning removes an invalid daily observation, the full
run stops after writing the quality audit so a missing session cannot silently
change the meaning of "next trading day".

EDA uses the training period only; each chart states its date range and sample
size. The final configured fraction stays in the holdout. One preceding row is
excluded from training, and `TimeSeriesSplit(n_splits=5, gap=1)` applies the same
boundary rule within training. This models a classifier frozen before the
evaluation period begins; training-label dates are recorded for verification.
See the [TimeSeriesSplit documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html).

RSI follows the example's recursive EWM initialization (`alpha=1/14`), not an
SMA-seeded implementation. MACD and its signal are normalized by current close;
ATR uses a 14-day simple mean of true range. `next_return` and `target_up` are
excluded from model inputs. Family importance is a sum of individual permutation
means, not joint-family permutation or a causal effect; correlated features and
weak predictive performance limit interpretation. See the
[permutation-importance documentation](https://scikit-learn.org/stable/modules/permutation_importance.html).

## Model comparison and explainability

The active implementation uses the original three fixed models requested by
the user. `build_models()` fits them on training data without hyperparameter
search or automatic model selection:

| Model | Fixed settings |
|---|---|
| Majority baseline | `DummyClassifier(strategy="most_frequent")` |
| Logistic Regression | Median imputation, StandardScaler, `C=1.0`, `class_weight="balanced"`, `max_iter=2000` |
| Random Forest | `n_estimators=700`, `max_depth=5`, `min_samples_leaf=12`, `class_weight="balanced_subsample"` |

Both learned models use `random_state` from `config.json`. Random Forest is the
explanation model by design. Its cross-validation uses five expanding training
folds with gap=1, and its permutation importance uses held-out balanced accuracy.
The config reader retains support for full-line `//` comments; strict JSON is
recommended for compatibility with other tools.

The active result files are:

- `outputs/tables/timeseries_cv_folds.csv` and `timeseries_cv_summary.csv`:
  Random Forest fold scores, temporal boundaries, means, and sample SDs.
- `outputs/tables/model_metrics.csv` and `test_predictions.csv`: the three
  fixed models' test scores, labels, and probabilities.
- `outputs/tables/permutation_importance.csv` and `group_importance.csv`:
  Random Forest feature importance and sums by indicator family.
- `outputs/tables/classification_report.json` and `confusion_matrix.csv`:
  Random Forest evaluation.

The previous model-search experiment has been archived under
`outputs/tables/archive/model-search-before-revert.t3yToI/`. It is not used by
the active pipeline. This archive preserves the experimental code and results.

The existing held-out period has already been inspected during development.
New results on it are exploratory comparisons. A genuinely new chronological
evaluation period is needed to confirm improvements; repeatedly searching for
higher test accuracy would make the test part of model selection.

## Expected outputs

After all development stages are complete, a successful run must produce:

- Immutable Nasdaq response: `data/raw/qqq_nasdaq_raw.json`
- Retrieval metadata and raw checksum: `data/raw/qqq_nasdaq_raw.metadata.json`
- Processed feature dataset: `data/processed/qqq_features.csv`
- Executed notebook: `notebooks/qqq_feature_importance.ipynb`
- Figures: `outputs/figures/`
- Machine-readable result tables: `outputs/tables/`
- Written report: `deliverables/report/QQQ_Technical_Indicator_Report.docx` and `.pdf`
- Presentation: `deliverables/presentation/QQQ_Technical_Indicator_Presentation.pptx`

The final notebook must run top-to-bottom, the dataset must contain at least 500 original observations, and the EDA must include at least five different meaningful chart types.

The downloader must refuse to overwrite an existing raw response unless the
operator explicitly requests a refresh with `--force`. The raw response body
is validated and then saved byte-for-byte; retrieval time, final request URL, and
SHA-256 checksum are stored in the separate metadata file.

## Methodological guardrails

- Check schema, missing values, duplicates, formats, and chronological ordering before analysis.
- Never randomly shuffle observations before splitting time-series data.
- Keep the final 20% as an unseen chronological test set.
- Use expanding time-series splits within training to evaluate the fixed Random Forest.
- Compare Random Forest with a simple baseline classifier.
- Evaluate at least accuracy, precision, recall, F1-score, and confusion matrix.
- Calculate feature importance without using future observations.
- Treat feature importance as model reliance, not evidence of causality.
- Do not present classification performance as evidence of trading profitability.

## Assignment deliverables

- A well-commented notebook or Python code
- A 5-10 page written report covering the required assignment sections
- A presentation designed for a 10-15 minute session
- Explicit answers to all three research questions
- At least five different chart types with titles, axes, units, and interpretations

See `docs/Big_Data_Analytics_Project_Assignment.pdf` for the authoritative assignment requirements and `docs/PROJECT_PLAN_AND_RUBRIC.md` for the implementation checklist.
