# Project Plan and Rubric

This file translates the official requirements in
`docs/Big_Data_Analytics_Project_Assignment.pdf` into an implementation and
review checklist for this repository. The PDF remains the authoritative source
if this summary and the assignment differ.

## Assignment requirements

- Use a real-world topic and dataset approved by the instructor.
- Include at least 500 data rows and cite the data source clearly.
- Document columns, data types, and meanings.
- Handle missing values, duplicates, incorrect formats, and required
  transformations or feature engineering.
- Include descriptive statistics and at least five different chart types.
- Apply at least one statistical or machine-learning technique.
- Answer at least three research questions with findings and possible
  recommendations.
- Submit a commented code notebook or script.
- Submit a 5–10 page PDF or DOCX report.
- Submit a presentation designed for a 10–15 minute session.
- Final submission and presentation are due in Week 12.

## Authoritative data-source decision

- Provider: Nasdaq Historical Quotes for QQQ.
- Collection method: direct requests to Nasdaq's public historical quote
  backend API, not HTML scraping.
- API endpoint: `https://api.nasdaq.com/api/quote/QQQ/historical`.
- Request parameters: `assetclass`, `fromdate`, `todate`, and `limit`.
- Human-readable citation:
  `https://www.nasdaq.com/market-activity/etf/qqq/historical`.
- The raw API response must be preserved without manual modification.

## Planned phases

1. Reproducible project setup
2. Nasdaq data acquisition and validation
3. Technical-indicator feature engineering
4. Time-aware modeling and baseline comparison
5. Held-out permutation-importance analysis
6. Notebook, report, and presentation delivery

## Required report sections

1. Title and group members
2. Introduction and objectives
3. Data description and source
4. Data-cleaning process
5. Exploratory data analysis with charts
6. Analytical methods
7. Results and discussion
8. Conclusion and recommendations
9. References

## Evaluation rubric

| Criterion | Points | Evidence expected in this project |
|---|---:|---|
| Dataset quality and relevance | 10 | Nasdaq source citation, raw extract, schema, and at least 500 valid rows |
| Data cleaning and preprocessing | 15 | Auditable checks, corrections, transformations, and technical indicators |
| EDA and visualization | 20 | Descriptive statistics and at least five distinct chart types |
| Analytical/modeling methods | 20 | Time-aware classification, baselines, cross-validation, and held-out evaluation |
| Insights and conclusions | 15 | Direct answers to all three research questions with limitations |
| Report clarity and structure | 10 | A well-structured 5–10 page DOCX/PDF report |
| Presentation quality | 10 | A focused slide deck suitable for a 10–15 minute presentation |
| **Total** | **100** | |

## Development gates

- Do not publish analytical claims before source, schema, row count, missing
  values, duplicates, date ordering, and price formats have been checked.
- Do not complete EDA until five genuinely different chart types are present
  and interpretable.
- Do not report model performance without chronological holdout evaluation and
  simple baselines.
- Do not finalize deliverables until the notebook runs top-to-bottom and the
  report and presentation cover every required section above.

## Eight-function pipeline acceptance record (2026-09-18)

Historical snapshot before the model-search iteration below.

Scope: adapt the supplied Python example in `src/pipeline.py`. Inputs are
`config.json`, the preserved Nasdaq response, and its metadata. Outputs are
`data/processed/qqq_features.csv`, `outputs/tables/`, and `outputs/figures/`.
This implements the code needed for dataset quality (10 points), preprocessing
(15), EDA (20), and modeling (20); it does not declare those rubric scores earned
or the notebook/report/presentation stages complete.

- [x] All eight required functions are implemented with documented inputs and
  outputs: `download_data`, `parse_raw`, `rsi`, `add_features`, `metric_row`,
  `build_models`, `save_eda`, and `run`. See the function table in `README.md`.
- [x] `build_models` matches the example's `(fitted, metrics)` return contract;
  all three classifiers are fitted. `run` saves predictions and probabilities.
- [x] Raw provenance is preserved: SHA-256 remains
  `26496d23be919c5dfd6be51de28d3761d46772c34124c0e4a83a788c368b9947`.
  The real run reused the existing response; download behavior was tested with
  a mock response without replacing it.
- [x] Row counts reconcile in `outputs/tables/data_quality.json`:
  1,432 raw = 1,432 valid OHLCV = 1,411 model rows + 20 warm-up + 1 unknown
  final target. No other feature rows were removed.
- [x] Targets were checked against next-day closes from raw JSON. Regression
  tests verify up/down/flat labels, RSI edge cases, duplicate conflicts,
  nonfinite values, and invariance of past features to future-data changes.
- [x] Chronological boundaries are recorded: 1,127 train rows through
  2025-07-29, one excluded boundary row on 2025-07-30, and 283 test rows from
  2025-07-31 through 2026-09-15. Training-label dates precede the test period.
- [x] Five expanding Random Forest CV folds use a one-row gap within training;
  dates and metrics are saved in `timeseries_cv_folds.csv`. The means and sample
  standard deviations in `timeseries_cv_summary.csv` match those folds.
- [x] Six EDA chart types use only the 1,127 development rows. All ten saved
  PNGs were opened and visually inspected; captions identify EDA scope and
  permutation error bars. Descriptive counts agree with the plotted population.
- [x] Saved prediction accuracy was independently recomputed for all models;
  individual feature importance sums reconcile to `group_importance.csv`.
- [x] `.venv/bin/python -m unittest discover -s tests -v`: 13 tests passed.
- [x] `.venv/bin/python src/pipeline.py`: completed successfully on saved data.

Assumptions and remaining limits: the classifier is frozen before its evaluation
period, so a boundary gap is intentional. Indicators retain the example's
formulas, including recursive-EWM RSI, normalized MACD, and simple-mean ATR.
The recorded close is used as supplied by Nasdaq; no adjusted-close or
total-return claim is made. Removing an invalid quote stops the full workflow
for review so a missing session is not silently treated as a one-day return.
Family importance is a sum of individual estimates and is sensitive to
correlation and family size. The Random Forest's held-out accuracy is 0.522968
versus the majority baseline's 0.544170, with ROC AUC 0.498389; these results
do not establish useful predictive performance. The holdout has already been
inspected during development; changing model choices in response to its results
would require a new untouched evaluation period.

Decision: **STOP** after the requested pipeline/function refactor. Notebook,
report, presentation, and final-submission gates remain separate work.

## Model-comparison iteration (2026-09-18)

Historical experiment, superseded by the user-requested fixed-model restoration
below. Its code and results are retained under
`outputs/tables/archive/model-search-before-revert.t3yToI/`; file references in
this historical section describe that experimental run, not current outputs.

Scope: continue `pipeline.py` with the original next-day Up/Down target and
existing Nasdaq dataset. Compare model families and class-weight settings using
training CV, then explain the selected learned classifier. Preserve the eight
main function names and the two-value `build_models` return contract.

- [x] Baseline, Logistic Regression, Random Forest, and HistGradientBoosting
  share five expanding folds with gap=1. `model_search_results.csv` records
  25 predeclared configurations; all searches fit only training data.
- [x] `model_selection.json` records selection by mean training-CV accuracy:
  baseline 0.536898, Logistic Regression 0.535829, Random Forest 0.530481,
  HistGradientBoosting 0.504813. Logistic Regression is the best learned
  candidate, but **none exceeds the baseline in this selection comparison**.
- [x] Selected logistic parameters are `C=0.1`, `class_weight=None`. The full
  pipeline executes successfully on the unchanged 1,432-row raw extract and
  1,411-row feature dataset (1,127 train, one gap, 283 test).
- [x] Test accuracy in `model_metrics.csv`: baseline 0.544170, Logistic Regression
  0.522968, Random Forest 0.501767, HistGradientBoosting 0.508834. Prior fixed-model
  scores are preserved in `fixed_model_metrics_reference.csv`; the tuned logistic
  accuracy improves by 0.021201, without beating the majority reference.
- [x] Explainability now uses the CV-selected Logistic Regression. Importance
  files identify the estimator and score, with separate accuracy, balanced
  accuracy, and ROC AUC results. No test-importance ranking selects features.
- [x] `.venv/bin/python -m unittest discover -s tests -v`: 16 tests passed,
  including a held-out-label perturbation test that leaves model selection,
  hyperparameters, and CV results unchanged.

Limits: all CV scores used for tuning are selection estimates. The previously
inspected holdout supplies exploratory results, not an untouched confirmation.
Top accuracy-based importance for the selected model is `obv_change_5`, mean
0.018139 with permutation SD 0.020497; variation is substantial, and the model's
test ROC AUC is only 0.481325. Neither the ranking nor increased logistic
accuracy establishes robust predictive usefulness. The raw source, target,
chronological holdout, and future-information restrictions remain the same.

Decision: **STOP** after the authorized model-comparison implementation and
validation. No further tuning was performed in response to these test scores.

## Restore original three fixed models (2026-09-18, current)

Scope: user-requested return to the earlier model configuration and reported
Random Forest accuracy of 52.30%. The experimental search code and outputs are
preserved in `outputs/tables/archive/model-search-before-revert.t3yToI/`.

- [x] Restored exactly Majority baseline, Logistic Regression (`C=1.0`, balanced
  class weights), and Random Forest (700 trees, depth 5, minimum leaf size 12,
  balanced-subsample class weights). Both learned models retain seed 42.
- [x] Removed HGB, grid search, and automatic model selection from active code.
  Raw data, features, target, and the chronological train/gap/test split are
  unchanged. Config-comment support remains compatible with the user's file.
- [x] `run()` again evaluates Random Forest with five expanding CV folds and
  produces Random Forest confusion/importance plots using balanced accuracy.
- [x] `.venv/bin/python src/pipeline.py` succeeded. Regenerated `model_metrics.csv`
  is byte-for-byte identical to `fixed_model_metrics_reference.csv`:
  baseline accuracy 0.5441696113, logistic accuracy 0.5017667845, and Random
  Forest accuracy 0.5229681979, balanced accuracy 0.5170139938, AUC 0.4983892077.
- [x] `.venv/bin/python -m unittest discover -s tests -v`: 15 tests passed,
  including fixed-parameter assertions and held-out-label independence.
- [x] Independent artifact checks confirmed raw SHA-256, target arithmetic,
  row counts, test-prediction metrics, CV aggregates, and feature-family sums.
  All ten regenerated figures were opened and visually reviewed.
- [x] Obsolete search-only files were moved to the archive, and README now
  describes the active three-model workflow.

Decision: **STOP**. The requested prior configuration and numerical results are
restored; earlier exploratory experiments remain recorded in project history.
