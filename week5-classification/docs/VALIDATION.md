# Validation record

- Source: pinned Kaggle version 2, 25,305,108-byte CSV; SHA-256 verified.
- Algorithm tests: 5 passed using standard-library unittest, including lecture weather data, conjunction learning, contradictory labels, unseen categories and training-only binning.
- Notebook: all 11 code cells executed sequentially without error; three figures retained in outputs.
- Tables: class support totals, chronological separation, event separation, rule purity, test prediction counts and rule interpretations reconciled.
- Visual QA: inspected all three exported figure PNGs; labels, matrix values and plot scales are readable.
- HTML: generated from the executed notebook, with three embedded plot images and all tables. Full browser layout inspection was blocked: no browser integration was available and native Computer Use awaited Accessibility/Screen Recording permission. Open `report/solar_pv_classification.html` locally to complete this check.
- Direct dependency versions: `requirements-lock.txt`; Python 3.14.7. This is a direct-dependency snapshot, not a complete transitive environment lock.

Recheck with the commands in README. No evaluation score is claimed for real PV field measurements.
