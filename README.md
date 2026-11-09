# AgriVision-Crop — Synthetic Crop Suitability Classification

**Sector:** Agriculture  
**Task:** Multi-class classification: six illustrative crop labels

Maps generated nutrient and climate profiles to one of six labels (rice, wheat, maize, cotton, banana, coffee). The class profiles are hand-specified for an offline coding demonstration. Confusion-matrix class order is rice, wheat, maize, cotton, banana, coffee.

## Data and provenance

The offline demo uses a deterministic synthetic generator in `agrivision/data.py`. No public or private dataset is downloaded or redistributed. The generated values are illustrative and do not represent real people, organizations, or operational systems. Synthetic output is not evidence of real-world model performance.

## Method

- **Baseline:** majority-class prior classifier.
- **Stronger model:** histogram gradient-boosting classifier.
- **Split:** Stratified random holdout. Rows are sampled independently around crop profiles.. The seed is fixed by default and configurable.
- **Evaluation:** accuracy, balanced accuracy, macro F1, confusion matrix, and (for binary tasks) ROC-AUC, average precision, precision, and recall at threshold 0.5.
- **Error analysis:** confusion matrix and misclassification count are emitted with the metrics.

This is not agronomic advice. The profile distributions are illustrative, omit local soil, cultivar, management, and measurement variation, and must not guide planting decisions.

## Run locally

    python3 -m venv .venv
    . .venv/bin/activate
    python -m pip install -e ".[test]"
    python -m agrivision.train --seed 42 --samples 2400
    python -m agrivision.train --seed 42 --samples 2400 --json
    python -m pytest -q

`make setup`, `make train`, and `make test` provide shortcuts. Python 3.9 or newer is required. No network access or data download is needed after dependencies are installed.

## Repository layout

- `agrivision/data.py` — validated domain-specific synthetic generator.
- `agrivision/split.py` — repeatable holdout split.
- `agrivision/model.py` — baseline, stronger model, metrics, and diagnostic summaries.
- `agrivision/train.py` — train/evaluate CLI.
- `tests/` — validation, split, model, and CLI tests.
- `.github/workflows/ci.yml` — automated test and CLI smoke run.

## Intended use

Educational and research prototype only. Do not use the synthetic benchmark to make decisions about real people or operations. Review the data assumptions, error patterns, and subgroup diagnostics before describing any findings.
