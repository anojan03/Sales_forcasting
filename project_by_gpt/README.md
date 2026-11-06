# Convenience-store sales forecasting research

Fresh research implementation comparing Prophet, XGBoost and LSTM with and without payroll proxies. The preserved original study is **rolling one-hour-ahead net retail sales forecasting**, issued at the beginning of the target hour. Extensions add direct upcoming-day forecasts, a validation-weighted hybrid, department analysis, holidays, dependent uncertainty and planning scenarios. The daily forecast is issued at local midnight before any target-day sales. Read IMPLEMENTATION.md and both reports for their distinct methods; hourly and daily scores must not be ranked together.

## Data and accounting

The folder contains **five overlapping source datasets and one previous prediction export**. They are never concatenated. The canonical source is `Master transaction-level research dataset.csv`, aggregating signed `Line_Total` by `Transaction_Time`. Explicit `-PAYOUT` supplier entries are excluded; product sales, refunds, promotion adjustments and voucher discounts are retained. `Promotion` is not subtracted a second time. Negative sales are legitimate net revenue observations, not automatically outliers.

The cleaned headerless CSV exactly matches the nonnegative master lines on shared transaction fields. Both hourly exports and the daily department export reconcile to these positive lines. Their gross total of GBP 457,238.05 differs from the full extracted net-retail total of GBP 451,755.92. `Final_Model_Comparison.csv` is audited for provenance only and excluded from all new experiments. No previous notebooks or model settings were read.

Original CSVs are preserved. Hashes, column schemas, dates, missing counts, duplicate counts, receipt overlap and aggregate reconciliations are in `outputs/audit/`. Full-row master duplicates are absent; repeated receipt IDs are legitimate line items. Missing department names do not cause deletion.

## Assumptions and limitations

- Timestamps are assumed to be Europe/London local shop time; the export has no timezone metadata. UTC aggregation and a regular elapsed-hour grid handle daylight saving. Ambiguous or nonexistent recorded local times fail rather than being guessed.
- Currency is treated as GBP. Tax basis is unverified. Three negative voucher lines are interpreted as retail discounts from their labels. Shop accounting confirmation remains desirable.
- The last three and first three calendar days of each month form a **payday proxy**, not actual customer paydays. Window widths are configurable and not tuned on the test set.
- Missing hours remain unknown. No opening-hours calendar or extraction completeness proof is available; metrics cover recorded retained-line hours and may exclude genuine zero-transaction trading hours.
- Analysis starts 1 January 2025 to exclude sparse startup records; the final extraction date, 15 July 2026, is excluded as potentially incomplete.
- The original hourly study uses one store, one holdout and one seed. Daily extensions add pre-test XGBoost temporal folds, three LSTM seeds and block-bootstrap sensitivity. No causal or universal statistical-significance claim is made. Refer to both reports for subgroup effects and drift limitations.

## Setup

Python 3.11 was used. In PowerShell, start in this folder:

```powershell
cd C:\Users\user\Downloads\datasets_project\project_by_gpt
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run_all.py
```

The already-executed environment is the parent folder's `.venv`. To rerun with that environment:

```powershell
..\.venv\Scripts\python.exe run_all.py
```

Core dependencies are pinned to executed versions. `outputs/logs/requirements.lock.txt` records the full forecasting-environment package freeze. Word/PDF export uses `REPORT_PYTHON` if set; otherwise it prefers the available Codex bundled artifact runtime and falls back to the active Python executable. Its package versions are recorded separately in `outputs/logs/artifact_environment.json`. No GPU is required. TensorFlow uses two CPU threads, deterministic operations, disabled oneDNN custom optimizations and no sample shuffling.

## Independent execution

Run preparation before modelling. Each main script trains and saves both payday variants:

```powershell
..\.venv\Scripts\python.exe src\data_preparation.py --audit-only
..\.venv\Scripts\python.exe src\data_preparation.py
..\.venv\Scripts\python.exe src\baselines.py
..\.venv\Scripts\python.exe src\prophet_model.py
..\.venv\Scripts\python.exe src\xgboost_model.py
..\.venv\Scripts\python.exe src\lstm_model.py
..\.venv\Scripts\python.exe src\model_comparison.py
..\.venv\Scripts\python.exe src\verify_workflow.py
..\.venv\Scripts\python.exe src\research_report.py
```

With a newly created local environment, replace `..\.venv` with `.\.venv`. **Comparison loads saved predictions and never retrains.** Reporting also consumes saved results. `run_all.py` calls the independent scripts in that order, records commands and execution logs, and fails visibly if any step fails.

## Experimental design

`config.json` controls source path, target, timestamp, timezone, excluded accounting classes, split dates, proxy width, seed and model settings. Boundary dates are local UK dates:

| Period | Dates |
| --- | --- |
| Training | 1 January to 31 December 2025 |
| Validation | 1 January to 31 March 2026 |
| Test | 1 April to 14 July 2026 |

The first 24 grid hours are excluded from eligible targets consistently across all models. Imputers and scalers fit only eligible training observations. Models are frozen after validation selection and are not refitted on validation. Shared hyperparameter selection minimizes the mean validation RMSE of the two variants in each family. Prophet tests changepoint priors 0.01/0.05; XGBoost tests depth 3/5 with 300 trees; LSTM tests 16/32 units with at most 25 epochs and validation early stopping. Candidate scores and selected settings are saved.

XGBoost and Prophet use shifted past-sales lags 1/24/168 and a prior-24-hour rolling mean/count. Calendar predictors are known at hour start. Prophet supplies its own daily/weekly Fourier terms and trend. LSTM encodes 24 **elapsed clock hours**, including missing-hour slots; scaled unknown sales are replaced with the training mean and an observation flag. A separate input supplies target-hour calendar features. Predictions inverse-transform only the target channel into GBP. No current-hour quantity, promotion amount or transaction count enters predictors.

The naive, daily seasonal-naive and weekly seasonal-naive baselines use the relevant lag and a training-median fallback when it is unknown. Offsets mean elapsed UTC hours; they may differ from identical local-hour offsets across DST changes.

## Outputs and interpretation

- `outputs/prepared/hourly_grid.csv`: UTC clock grid, local timestamp, net target, observed flag, proxy, calendar and historical features, chronological split.
- `outputs/models/`: fitted Prophet JSON, XGBoost JSON, LSTM Keras models and preprocessing objects; LSTM candidate checkpoints are also retained.
- `outputs/predictions/`: full-precision validation/test forecasts, provenance metadata, common test comparison and SHAP contributions.
- `outputs/metrics/model_metrics.csv`: common and model-specific full-test MAE/RMSE, sizes, UTC start/end dates, payday and non-payday segments.
- `outputs/metrics/payday_effects.csv`: error **with minus without** payday and percent changes; negative denotes improvement.
- `outputs/charts/`: exploratory coverage, forecasts, error comparison, proxy effects, residuals and Tree SHAP plots.
- `outputs/logs/`: exact package versions, model settings, validation candidates, LSTM histories, stage execution logs, workflow status and verification evidence.
- `reports/research_report.md`, `.docx`, `.pdf`: reports built from actual outputs.

The comparison validates timestamp uniqueness/order, finite values, actual-sales agreement, proxy agreement, test membership and prepared-data/configuration provenance. It evaluates the six variants on the exact intersection and also retains their full-test results. A mutation check replaces target and future observations and confirms origin features are unchanged. Train-only imputer statistics and LSTM scaler inversion are verified. Changing configuration requires preparation and new model outputs; stale predictions are rejected.

SHAP uses TreeExplainer with `tree_path_dependent` on up to 500 deterministic test samples. Additivity is verified. Mean absolute contributions show fitted predictive influence, not causation. No independent-hour statistical test is applied to temporally dependent errors.

Forecast CSVs preserve 17 significant digits; only report tables round. Results are specific to the configured target and split. The legacy 341-row export is not comparable evidence for these fresh experiments.

## Privacy

Raw timestamps, receipt IDs, customer IDs and product combinations may permit indirect identification. Features and reports exclude customer identity. Keep source data and model artifacts access-controlled, obtain store/institutional authorization before sharing, and publish aggregated outputs with an appropriate retention policy. Small shop-level aggregates remain commercially sensitive.

## Method documentation

See primary [Prophet documentation](https://facebook.github.io/prophet/docs/seasonality,_holiday_effects,_and_regressors.html), [XGBoost parameters](https://xgboost.readthedocs.io/en/stable/parameter.html), [TensorFlow LSTM](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM), [Tree SHAP](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html) and [data leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage). These document implementations; the store results come from local execution.
# Research extensions

The proposal-alignment extensions preserve the original hourly experiment and add a separate upcoming-day study. Read [IMPLEMENTATION.md](IMPLEMENTATION.md) for choices, limitations and commands. Run `python run_extensions.py` for the additions only; `python run_all.py` now runs both workflows. Results are in `outputs/extensions/` and `reports/research_extensions.md`, `.docx` and `.pdf`. Daily and hourly scores have different target units and must not be ranked together. England holidays use a cached official calendar; payroll features are explicitly labelled configurable proxies. The new hybrid is a validation-weighted ensemble, not a sequential residual cascade.

The 3 November completion adds `sequential_hybrid_model.py` alongside that ensemble, plus `online_payroll_analysis.py`, `privacy_export.py` and `research_completion.py`. `cascade_config.json` fixes chronological residual-stage settings; `payroll_sources.json` records official NAO/UHL/Bath sources. Payroll dates are employer-specific and coverage-aware, not observed customer income. `reports/literature_review.md` contains the developed targeted narrative review. `governance/` provides proposed data-management and prospective pilot protocols with unconfirmed evidence fields. No release is automatically authorised. The dissertation is deferred by the user's instruction.

Run the new stages in order after the existing daily preparation/base forecasts: `python src/sequential_hybrid_model.py`, `python src/online_payroll_analysis.py`, `python src/privacy_export.py`, `python src/next_day_comparison.py`, `python src/research_completion.py`, `python src/extension_report.py`. `python run_all.py --from-step sequential_hybrid_model` resumes the integrated workflow there. New reports are `reports/research_completion.md`, `.docx` and `.pdf`. All comparisons and report generation load saved predictions without retraining the base models.

