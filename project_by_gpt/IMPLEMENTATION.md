# Research extension implementation

The working one-hour workflow and its results are preserved. The new upcoming-day experiment is separate because daily and hourly forecast errors are not comparable. All new research results are generated under `outputs/extensions`; its report is `reports/research_extensions` in Markdown, Word and PDF.

## Experimental choices

- Issue a direct forecast at midnight for the upcoming local day. Predict recorded net revenue; shop opening hours and complete recording remain unverified. No missing hour is replaced with zero.
- Keep 2025 training, January–March 2026 validation and April–14 July 2026 test. Frozen base model weights. Previous completed days become history only at later origins.
- Use England/Wales official bank holidays following the user's confirmation that the shop is in England. Cache and hash the source calendar for offline reruns.
- Add configurable monthly, Friday/Saturday weekly and last-working-day payroll proxies. The last-working-day scenario is illustrative, not a verified public-sector employer rule.
- Pair with/without variants using the same data and shared structural tuning. All variants contain holiday features; only payroll features differ.
- Add a convex Prophet/XGBoost/LSTM hybrid with shared weights learned from validation, not test. This is a justified ensemble alternative to the literal slide arrow suggesting a residual cascade. An ensemble is not a sequential residual pipeline.
- Expand category/hour/weekday/payday analysis. Keep unattributable receipt adjustments in an explicit unknown group rather than invent category allocations.
- Evaluate positive-target daily MAPE with denominator counts; retain MAE/RMSE as primary measures. Do not reinterpret hourly net-sales MAPE as valid.
- Add pre-test expanding XGBoost validation, validation-only monthly-window sensitivity, monthly test results for every model and three LSTM seeds. No test-based seed selection.
- Use daily circular block-bootstrap loss differences at 7/14-day block lengths. Approximate stationarity, few payroll cycles and drift limit interpretation.
- Evaluate training-threshold high-revenue-day precision/recall and over/underforecast revenue scenarios. These are workload proxies, not measured staff requirements, SKU stockouts or profit.
- Derived files omit customer/receipt/barcode/payment identifiers. Preserve raw originals. This does not certify consent, institutional approval, access controls or full raw-data anonymisation.

## Remaining evidence and scope

Actual customer/employer payday dates, complete capture/opening-hours metadata, stock/service-time/cost datasets, store/institution permissions and storage/retention policy cannot be invented. These are explicit remaining requirements. The supplement contains a short verified methodological literature discussion, not a systematic literature review or university-approved final dissertation. SHAP is implemented for XGBoost, not claimed for Prophet/LSTM. Statistical intervals do not establish causal income effects. Presentation results must be reviewed against the saved outputs; the original proposal and uploaded slides are not overwritten.

## Commands

From `project_by_gpt`, use `..\.venv\Scripts\python.exe run_extensions.py` for the additions only. `..\.venv\Scripts\python.exe run_all.py` executes both studies. Independently run the same interpreter with:

1. `src\extension_preparation.py`
2. `src\department_analysis.py`
3. `src\next_day_prophet.py`
4. `src\next_day_xgboost.py`
5. `src\next_day_lstm.py`
6. `src\hybrid_model.py`
7. `src\next_day_comparison.py`
8. `src\extension_validation.py`
9. `src\extension_report.py`
10. `src\research_notebook.py` (execute saved-result exploration)

`reports/research_results.pptx` is generated with the supplement. `research_results.ipynb` is executed with a project-local kernel specification. Raw source data are not embedded in either artifact. Hourly high-revenue peak precision/recall is also saved separately from daily planning scenarios.

The hybrid requires saved base validation/test predictions. Comparison requires saved forecasts only; it does not retrain. Report generation requires saved analyses. Update `extension_config.json` then rerun preparation and models if changing settings. Existing hourly `config.json` remains intact.
