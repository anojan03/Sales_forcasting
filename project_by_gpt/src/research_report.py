"""Generate an evidence-based report from saved audits and model outputs."""
from data_preparation import *
import os,subprocess

def table_block(df, columns=None):
    if columns is not None: df=df[columns]
    head=[str(x).replace('_',' ') for x in df.columns]
    rows=[]
    for row in df.itertuples(index=False,name=None):
        rows.append([f'{v:.3f}' if isinstance(v,(float,np.floating)) else str(v).replace('_',' ') for v in row])
    return dict(kind='table',headers=head,rows=rows)

def run():
    folders(); c=config(); d=load_data()
    audit=json.loads((OUT/'audit'/'dataset_audit.json').read_text()); prep=json.loads((OUT/'audit'/'preparation_decisions.json').read_text())
    metrics=pd.read_csv(OUT/'metrics'/'model_metrics.csv'); effects=pd.read_csv(OUT/'metrics'/'payday_effects.csv')
    common=metrics[(metrics.scope=='common')&(metrics.group=='all')].copy()
    logs={k:json.loads((OUT/'logs'/f'{k}.json').read_text()) for k in ['prophet','xgboost','lstm']}
    cv=json.loads((OUT/'logs'/'comparison_validation.json').read_text())
    versions=json.loads((OUT/'logs'/'environment.json').read_text())
    best=common.sort_values('RMSE').iloc[0]
    best_main=common[~common.model.str.contains('Naive')].sort_values('RMSE').iloc[0]
    blocks=[]
    def h(text,level=1): blocks.append(dict(kind='heading',text=text,level=level))
    def p(text): blocks.append(dict(kind='paragraph',text=text))
    def fig(name,caption): blocks.append(dict(kind='figure',path=str(OUT/'charts'/name),caption=caption))
    h('Hourly convenience store sales forecasting',0)
    p('Research report on Prophet, XGBoost and LSTM with a configurable calendar payday proxy. Fresh experiments use net retail sales and a chronological holdout. All numerical tables and figures below derive from saved execution outputs.')
    h('Research aim and principal findings')
    p(f'The study asks whether a calendar proxy for payday improves one-hour-ahead sales forecasts at a small independent convenience store. The common test contains {cv["common_n"]:,} observed hours, including {cv["payday_n"]:,} payday-proxy and {cv["non_payday_n"]:,} non-proxy hours. The lowest test RMSE across the six main variants is {best_main.model.replace("_"," ")} at GBP {best_main.RMSE:.3f}; the lowest RMSE including baselines is {best.model.replace("_"," ")} at GBP {best.RMSE:.3f}. These rankings apply to this holdout and recorded hours only.')
    for row in effects[effects.group=='all'].itertuples():
        p(f'For {row.family}, adding the payday indicator changes MAE by GBP {row.MAE_change:+.3f} ({row.MAE_change_percent:+.2f}%) and RMSE by GBP {row.RMSE_change:+.3f} ({row.RMSE_change_percent:+.2f}%). A negative change denotes improvement. The indicators describe calendar association, not customers\' observed paydays or a causal effect.')
    h('Data sources and provenance')
    p('Five supplied source CSVs and one additional prediction export were audited. No datasets were concatenated. All sources remain unchanged, and SHA256 hashes are stored in outputs/audit/dataset_audit.json. Store identity, extraction software, completeness guarantees and tax basis are not established by the CSVs. Currency is interpreted as GBP from the research context.')
    summary=pd.DataFrame([{'Source':r['file'],'Rows':r['rows'],'Total GBP':r['total'],'Negative rows':r['negative_rows']} for r in audit])
    blocks.append(table_block(summary))
    p('The master transaction file contains 139,919 line items across receipts, rather than 139,919 distinct transactions. Transaction_ID repeats within receipts and must not be a deduplication key. No exact full-row duplicates occur in the master. The cleaned headerless file contains 134,866 records and 53 repeated projections; a multiset check confirms that it exactly matches the nonnegative master lines on receipt, timestamp, product, quantity, unit price and line total. Repeated product lines are retained.')
    p('The hourly total file contains 8,616 distinct hour starts; hourly departments contain 49,816 department-hour records; daily departments contain 8,131 department-day records. Their GBP 457,238.05 totals and timestamp-level reconciliation identify them as overlapping positive-sales aggregates. The maximum reconciliation discrepancy is below GBP 0.000000000001. Their Transaction_Count should be interpreted as exported row counts until receipt-count semantics are established.')
    p('All source date ranges begin on 19 December 2024 and end on 15 July 2026. Transaction timestamps include millisecond precision, such as YYYY-MM-DD HH:MM:SS.sss; hourly exports append .000 and daily exports use YYYY-MM-DD. No invalid source timestamps were found. The cleaned file has no header: its 16 field meanings were inferred by matching the master, while its calendar fields are not used for modelling.')
    p('Final_Model_Comparison.csv is a prior rounded prediction export with 341 rows between 29 March 2026 20:00 and 14 July 2026 21:00 and a 255/86 non-payday/payday partition. Its actuals match positive hourly aggregates, but its training periods, feature definitions and forecast alignment are unknown. It is excluded from fitting, tuning and the new performance comparison. These experiments do not reproduce those models or force their sample counts.')
    h('Audit and cleaning decisions')
    classes=pd.read_csv(OUT/'audit'/'transaction_classes.csv'); blocks.append(table_block(classes))
    p('Net retail revenue is the signed sum of Line_Total, excluding 32 explicitly labelled -PAYOUT supplier entries totalling GBP -1,957.75. Retained classes include 4,917 promotion adjustment lines (GBP -3,775.80), 101 refunds (GBP -1,696.25), three negative voucher lines (GBP -10.08), and 13 other records with department ID zero (GBP 52.49). Promotion totals are already represented by negative line entries: the receipt-level Promotion field is not subtracted again. Quantity multiplied by Unit_Price matches Line_Total within floating-point tolerance, supporting use of the line amount.')
    p(f'The full extracted net retail total after excluding payouts is GBP {prep["net_source_total"]:,.2f}; the modelling period totals GBP {prep["selected_period_total"]:,.2f}. Accounting treatment of vouchers is an explicit interpretation of the labels; timezone and GBP are assumptions rather than merchant-confirmed metadata. These assumptions are configurable and should be checked against the shop\'s accounting and till documentation before operational use.')
    p('Master Department_ID has no nulls, but zero is an unknown or adjustment code for 4,965 rows. Department_Name is missing on 4,965 master rows and on a small number of aggregated department records. Missing barcodes are not missing sales. Master has 11,610 missing barcodes; clean has 6,632 and 13 missing department names. No imputation of product identity or deletion by department was needed for total-shop aggregation. Complete column-wise missing counts and dtypes are in the machine-readable audit.')
    p('No outlier trimming, winsorisation or removal of small positive sales was performed. The sparse December startup period is excluded by a fixed analysis start of 1 January 2025; it may include setup transactions, but this is unconfirmed. The entire final extraction date, 15 July 2026, is excluded because recording stops at 16:27 and it may be incomplete. No other observed hour is removed for low or negative sales.')
    h('Clock hours missingness and daylight saving')
    p(f'Timestamps are treated as Europe/London wall time and converted to UTC before hourly aggregation. Localization fails on ambiguous or nonexistent local times rather than guessing. The selected grid contains {len(d):,} elapsed clock hours; {prep["unknown_hours"]:,} have no retained observations and remain missing. There are {prep["observed_zero_hours"]} observed zero-total hours and {prep["observed_negative_hours"]} observed negative-total hours. A negative net hour can arise from refunds and is retained.')
    p('A missing record could mean closure, no transactions, an extraction gap or downtime. The exports alone cannot distinguish these explanations. Nighttime sparsity is compatible with closure but does not prove a trading calendar. No missing targets are set to zero. Metrics condition on hours with recorded retained lines; zero-transaction trading hours may therefore be underrepresented. This is a material limitation of the estimand.')
    p('Calendar features use local shop time, while each LSTM step and lag offset uses one elapsed UTC hour. The spring clock change makes the validation grid 2,159 hours rather than 2,160. Lags of 24 and 168 elapsed hours may differ from the same local hour across a DST change. This convention is explicit; the April-to-July test contains no clock change.')
    h('Exploratory analysis')
    monthly=pd.read_csv(OUT/'audit'/'monthly_sales.csv'); p(f'Observed monthly net sales vary from GBP {monthly.sales.min():,.2f} to GBP {monthly.sales.max():,.2f} in the selected period; the minimum and maximum include the partial July endpoint if applicable. Monthly comparisons mix trading-day counts, coverage and demand, so they do not isolate customer spending effects. Increasing late-period sales can produce drift relative to a model fitted on 2025.')
    fig('exploratory.png','Figure 1. Observed monthly net sales and local-hour recording coverage. Missing hours are not assigned zero.')
    h('Payday proxy and forecasting task')
    p('The proxy is 1 on the last three calendar days of each month and the first three days of the next month, otherwise 0. A symmetric six-day boundary window allows exploratory association with monthly payroll arriving near month end and spending spilling into the next month. The exact width is a pragmatic hypothesis fixed before model evaluation, not an empirically established payroll rule. No weekend, bank-holiday or business-day payday adjustment is applied. Weekly wages, benefits, customer-specific pay cycles and actual income receipts are not observed.')
    p('Forecasts are issued at the start of clock hour t for net sales during [t,t+1 hour). The operational assumption is that all recorded sales from completed earlier hours have been finalized and are available at that origin. Validation and test observations may update lag or sequence inputs only after their hours complete. Weights, imputers and scalers remain fixed from training. This is rolling one-hour-ahead evaluation, not a multi-hour recursive forecast issued at the beginning of the whole test period. All families have the same information cutoff, although their inductive structures differ.')
    h('Chronological design and feature preparation')
    split=pd.read_csv(OUT/'audit'/'split_counts.csv'); blocks.append(table_block(split))
    p('Training targets run from 1 January through 31 December 2025. Validation targets run from 1 January through 31 March 2026. Test targets run from 1 April through 14 July 2026, with boundary dates interpreted locally. The first 24 grid hours cannot construct a complete LSTM time window and are excluded consistently from training and evaluation for all algorithms. No time-series samples are randomly shuffled, and no fitting uses test targets. Validation is used for small candidate grids and LSTM early stopping only; final models are not refitted on validation.')
    p('Tabular predictors comprise local hour, weekday and month sine/cosine pairs; weekend flag; elapsed-day trend; past sales at 1, 24 and 168 hours; missing indicators for those lags; and the mean and count of recorded sales in the previous 24 clock hours. All historical features shift before rolling. Missing tabular predictors use medians learned from eligible training rows only. Missing lag indicators and rolling counts make the information gaps visible. Current transaction count, quantity, promotions and department totals are excluded because they would be unavailable at hour start.')
    p('Each family chooses one shared hyperparameter candidate by averaging the two variants\' validation RMSE. This controls the paired payday comparison: both variants use the same chosen structural setting, source data, cutoffs and seed. LSTM early stopping may select different epochs because each variant has its own validation loss. Each pair has the same non-payday inputs; only the proxy column is added. Cross-family differences in representation are reported rather than hidden.')
    h('Model architectures and selected settings')
    p(f'Prophet uses an additive piecewise linear trend, daily Fourier order {c["prophet"]["daily_fourier_order"]} and weekly Fourier order {c["prophet"]["weekly_fourier_order"]}, with yearly seasonality disabled because training covers only one year. Candidates for changepoint prior scale are {c["prophet"]["changepoint_prior_scale_candidates"]}; validation selected {logs["prophet"]["chosen_changepoint_prior_scale"]}. Past-sales features, missingness, monthly cyclic terms and weekend are additive regressors with prior scale {c["prophet"]["regressor_prior_scale"]}. Duplicate hour/weekday harmonics and the explicit trend column are omitted from Prophet regressors because Prophet already supplies those structures. Internal regressor standardization learns from training. Uncertainty simulation is disabled; forecasts are point estimates.')
    p(f'XGBoost uses squared-error histogram gradient boosting, {c["xgboost"]["n_estimators"]} trees, learning rate {c["xgboost"]["learning_rate"]}, min_child_weight {c["xgboost"]["min_child_weight"]}, full row and column sampling and two CPU threads. Depth candidates {c["xgboost"]["max_depth_candidates"]} give selected depth {logs["xgboost"]["chosen_max_depth"]}. All tabular predictors are included. Tree count is fixed and selected depth uses only validation.')
    p(f'LSTM encodes the previous {c["lstm"]["sequence_hours"]} elapsed clock hours. At each step it receives standardized sales, an observation flag and known calendar predictors; the payday variant also receives the historical proxy. A second input provides the target-hour calendar and, for the payday variant, its proxy. One LSTM layer with selected {logs["lstm"]["chosen_units"]} units feeds a 16-unit ReLU dense layer and a linear output. Candidates are {c["lstm"]["units_candidates"]} units. Adam learning rate is {c["lstm"]["learning_rate"]}, batch size {c["lstm"]["batch_size"]}, maximum {c["lstm"]["max_epochs"]} epochs, patience {c["lstm"]["patience"]}. Model training uses shuffle=False and restored best validation weights.')
    p('The LSTM target scaler learns the mean and standard deviation of observed eligible training sales only. Missing historical sales are represented by the training mean (scaled value zero) together with observed=0; this is a modelling imputation and does not create zero-sales targets. True standardized zero with observed=1 means an observed value equal to the training mean. All 24 steps remain on the time grid even overnight. Predictions inverse-transform the single target channel to GBP, never a feature matrix.')
    for family in ['prophet','xgboost','lstm']:
        h(f'{dict(prophet="Prophet",xgboost="XGBoost",lstm="LSTM")[family]} validation candidates',2); blocks.append(table_block(pd.DataFrame(logs[family]['trials'])))
    p('Naive, daily seasonal-naive and weekly seasonal-naive baselines use sales from t-1, t-24 and t-168 elapsed hours respectively. If the corresponding historical hour is unrecorded, they use the eligible training median and report fallback counts. They use the same forecast origins and test targets as the main models. Baselines do not use payday features.')
    h('Prediction validation and error measures')
    p(f'Each saved forecast contains UTC timestamp, offset-labelled local timestamp, actual net sales, predicted sales, model variant and proxy indicator. Full-precision CSV serialization uses 17 significant digits. Before joining, the comparison validates unique ordered timestamps, finite values, test membership, labels, identical proxy values and actual-sales agreement at rtol=1e-10 and atol=1e-8. It also checks prepared-data SHA256 and configuration provenance. Common dates run from {cv["start_utc"]} to {cv["end_utc"]}; every eligible observed test hour is shared by the six variants.')
    p('MAE is the mean absolute forecast error; RMSE is the square root of mean squared error. Both are in GBP per recorded hour and use unrounded predictions and actuals. RMSE places greater weight on large errors. Change equals error with payday minus error without payday; percentage change divides by the without-payday error. No percentage accuracy or MAPE is used because net sales can be near zero or negative. Tables round only presentation values.')
    h('Common test results')
    blocks.append(table_block(common[['model','n','MAE','RMSE']]))
    fig('error_comparison.png','Figure 2. MAE and RMSE across all main variants and baselines on the identical observed test hours.')
    h('Payday and non payday test results')
    for group in ['payday','non_payday']:
        h(group.replace('_',' ').capitalize(),2)
        rows=metrics[(metrics.scope=='common')&(metrics.group==group)]
        blocks.append(table_block(rows[['model','n','MAE','RMSE']]))
    h('Change after adding the payday proxy')
    effect_table=effects[['family','group','n','MAE_change','MAE_change_percent','RMSE_change','RMSE_change_percent']].rename(columns={'MAE_change':'MAE delta GBP','MAE_change_percent':'MAE delta %','RMSE_change':'RMSE delta GBP','RMSE_change_percent':'RMSE delta %'})
    b=table_block(effect_table); b['width_fractions']=[.16,.17,.07,.15,.15,.15,.15]; blocks.append(b)
    fig('payday_effect.png','Figure 3. Paired error changes over all common test hours. Negative bars indicate improvement and positive bars indicate deterioration.')
    h('Full test coverage and baseline fallbacks')
    full=metrics[(metrics.scope=='full_test')&(metrics.group=='all')]
    blocks.append(table_block(full[['model','n','MAE','RMSE']]))
    p('All full-test sample sizes equal the common sample in this execution. The separately saved full-test rows remain useful if later variants have different coverage; the comparison does not infer equal tasks merely from timestamp overlap. Model-specific segment metrics and UTC evaluation start/end dates are saved in outputs/metrics/model_metrics.csv. Exact per-hour local dates are in the predictions.')
    base=json.loads((OUT/'logs'/'baselines.json').read_text())
    blocks.append(table_block(pd.DataFrame([dict(model=k.replace('_test',''),fallback_hours=v['fallback_hours'],fallback_value=v['fallback_value']) for k,v in base.items() if k.endswith('_test')])))
    h('Forecast and residual diagnostics')
    fig('actual_vs_predicted.png','Figure 4. First seven test days. Points show recorded hours without drawing lines across unknown overnight sales.')
    p('Residuals are predicted minus actual net sales. Large negative residuals indicate underprediction. The scatter plots assess whether error magnitude changes with sales level; daily mean residuals reveal sustained bias or drift. They are descriptive diagnostics, not proof of independent errors or calibrated probabilistic uncertainty.')
    diagnostics=pd.read_csv(OUT/'metrics'/'residual_diagnostics.csv')
    blocks.append(table_block(diagnostics[['model','mean_residual','lag1_clock_hour_correlation','negative_predictions']]))
    training_mean=d.loc[mask(d,'train'),'sales'].mean(); test_mean=d.loc[mask(d,'test'),'sales'].mean()
    p(f'The mean eligible training target is GBP {training_mean:.3f}, compared with GBP {test_mean:.3f} in the test period. All six models have negative mean residuals in this holdout, indicating systematic underprediction. The residual plots show especially large underprediction for high-sales hours and sustained late-test negative bias. This is compatible with distribution drift and regression toward typical training sales, but cannot identify a causal explanation. The lag-one correlations pair only adjacent elapsed clock hours with recorded residuals; gaps are not treated as adjacent samples.')
    fig('residuals.png','Figure 5. Residual versus actual net sales for each main model variant.')
    fig('residual_time.png','Figure 6. Daily mean residuals across the test period; each daily mean covers recorded hours only.')
    h('XGBoost SHAP analysis')
    p('Tree SHAP explains fitted XGBoost predictions using the tree_path_dependent convention. Explanations use up to 500 evenly spaced common test hours after model selection. The implementation verifies that base value plus the sum of contributions reconstructs predictions within GBP 0.01. Contributions are in GBP, and the reported mean absolute contributions summarize influence in this sample. Correlated lag, calendar and payday features can share attribution; rankings depend on the fitted model and explanation convention.')
    for pay in ['Without_Payday','With_Payday']:
        name='XGBoost_'+pay
        ranks=pd.read_csv(OUT/'metrics'/f'{name}_shap_importance.csv')
        top=', '.join(ranks.feature.head(3))
        p(f'{name.replace("_"," ")}: the three largest mean absolute SHAP contributors are {top}.')
        if pay=='With_Payday':
            payday_row=ranks[ranks.feature=='payday'].iloc[0]
            p(f'The payday indicator has mean absolute contribution GBP {payday_row.mean_absolute_shap:.3f} and rank {int(np.flatnonzero(ranks.feature.eq("payday"))[0])+1} of {len(ranks)} predictors. Attribution does not establish a causal payday response or prove out-of-sample usefulness; the paired holdout errors provide the predictive comparison.')
        blocks.append(table_block(ranks.head(8)))
        fig(f'{name}_shap.png',f'Figure {7 if pay=="Without_Payday" else 8}. {name.replace("_"," ")} Tree SHAP on recorded test hours.')
    h('Reproducibility and execution evidence')
    p(f'Execution uses Python {versions["python"].split()[0]} on {versions["platform"]}, seed {c["seed"]}, fixed chronological masks, train-only learned transformations and CPU thread limits. TensorFlow deterministic operations are enabled and oneDNN custom optimizations are disabled. Exact package versions, candidate scores, training histories, trained artifacts and prediction provenance are saved. Bitwise equality across different hardware or numerical libraries is not guaranteed.')
    blocks.append(table_block(pd.DataFrame(list(versions['packages'].items()),columns=['Package','Version'])))
    verify=json.loads((OUT/'logs'/'verification.json').read_text())
    p('Executed verification checks: '+ '; '.join(k.replace('_',' ') for k,v in verify.items() if v)+'. The causality check replaces target and future observations at representative origins in each split and confirms that origin predictors are unchanged. The LSTM inverse-transform roundtrip and training-only imputer statistics are checked explicitly.')
    rep=OUT/'logs'/'reproducibility_check.json'
    if rep.exists():
        reproduction=json.loads(rep.read_text())
        same_data=reproduction['prepared_sha256']==hashlib.sha256((OUT/'prepared'/'hourly_grid.csv').read_bytes()).hexdigest()
        if reproduction['config']==c and same_data:
            maximum=max(v['max_absolute_prediction_difference'] for v in reproduction['variants'].values())
            p(f'A second complete execution compared the saved main-model forecasts with the first execution. Timestamps and actual sales were identical for all six variants; the maximum absolute prediction difference across variants was GBP {maximum:.17g}. This supports numerical reproducibility in the same environment, without establishing portability to other machines.')
    h('Limitations privacy and conclusions')
    p('The evidence covers one store, one temporal holdout and observed sales-bearing or adjustment-bearing hours. Unknown trading hours, recording gaps, setup records, supplier accounting, voucher treatment, timezone and tax metadata limit external validity. No actual customer payday data, weather, local events, stock availability or prospective promotion schedule were available. Late-period demand shifts can disadvantage frozen training weights. Candidate grids and single-seed LSTM runs are intentionally modest; family rankings may change with broader validation or repeated seeds.')
    p('No inferential hypothesis test or statistical significance claim is included. Hourly forecast errors can be serially dependent, and payday windows repeat in clusters. A future comparison could use paired date-level loss differences with a block bootstrap and sensitivity to block length, after checking approximate stability; abrupt drift and a small number of payday cycles would still limit inference. Ordinary independent-sample tests on these hourly errors would be inappropriate.')
    p('The transaction export includes timestamps, product descriptions, barcodes, receipt IDs, payment-type codes and a Customer_ID field. Even small integer identifiers should not be presumed anonymous. Products and detailed purchase timing can enable indirect identification when linked to other information. Research sharing should exclude raw customer and receipt identifiers, restrict access to originals, publish aggregated outputs, document consent or other lawful basis with the institution, and apply a retention policy. Reports and model features here omit customer identifiers; source files remain locally preserved. Small hourly aggregates and trained models can still reveal business-sensitive patterns, so they should not be published without store authorization.')
    p(f'For this defined task, {best_main.model.replace("_"," ")} gives the lowest main-model RMSE. The numerical payday changes above determine whether the proxy helped each family; predictive gain should not be generalized to customer income causality. Completed outputs include the five-source audit, six trained variants, three baselines, validation selection, aligned full-precision test predictions, subgroup errors, Tree SHAP and descriptive charts. Merchant-confirmed opening hours and metadata, additional temporal folds, payday-window sensitivity, broader tuning and prospective monitoring remain future work.')
    h('Method documentation references')
    p('References below are primary software documentation consulted for method implementation, not sources for the store-specific empirical results. Accessed 2 October 2026. The six-day payday window is an explicit research assumption and is not attributed to these sources.')
    references=[('Prophet additive regressors and seasonality','https://facebook.github.io/prophet/docs/seasonality,_holiday_effects,_and_regressors.html'),('XGBoost parameter documentation','https://xgboost.readthedocs.io/en/stable/parameter.html'),('TensorFlow LSTM layer documentation','https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM'),('SHAP TreeExplainer documentation','https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html'),('Scikit learn data leakage guidance','https://scikit-learn.org/stable/common_pitfalls.html#data-leakage')]
    for title,url in references: p(f'{title}: {url}')
    h('Appendix source schema and detailed audit')
    for r in audit:
        h(r['file'].replace('.csv','').replace('_',' '),2)
        p(f'Role: {r["role"]}. Timestamp column: {r["timestamp_column"]}. Coverage: {r["start"]} to {r["end"]}. Exact row duplicates: {r["exact_duplicates"]}. Timestamp repetitions: {r["timestamp_duplicates"]}; repetition is legitimate for line items and department records, but is checked as an error for total-shop predictions.')
        blocks.append(table_block(pd.DataFrame([{'Column':col,'Type':r['dtypes'][col],'Missing':r['missing'][col]} for col in r['columns']])))
    h('Appendix reproducible commands')
    p('From project_by_gpt, create a Python 3.11 environment and install requirements.txt. Run python run_all.py for the full workflow. Independent commands are python src/data_preparation.py, python src/baselines.py, python src/prophet_model.py, python src/xgboost_model.py, python src/lstm_model.py, python src/model_comparison.py, python src/verify_workflow.py and python src/research_report.py. The comparison and reporting commands do not retrain models. Changing config requires re-preparing data and rerunning affected models; provenance validation rejects stale forecasts.')
    dump(OUT/'logs'/'report_payload.json',blocks)
    md=[]
    for b in blocks:
        if b['kind']=='heading': md.append('#'*(b['level']+1)+' '+b['text'])
        elif b['kind']=='paragraph': md.append(b['text'])
        elif b['kind']=='table':
            md.append('| '+' | '.join(b['headers'])+' |\n| '+' | '.join(['---']*len(b['headers']))+' |\n'+'\n'.join('| '+' | '.join(row)+' |' for row in b['rows']))
        elif b['kind']=='figure': md.append(f'![{b["caption"]}](../outputs/charts/{Path(b["path"]).name})\n\n{b["caption"]}')
    (ROOT/'reports'/'research_report.md').write_text('\n\n'.join(md)+'\n',encoding='utf-8')
    bundled=Path.home()/'.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
    python=os.environ.get('REPORT_PYTHON',str(bundled) if bundled.exists() else sys.executable)
    subprocess.run([python,str(ROOT/'src'/'report_artifacts.py')],check=True,cwd=ROOT)
    print('Markdown, Word and PDF report generated from saved outputs.')

if __name__=='__main__': run()
