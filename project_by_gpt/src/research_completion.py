"""Build a verified methods update, targeted review and governance pack; no dissertation."""
from extension_preparation import *
from pathlib import Path
import os,sys,subprocess
import matplotlib
matplotlib.use('Agg')
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def markdown_blocks(path):
    blocks=[]
    for paragraph in path.read_text(encoding='utf-8').split('\n\n'):
        paragraph=paragraph.strip()
        if not paragraph: continue
        if paragraph.startswith('#'):
            heading=paragraph.lstrip('#').strip(); blocks.append({'kind':'heading','text':heading,'level':2})
        elif paragraph.startswith('- '):
            for entry in paragraph.split('\n'):
                if entry.strip(): blocks.append({'kind':'paragraph','text':entry.strip().removeprefix('- ')})
        else: blocks.append({'kind':'paragraph','text':paragraph.replace('\n',' ')})
    return blocks

def render_markdown(blocks):
    lines=[]
    for b in blocks:
        if b['kind']=='heading': lines.append('#'*(b['level']+1)+' '+b['text'])
        elif b['kind']=='paragraph': lines.append(b['text'])
        elif b['kind']=='table': lines.append('| '+' | '.join(b['headers'])+' |\n| '+' | '.join(['---']*len(b['headers']))+' |\n'+'\n'.join('| '+' | '.join(row)+' |' for row in b['rows']))
        else: lines.append(f"![{b['caption']}](../outputs/extensions/charts/{Path(b['path']).name})")
    return '\n\n'.join(lines)+'\n'

def run():
    f=load(); blocks=[]
    def h(text,level=1): blocks.append({'kind':'heading','text':text,'level':level})
    def p(text): blocks.append({'kind':'paragraph','text':text})
    def table(df): blocks.append({'kind':'table','headers':[str(c).replace('_',' ') for c in df.columns],'rows':[[f'{v:.3f}' if isinstance(v,(float,np.floating)) else str(v) for v in row] for row in df.itertuples(index=False,name=None)]})
    def fig(name,caption): blocks.append({'kind':'figure','path':str(EXT/'charts'/name),'caption':caption})
    metrics=pd.read_csv(EXT/'metrics'/'daily_metrics.csv'); overall=metrics[metrics.group.eq('all')]
    cascade=overall[overall.model.str.startswith('Sequential_Hybrid')]; online=pd.read_csv(EXT/'metrics'/'online_payroll_metrics.csv')
    runlog=json.loads((EXT/'logs'/'sequential_hybrid.json').read_text()); gov=json.loads((EXT/'logs'/'governance_readiness.json').read_text())
    h('Retail forecasting methods and evidence update',0)
    p('This update completes the sequential residual hybrid, records online employer-payroll sources, develops the research literature discussion and supplies proposed governance/operational protocols. It is not a dissertation: preparation of the dissertation was explicitly deferred by the researcher. Results below derive from saved executed outputs; customer payroll exposure, permission and realised operational benefits are not invented.')
    h('Executed sequential hybrid')
    p('The forecast is Prophet prediction plus XGBoost residual correction plus LSTM remaining-residual correction. The existing convex ensemble is preserved separately. Both cascade variants use the same local-day forecast origin, target, test dates, structural settings and seed. Only the three payroll proxies differ. Source CSVs and the earlier base forecasts are preserved.')
    p('To avoid learning later stages from earlier stages\' in-sample residuals, Prophet fits the eligible January-June 2025 block (153 days), XGBoost fits July-September (92 days), and LSTM fits October-December (92 days). Earlier stages remain frozen. Historical residuals are available only after their days complete. Stage-specific imputers and residual scaling fit their respective training blocks. January-March 2026 supplies LSTM stopping information, never final-test outcomes.')
    p('Prophet changepoint prior scale is 0.05 with weekly seasonality and historical/calendar regressors. XGBoost has 100 depth-2 trees, learning rate 0.03 and minimum child weight 5. The LSTM has eight units, an eight-unit ReLU dense layer, seven prior local-day residual steps, Adam 0.001, batch size 16, at most 40 epochs and patience 5. These manageable settings were fixed before this run. Fewer training samples per stage are a necessary trade-off of the disjoint-block design, so this is not a same-training-budget architecture contest.')
    p('The original holdout had already been inspected before the cascade was requested. Its assessment is therefore a later exploratory extension, not a newly blinded confirmatory test. No new test-score search was used to choose its settings. Fresh future data would be needed for prospective confirmation.')
    table(cascade[['model','n','MAE','RMSE','MAPE_positive_only']])
    best=overall[~overall.model.str.contains('seed')].sort_values('RMSE').iloc[0]
    p(f'The lowest common daily RMSE remains {best.model} at GBP {best.RMSE:.3f}. The cascade does not outperform it. Payday changes must be interpreted from the paired errors, not assumed beneficial from the architecture.')
    stage_rows=[]
    fig1,axes=plt.subplots(2,1,figsize=(10,7),sharex=True)
    for ax,pay in zip(axes,['Without_Payday','With_Payday']):
        name='Sequential_Hybrid_'+pay; s=pd.read_csv(EXT/'predictions'/f'{name}_test_stages.csv',parse_dates=['timestamp']); z=pd.read_csv(EXT/'predictions'/f'{name}_test.csv')
        np.testing.assert_allclose(s.prophet+s.xgboost_residual_correction+s.lstm_remaining_correction,z.predicted_sales,atol=1e-8)
        np.testing.assert_allclose(z.actual_sales,f.iloc[indices(f,'test')].sales)
        ax.plot(s.timestamp,z.actual_sales,color='black',label='Actual recorded sales'); ax.plot(s.timestamp,s.prophet,label='Prophet stage'); ax.plot(s.timestamp,s.prophet+s.xgboost_residual_correction,label='After XGBoost'); ax.plot(s.timestamp,s.final_prediction,label='Final LSTM correction'); ax.set(title=pay,ylabel='GBP/day'); ax.legend(fontsize=7)
        for label,pred in [('Prophet',s.prophet),('Prophet + XGBoost',s.prophet+s.xgboost_residual_correction),('Full cascade',s.final_prediction)]: stage_rows.append({'variant':pay,'stage':label,**score(z.actual_sales,pred)})
    fig1.autofmt_xdate(); fig1.tight_layout(); fig1.savefig(EXT/'charts'/'cascade_stages.png',dpi=150); plt.close(fig1)
    stage_table=pd.DataFrame(stage_rows); stage_table.to_csv(EXT/'metrics'/'cascade_stage_metrics.csv',index=False,float_format='%.17g'); table(stage_table[['variant','stage','n','MAE','RMSE']]); fig('cascade_stages.png','Figure 1. Frozen sequential stages and actual recorded daily revenue. Adding a correction is not guaranteed to reduce held-out error.')
    h('Online payroll evidence and retrospective experiment')
    p('Official sources identify employer-specific schedules rather than one national public-sector payday. The NAO manual predates the experiment and states a last-working-day rule; its continuation is assumed. UHL resident-doctor guidance states a 27th-of-month rule with banking-day adjustment, but December is an unspecified exception. Bath lists exact payment dates from November 2025 onwards. URLs, retrieval date, derivation type and unknown coverage are recorded in payroll_sources.json and online_payroll_dates.csv. Payroll submission deadlines are not used as payment dates.')
    p('The code leaves UHL December and potentially affected January spillover days unknown, and leaves Bath\'s uncovered earlier months unknown. Exact Bath dates are audited, not fitted as a fictional full calendar. The exploratory XGBoost pair uses only common NAO/UHL coverage, 306 training days and 105 test days, with identical training rows and shared validation-selected depth. The without variant has calendar/history inputs; the with variant adds the two sourced-employer windows. This differs from the primary 337-day training set, so comparisons across those experiments do not isolate only the feature change.')
    p('The UHL page has no established historical publication/effective date. Its historical application is explicitly retrospective and is not certified as information available at every 2025 forecast origin. No source proves these employers are represented among customers. The exact Bath calendar and other employer rules strengthen documentation while retaining these exposure and availability limitations.')
    table(online[online.group.eq('all')][['model','train_n','n','MAE','RMSE']]); fig('online_payroll_comparison.png','Figure 2. Matched sourced-employer calendar scenario; positive-target revenue forecasts, not customer payday identification.')
    table(pd.read_csv(EXT/'metrics'/'online_payroll_effects.csv')); table(pd.read_csv(EXT/'metrics'/'online_payroll_bootstrap.csv')[['block_days','mean_absolute_loss_change','ci_2_5','ci_97_5']])
    p('Employer-calendar error changes are with minus without: negative means improvement. Circular daily block-bootstrap intervals use 1,000 resamples and 7/14-day blocks under an approximate stability assumption. They are exploratory loss-difference intervals, not causal or prospective confirmation.')
    h('Targeted critical literature review')
    blocks.extend(markdown_blocks(ROOT/'reports'/'literature_review.md'))
    h('Data management and disclosure review')
    p(f'A local monthly department-coded review candidate contains {gov["candidate_rows"]} cells, with {gov["suppressed_cells"]} cells suppressed below ten distinct receipts. Values are rounded to whole pounds. Direct customer/receipt/product/payment identifiers are excluded. This threshold is not a proof of anonymity, and totals remain commercially sensitive. No file was published and release_authorised remains false.')
    blocks.extend(markdown_blocks(ROOT/'governance'/'data_management_protocol.md'))
    h('Operational evaluation readiness')
    blocks.extend(markdown_blocks(ROOT/'governance'/'operational_validation_protocol.md'))
    h('Verification and remaining evidence')
    p('Verified code checks include disjoint cascade training blocks, out-of-earlier-stage-training residual targets, training-only scaler inversion, future-residual sequence invariance, additive reconstruction of saved forecasts, actual-sales agreement, explicit payroll unknown coverage and identifier/suppression checks for the review candidate. The final status files distinguish technical completion from missing institutional and operational evidence.')
    p('Remaining items are evidence-dependent: store permission, institutional ethics/lawful-basis decisions, verified storage/access/retention arrangements, actual customer income exposure, complete recording/opening hours, and a prospective operational pilot. Empty templates are not pilot results. Dissertation work is deferred at the researcher\'s request. Online employer schedules do not remove those limits.')
    p('Run python src/sequential_hybrid_model.py, python src/online_payroll_analysis.py, python src/privacy_export.py, python src/next_day_comparison.py and python src/research_completion.py, using the project environment. The comparison and completion report read saved predictions. Full run_all.py and run_extensions.py include the new stages.')
    dump(EXT/'logs'/'completion_payload.json',blocks); (ROOT/'reports'/'research_completion.md').write_text(render_markdown(blocks),encoding='utf-8')
    bundled=Path.home()/'.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'; python=os.environ.get('REPORT_PYTHON',str(bundled) if bundled.exists() else sys.executable)
    subprocess.run([python,str(ROOT/'src'/'report_artifacts.py'),str(EXT/'logs'/'completion_payload.json'),'research_completion'],check=True)
    dump(EXT/'logs'/'completion_verification.json',{'cascade_forecast_stage_sum_matches':True,'cascade_actuals_match_prepared_test':True,'literature_review_exists':True,'governance_protocols_exist':True,'dissertation_deferred_by_user':True,'customer_paydays_verified':False,'release_authorised':False,'model_settings_and_sources_frozen_before_this_run':True})
    print('Research methods/evidence update generated; dissertation deferred')
if __name__=='__main__': run()
