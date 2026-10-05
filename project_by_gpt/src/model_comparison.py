"""Validate and compare saved forecasts. This module never trains a model."""
from data_preparation import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MODELS=[f'{family}_{pay}' for family in ['Prophet','XGBoost','LSTM'] for pay in ['Without_Payday','With_Payday']]
BASELINES=['Naive','Daily_Seasonal_Naive','Weekly_Seasonal_Naive']

def read_prediction(name,d):
    p=OUT/'predictions'/f'{name}_test.csv'
    f=pd.read_csv(p); f['timestamp']=pd.to_datetime(f.timestamp,utc=True)
    meta=json.loads(p.with_suffix('.metadata.json').read_text())
    if meta['config']!=config(): raise ValueError(f'Stale config: {name}')
    if meta['prepared_sha256']!=hashlib.sha256((OUT/'prepared'/'hourly_grid.csv').read_bytes()).hexdigest(): raise ValueError(f'Stale prepared data: {name}')
    if meta['forecast_horizon_hours']!=1 or meta['train_observations']!=int(mask(d,'train').sum()) or not meta['no_refit_on_validation']:
        raise ValueError(f'Incompatible forecasting design: {name}')
    if pd.Timestamp(meta['train_end'])!=d.loc[mask(d,'train'),'timestamp'].max():
        raise ValueError(f'Incompatible training cutoff: {name}')
    if f.timestamp.duplicated().any() or not np.isfinite(f[['actual_sales','predicted_sales','payday']].to_numpy()).all(): raise ValueError(f'Invalid values: {name}')
    if not f.model.eq(name).all(): raise ValueError(f'Model label mismatch: {name}')
    expected=d.set_index('timestamp').reindex(f.timestamp)
    if not expected.split.eq('test').all() or not expected.sales.notna().all(): raise ValueError(f'Invalid test timestamps: {name}')
    if not np.allclose(expected.sales,f.actual_sales,rtol=1e-10,atol=1e-8): raise ValueError(f'Actual sales disagree: {name}')
    if not np.array_equal(expected.payday.to_numpy(),f.payday.to_numpy()): raise ValueError(f'Payday disagreement: {name}')
    if not f.timestamp.is_monotonic_increasing: raise ValueError(f'Unsorted timestamps: {name}')
    return f.set_index('timestamp')

def rows_for(f,name,scope):
    rows=[]
    for group in ['all','payday','non_payday']:
        z=f if group=='all' else f[f.payday.eq(1 if group=='payday' else 0)]
        if not len(z): continue
        rows.append(dict(model=name,scope=scope,group=group,start_utc=z.index.min(),end_utc=z.index.max(),**metric(z.actual_sales,z.predicted_sales)))
    return rows

def charts(common,metrics,effects,d):
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(3,1,figsize=(12,9),sharex=True)
    recent=common[common.index<common.index.min()+pd.Timedelta(days=7)]
    # Scatter avoids lines interpolating overnight unknown hours.
    for ax,family in zip(axs,['Prophet','XGBoost','LSTM']):
        ax.scatter(recent.index,recent.actual_sales,s=9,c='black',label='Actual net sales')
        for pay,col in [('Without_Payday','#2671ad'),('With_Payday','#d37a21')]:
            ax.scatter(recent.index,recent[f'{family}_{pay}'],s=7,alpha=.7,color=col,label=pay.replace('_',' '))
        ax.set_ylabel('Net sales (GBP)'); ax.set_title(f'{family}: first seven test days, observed hours'); ax.legend(ncol=3,fontsize=8)
    axs[-1].set_xlabel('Hour start (UTC)'); fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(OUT/'charts'/'actual_vs_predicted.png',dpi=160); plt.close(fig)
    allmetrics=metrics[(metrics.scope=='common')&(metrics.group=='all')].set_index('model').reindex(MODELS+BASELINES)
    fig,axs=plt.subplots(1,2,figsize=(12,5))
    for ax,met in zip(axs,['MAE','RMSE']):
        ax.barh(allmetrics.index.str.replace('_',' '),allmetrics[met],color=['#2671ad','#d37a21']*3+['#777777']*3)
        ax.invert_yaxis(); ax.set_xlabel(f'{met} (GBP)'); ax.set_title('Common observed test hours')
    fig.tight_layout(); fig.savefig(OUT/'charts'/'error_comparison.png',dpi=160); plt.close(fig)
    ef=effects[effects.group=='all']
    fig,ax=plt.subplots(figsize=(8,4)); x=np.arange(len(ef)); ax.bar(x-.18,ef.MAE_change,width=.36,label='MAE change'); ax.bar(x+.18,ef.RMSE_change,width=.36,label='RMSE change')
    ax.axhline(0,color='black',lw=.8); ax.set_xticks(x,ef.family); ax.set_ylabel('Error with payday minus without (GBP)'); ax.set_title('Payday proxy effect: negative means lower error'); ax.legend(); fig.tight_layout(); fig.savefig(OUT/'charts'/'payday_effect.png',dpi=160); plt.close(fig)
    fig,axs=plt.subplots(3,2,figsize=(12,9))
    for ax,name in zip(axs.flat,MODELS):
        residual=common[name]-common.actual_sales
        ax.scatter(common.actual_sales,residual,s=5,alpha=.4); ax.axhline(0,color='black',lw=.8); ax.set_title(name.replace('_',' ')); ax.set_xlabel('Actual net sales (GBP)'); ax.set_ylabel('Predicted minus actual (GBP)')
    fig.tight_layout(); fig.savefig(OUT/'charts'/'residuals.png',dpi=160); plt.close(fig)
    fig,axs=plt.subplots(2,1,figsize=(11,6))
    monthly=pd.read_csv(OUT/'audit'/'monthly_sales.csv'); axs[0].plot(monthly.month,monthly.sales,marker='o'); axs[0].tick_params(axis='x',rotation=60); axs[0].set_ylabel('Observed net sales (GBP)'); axs[0].set_title('Monthly totals; July is a partial month')
    hour=pd.read_csv(OUT/'audit'/'hour_coverage.csv'); axs[1].bar(hour.hour,hour.observed_hours/hour.clock_hours); axs[1].set_xticks(range(24)); axs[1].set_ylabel('Proportion recorded'); axs[1].set_xlabel('Local shop hour'); axs[1].set_title('Coverage does not establish opening hours')
    fig.tight_layout(); fig.savefig(OUT/'charts'/'exploratory.png',dpi=160); plt.close(fig)
    fig,axs=plt.subplots(3,1,figsize=(11,8),sharex=True)
    for ax,family in zip(axs,['Prophet','XGBoost','LSTM']):
        for pay in ['Without_Payday','With_Payday']:
            e=common[f'{family}_{pay}']-common.actual_sales
            daily=e.groupby(e.index.tz_convert(config()['timezone']).date).mean()
            ax.plot(pd.to_datetime(daily.index),daily.values,label=pay.replace('_',' '))
        ax.axhline(0,color='black',lw=.8); ax.set_ylabel('Mean residual (GBP)'); ax.set_title(family); ax.legend(fontsize=8)
    axs[-1].set_xlabel('Local date'); fig.tight_layout(); fig.savefig(OUT/'charts'/'residual_time.png',dpi=160); plt.close(fig)

def run():
    d=load_data(); forecasts={name:read_prediction(name,d) for name in MODELS+BASELINES}
    common_index=forecasts[MODELS[0]].index
    for name in MODELS[1:]: common_index=common_index.intersection(forecasts[name].index)
    if not len(common_index): raise ValueError('No common valid test hours')
    common_index=common_index.sort_values(); expected_index=pd.DatetimeIndex(d.loc[mask(d,'test'),'timestamp'])
    common=forecasts[MODELS[0]].loc[common_index,['actual_sales','payday','local_timestamp']].copy()
    records=[]
    for name,f in forecasts.items():
        records.extend(rows_for(f,name,'full_test'))
        if not common_index.isin(f.index).all(): raise ValueError(f'Baseline missing common hours: {name}')
        z=f.loc[common_index]; records.extend(rows_for(z,name,'common'))
        common[name]=z.predicted_sales
    common.to_csv(OUT/'predictions'/'common_test_comparison.csv',float_format='%.17g')
    metrics=pd.DataFrame(records); metrics.to_csv(OUT/'metrics'/'model_metrics.csv',index=False,float_format='%.17g')
    effects=[]
    for family in ['Prophet','XGBoost','LSTM']:
        for group in ['all','payday','non_payday']:
            rows=metrics[(metrics.scope=='common')&(metrics.group==group)].set_index('model')
            a=rows.loc[family+'_Without_Payday']; b=rows.loc[family+'_With_Payday']
            effects.append(dict(family=family,group=group,n=int(a.n),MAE_without=a.MAE,MAE_with=b.MAE,MAE_change=b.MAE-a.MAE,MAE_change_percent=100*(b.MAE-a.MAE)/a.MAE,RMSE_without=a.RMSE,RMSE_with=b.RMSE,RMSE_change=b.RMSE-a.RMSE,RMSE_change_percent=100*(b.RMSE-a.RMSE)/a.RMSE))
    ef=pd.DataFrame(effects); ef.to_csv(OUT/'metrics'/'payday_effects.csv',index=False,float_format='%.17g')
    # Daily residual summaries describe dependence without assuming iid errors.
    residual=[]
    for name in MODELS:
        e=common[name]-common.actual_sales
        groups=e.groupby(e.index.tz_convert(config()['timezone']).date)
        for day,v in groups: residual.append(dict(model=name,date=day,n=len(v),mean_residual=float(v.mean()),MAE=float(v.abs().mean())))
    pd.DataFrame(residual).to_csv(OUT/'metrics'/'daily_residuals.csv',index=False)
    diagnostics=[]
    test_grid=pd.DatetimeIndex(d.loc[d.split.eq('test'),'timestamp'])
    for name in MODELS:
        e=common[name]-common.actual_sales
        clock=e.reindex(test_grid)
        diagnostics.append(dict(model=name,n=len(e),mean_residual=float(e.mean()),lag1_clock_hour_correlation=float(clock.corr(clock.shift(1))),negative_predictions=int(common[name].lt(0).sum()),max_absolute_error=float(e.abs().max())))
    pd.DataFrame(diagnostics).to_csv(OUT/'metrics'/'residual_diagnostics.csv',index=False)
    dump(OUT/'logs'/'comparison_validation.json',dict(common_n=len(common),expected_observed_test_n=len(expected_index),all_expected_hours_common=common_index.equals(expected_index),start_utc=common.index.min(),end_utc=common.index.max(),payday_n=int(common.payday.sum()),non_payday_n=int(common.payday.eq(0).sum()),actual_agreement_tolerance=dict(rtol=1e-10,atol=1e-8),provenance_checked=True,statistical_tests='none; descriptive errors only'))
    charts(common,metrics,ef,d)
    print(metrics[(metrics.scope=='common')&(metrics.group=='all')][['model','n','MAE','RMSE']].to_string(index=False))
    return metrics,ef

if __name__=='__main__': run()
