"""Compare saved daily forecasts, block-dependent uncertainty and planning scenarios."""
from extension_preparation import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
FAMILIES=['Prophet','XGBoost','LSTM','Hybrid']
def read(name,split,f):
    p=EXT/'predictions'/f'{name}_{split}.csv'; z=pd.read_csv(p,parse_dates=['timestamp']).set_index('timestamp')
    meta=json.loads(p.with_suffix('.metadata.json').read_text()); provenance=json.loads((EXT/'logs'/'provenance.json').read_text())
    if meta['provenance']!=provenance: raise ValueError('Stale forecast provenance')
    expected=f.iloc[indices(f,split)]
    if not z.index.equals(expected.index) or not z.model.eq(name).all() or not np.isfinite(z[['actual_sales','predicted_sales']]).all().all(): raise ValueError('Alignment or values')
    if not np.allclose(z.actual_sales,expected.sales,rtol=1e-10,atol=1e-8): raise ValueError('Actual mismatch')
    for col in PAY:
        if not np.array_equal(z[col],expected[col]): raise ValueError('Proxy mismatch')
    return z
def run():
    f=load(); tr=indices(f,'train'); te=indices(f,'test'); median=f.sales.iloc[tr].median()
    for lag,name in [(1,'Naive'),(7,'Weekly_Seasonal_Naive')]:
        for split in ['validation','test']:
            ix=indices(f,split); save(f,ix,f.sales.shift(lag).iloc[ix].fillna(median),name,split)
    names=[fam+'_'+pay for fam in FAMILIES for pay in ['Without_Payday','With_Payday']]+['Naive','Weekly_Seasonal_Naive']+[f'LSTM_{pay}_seed_{seed}' for seed in [43,44] for pay in ['Without_Payday','With_Payday']]
    frames={name:read(name,'test',f) for name in names}; rows=[]; periods=[]; operations=[]; effects=[]
    if not (next(iter(frames.values())).index.to_series().diff().dropna()==pd.Timedelta(days=1)).all():
        raise ValueError('Daily block bootstrap requires contiguous test dates')
    threshold=f.sales.iloc[tr].quantile(settings()['staffing_peak_quantile'])
    for name,z in frames.items():
        for group in ['all','payday','non_payday']:
            a=z if group=='all' else z[z.payday.eq(int(group=='payday'))]
            rows.append({'model':name,'scope':'common_and_full','group':group,'start':a.index.min(),'end':a.index.max(),**score(a.actual_sales,a.predicted_sales)})
        for month,g in z.groupby(z.index.strftime('%Y-%m')): periods.append({'model':name,'month':month,**score(g.actual_sales,g.predicted_sales)})
        actual=z.actual_sales>=threshold; pred=z.predicted_sales>=threshold
        tp=int((actual&pred).sum()); operations.append({'model':name,'training_threshold_GBP':threshold,'peak_precision':tp/max(1,int(pred.sum())),'peak_recall':tp/max(1,int(actual.sum())),'underforecast_GBP':float(np.maximum(z.actual_sales-z.predicted_sales,0).sum()),'overforecast_GBP':float(np.maximum(z.predicted_sales-z.actual_sales,0).sum()),'scope':'illustrative revenue workload proxy, not staffing costs or stock units'})
    metrics=pd.DataFrame(rows); metrics.to_csv(EXT/'metrics'/'daily_metrics.csv',index=False,float_format='%.17g'); pd.DataFrame(periods).to_csv(EXT/'metrics'/'monthly_stability.csv',index=False); pd.DataFrame(operations).to_csv(EXT/'metrics'/'planning_scenarios.csv',index=False)
    hourly=pd.read_csv(OUT/'prepared'/'hourly_grid.csv'); threshold_hour=hourly.loc[hourly.split.eq('train')&hourly.sales.notna(),'sales'].quantile(.75); hour_ops=[]
    for fam in ['Prophet','XGBoost','LSTM']:
        for pay in ['Without_Payday','With_Payday']:
            name=fam+'_'+pay; path=OUT/'predictions'/f'{name}_test.csv'
            if not path.exists(): continue
            z=pd.read_csv(path)
            actual=z.actual_sales>=threshold_hour; predicted=z.predicted_sales>=threshold_hour; tp=int((actual&predicted).sum())
            hour_ops.append({'model':name,'n':len(z),'training_hour_threshold_GBP':threshold_hour,'peak_hour_precision':tp/max(1,int(predicted.sum())),'peak_hour_recall':tp/max(1,int(actual.sum())),'scope':'one-hour workload proxy; not required staff counts'})
    pd.DataFrame(hour_ops,columns=['model','n','training_hour_threshold_GBP','peak_hour_precision','peak_hour_recall','scope']).to_csv(EXT/'metrics'/'hourly_planning_scenarios.csv',index=False)
    rng=np.random.default_rng(42)
    for fam in FAMILIES:
        a=frames[fam+'_Without_Payday']; b=frames[fam+'_With_Payday']
        for loss in ['absolute','squared']:
            ea=a.predicted_sales-a.actual_sales; eb=b.predicted_sales-b.actual_sales; delta=(np.abs(eb)-np.abs(ea)).to_numpy() if loss=='absolute' else (eb**2-ea**2).to_numpy(); n=len(delta)
            for block in settings()['bootstrap_block_days']:
                boot=[]
                for _ in range(settings()['bootstrap_replicates']):
                    starts=rng.integers(0,n,size=int(np.ceil(n/block))); ix=np.concatenate([(start+np.arange(block))%n for start in starts])[:n]; boot.append(delta[ix].mean())
                effects.append({'family':fam,'loss':loss,'block_days':block,'n':n,'mean_with_minus_without':delta.mean(),'ci_2_5':np.quantile(boot,.025),'ci_97_5':np.quantile(boot,.975),'assumption':'circular blocks assume approximately stable dependent daily losses; drift and few cycles limit inference'})
    pd.DataFrame(effects).to_csv(EXT/'metrics'/'block_bootstrap.csv',index=False,float_format='%.17g')
    merge=next(iter(frames.values()))[['actual_sales','payday']].copy()
    for name,z in frames.items(): merge[name]=z.predicted_sales
    merge.to_csv(EXT/'predictions'/'daily_common_comparison.csv',float_format='%.17g')
    paired=[]
    for fam in FAMILIES:
        for group in ['all','payday','non_payday']:
            a=metrics[(metrics.model==fam+'_Without_Payday')&(metrics.group==group)].iloc[0]; b=metrics[(metrics.model==fam+'_With_Payday')&(metrics.group==group)].iloc[0]
            paired.append({'family':fam,'group':group,'n':a.n,'MAE_change':b.MAE-a.MAE,'RMSE_change':b.RMSE-a.RMSE,'MAE_percent':100*(b.MAE/a.MAE-1),'RMSE_percent':100*(b.RMSE/a.RMSE-1)})
    pd.DataFrame(paired).to_csv(EXT/'metrics'/'payday_effects.csv',index=False,float_format='%.17g')
    fig,axes=plt.subplots(4,1,figsize=(11,10),sharex=True)
    for fam,ax in zip(FAMILIES,axes):
        ax.plot(merge.index,merge.actual_sales,c='black',label='Actual recorded daily sales')
        for pay in ['Without_Payday','With_Payday']: ax.plot(merge.index,merge[fam+'_'+pay],label=pay)
        ax.set(title=fam,ylabel='GBP/day'); ax.legend(fontsize=7)
    fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(EXT/'charts'/'daily_actual_predictions.png',dpi=150); plt.close(fig)
    main=metrics[(metrics.group=='all')&~metrics.model.str.contains('seed')]
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for ax,col in zip(axes,['MAE','RMSE']): ax.barh(main.model,main[col]); ax.set(xlabel=f'{col} GBP/day',title='Same upcoming-day test targets')
    fig.tight_layout(); fig.savefig(EXT/'charts'/'daily_errors.png',dpi=150); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4)); effect=pd.DataFrame(paired).query("group=='all'")
    for ax,col in zip(axes,['MAE_change','RMSE_change']): ax.bar(effect.family,effect[col]); ax.axhline(0,c='black'); ax.set(title=col,ylabel='With minus without GBP/day')
    fig.tight_layout(); fig.savefig(EXT/'charts'/'daily_payday_effect.png',dpi=150); plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for ax,fam in zip(axes.flat,FAMILIES):
        for pay in ['Without_Payday','With_Payday']: ax.plot(merge.index,merge[fam+'_'+pay]-merge.actual_sales,label=pay)
        ax.axhline(0,c='black'); ax.set(title=fam,ylabel='Residual GBP/day'); ax.legend(fontsize=6)
    fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(EXT/'charts'/'daily_residuals.png',dpi=150); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5)); temporal=pd.DataFrame(periods)
    for fam in FAMILIES:
        z=temporal[temporal.model.eq(fam+'_With_Payday')]; axes[0].plot(z.month,z.RMSE,marker='o',label=fam)
    axes[0].set(title='Monthly test stability, payday variants',ylabel='RMSE GBP/day'); axes[0].legend(fontsize=8)
    seed_rows=metrics[(metrics.group=='all')&metrics.model.str.startswith('LSTM')]
    for k,pay in enumerate(['Without_Payday','With_Payday']):
        z=seed_rows[seed_rows.model.str.contains(pay)]; axes[1].scatter([42,43,44],z.RMSE,label=pay)
    axes[1].set(title='LSTM seeds reported without test selection',xlabel='Seed',ylabel='RMSE GBP/day'); axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(EXT/'charts'/'stability.png',dpi=150); plt.close(fig)
    dump(EXT/'logs'/'comparison.json',{'validated_models':names,'n_common':len(merge),'positive_MAPE_only':True,'test_selection':False,'all_test_dates_contiguous':bool((merge.index.to_series().diff().dropna()==pd.Timedelta(days=1)).all())})
    print(main[['model','n','MAE','RMSE']].to_string(index=False))
if __name__=='__main__': run()
