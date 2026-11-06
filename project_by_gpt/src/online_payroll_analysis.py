"""Public employer schedules, explicit unknown coverage, paired exploratory XGBoost."""
from extension_preparation import *
from pathlib import Path
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Transcribed from the source's payment-date column (not either deadline column).
BATH={'2025-11':25,'2025-12':22,'2026-01':23,'2026-02':25,'2026-03':25,'2026-04':24,'2026-05':22,'2026-06':25,'2026-07':24,'2026-08':25,'2026-09':25,'2026-10':23,'2026-11':25,'2026-12':22,'2027-01':25,'2027-02':25,'2027-03':25}

def build_calendar(f):
    cfg=json.loads((ROOT/'payroll_sources.json').read_text()); hs=holidays(); rows=[]
    months=pd.date_range(f.index.min()-pd.Timedelta(days=35),f.index.max()+pd.Timedelta(days=32),freq='MS')
    def prior_banking(day):
        while day.dayofweek>=5 or day.strftime('%Y-%m-%d') in hs: day-=pd.Timedelta(days=1)
        return day
    for m in months:
        for source in cfg['sources']:
            sid=source['id']; date=None
            if sid=='nao': date=prior_banking(m+pd.offsets.MonthEnd(0))
            elif sid=='uhl' and m.month!=12: date=prior_banking(m.replace(day=27))
            elif sid=='bath' and m.strftime('%Y-%m') in BATH: date=m.replace(day=BATH[m.strftime('%Y-%m')])
            rows.append({'source_id':sid,'employer':source['employer'],'month':m.strftime('%Y-%m'),'pay_date':date,'coverage_known':date is not None,'date_type':'published_calendar' if sid=='bath' else 'rule_derived_scenario','source_url':source['url'],'retrieved_on':cfg['retrieved_on'],'customer_exposure_verified':False})
    dates=pd.DataFrame(rows); dates.to_csv(EXT/'audit'/'online_payroll_dates.csv',index=False)
    features=pd.DataFrame(index=f.index)
    for sid in ['nao','uhl','bath']:
        subset=dates[dates.source_id.eq(sid)]; known_months=set(subset[subset.coverage_known].month)
        mapping=dict(zip(subset.month,subset.pay_date))
        values=[]; availability=[]
        for day in f.index:
            month=day.strftime('%Y-%m'); prev=(day.replace(day=1)-pd.Timedelta(days=1)).strftime('%Y-%m')
            known=month in known_months and (day.day>=cfg['window_days'] or prev in known_months)
            active=any(pd.notna(paydate) and 0<=(day-pd.Timestamp(paydate)).days<cfg['window_days'] for paydate in mapping.values())
            values.append(float(active) if known else np.nan); availability.append(int(known))
        features[sid+'_payday']=values; features[sid+'_known']=availability
    features.to_csv(EXT/'prepared'/'online_payroll_features.csv',float_format='%.17g')
    # A documented December exception is unknown, never silently a non-payday.
    december=f.index.month==12
    assert features.loc[december,'uhl_payday'].isna().all()
    assert features.loc[features.bath_known.eq(0),'bath_payday'].isna().all()
    assert features.loc['2026-01-02','uhl_known']==0
    assert dates[(dates.source_id=='bath')&(dates.month=='2026-01')].pay_date.iloc[0]==pd.Timestamp('2026-01-23')
    assert dates[(dates.source_id=='uhl')&(dates.month=='2026-06')].pay_date.iloc[0]==pd.Timestamp('2026-06-26')
    return cfg,dates,features

def run():
    f=load(); cfg,dates,features=build_calendar(f)
    # Bath covers only two training months, so it is audited/described, not used
    # as an invented complete training calendar. UHL/NAO pair uses common coverage.
    eligible=features[['uhl_known','nao_known']].eq(1).all(axis=1).to_numpy()
    split_ix={split:np.array([i for i in indices(f,split) if eligible[i]]) for split in ['train','validation','test']}
    tr,va,te=[split_ix[s] for s in ['train','validation','test']]
    xframe=f.join(features); trials=[]; models={}; names=[]
    for depth in [2,3]:
        for pay in [False,True]:
            cols=columns(False)+(['uhl_payday','nao_payday'] if pay else [])
            imp=SimpleImputer(strategy='median',keep_empty_features=True).fit(xframe.iloc[tr][cols]); x=imp.transform(xframe[cols])
            m=XGBRegressor(n_estimators=200,max_depth=depth,learning_rate=.03,min_child_weight=5,n_jobs=2,tree_method='hist',random_state=42)
            m.fit(x[tr],f.sales.iloc[tr]); vp=m.predict(x[va]); trials.append({'depth':depth,'pay':pay,**score(f.sales.iloc[va],vp)}); models[(depth,pay)]=(m,imp,x,vp,cols)
    chosen=min([2,3],key=lambda dep:np.mean([r['RMSE'] for r in trials if r['depth']==dep])); metrics=[]; frames=[]
    for pay in [False,True]:
        name='Online_Payroll_XGBoost_'+('With_Payday' if pay else 'Without_Payday'); names.append(name); model,imp,x,vp,cols=models[(chosen,pay)]
        for split,ix,pred in [('validation',va,vp),('test',te,model.predict(x[te]))]:
            save(f,ix,pred,name,split,{'payroll_sources':cfg,'train_n':len(tr),'feature_columns':cols,'calendar_sha256':hashlib.sha256((EXT/'audit'/'online_payroll_dates.csv').read_bytes()).hexdigest(),'retrospective_scenario':True})
        model.save_model(EXT/'models'/f'{name}.json'); joblib.dump({'imputer':imp,'columns':cols},EXT/'models'/f'{name}_preprocessing.joblib')
        z=pd.read_csv(EXT/'predictions'/f'{name}_test.csv'); frames.append(z)
        for group in ['all','sourced_window','outside_sourced_window']:
            active=features.iloc[te][['uhl_payday','nao_payday']].eq(1).any(axis=1).to_numpy(); select=np.ones(len(te),bool) if group=='all' else (active if group=='sourced_window' else ~active)
            metrics.append({'model':name,'group':group,'train_n':len(tr),**score(z.actual_sales[select],z.predicted_sales[select])})
    assert frames[0].timestamp.equals(frames[1].timestamp) and np.allclose(frames[0].actual_sales,frames[1].actual_sales)
    assert np.allclose(frames[0].actual_sales,f.sales.iloc[te])
    pd.DataFrame(metrics).to_csv(EXT/'metrics'/'online_payroll_metrics.csv',index=False,float_format='%.17g')
    effects=[]; rng=np.random.default_rng(42)
    for group in ['all','sourced_window','outside_sourced_window']:
        pair=[next(r for r in metrics if r['model']==name and r['group']==group) for name in names]
        effects.append({'group':group,'n':pair[0]['n'],'MAE_change':pair[1]['MAE']-pair[0]['MAE'],'RMSE_change':pair[1]['RMSE']-pair[0]['RMSE']})
    pd.DataFrame(effects).to_csv(EXT/'metrics'/'online_payroll_effects.csv',index=False,float_format='%.17g')
    loss_difference=(np.abs(frames[1].predicted_sales-frames[1].actual_sales)-np.abs(frames[0].predicted_sales-frames[0].actual_sales)).to_numpy(); n=len(loss_difference); intervals=[]
    assert (pd.to_datetime(frames[0].timestamp).diff().dropna()==pd.Timedelta(days=1)).all()
    for block in [7,14]:
        resampled=[]
        for _ in range(1000):
            starts=rng.integers(0,n,int(np.ceil(n/block))); ix=np.concatenate([(start+np.arange(block))%n for start in starts])[:n]; resampled.append(loss_difference[ix].mean())
        intervals.append({'block_days':block,'mean_absolute_loss_change':loss_difference.mean(),'ci_2_5':np.quantile(resampled,.025),'ci_97_5':np.quantile(resampled,.975),'scope':'retrospective paired daily losses; approximate stability assumption'})
    pd.DataFrame(intervals).to_csv(EXT/'metrics'/'online_payroll_bootstrap.csv',index=False,float_format='%.17g')
    plot=pd.DataFrame(metrics).query("group=='all'"); fig,ax=plt.subplots(figsize=(8,4)); ax.bar(['Without employer flags','With employer flags'],plot.RMSE); ax.set(title='Retrospective sourced-employer scenario',ylabel='RMSE GBP/day'); fig.tight_layout(); fig.savefig(EXT/'charts'/'online_payroll_comparison.png',dpi=150); plt.close(fig)
    coverage=features.groupby(f.split).agg({v:'sum' for v in ['uhl_known','nao_known','bath_known']}); coverage.to_csv(EXT/'audit'/'online_payroll_coverage.csv')
    dump(EXT/'logs'/'online_payroll.json',{'sources':cfg,'chosen_depth':chosen,'validation_trials':trials,'split_n':{s:len(ix) for s,ix in split_ix.items()},'verified':{'December_exception_unknown':True,'Bath_uncovered_months_unknown':True,'published_January_date_matches':True,'NHS_weekend_adjustment_matches':True,'paired_test_alignment_and_actuals':True},'limits':'Current UHL rule applied retrospectively; no proof of pre-origin publication or customer employer exposure. Different training coverage from primary comparison. Exact Bath calendar audited but not fitted due sparse training coverage.'})
    print(plot[['model','train_n','n','MAE','RMSE']].to_string(index=False))
if __name__=='__main__': run()
