"""Pre-test expanding validation, payday sensitivity, meaningful leakage tests."""
from extension_preparation import *
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
import joblib
def features_again(f):
    x=f.copy()
    for lag in [1,7,14,28]:
        x[f'lag_{lag}']=x.sales.shift(lag); x[f'lag_{lag}_missing']=x[f'lag_{lag}'].isna().astype(int)
    x['rolling_7']=x.sales.shift(1).rolling(7,min_periods=1).mean()
    return x
def run():
    f=load(); origin=f.index.get_loc(pd.Timestamp(config()['test_start'])); changed=f.copy(); changed.iloc[origin:,changed.columns.get_loc('sales')]=1e9
    a=features_again(f); b=features_again(changed)
    hist=[v for v in columns(False) if v.startswith('lag_') or v=='rolling_7']
    assert np.allclose(a.iloc[origin][hist].to_numpy(float),b.iloc[origin][hist].to_numpy(float),equal_nan=True)
    assert f.index.is_unique and (f.index.to_series().diff().dropna()==pd.Timedelta(days=1)).all()
    # Daily steps are local days. DST days have 23/25 clock hours in hourly grid.
    tr=indices(f,'train'); va=indices(f,'validation'); trials=[]
    scaler=joblib.load(EXT/'models'/'lstm_scaler.joblib')
    assert np.isclose(scaler.mean_[0],f.sales.iloc[tr].mean())
    sample=f.sales.iloc[tr].to_numpy().reshape(-1,1)
    assert np.allclose(scaler.inverse_transform(scaler.transform(sample)),sample)
    q1,q3=f.sales.iloc[tr].quantile([.25,.75]); iqr=q3-q1
    flags=f[['sales','observed_hours','clock_hours','split']].copy()
    flags['training_IQR_extreme_flag']=(flags.sales<q1-3*iqr)|(flags.sales>q3+3*iqr)
    flags['removed']=False; flags.to_csv(EXT/'audit'/'daily_quality_flags.csv')
    for width in [1,3,5]:
        x=f.copy(); x['monthly_payday']=((x.index.day>x.index.days_in_month-width)|(x.index.day<=width)).astype(int)
        cols=columns(True); imp=SimpleImputer(strategy='median',keep_empty_features=True).fit(x.iloc[tr][cols])
        model=XGBRegressor(n_estimators=200,max_depth=2,learning_rate=.03,n_jobs=2,random_state=42)
        model.fit(imp.transform(x.iloc[tr][cols]),x.sales.iloc[tr]); pred=model.predict(imp.transform(x.iloc[va][cols]))
        trials.append({'month_boundary_width':width,'split':'validation_only','not_used_to_change_test_design':True,**score(x.sales.iloc[va],pred)})
    pd.DataFrame(trials).to_csv(EXT/'metrics'/'payday_window_sensitivity_validation.csv',index=False)
    folds=[]
    for start,end in [('2025-10-01','2026-01-01'),('2026-01-01','2026-02-01'),('2026-02-01','2026-03-01'),('2026-03-01','2026-04-01')]:
        train=np.flatnonzero((f.index<pd.Timestamp(start))&f.sales.notna()&(np.arange(len(f))>=28)); val=np.flatnonzero((f.index>=pd.Timestamp(start))&(f.index<pd.Timestamp(end))&f.sales.notna())
        for pay in [False,True]:
            cols=columns(pay); imp=SimpleImputer(strategy='median',keep_empty_features=True).fit(f.iloc[train][cols]); m=XGBRegressor(n_estimators=200,max_depth=2,learning_rate=.03,n_jobs=2,random_state=42)
            m.fit(imp.transform(f.iloc[train][cols]),f.sales.iloc[train]); pred=m.predict(imp.transform(f.iloc[val][cols]))
            folds.append({'family':'XGBoost','payday':pay,'train_end':f.index[train[-1]],'validation_start':start,'validation_end_exclusive':end,**score(f.sales.iloc[val],pred)})
    pd.DataFrame(folds).to_csv(EXT/'metrics'/'expanding_validation.csv',index=False)
    # Original CSV hashes and original numerical outputs stay unchanged.
    original=json.loads((OUT/'audit'/'dataset_audit.json').read_text())
    assert all(hashlib.sha256((ROOT/r['file']).read_bytes()).hexdigest()==r['sha256'] for r in original)
    p=json.loads((EXT/'logs'/'provenance.json').read_text()); assert p['daily_sha256']==hashlib.sha256((EXT/'prepared'/'daily.csv').read_bytes()).hexdigest()
    # Bank-holiday boundary case: 31 August 2026 is a holiday; proxy starts Friday 28.
    cal=calendar(pd.date_range('2026-08-27','2026-09-02'))
    assert cal.loc['2026-08-31','holiday']==1 and cal.loc['2026-08-28','last_working_payday']==1
    dump(EXT/'logs'/'verification.json',{'daily_feature_future_mutation_invariance':True,'local_calendar_day_alignment':True,'holiday_adjustment_boundary':True,'original_csv_hashes_unchanged':True,'daily_hash_matches':True,'pretest_only_sensitivity_and_expanding_folds':True,'lstm_training_only_scaler_inverse_roundtrip':True,'extreme_days_flagged_not_removed':int(flags.training_IQR_extreme_flag.sum()),'stability_scope':'Expanding folds XGBoost; monthly test errors all models; repeated seeds LSTM. Not all-family nested cross-validation.'})
if __name__=='__main__': run()
