"""Audit sources and prepare a clock-hour grid without inventing zero sales."""
from pathlib import Path
import json, hashlib, sys, platform, importlib.metadata
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'
CLEAN_COLUMNS = ['Transaction_ID','Transaction_Time','Sale_Day','Sale_Hour','Day_Name','Day_Number','Month','Year','Product_ID','Barcode','Product_Name','Department_ID','Department_Name','Quantity','Unit_Price','Line_Total']

def config():
    return json.loads((ROOT / 'config.json').read_text())

def folders():
    for name in ['prepared','models','predictions','metrics','charts','logs','audit']:
        (OUT / name).mkdir(parents=True, exist_ok=True)
    (ROOT / 'reports').mkdir(exist_ok=True)

def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, default=str), encoding='utf-8')

def read_source(path):
    if path.name.startswith('clean version'):
        return pd.read_csv(path, header=None, names=CLEAN_COLUMNS, dtype={'Barcode':'string'})
    return pd.read_csv(path, dtype={'Barcode':'string'})

def audit():
    folders()
    datasets = {}; records = []
    for path in sorted(ROOT.glob('*.csv')):
        d = read_source(path); datasets[path.name] = d
        tc = next(c for c in ['Transaction_Time','Hours_Start','Hour_Start','Sale_Day'] if c in d)
        dt = pd.to_datetime(d[tc], format='mixed', errors='coerce')
        sales = next(c for c in ['Line_Total','Hourly_Sales','Department_Hourly_Sales','Department_Daily_Sales','Actual_Sales'] if c in d)
        role = 'prediction export excluded from experiments' if 'Comparison' in path.name else ('transaction lines' if 'Line_Total' in d else 'aggregate')
        r = dict(file=path.name, role=role, rows=len(d), columns=list(d), dtypes=d.dtypes.astype(str).to_dict(), timestamp_column=tc,
                 timestamp_examples=d[tc].head(3).tolist(), start=dt.min(), end=dt.max(), invalid_timestamps=int(dt.isna().sum()),
                 timestamp_duplicates=int(dt.duplicated().sum()), exact_duplicates=int(d.duplicated().sum()), missing=d.isna().sum().to_dict(),
                 sales_column=sales, total=float(d[sales].sum()), negative_rows=int(d[sales].lt(0).sum()), zero_rows=int(d[sales].eq(0).sum()),
                 sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        if 'Department_ID' in d:
            r['unknown_department_zero'] = int(d.Department_ID.eq(0).sum())
        if 'Transaction_ID' in d:
            r['duplicate_transaction_ids'] = int(d.Transaction_ID.duplicated().sum())
        records.append(r)
    dump(OUT/'audit'/'dataset_audit.json', records)
    pd.DataFrame([{k:r[k] for k in ['file','role','rows','start','end','exact_duplicates','total','negative_rows']} for r in records]).to_csv(OUT/'audit'/'dataset_summary.csv', index=False)
    m = datasets['Master transaction-level research dataset.csv']
    name = m.Product_Name.str.strip()
    classes = np.select([name.str.startswith('-PAYOUT', na=False), m.Refund.eq(1), name.str.startswith('***', na=False), name.str.startswith('*Voucher', na=False), m.Department_ID.eq(0)], ['supplier_payout','refund','promotion_adjustment','voucher','other_unknown_department'], default='product_sale')
    m = m.assign(record_class=classes)
    summary = m.groupby('record_class').agg(rows=('Line_Total','size'), sales=('Line_Total','sum'), negatives=('Line_Total',lambda x:int(x.lt(0).sum())))
    summary.to_csv(OUT/'audit'/'transaction_classes.csv')
    m.loc[m.Department_ID.eq(0)].groupby(['Product_Name','record_class'],dropna=False).agg(rows=('Line_Total','size'),sales=('Line_Total','sum')).to_csv(OUT/'audit'/'unknown_department_records.csv')
    reconciliation = []
    clean = datasets['clean version for modelling.csv']
    pos = m[m.Line_Total.ge(0)]
    key = ['Transaction_ID','Transaction_Time','Product_ID','Quantity','Unit_Price','Line_Total']
    # Multiset reconciliation preserves legitimate repeated line items.
    a = pos.groupby(key,dropna=False).size(); b=clean.groupby(key,dropna=False).size()
    both = pd.concat([a.rename('master_nonnegative'),b.rename('clean')],axis=1).fillna(0)
    dump(OUT/'audit'/'clean_master_overlap.json',dict(master_nonnegative_rows=len(pos),clean_rows=len(clean), unmatched_multiplicity=int((both.iloc[:,0]-both.iloc[:,1]).abs().sum())))
    h = datasets['Hourly total-sales dataset.csv'].assign(t=lambda d:pd.to_datetime(d.Hours_Start,format='mixed')).set_index('t').Hourly_Sales
    for filename,tc,sc,positive in [('Master transaction-level research dataset.csv','Transaction_Time','Line_Total',True),('clean version for modelling.csv','Transaction_Time','Line_Total',False),('Hourly sales by department.csv','Hour_Start','Department_Hourly_Sales',False)]:
        d = datasets[filename]
        if positive: d=d[d[sc].ge(0)]
        a=d.assign(t=pd.to_datetime(d[tc],format='mixed').dt.floor('h')).groupby('t')[sc].sum()
        j=pd.concat([h.rename('hourly'),a.rename('source')],axis=1)
        reconciliation.append(dict(source=filename,master_nonnegative_only=positive,union_hours=len(j),missing_values=int(j.isna().sum().sum()),max_abs_difference=float((j.hourly-j.source).abs().max())))
    daily=datasets['Daily sales by department.csv'].groupby('Sale_Day').Department_Daily_Sales.sum()
    hd=datasets['Hourly total-sales dataset.csv'].groupby('Sale_Day').Hourly_Sales.sum()
    reconciliation.append(dict(source='Daily sales by department.csv',max_abs_difference=float((daily-hd).abs().max()),union_days=len(daily.index.union(hd.index))))
    dump(OUT/'audit'/'aggregate_reconciliation.json',reconciliation)
    # Audit the legacy export only; no old scores enter model selection.
    old=datasets['Final_Model_Comparison.csv']; ot=pd.to_datetime(old.Hours_Start)
    check=h.reindex(ot).to_numpy()
    dump(OUT/'audit'/'legacy_export_audit.json',dict(rows=len(old),start=ot.min(),end=ot.max(),payday_counts=old.Payday_Window.value_counts().to_dict(),actual_vs_gross_max_abs_difference=float(np.max(np.abs(check-old.Actual_Sales))),rounded_predictions=True,training_design_unknown=True))
    print(pd.read_csv(OUT/'audit'/'dataset_summary.csv').to_string(index=False))
    print(summary.to_string())
    return m

def payday(local, cfg):
    """Last N calendar days and first M calendar days, a configurable proxy."""
    return ((local.day > local.days_in_month-cfg['last_days']) | (local.day <= cfg['first_days'])).astype(int)

def history_features(data):
    data=data.copy()
    for lag in config()['lags']:
        data[f'lag_{lag}']=data.sales.shift(lag)
        data[f'lag_{lag}_missing']=data[f'lag_{lag}'].isna().astype(int)
    data['rolling_24_mean']=data.sales.shift(1).rolling(24,min_periods=1).mean()
    data['rolling_24_count']=data.sales.shift(1).rolling(24,min_periods=1).count()
    return data

def prepare():
    c=config(); m=audit()
    if not c['decisions_resolved']:
        raise ValueError('Accounting and timezone decisions are pending; audit is available.')
    if c['forecast_horizon_hours'] != 1:
        raise ValueError('This implementation supports rolling one-hour-ahead forecasts only.')
    if c['source_file'] != 'Master transaction-level research dataset.csv':
        raise ValueError('This audited accounting pipeline requires the master transaction source.')
    boundaries=[pd.Timestamp(c[k]) for k in ['analysis_start','validation_start','test_start','analysis_end_exclusive']]
    if not all(a<b for a,b in zip(boundaries,boundaries[1:])):
        raise ValueError('Split dates must be strictly chronological.')
    dt=pd.to_datetime(m[c['timestamp_column']],format='mixed',errors='raise')
    # Fail rather than guess at duplicated or nonexistent local DST times.
    utc=dt.dt.tz_localize(c['timezone'],ambiguous='raise',nonexistent='raise').dt.tz_convert('UTC')
    excluded=m.record_class.isin(c['exclude_record_classes'])
    keep=m.loc[~excluded].assign(timestamp=utc.loc[~excluded].dt.floor('h'))
    s=keep.groupby('timestamp')[c['target_column']].sum()
    first=pd.Timestamp(c['analysis_start'],tz=c['timezone']).tz_convert('UTC')
    end=pd.Timestamp(c['analysis_end_exclusive'],tz=c['timezone']).tz_convert('UTC')
    grid=pd.date_range(first,end,freq='h',inclusive='left')
    data=pd.DataFrame({'timestamp':grid,'sales':s.reindex(grid).to_numpy()})
    local=grid.tz_convert(c['timezone'])
    data['local_timestamp']=local.astype(str)
    data['observed']=data.sales.notna().astype(int)
    data['payday']=payday(local,c['payday'])
    data['hour_sin']=np.sin(2*np.pi*local.hour/24); data['hour_cos']=np.cos(2*np.pi*local.hour/24)
    data['dow_sin']=np.sin(2*np.pi*local.dayofweek/7); data['dow_cos']=np.cos(2*np.pi*local.dayofweek/7)
    data['month_sin']=np.sin(2*np.pi*(local.month-1)/12); data['month_cos']=np.cos(2*np.pi*(local.month-1)/12)
    data['weekend']=(local.dayofweek>=5).astype(int)
    data['trend_days']=(grid-grid[0]).total_seconds()/86400
    data['hour']=local.hour; data['dayofweek']=local.dayofweek
    # The previous 24 clock hours, never the target observation.
    data=history_features(data)
    val=pd.Timestamp(c['validation_start'],tz=c['timezone']).tz_convert('UTC')
    test=pd.Timestamp(c['test_start'],tz=c['timezone']).tz_convert('UTC')
    data['split']=np.select([grid<val,grid<test],['train','validation'],default='test')
    assert data.timestamp.is_unique
    data.to_csv(OUT/'prepared'/'hourly_grid.csv',index=False)
    counts=data.groupby('split').agg(clock_hours=('sales','size'),observed_hours=('sales','count'),sales=('sales','sum'))
    counts['eligible_forecast_hours']=[int(mask(data,split).sum()) for split in counts.index]
    counts.to_csv(OUT/'audit'/'split_counts.csv')
    data.assign(month=local.strftime('%Y-%m')).groupby('month').agg(observed_hours=('sales','count'),sales=('sales','sum')).to_csv(OUT/'audit'/'monthly_sales.csv')
    data.groupby('hour').agg(clock_hours=('sales','size'),observed_hours=('sales','count'),mean_sales=('sales','mean')).to_csv(OUT/'audit'/'hour_coverage.csv')
    excluded_summary=m.loc[excluded].groupby('record_class').Line_Total.agg(['size','sum']).to_dict()
    dump(OUT/'audit'/'preparation_decisions.json',dict(excluded_classes=c['exclude_record_classes'],excluded_summary=excluded_summary,net_source_total=float(keep.Line_Total.sum()),selected_period_total=float(data.sales.sum()),unknown_hours=int(data.sales.isna().sum()),observed_zero_hours=int(data.sales.eq(0).sum()),observed_negative_hours=int(data.sales.lt(0).sum()),timezone=c['timezone'],dst_policy='localize with errors on ambiguity/nonexistence; model grid in UTC; calendar in local time',duplicate_policy='retain repeated line items; master exact duplicates zero; transaction ID is a receipt not a unique line',target='net retail line totals excluding explicit supplier payouts; promotion field not subtracted again'))
    versions={}
    for package in ['numpy','pandas','scikit-learn','prophet','cmdstanpy','xgboost','shap','tensorflow','tensorflow_cpu','tensorflow-intel','keras','matplotlib','joblib']:
        try: versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: pass
    dump(OUT/'logs'/'environment.json',dict(python=sys.version,platform=platform.platform(),packages=versions,config=c))
    print(counts.to_string())
    return data

CALENDAR=['hour_sin','hour_cos','dow_sin','dow_cos','month_sin','month_cos','weekend','trend_days']
def feature_names(with_payday=False):
    c=config(); cols=CALENDAR.copy()
    cols += [v for lag in c['lags'] for v in [f'lag_{lag}',f'lag_{lag}_missing']]+['rolling_24_mean','rolling_24_count']
    return cols+(['payday'] if with_payday else [])

def load_data():
    c=config(); path=OUT/'prepared'/'hourly_grid.csv'
    if not path.exists(): prepare()
    d=pd.read_csv(path); d['timestamp']=pd.to_datetime(d.timestamp,utc=True)
    # Guard against stale data after configuration changes.
    saved=json.loads((OUT/'logs'/'environment.json').read_text())['config']
    if saved != c: raise ValueError('Prepared data config differs. Rerun data_preparation.py.')
    return d

def mask(d, split):
    return d.split.eq(split)&d.sales.notna()&(np.arange(len(d))>=config()['lstm']['sequence_hours'])

def metric(actual,pred):
    e=np.asarray(pred)-np.asarray(actual)
    return dict(n=len(e),MAE=float(np.mean(np.abs(e))),RMSE=float(np.sqrt(np.mean(e**2))))

def save_predictions(d, indices, pred, model, split='test'):
    r=d.loc[indices,['timestamp','local_timestamp','sales','payday']].rename(columns={'sales':'actual_sales'})
    r['predicted_sales']=np.asarray(pred,dtype=float); r['model']=model
    if not r.timestamp.is_unique or not np.isfinite(r[['actual_sales','predicted_sales']].to_numpy()).all():
        raise ValueError('Invalid prediction output')
    r.to_csv(OUT/'predictions'/f'{model}_{split}.csv',index=False,float_format='%.17g')
    dump(OUT/'predictions'/f'{model}_{split}.metadata.json',dict(model=model,split=split,config=config(),prepared_sha256=hashlib.sha256((OUT/'prepared'/'hourly_grid.csv').read_bytes()).hexdigest(),forecast_horizon_hours=1,forecast_origin='start of target hour; observed prior hours available',train_observations=int(mask(d,'train').sum()),train_end=d.loc[mask(d,'train'),'timestamp'].max(),no_refit_on_validation=True))
    return r

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('--audit-only',action='store_true'); args=p.parse_args()
    audit() if args.audit_only else prepare()
