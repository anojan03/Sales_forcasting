"""Shared upcoming-day data and calendar proxies; preserves original experiments."""
from data_preparation import ROOT, OUT, config, dump
import hashlib, json, urllib.request
import numpy as np
import pandas as pd

EXT=OUT/'extensions'
def settings(): return json.loads((ROOT/'extension_config.json').read_text())
def directories():
    for p in ['prepared','predictions','models','metrics','charts','logs','audit']: (EXT/p).mkdir(parents=True,exist_ok=True)
def holidays():
    directories(); path=EXT/'audit'/'govuk_bank_holidays.json'
    if not path.exists():
        with urllib.request.urlopen('https://www.gov.uk/bank-holidays.json',timeout=30) as response: raw=response.read()
        path.write_bytes(raw)
    events=json.loads(path.read_text())[settings()['holiday_region']]['events']
    dates={e['date'] for e in events}
    if not {2025,2026}.issubset({int(x[:4]) for x in dates}): raise ValueError('Holiday snapshot does not cover study years')
    return dates
PAY=['monthly_payday','weekly_payday','last_working_payday']
def calendar(index):
    idx=pd.DatetimeIndex(index); hs=holidays(); s=settings()
    f=pd.DataFrame(index=idx)
    f['dow_sin']=np.sin(2*np.pi*idx.dayofweek/7); f['dow_cos']=np.cos(2*np.pi*idx.dayofweek/7)
    f['month_sin']=np.sin(2*np.pi*(idx.month-1)/12); f['month_cos']=np.cos(2*np.pi*(idx.month-1)/12)
    f['holiday']=idx.strftime('%Y-%m-%d').isin(hs).astype(int)
    f['working_day']=((idx.dayofweek<5)&~idx.strftime('%Y-%m-%d').isin(hs)).astype(int)
    f['monthly_payday']=((idx.day>idx.days_in_month-s['monthly_window_days'])|(idx.day<=s['monthly_window_days'])).astype(int)
    f['weekly_payday']=((idx.dayofweek-s['weekly_payday_weekday'])%7<s['weekly_window_days']).astype(int)
    starts=[]
    for month in pd.date_range(idx.min()-pd.Timedelta(days=35),idx.max(),freq='MS'):
        day=month+pd.offsets.MonthEnd(0)
        while day.dayofweek>=5 or day.strftime('%Y-%m-%d') in hs: day-=pd.Timedelta(days=1)
        starts.extend([day+pd.Timedelta(days=k) for k in range(3)])
    f['last_working_payday']=idx.isin(starts).astype(int)
    return f
def prepare():
    directories(); c=config()
    if not (OUT/'prepared'/'hourly_grid.csv').exists():
        from data_preparation import prepare as prepare_hourly
        prepare_hourly()
    d=pd.read_csv(OUT/'prepared'/'hourly_grid.csv')
    d['date']=pd.to_datetime(d.local_timestamp.str[:10]); grid=pd.date_range(c['analysis_start'],pd.Timestamp(c['analysis_end_exclusive'])-pd.Timedelta(days=1),freq='D')
    g=d.groupby('date').agg(sales=('sales',lambda z:z.sum(min_count=1)),observed_hours=('sales','count'),clock_hours=('sales','size'))
    f=g.reindex(grid).join(calendar(grid)); f.index.name='timestamp'
    f['trend_days']=np.arange(len(f))
    for lag in [1,7,14,28]:
        f[f'lag_{lag}']=f.sales.shift(lag); f[f'lag_{lag}_missing']=f[f'lag_{lag}'].isna().astype(int)
    f['rolling_7']=f.sales.shift(1).rolling(7,min_periods=1).mean()
    f['previous_recorded_hours']=f.observed_hours.shift(1)
    f['split']=np.select([f.index<pd.Timestamp(c['validation_start']),f.index<pd.Timestamp(c['test_start'])],['train','validation'],default='test')
    f.reset_index().to_csv(EXT/'prepared'/'daily.csv',index=False,float_format='%.17g')
    dump(EXT/'logs'/'provenance.json',{'original_config':c,'extension_config':settings(),'hourly_sha256':hashlib.sha256((OUT/'prepared'/'hourly_grid.csv').read_bytes()).hexdigest(),'daily_sha256':hashlib.sha256((EXT/'prepared'/'daily.csv').read_bytes()).hexdigest(),'holiday_sha256':hashlib.sha256((EXT/'audit'/'govuk_bank_holidays.json').read_bytes()).hexdigest(),'origin':'00:00 local target day; daily horizon ends next local midnight, 23/24/25 elapsed hours','target':'sum of recorded net retail lines, unknown within-day completeness','privacy':'derived features exclude customer, receipt, barcode and payment identifiers'})
    f.groupby('split').agg(days=('sales','count'),sales=('sales','sum')).to_csv(EXT/'audit'/'daily_splits.csv')
    return f
def load():
    f=pd.read_csv(EXT/'prepared'/'daily.csv',parse_dates=['timestamp']).set_index('timestamp')
    p=json.loads((EXT/'logs'/'provenance.json').read_text())
    if p['extension_config']!=settings() or p['original_config']!=config(): raise ValueError('Stale preparation configuration')
    if p['hourly_sha256']!=hashlib.sha256((OUT/'prepared'/'hourly_grid.csv').read_bytes()).hexdigest(): raise ValueError('Changed hourly input')
    if p['daily_sha256']!=hashlib.sha256((EXT/'prepared'/'daily.csv').read_bytes()).hexdigest(): raise ValueError('Changed daily data')
    if p['holiday_sha256']!=hashlib.sha256((EXT/'audit'/'govuk_bank_holidays.json').read_bytes()).hexdigest(): raise ValueError('Changed holiday snapshot')
    return f
def columns(pay):
    return ['dow_sin','dow_cos','month_sin','month_cos','holiday','working_day','trend_days']+[v for lag in [1,7,14,28] for v in [f'lag_{lag}',f'lag_{lag}_missing']]+['rolling_7','previous_recorded_hours']+(PAY if pay else [])
def indices(f,split): return np.flatnonzero((f.split.eq(split)&f.sales.notna()).to_numpy()&(np.arange(len(f))>=28))
def score(y,p):
    y=np.asarray(y); p=np.asarray(p); e=p-y; good=y>0
    return {'n':len(y),'MAE':float(np.abs(e).mean()),'RMSE':float(np.sqrt(np.mean(e**2))),'MAPE_positive_only':float(np.mean(np.abs(e[good]/y[good]))*100) if good.any() else None,'MAPE_n':int(good.sum()),'bias':float(e.mean())}
def save(f,ix,p,name,split,extra=None):
    z=f.iloc[ix][['sales','monthly_payday','weekly_payday','last_working_payday']].rename(columns={'sales':'actual_sales'}).copy()
    z['predicted_sales']=np.asarray(p); z['model']=name; z['payday']=z.monthly_payday
    z['forecast_origin']=z.index.tz_localize(config()['timezone']).astype(str)
    if not z.index.is_unique or not np.isfinite(z[['actual_sales','predicted_sales']]).all().all(): raise ValueError('Invalid predictions')
    path=EXT/'predictions'/f'{name}_{split}.csv'; z.to_csv(path,float_format='%.17g')
    dump(path.with_suffix('.metadata.json'),{'provenance':json.loads((EXT/'logs'/'provenance.json').read_text()),'split':split,'model':name,'extra':extra})
if __name__=='__main__': prepare()
