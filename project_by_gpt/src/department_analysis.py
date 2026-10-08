"""Department patterns, payday associations and privacy-minimised aggregates."""
from extension_preparation import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def run():
    directories(); c=config(); m=pd.read_csv(ROOT/c['source_file']); n=m.Product_Name.str.strip(); m=m[~n.str.startswith('-PAYOUT',na=False)].copy()
    t=pd.to_datetime(m.Transaction_Time,format='mixed'); m['date']=t.dt.normalize(); m['hour']=t.dt.hour; m['weekday']=t.dt.dayofweek
    m=m[(m.date>=c['analysis_start'])&(m.date<c['analysis_end_exclusive'])]
    m['category']=m.Department_Name.fillna('Unknown or receipt adjustment'); m.loc[m.Department_ID.eq(0),'category']='Unknown or receipt adjustment'
    daily=m.groupby(['date','category']).Line_Total.sum().unstack()
    # Absence of category lines on a recorded shop day means no recorded category sales,
    # not proof of complete capture or zero latent demand.
    daily=daily.reindex(load().index).fillna(0).where(load().sales.notna(),axis=0)
    daily.to_csv(EXT/'prepared'/'department_daily_recorded_sales.csv')
    profiles=m.groupby(['category','hour','weekday']).agg(sales=('Line_Total','sum'),lines=('Line_Total','size')).reset_index(); profiles.to_csv(EXT/'audit'/'department_profiles.csv',index=False)
    cal=calendar(daily.index); rows=[]
    for cat in daily:
        for proxy in PAY:
            for v in [0,1]:
                z=daily.loc[cal[proxy].eq(v),cat].dropna(); rows.append({'category':cat,'proxy':proxy,'indicator':v,'recorded_shop_days':len(z),'mean_recorded_daily_sales':z.mean(),'total_sales':z.sum()})
    pd.DataFrame(rows).to_csv(EXT/'metrics'/'department_payday_associations.csv',index=False)
    top=m.groupby('category').Line_Total.sum().sort_values(ascending=False).head(8).index
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for cat in top:
        z=m[m.category.eq(cat)]
        axes[0].plot(z.groupby('hour').Line_Total.sum().reindex(range(24)),label=cat)
        axes[1].plot(z.groupby('weekday').Line_Total.sum().reindex(range(7)),label=cat)
    axes[0].set(xlabel='Local hour',ylabel='Recorded net sales across study (GBP)',title='Department hourly totals')
    axes[1].set(xlabel='Weekday: Monday=0',ylabel='Recorded net sales across study (GBP)',title='Department weekday totals'); axes[1].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(EXT/'charts'/'department_patterns.png',dpi=150); plt.close(fig)
    associations=pd.DataFrame(rows); fig,ax=plt.subplots(figsize=(11,6)); positions=np.arange(len(top))
    for k,proxy in enumerate(PAY):
        pivot=associations[associations.proxy.eq(proxy)].pivot(index='category',columns='indicator',values='mean_recorded_daily_sales').reindex(top)
        ax.barh(positions+(k-1)*.24,pivot[1]-pivot[0],height=.24,label=proxy)
    ax.set_yticks(positions,top); ax.axvline(0,c='black'); ax.set(xlabel='Proxy-day minus other-day mean recorded sales GBP/day',title='Unadjusted department associations, not causal effects'); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(EXT/'charts'/'department_payday.png',dpi=150); plt.close(fig)
    f=load(); fig,axes=plt.subplots(1,2,figsize=(10,4))
    f.assign(dow=f.index.dayofweek).groupby('dow').sales.mean().plot.bar(ax=axes[0]); axes[0].set(title='Recorded daily revenue by weekday',ylabel='Mean GBP',xlabel='Monday=0')
    for proxy in PAY: axes[1].plot([0,1],f.groupby(proxy).sales.mean(),marker='o',label=proxy)
    axes[1].set(title='Descriptive proxy associations',xlabel='Proxy indicator',ylabel='Mean recorded revenue GBP'); axes[1].legend(fontsize=7); fig.tight_layout(); fig.savefig(EXT/'charts'/'weekday_payday_eda.png',dpi=150); plt.close(fig)
    dump(EXT/'audit'/'privacy.json',{'derived_columns':list(daily.columns),'identifiers_excluded':['Customer_ID','Transaction_ID','Barcode','Payment_Type'],'raw_files_unchanged':True,'access_controls_encryption_consent_retention':'Requires owner/institution confirmation; not inferred from local file storage','causal_warning':'Unadjusted department group means confounded by weekdays, trend, coverage and promotions'})
if __name__=='__main__': run()
