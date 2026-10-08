"""Direct upcoming-day Prophet paired experiments."""
from extension_preparation import *
from prophet import Prophet
from prophet.serialize import model_to_json
from sklearn.impute import SimpleImputer
import joblib
def run():
    f=load(); tr=indices(f,'train'); va=indices(f,'validation'); te=indices(f,'test'); trials=[]; candidates={}
    for cp in [.01,.05]:
        for pay in [False,True]:
            cols=[x for x in columns(pay) if x not in ['dow_sin','dow_cos','trend_days']]
            imp=SimpleImputer(keep_empty_features=True,strategy='median').fit(f.iloc[tr][cols]); x=pd.DataFrame(imp.transform(f[cols]),columns=cols)
            x['ds']=f.index; x['y']=f.sales.to_numpy()
            m=Prophet(daily_seasonality=False,weekly_seasonality=True,yearly_seasonality=False,changepoint_prior_scale=cp,uncertainty_samples=0)
            for col in cols: m.add_regressor(col,prior_scale=.5)
            m.fit(x.iloc[tr],seed=42); vp=m.predict(x.iloc[va]).yhat.to_numpy()
            trials.append({'cp':cp,'pay':pay,**score(f.sales.iloc[va],vp)}); candidates[(cp,pay)]=(m,imp,x,vp)
    cp=min([.01,.05],key=lambda v:np.mean([r['RMSE'] for r in trials if r['cp']==v]))
    for pay in [False,True]:
        name='Prophet_'+('With_Payday' if pay else 'Without_Payday'); m,imp,x,vp=candidates[(cp,pay)]
        save(f,va,vp,name,'validation'); save(f,te,m.predict(x.iloc[te]).yhat,name,'test')
        (EXT/'models'/f'{name}.json').write_text(model_to_json(m)); joblib.dump(imp,EXT/'models'/f'{name}_imputer.joblib')
    dump(EXT/'logs'/'prophet.json',{'trials':trials,'chosen_cp':cp})
if __name__=='__main__': run()
