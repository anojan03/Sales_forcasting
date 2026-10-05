"""Prophet with matched validation selection and past-sales regressors."""
from data_preparation import *
from prophet import Prophet
from prophet.serialize import model_to_json
from sklearn.impute import SimpleImputer
import joblib, time

def run():
    start=time.time(); d=load_data(); c=config(); settings=c['prophet']
    train=d.index[mask(d,'train')]; val=d.index[mask(d,'validation')]; test=d.index[mask(d,'test')]
    # Prophet requires naive dates. These local observed hours are unambiguous
    # (localization was already checked); lag computation remains on UTC grid.
    ds=pd.DatetimeIndex(d.timestamp).tz_convert(c['timezone']).tz_localize(None)
    trials=[]; candidates={}
    for cp in settings['changepoint_prior_scale_candidates']:
        for pay in [False,True]:
            name='Prophet_'+('With_Payday' if pay else 'Without_Payday')
            cols=feature_names(pay)
            imputer=SimpleImputer(strategy='median',keep_empty_features=True)
            imputer.fit(d.loc[train,cols]); x=imputer.transform(d[cols])
            f=pd.DataFrame(x,columns=cols); f['ds']=ds; f['y']=d.sales
            model=Prophet(changepoint_prior_scale=cp,daily_seasonality=False,weekly_seasonality=False,yearly_seasonality=False,uncertainty_samples=0)
            model.add_seasonality('daily',period=1,fourier_order=settings['daily_fourier_order'])
            model.add_seasonality('weekly',period=7,fourier_order=settings['weekly_fourier_order'])
            # Daily/weekly harmonics already represent these cyclic calendars.
            # Avoid duplicate harmonic columns in Prophet regressors.
            regressors=[v for v in cols if v not in ['hour_sin','hour_cos','dow_sin','dow_cos','trend_days']]
            for col in regressors: model.add_regressor(col,prior_scale=settings['regressor_prior_scale'],standardize='auto')
            model.fit(f.loc[train,['ds','y']+regressors],seed=c['seed'])
            pred=model.predict(f.loc[val,['ds']+regressors]).yhat.to_numpy()
            score=metric(d.loc[val,'sales'],pred)
            trials.append(dict(model=name,changepoint_prior_scale=cp,**score))
            candidates[(cp,pay)]=(model,imputer,f,regressors,pred)
    chosen=min(settings['changepoint_prior_scale_candidates'],key=lambda cp:np.mean([r['RMSE'] for r in trials if r['changepoint_prior_scale']==cp]))
    for pay in [False,True]:
        name='Prophet_'+('With_Payday' if pay else 'Without_Payday')
        model,imp,f,cols,vpred=candidates[(chosen,pay)]
        save_predictions(d,val,vpred,name,'validation')
        pred=model.predict(f.loc[test,['ds']+cols]).yhat.to_numpy()
        save_predictions(d,test,pred,name)
        (OUT/'models'/f'{name}.json').write_text(model_to_json(model))
        joblib.dump(dict(imputer=imp,input_columns=feature_names(pay),regressors=cols),OUT/'models'/f'{name}_preprocessing.joblib')
    dump(OUT/'logs'/'prophet.json',dict(trials=trials,chosen_changepoint_prior_scale=chosen,selection='mean validation RMSE of paired variants; no refit on validation',settings=settings,elapsed_seconds=time.time()-start))
    print('Prophet completed',chosen)

if __name__=='__main__': run()
