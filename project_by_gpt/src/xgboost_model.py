"""XGBoost matched experiments, train-only imputation and Tree SHAP."""
from data_preparation import *
from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer
import joblib,time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def run():
    start=time.time(); d=load_data(); c=config(); settings=c['xgboost']
    train=d.index[mask(d,'train')]; val=d.index[mask(d,'validation')]; test=d.index[mask(d,'test')]
    trials=[]; candidates={}
    for depth in settings['max_depth_candidates']:
        for pay in [False,True]:
            name='XGBoost_'+('With_Payday' if pay else 'Without_Payday'); cols=feature_names(pay)
            imp=SimpleImputer(strategy='median',keep_empty_features=True).fit(d.loc[train,cols]); x=imp.transform(d[cols])
            model=XGBRegressor(objective='reg:squarederror',tree_method='hist',random_state=c['seed'],max_depth=depth,**{k:settings[k] for k in ['n_estimators','learning_rate','min_child_weight','subsample','colsample_bytree','n_jobs']})
            model.fit(x[train],d.loc[train,'sales'])
            vp=model.predict(x[val]); score=metric(d.loc[val,'sales'],vp)
            trials.append(dict(model=name,max_depth=depth,**score)); candidates[(depth,pay)]=(model,imp,x,cols,vp)
    chosen=min(settings['max_depth_candidates'],key=lambda dep:np.mean([r['RMSE'] for r in trials if r['max_depth']==dep]))
    shap_info={}
    for pay in [False,True]:
        name='XGBoost_'+('With_Payday' if pay else 'Without_Payday')
        model,imp,x,cols,vp=candidates[(chosen,pay)]
        save_predictions(d,val,vp,name,'validation'); save_predictions(d,test,model.predict(x[test]),name)
        model.save_model(OUT/'models'/f'{name}.json'); joblib.dump(dict(imputer=imp,columns=cols),OUT/'models'/f'{name}_preprocessing.joblib')
        import shap
        # Deterministic evenly spaced test sample, explanations only after model selection.
        sample=test[np.linspace(0,len(test)-1,min(settings['shap_max_rows'],len(test)),dtype=int)]
        frame=pd.DataFrame(x[sample],columns=cols)
        explainer=shap.TreeExplainer(model,feature_perturbation='tree_path_dependent')
        values=explainer(frame)
        error=float(np.max(np.abs(values.values.sum(axis=1)+values.base_values-model.predict(frame))))
        if error>0.01: raise ValueError(f'SHAP additivity failure {error}')
        ranks=pd.DataFrame({'feature':cols,'mean_absolute_shap':np.abs(values.values).mean(axis=0)}).sort_values('mean_absolute_shap',ascending=False)
        ranks.to_csv(OUT/'metrics'/f'{name}_shap_importance.csv',index=False)
        detail=frame.copy(); detail.columns=['input_'+v for v in cols]
        for i,col in enumerate(cols): detail['shap_'+col]=values.values[:,i]
        detail['timestamp']=d.loc[sample,'timestamp'].astype(str).to_numpy(); detail['base_value']=values.base_values
        detail.to_csv(OUT/'predictions'/f'{name}_shap.csv',index=False)
        shap.plots.beeswarm(values,max_display=12,show=False); plt.title(name.replace('_',' ')+' - Tree SHAP'); plt.tight_layout(); plt.savefig(OUT/'charts'/f'{name}_shap.png',dpi=160,bbox_inches='tight'); plt.close('all')
        shap_info[name]=dict(n=len(sample),max_additivity_error=error,method='Tree SHAP tree_path_dependent',top_features=ranks.head(5).to_dict('records'))
    dump(OUT/'logs'/'xgboost.json',dict(trials=trials,chosen_max_depth=chosen,selection='mean validation RMSE of paired variants; no test tuning',settings=settings,selected_model_parameters=model.get_params(),shap=shap_info,elapsed_seconds=time.time()-start))
    print('XGBoost completed',chosen)

if __name__=='__main__': run()
