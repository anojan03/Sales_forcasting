"""Direct upcoming-day XGBoost and checked SHAP contributions."""
from extension_preparation import *
from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer
import joblib, shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def run():
    f=load(); tr=indices(f,'train'); va=indices(f,'validation'); te=indices(f,'test'); trials=[]; candidates={}
    for depth in [2,3]:
        for pay in [False,True]:
            cols=columns(pay); imp=SimpleImputer(keep_empty_features=True,strategy='median').fit(f.iloc[tr][cols]); x=imp.transform(f[cols])
            m=XGBRegressor(n_estimators=200,max_depth=depth,learning_rate=.03,min_child_weight=5,n_jobs=2,random_state=42,tree_method='hist')
            m.fit(x[tr],f.sales.iloc[tr]); vp=m.predict(x[va]); trials.append({'depth':depth,'pay':pay,**score(f.sales.iloc[va],vp)}); candidates[(depth,pay)]=(m,imp,x,cols,vp)
    dep=min([2,3],key=lambda v:np.mean([r['RMSE'] for r in trials if r['depth']==v]))
    for pay in [False,True]:
        name='XGBoost_'+('With_Payday' if pay else 'Without_Payday'); m,imp,x,cols,vp=candidates[(dep,pay)]
        save(f,va,vp,name,'validation'); save(f,te,m.predict(x[te]),name,'test'); m.save_model(EXT/'models'/f'{name}.json'); joblib.dump({'imputer':imp,'columns':cols},EXT/'models'/f'{name}_preprocessing.joblib')
        frame=pd.DataFrame(x[te],columns=cols); sv=shap.TreeExplainer(m)(frame)
        err=float(np.max(np.abs(sv.values.sum(axis=1)+sv.base_values-m.predict(frame))))
        if err>.01: raise ValueError('SHAP additivity')
        pd.DataFrame({'feature':cols,'mean_absolute_shap':np.abs(sv.values).mean(axis=0)}).sort_values('mean_absolute_shap',ascending=False).to_csv(EXT/'metrics'/f'{name}_shap.csv',index=False)
        shap.plots.beeswarm(sv,show=False,max_display=14); plt.title('Upcoming-day '+name); plt.tight_layout(); plt.savefig(EXT/'charts'/f'{name}_shap.png',dpi=150,bbox_inches='tight'); plt.close('all')
    dump(EXT/'logs'/'xgboost.json',{'trials':trials,'chosen_depth':dep,'shap_max_additivity_error':err})
if __name__=='__main__': run()
