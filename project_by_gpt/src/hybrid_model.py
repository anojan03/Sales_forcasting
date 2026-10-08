"""Validation-weighted Prophet/XGBoost/LSTM ensemble; not a residual cascade."""
from extension_preparation import *
from scipy.optimize import minimize
def run():
    f=load(); va=indices(f,'validation'); te=indices(f,'test'); pairs={}; objective=[]
    for pay in ['Without_Payday','With_Payday']:
        arrays=[]
        for split in ['validation','test']:
            frames=[pd.read_csv(EXT/'predictions'/f'{fam}_{pay}_{split}.csv') for fam in ['Prophet','XGBoost','LSTM']]
            assert all(frames[0].timestamp.equals(v.timestamp) and np.allclose(frames[0].actual_sales,v.actual_sales) for v in frames)
            arrays.append(np.column_stack([v.predicted_sales for v in frames]))
        pairs[pay]=arrays; objective.append(arrays[0])
    y=f.sales.iloc[va].to_numpy()
    # Shared nonnegative weights keep the paired payday ensemble comparable.
    fit=minimize(lambda w:np.mean([np.mean((x@w-y)**2) for x in objective])/np.var(y),np.ones(3)/3,bounds=[(0,1)]*3,constraints={'type':'eq','fun':lambda w:w.sum()-1},method='SLSQP',options={'maxiter':1000,'ftol':1e-7})
    if not fit.success: raise ValueError(fit.message)
    for pay,arr in pairs.items():
        for split,ix,x in zip(['validation','test'],[va,te],arr): save(f,ix,x@fit.x,'Hybrid_'+pay,split,{'validation_fitted_weights':fit.x.tolist(),'validation_score_is_in_sample_for_weights':True})
    dump(EXT/'models'/'hybrid_weights.json',{'families':['Prophet','XGBoost','LSTM'],'weights':fit.x.tolist(),'fit_period':'validation only','form':'convex ensemble; not Prophet-to-XGBoost-to-LSTM cascade'})
if __name__=='__main__': run()
