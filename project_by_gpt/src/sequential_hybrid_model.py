"""Prophet -> XGBoost residuals -> LSTM remaining residuals, chronological stages."""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
os.environ['TF_DETERMINISTIC_OPS']='1'
from extension_preparation import *
from pathlib import Path
from prophet import Prophet
from prophet.serialize import model_to_json
from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import joblib

def cascade_config(): return json.loads((ROOT/'cascade_config.json').read_text())
def stage_indices(f,c):
    eligible=f.sales.notna().to_numpy()&(np.arange(len(f))>=28)
    a=np.flatnonzero(eligible&(f.index<pd.Timestamp(c['prophet_fit_end_exclusive'])))
    b=np.flatnonzero(eligible&(f.index>=pd.Timestamp(c['prophet_fit_end_exclusive']))&(f.index<pd.Timestamp(c['xgboost_fit_end_exclusive'])))
    cc=np.flatnonzero(eligible&(f.index>=pd.Timestamp(c['xgboost_fit_end_exclusive']))&(f.split.eq('train').to_numpy()))
    if not min(len(a),len(b),len(cc))>30 or not a.max()<b.min() or not b.max()<cc.min(): raise ValueError('Insufficient or overlapping cascade training blocks')
    if not all(f.iloc[ix].split.eq('train').all() for ix in [a,b,cc]): raise ValueError('Every cascade fitting block must precede validation')
    return a,b,cc

def sequence_inputs(history,known,ix,steps):
    if min(ix)<steps: raise ValueError('Insufficient sequence history')
    return [np.stack([history[i-steps:i] for i in ix]),known[ix]]

def run():
    directories(); f=load(); c=cascade_config(); a,b,cc=stage_indices(f,c); va=indices(f,'validation'); te=indices(f,'test'); steps=c['sequence_days']
    tf.config.threading.set_intra_op_parallelism_threads(2); tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.experimental.enable_op_determinism(); logs=[]
    for pay in [False,True]:
        tf.keras.backend.clear_session(); tf.keras.utils.set_random_seed(c['seed'])
        name='Sequential_Hybrid_'+('With_Payday' if pay else 'Without_Payday')
        # Stage A has access only to its own earlier block.
        cols=[v for v in columns(pay) if v not in ['dow_sin','dow_cos','trend_days']]
        imp_a=SimpleImputer(strategy='median',keep_empty_features=True).fit(f.iloc[a][cols])
        frame=pd.DataFrame(imp_a.transform(f[cols]),columns=cols); frame['ds']=f.index
        training=frame.iloc[a].copy(); training['y']=f.sales.iloc[a].to_numpy()
        prophet=Prophet(daily_seasonality=False,weekly_seasonality=True,yearly_seasonality=False,changepoint_prior_scale=c['prophet_changepoint_prior_scale'],uncertainty_samples=0)
        for col in cols: prophet.add_regressor(col,prior_scale=.5)
        prophet.fit(training,seed=c['seed']); base=prophet.predict(frame).yhat.to_numpy()
        residual_a=f.sales.to_numpy()-base
        # Stage B targets are genuinely outside stage A's fit period.
        bcols=columns(pay); imp_b=SimpleImputer(strategy='median',keep_empty_features=True).fit(f.iloc[b][bcols])
        x=imp_b.transform(f[bcols]); x=np.column_stack([x,base])
        booster=XGBRegressor(**c['xgboost'],objective='reg:squarederror',tree_method='hist',n_jobs=2,random_state=c['seed'])
        booster.fit(x[b],residual_a[b]); correction_b=booster.predict(x).astype(float); two_stage=base+correction_b
        remaining=f.sales.to_numpy()-two_stage
        # Stage C targets are outside both earlier stages' training blocks.
        scaler=StandardScaler().fit(remaining[cc].reshape(-1,1)); scaled=scaler.transform(remaining.reshape(-1,1)).ravel()
        calendar_cols=['dow_sin','dow_cos','month_sin','month_cos','holiday','working_day']+(PAY if pay else [])
        known=f[calendar_cols].to_numpy(np.float32)
        hist=np.column_stack([np.nan_to_num(scaled,nan=0),np.isfinite(remaining).astype(int),known]).astype(np.float32)
        ti=sequence_inputs(hist,known,cc,steps); vi=sequence_inputs(hist,known,va,steps)
        seq=tf.keras.Input((steps,hist.shape[1])); future=tf.keras.Input((known.shape[1],))
        encoded=tf.keras.layers.LSTM(c['lstm']['units'])(seq)
        joined=tf.keras.layers.Concatenate()([encoded,future]); dense=tf.keras.layers.Dense(c['lstm']['dense_units'],activation='relu')(joined)
        model=tf.keras.Model([seq,future],tf.keras.layers.Dense(1)(dense))
        model.compile(optimizer=tf.keras.optimizers.Adam(c['lstm']['learning_rate']),loss='mse')
        training_history=model.fit(ti,scaled[cc],validation_data=(vi,scaled[va]),epochs=c['lstm']['epochs'],batch_size=c['lstm']['batch_size'],shuffle=False,callbacks=[tf.keras.callbacks.EarlyStopping(patience=c['lstm']['patience'],restore_best_weights=True)],verbose=0)
        for split,ix in [('validation',va),('test',te)]:
            correction_c=scaler.inverse_transform(model.predict(sequence_inputs(hist,known,ix,steps),verbose=0)).ravel()
            final=two_stage[ix]+correction_c
            save(f,ix,final,name,split,{'cascade_config':c,'training_blocks':{'prophet':str(f.index[a[-1]]),'xgboost':str(f.index[b[-1]]),'lstm':str(f.index[cc[-1]])}})
            pd.DataFrame({'timestamp':f.index[ix],'prophet':base[ix],'xgboost_residual_correction':correction_b[ix],'lstm_remaining_correction':correction_c,'final_prediction':final}).to_csv(EXT/'predictions'/f'{name}_{split}_stages.csv',index=False,float_format='%.17g')
        (EXT/'models'/f'{name}_prophet.json').write_text(model_to_json(prophet)); booster.save_model(EXT/'models'/f'{name}_xgboost.json'); model.save(EXT/'models'/f'{name}_lstm.keras')
        joblib.dump({'stage_a_imputer':imp_a,'stage_a_columns':cols,'stage_b_imputer':imp_b,'stage_b_columns':bcols,'stage_c_scaler':scaler,'calendar_columns':calendar_cols,'sequence_days':steps},EXT/'models'/f'{name}_preprocessing.joblib')
        pd.DataFrame(training_history.history).to_csv(EXT/'logs'/f'{name}_training.csv',index=False)
        # Deliberately mutate target/future values: historical sequence at origin is unchanged.
        origin=te[0]; mutated=hist.copy(); mutated[origin:,0]=1e9
        assert np.array_equal(sequence_inputs(hist,known,[origin],steps)[0],sequence_inputs(mutated,known,[origin],steps)[0])
        assert np.isclose(scaler.mean_[0],remaining[cc].mean())
        np.testing.assert_allclose(scaler.inverse_transform(scaler.transform(remaining[cc].reshape(-1,1))),remaining[cc].reshape(-1,1),atol=1e-10)
        logs.append({'model':name,'stage_training_n':[len(a),len(b),len(cc)],'stage_training_end':[str(f.index[ix[-1]]) for ix in [a,b,cc]],'epochs_run':len(training_history.history['loss']),'best_epoch':int(np.argmin(training_history.history['val_loss'])+1)})
    dump(EXT/'logs'/'sequential_hybrid.json',{'config':c,'models':logs,'verified':{'disjoint_stage_fit_periods':True,'residual_targets_out_of_earlier_stage_training':True,'training_only_scaler_inverse_roundtrip':True,'future_residual_mutation_invariance':True},'limitation':'Chronological stage splitting reduces per-stage sample size. This is a later exploratory experiment: test outcomes were already known from earlier models; no settings were chosen from new test scores.'})
    print('Sequential residual cascade completed')
if __name__=='__main__': run()
