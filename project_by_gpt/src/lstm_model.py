"""Clock-aligned LSTM histories; missing hours masked, never zero-sales fill."""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
os.environ['TF_DETERMINISTIC_OPS']='1'
from data_preparation import *
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib,time

def run():
    start=time.time(); d=load_data(); c=config(); settings=c['lstm']; steps=settings['sequence_hours']
    tf.config.threading.set_intra_op_parallelism_threads(settings['threads'])
    tf.config.threading.set_inter_op_parallelism_threads(settings['threads'])
    tf.config.experimental.enable_op_determinism()
    train=d.index[mask(d,'train')].to_numpy(); val=d.index[mask(d,'validation')].to_numpy(); test=d.index[mask(d,'test')].to_numpy()
    # Scaler learns only observed training targets; unknown history is replaced
    # by training mean (scaled 0) accompanied by an explicit observed indicator.
    scaler=StandardScaler().fit(d.loc[train,['sales']])
    sales=(d.sales.to_numpy()-scaler.mean_[0])/scaler.scale_[0]
    filled=np.nan_to_num(sales,nan=0.)
    trials=[]; candidates={}
    for units in settings['units_candidates']:
        for pay in [False,True]:
            tf.keras.backend.clear_session(); tf.keras.utils.set_random_seed(c['seed'])
            name='LSTM_'+('With_Payday' if pay else 'Without_Payday')
            cols=[v for v in CALENDAR if v!='trend_days']+(['payday'] if pay else [])
            future=d[cols].to_numpy(dtype=np.float32)
            history=np.column_stack([filled,d.observed.to_numpy(),future]).astype(np.float32)
            def inputs(ix):
                seq=np.stack([history[i-steps:i] for i in ix])
                assert all((d.timestamp.iloc[i]-d.timestamp.iloc[i-steps]).total_seconds()==steps*3600 for i in ix)
                return [seq,future[ix]]
            ti=inputs(train); vi=inputs(val)
            seqin=tf.keras.Input(shape=(steps,history.shape[1]),name='previous_clock_hours')
            targetin=tf.keras.Input(shape=(len(cols),),name='known_target_calendar')
            encoded=tf.keras.layers.LSTM(units)(seqin)
            joined=tf.keras.layers.Concatenate()([encoded,targetin])
            dense=tf.keras.layers.Dense(16,activation='relu')(joined)
            out=tf.keras.layers.Dense(1)(dense)
            model=tf.keras.Model([seqin,targetin],out)
            model.compile(optimizer=tf.keras.optimizers.Adam(settings['learning_rate']),loss='mse')
            early=tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=settings['patience'],restore_best_weights=True)
            hist=model.fit(ti,sales[train],validation_data=(vi,sales[val]),epochs=settings['max_epochs'],batch_size=settings['batch_size'],shuffle=False,callbacks=[early],verbose=0)
            # Explicitly invert the target scaler, independent of feature dimensions.
            vp=scaler.inverse_transform(model.predict(vi,verbose=0)).ravel()
            score=metric(d.loc[val,'sales'],vp)
            trials.append(dict(model=name,units=units,epochs_run=len(hist.history['loss']),best_epoch=int(np.argmin(hist.history['val_loss'])+1),**score))
            path=OUT/'models'/f'{name}_candidate_{units}.keras'; model.save(path)
            candidates[(units,pay)]=(path,cols,history,future,vp,hist.history)
    chosen=min(settings['units_candidates'],key=lambda units:np.mean([r['RMSE'] for r in trials if r['units']==units]))
    for pay in [False,True]:
        name='LSTM_'+('With_Payday' if pay else 'Without_Payday')
        path,cols,history,future,vp,hist=candidates[(chosen,pay)]
        model=tf.keras.models.load_model(path)
        seq=np.stack([history[i-steps:i] for i in test])
        pred=scaler.inverse_transform(model.predict([seq,future[test]],verbose=0)).ravel()
        save_predictions(d,val,vp,name,'validation'); save_predictions(d,test,pred,name)
        model.save(OUT/'models'/f'{name}.keras')
        joblib.dump(dict(target_scaler=scaler,calendar_columns=cols,sequence_hours=steps,history_columns=['scaled_sales','observed']+cols),OUT/'models'/f'{name}_preprocessing.joblib')
        pd.DataFrame(hist).assign(epoch=lambda f:np.arange(1,len(f)+1)).to_csv(OUT/'logs'/f'{name}_training_history.csv',index=False)
    dump(OUT/'logs'/'lstm.json',dict(trials=trials,chosen_units=chosen,selection='mean paired validation RMSE; early stopping per variant on validation only',settings=settings,scaler_mean=float(scaler.mean_[0]),scaler_scale=float(scaler.scale_[0]),elapsed_seconds=time.time()-start))
    print('LSTM completed',chosen)

if __name__=='__main__': run()
