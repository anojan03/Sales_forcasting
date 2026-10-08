"""Seven local calendar-day LSTM histories and known target-day calendars."""
import os
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
os.environ['TF_DETERMINISTIC_OPS']='1'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from extension_preparation import *
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib
def run():
    tf.config.threading.set_intra_op_parallelism_threads(2); tf.config.threading.set_inter_op_parallelism_threads(2)
    f=load(); tr=indices(f,'train'); va=indices(f,'validation'); te=indices(f,'test'); steps=settings()['sequence_days']
    scaler=StandardScaler().fit(f.sales.iloc[tr].to_numpy().reshape(-1,1)); y=scaler.transform(f.sales.to_numpy().reshape(-1,1)).ravel(); trials=[]
    for seed in settings()['seeds']:
        for pay in [False,True]:
            tf.keras.backend.clear_session(); tf.keras.utils.set_random_seed(seed)
            cols=['dow_sin','dow_cos','month_sin','month_cos','holiday','working_day']+(PAY if pay else [])
            cal=f[cols].to_numpy(np.float32); history=np.column_stack([np.nan_to_num(y,nan=0),f.sales.notna().astype(int),cal]).astype(np.float32)
            def inp(ix): return [np.stack([history[i-steps:i] for i in ix]),cal[ix]]
            a=tf.keras.Input((steps,history.shape[1])); b=tf.keras.Input((len(cols),)); h=tf.keras.layers.LSTM(16)(a)
            out=tf.keras.layers.Dense(1)(tf.keras.layers.Dense(16,activation='relu')(tf.keras.layers.Concatenate()([h,b])))
            m=tf.keras.Model([a,b],out); m.compile(optimizer=tf.keras.optimizers.Adam(.001),loss='mse')
            hist=m.fit(inp(tr),y[tr],validation_data=(inp(va),y[va]),epochs=40,batch_size=32,shuffle=False,callbacks=[tf.keras.callbacks.EarlyStopping(patience=5,restore_best_weights=True)],verbose=0)
            vp=scaler.inverse_transform(m.predict(inp(va),verbose=0)).ravel(); pred=scaler.inverse_transform(m.predict(inp(te),verbose=0)).ravel()
            base='LSTM_'+('With_Payday' if pay else 'Without_Payday'); name=base if seed==42 else base+f'_seed_{seed}'
            save(f,va,vp,name,'validation',{'seed':seed}); save(f,te,pred,name,'test',{'seed':seed}); m.save(EXT/'models'/f'{name}.keras')
            pd.DataFrame(hist.history).to_csv(EXT/'logs'/f'{name}_history.csv',index=False)
            trials.append({'model':name,'seed':seed,'epochs':len(hist.history['loss']),'validation':score(f.sales.iloc[va],vp)})
    joblib.dump(scaler,EXT/'models'/'lstm_scaler.joblib'); dump(EXT/'logs'/'lstm.json',{'trials':trials,'units':16,'sequence_days':steps,'meaning':'local days, not 24 elapsed hours per step','no_test_seed_selection':True})
if __name__=='__main__': run()
