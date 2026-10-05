"""Causality mutation checks, scaler provenance, and source integrity."""
from data_preparation import *
import joblib

def run():
    d=load_data(); c=config(); checks={}
    delta=d.timestamp.diff().dropna().dt.total_seconds()
    assert delta.eq(3600).all(); checks['every_grid_step_is_one_elapsed_hour']=True
    original=history_features(d)
    cols=feature_names(True)
    # Replace target and future sales at multiple origins. Features for forecasts
    # issued at those origins must remain identical, including missing flags.
    for split in ['train','validation','test']:
        candidates=d.index[mask(d,split)]; cut=int(candidates[len(candidates)//2])
        changed=d.copy(); changed.loc[cut:,'sales']=1234567.89
        alternate=history_features(changed)
        np.testing.assert_allclose(original.loc[:cut,cols],alternate.loc[:cut,cols],equal_nan=True)
    checks['target_and_future_sales_mutation_cannot_change_origin_features']=True
    train=mask(d,'train')
    for family in ['Prophet','XGBoost']:
        for pay in [False,True]:
            name=family+'_ '+('With_Payday' if pay else 'Without_Payday'); name=name.replace('_ ','_')
            saved=joblib.load(OUT/'models'/f'{name}_preprocessing.joblib')
            names=feature_names(pay)
            np.testing.assert_allclose(saved['imputer'].statistics_,d.loc[train,names].median().to_numpy(),equal_nan=True)
    checks['imputer_statistics_equal_training_only_medians']=True
    for pay in [False,True]:
        name='LSTM_'+('With_Payday' if pay else 'Without_Payday')
        saved=joblib.load(OUT/'models'/f'{name}_preprocessing.joblib')
        scaler=saved['target_scaler']; np.testing.assert_allclose(scaler.mean_[0],d.loc[train,'sales'].mean())
        v=d.loc[train,['sales']].head(100)
        np.testing.assert_allclose(scaler.inverse_transform(scaler.transform(v)),v,rtol=1e-12,atol=1e-12)
    checks['lstm_train_only_scaler_and_inverse_roundtrip']=True
    for r in json.loads((OUT/'audit'/'dataset_audit.json').read_text()):
        assert hashlib.sha256((ROOT/r['file']).read_bytes()).hexdigest()==r['sha256']
    checks['source_csv_hashes_unchanged']=True
    valid=json.loads((OUT/'logs'/'comparison_validation.json').read_text())
    assert valid['all_expected_hours_common']; checks['all_six_cover_identical_expected_test_hours']=True
    dump(OUT/'logs'/'verification.json',checks)
    print(json.dumps(checks,indent=2))

if __name__=='__main__': run()
