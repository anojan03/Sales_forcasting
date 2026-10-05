"""Compare the two executions retained locally; no model retraining."""
from data_preparation import *
from model_comparison import MODELS

def run():
    checks={}
    for name in MODELS:
        first=pd.read_csv(OUT/'reproducibility_first'/f'{name}_test.csv')
        second=pd.read_csv(OUT/'predictions'/f'{name}_test.csv')
        assert first.timestamp.equals(second.timestamp)
        np.testing.assert_allclose(first.actual_sales,second.actual_sales,rtol=0,atol=0)
        delta=float(np.max(np.abs(first.predicted_sales-second.predicted_sales)))
        checks[name]=dict(n=len(second),max_absolute_prediction_difference=delta,identical_predictions=delta==0)
    dump(OUT/'logs'/'reproducibility_check.json',dict(config=config(),prepared_sha256=hashlib.sha256((OUT/'prepared'/'hourly_grid.csv').read_bytes()).hexdigest(),variants=checks))
    print(json.dumps(checks,indent=2))

if __name__=='__main__': run()
