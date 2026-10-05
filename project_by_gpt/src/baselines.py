"""Observed-lag baselines with training-median fallback, never zero fill."""
from data_preparation import *

def run():
    d=load_data(); med=d.loc[mask(d,'train'),'sales'].median()
    info={}
    for name,lag in [('Naive',1),('Daily_Seasonal_Naive',24),('Weekly_Seasonal_Naive',168)]:
        for split in ['validation','test']:
            ix=d.index[mask(d,split)]
            raw=d.loc[ix,f'lag_{lag}']
            pred=raw.fillna(med).to_numpy()
            save_predictions(d,ix,pred,name,split)
            info[f'{name}_{split}']=dict(**metric(d.loc[ix,'sales'],pred),fallback_hours=int(raw.isna().sum()),fallback_value=float(med))
    dump(OUT/'logs'/'baselines.json',info)

if __name__=='__main__': run()
