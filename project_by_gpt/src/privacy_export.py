"""Create local aggregate disclosure-review candidates, never auto-publish."""
from extension_preparation import *
from pathlib import Path
def run():
    c=config(); fields=['Transaction_Time','Transaction_ID','Department_ID','Line_Total','Product_Name']
    m=pd.read_csv(ROOT/c['source_file'],usecols=fields); dt=pd.to_datetime(m.Transaction_Time,format='mixed')
    keep=(dt>=pd.Timestamp(c['analysis_start']))&(dt<pd.Timestamp(c['analysis_end_exclusive']))&~m.Product_Name.str.strip().str.startswith('-PAYOUT',na=False)
    d=m.loc[keep,['Transaction_ID','Department_ID','Line_Total']].copy(); d['month']=dt[keep].dt.strftime('%Y-%m')
    g=d.groupby(['month','Department_ID']).agg(receipts=('Transaction_ID','nunique'),net_revenue=('Line_Total','sum')).reset_index()
    codes={value:f'C{i+1:02d}' for i,value in enumerate(sorted(g.Department_ID.unique()))}
    g['category_code']=g.Department_ID.map(codes); g['suppressed']=g.receipts.lt(10)
    g['rounded_revenue_GBP']=g.net_revenue.round(0).where(~g.suppressed)
    candidate=g[['month','category_code','suppressed','rounded_revenue_GBP']]
    path=EXT/'prepared'/'disclosure_review_candidate.csv'; candidate.to_csv(path,index=False)
    forbidden={'Customer_ID','Transaction_ID','Barcode','Payment_Type','Product_Name','Transaction_Time','Department_ID'}
    assert not forbidden.intersection(candidate.columns)
    assert candidate.loc[candidate.suppressed,'rounded_revenue_GBP'].isna().all()
    # Keep the code map separately in local restricted audit outputs, not the candidate.
    dump(EXT/'audit'/'disclosure_review_category_map.json',{str(key):value for key,value in codes.items()})
    evidence=json.loads((ROOT/'governance'/'evidence_status.json').read_text())
    dump(EXT/'logs'/'governance_readiness.json',{'candidate_path':str(path),'candidate_rows':len(candidate),'suppressed_cells':int(candidate.suppressed.sum()),'identifying_columns_excluded':True,'original_source_sha256':hashlib.sha256((ROOT/c['source_file']).read_bytes()).hexdigest(),'evidence_status':evidence,'release_authorised':False,'reason':'Owner/institution permission and contextual disclosure review are not established by code','warning':'A review candidate, not a certified anonymous dataset. Monthly totals remain commercially sensitive.'})
    print('Local disclosure review candidate generated; no data published')
if __name__=='__main__': run()
