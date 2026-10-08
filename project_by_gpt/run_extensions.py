"""Execute proposal extensions without retraining preserved hourly experiments."""
from pathlib import Path
import sys, subprocess, time, json
ROOT=Path(__file__).resolve().parent
def main():
    logs=ROOT/'outputs/extensions/logs'; logs.mkdir(parents=True,exist_ok=True); rows=[]
    for step in ['extension_preparation','department_analysis','next_day_prophet','next_day_xgboost','next_day_lstm','hybrid_model','next_day_comparison','extension_validation','extension_report','research_notebook']:
        start=time.time(); print('Running',step,flush=True)
        with (logs/f'{step}_execution.log').open('w',encoding='utf-8') as log: p=subprocess.run([sys.executable,str(ROOT/'src'/f'{step}.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        rows.append({'step':step,'returncode':p.returncode,'elapsed_seconds':time.time()-start}); (logs/'workflow_status.json').write_text(json.dumps(rows,indent=2))
        if p.returncode: print((logs/f'{step}_execution.log').read_text(encoding='utf-8')); raise SystemExit(p.returncode)
        print('Completed',step,flush=True)
if __name__=='__main__': main()
