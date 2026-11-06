"""Execute independent scripts, preserve logs, fail visibly on an error."""
from pathlib import Path
import sys, subprocess, time, json

ROOT=Path(__file__).resolve().parent

def main():
    logs=ROOT/'outputs'/'logs'; logs.mkdir(parents=True,exist_ok=True)
    steps=['data_preparation','baselines','prophet_model','xgboost_model','lstm_model','model_comparison','verify_workflow','research_report','extension_preparation','department_analysis','next_day_prophet','next_day_xgboost','next_day_lstm','hybrid_model','sequential_hybrid_model','online_payroll_analysis','privacy_export','next_day_comparison','extension_validation','research_completion','extension_report','research_notebook']
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument('--from-step',choices=steps); args=parser.parse_args()
    result=[]
    if args.from_step:
        earlier=steps[:steps.index(args.from_step)]
        status=logs/'workflow_status.json'
        result=[r for r in json.loads(status.read_text()) if r['step'] in earlier] if status.exists() else []
        steps=steps[steps.index(args.from_step):]
    for step in steps:
        start=time.time(); print(f'Running {step}',flush=True)
        command=[sys.executable,str(ROOT/'src'/f'{step}.py')]
        with (logs/f'{step}_execution.log').open('w',encoding='utf-8') as log:
            proc=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        result.append(dict(step=step,returncode=proc.returncode,elapsed_seconds=time.time()-start,command=command))
        (logs/'workflow_status.json').write_text(json.dumps(result,indent=2))
        if proc.returncode:
            print((logs/f'{step}_execution.log').read_text(encoding='utf-8'),flush=True)
            raise SystemExit(f'{step} failed; see outputs/logs/{step}_execution.log')
        print(f'Completed {step} in {time.time()-start:.1f}s',flush=True)

if __name__=='__main__': main()
