"""Build and execute a results exploration notebook without retraining."""
from pathlib import Path
import os,sys,subprocess
import nbformat
from nbclient import NotebookClient
ROOT=Path(__file__).resolve().parents[1]
def run():
    prefix=ROOT/'outputs/extensions/jupyter'
    subprocess.run([sys.executable,'-m','ipykernel','install','--prefix',str(prefix),'--name','retail-research'],check=True)
    os.environ['JUPYTER_PATH']=str(prefix/'share/jupyter')
    nb=nbformat.v4.new_notebook(); nb.metadata.kernelspec={'name':'retail-research','display_name':'Retail research Python','language':'python'}
    nb.cells=[nbformat.v4.new_markdown_cell('# Retail forecasting results exploration\nExecuted saved-output exploration. Models run in separate scripts. Hourly and daily scores are different tasks. Payroll dates are calendar proxies, not observed customer paydays.'),nbformat.v4.new_code_cell("from pathlib import Path\nimport pandas as pd\nfrom IPython.display import display, Image\nroot=Path.cwd()\next=root/'outputs/extensions'\nmetrics=pd.read_csv(ext/'metrics/daily_metrics.csv')\ndisplay(metrics.query(\"group=='all'\"))"),nbformat.v4.new_code_cell("display(pd.read_csv(ext/'metrics/payday_effects.csv'))\ndisplay(pd.read_csv(ext/'metrics/block_bootstrap.csv'))"),nbformat.v4.new_code_cell("display(pd.read_csv(ext/'metrics/expanding_validation.csv'))\ndisplay(pd.read_csv(ext/'metrics/planning_scenarios.csv'))"),nbformat.v4.new_code_cell("for filename in ['daily_actual_predictions.png','department_patterns.png','weekday_payday_eda.png','XGBoost_With_Payday_shap.png']:\n    display(Image(filename=str(ext/'charts'/filename)))"),nbformat.v4.new_markdown_cell('Unknown within-day capture limits daily totals. Department associations and SHAP are noncausal. Planning metrics are illustrative workload/revenue proxies, not measured staffing or inventory benefits. Read reports/research_extensions.md for assumptions and remaining evidence.')]
    NotebookClient(nb,timeout=180,kernel_name='retail-research',resources={'metadata':{'path':str(ROOT)}}).execute()
    nbformat.write(nb,ROOT/'research_results.ipynb')
    print('Executed research_results.ipynb')
if __name__=='__main__': run()
