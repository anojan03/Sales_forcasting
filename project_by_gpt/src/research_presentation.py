"""Editable evidence slides generated from saved outputs using artifact runtime."""
from pathlib import Path
import csv,json
from pptx import Presentation
from pptx.util import Inches,Pt
from pptx.dml.color import RGBColor
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]; EXT=ROOT/'outputs/extensions'
def run():
    p=Presentation(); p.slide_width=Inches(13.333); p.slide_height=Inches(7.5)
    def slide(title):
        s=p.slides.add_slide(p.slide_layouts[6]); box=s.shapes.add_textbox(Inches(.6),Inches(.3),Inches(12),Inches(.8)); r=box.text_frame.paragraphs[0]; r.text=title; r.font.size=Pt(30); r.font.bold=True; return s
    def text(s,lines):
        box=s.shapes.add_textbox(Inches(.7),Inches(1.6),Inches(11.8),Inches(5.3)); tf=box.text_frame; tf.word_wrap=True
        for i,line in enumerate(lines):
            z=tf.paragraphs[0] if i==0 else tf.add_paragraph(); z.text=line; z.font.size=Pt(24); z.space_after=Pt(20)
    def chart(s,name):
        path=EXT/'charts'/name
        with Image.open(path) as im: w,h=im.size
        width=min(11.8,5.5*w/h); height=width*h/w
        s.shapes.add_picture(str(path),Inches((13.333-width)/2),Inches(1.4),width=Inches(width),height=Inches(height))
    s=slide('Retail sales forecasting with calendar payday proxies'); text(s,['Anojan Gnanasegaram | MSc Data Science','Verified hourly study plus upcoming-day research extensions','One independent convenience store; recorded net revenue','Results and limitations generated from executed outputs'])
    s=slide('Chronological experimental design'); text(s,['Training: 2025. Validation: January–March 2026. Test: April–14 July 2026.','Daily forecasts issued at local midnight, before target-day sales.','Prophet, XGBoost and LSTM each run with and without payroll proxies.','Hybrid combines base predictions using shared validation-fitted weights.','Missing hours are not zero sales. Complete recording is unverified.'])
    s=slide('Upcoming-day results on the same 105 recorded days')
    data=[r for r in csv.DictReader((EXT/'metrics/daily_metrics.csv').open()) if r['group']=='all' and 'seed' not in r['model']]
    table=s.shapes.add_table(len(data)+1,3,Inches(.7),Inches(1.5),Inches(11.9),Inches(4.9)).table
    table.columns[0].width=Inches(7); table.columns[1].width=Inches(2.45); table.columns[2].width=Inches(2.45)
    for i,row in enumerate([['Model','MAE GBP/day','RMSE GBP/day']]+[[r['model'].replace('_',' '),f"{float(r['MAE']):.2f}",f"{float(r['RMSE']):.2f}"] for r in data]):
        for j,value in enumerate(row):
            table.cell(i,j).text=value
            for para in table.cell(i,j).text_frame.paragraphs: para.font.size=Pt(17)
    s=slide('Payroll proxies can improve or worsen forecasts'); chart(s,'daily_payday_effect.png')
    s=slide('Department behaviour is analysed from recorded sales'); chart(s,'department_patterns.png')
    s=slide('XGBoost explanations describe prediction contributions'); chart(s,'XGBoost_With_Payday_shap.png')
    s=slide('Sequential hybrid follows the residual-learning architecture'); text(s,['Prophet prediction + XGBoost residual correction + LSTM remaining correction.','Chronological stage blocks: January-June, July-September, October-December 2025.','Later stages learn on forecasts outside earlier stages\' fitting periods.','The existing weighted ensemble is preserved as a separate comparison.','This is an exploratory extension after earlier test outcomes were inspected.'])
    s=slide('Online payroll sources are employer-specific'); chart(s,'online_payroll_comparison.png')
    s=slide('What the evidence supports and what remains unknown'); text(s,['England bank holidays and working-day indicators are included.','NAO, Leicester NHS and Bath payroll sources are documented separately.','Published schedules do not identify the paydays of shop customers.','Literature review and governance/pilot protocols are prepared.','No measured stockout, staffing or profitability improvement is claimed.','Dissertation work is deferred; permissions and external evidence remain pending.'])
    for s in p.slides:
        for sh in s.shapes:
            if sh.has_text_frame:
                for para in sh.text_frame.paragraphs: para.font.name='Calibri'; para.font.color.rgb=RGBColor(0,0,0)
    p.save(ROOT/'reports/research_results.pptx'); print('Saved research_results.pptx')
if __name__=='__main__': run()
