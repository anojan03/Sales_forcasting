"""Render the shared evidence payload to Word and PDF using artifact runtime."""
from pathlib import Path
import json,html,sys,importlib.metadata,os
from docx import Document
from docx.shared import Inches,Pt,RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,KeepTogether
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from PIL import Image as PILImage

ROOT=Path(__file__).resolve().parents[1]

def run():
    payload=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'outputs/logs/report_payload.json'
    stem=sys.argv[2] if len(sys.argv)>2 else 'research_report'
    blocks=json.loads(payload.read_text())
    (ROOT/'outputs/logs/artifact_environment.json').write_text(json.dumps({'python':sys.version,'executable':sys.executable,'packages':{p:importlib.metadata.version(p) for p in ['python-docx','reportlab','Pillow']}},indent=2))
    doc=Document(); sec=doc.sections[0]
    sec.page_width=Inches(8.27); sec.page_height=Inches(11.69)
    sec.top_margin=sec.bottom_margin=Inches(.7); sec.left_margin=sec.right_margin=Inches(.7)
    styles=doc.styles
    for name in ['Normal','Title','Heading 1','Heading 2']:
        styles[name].font.name='Calibri'; styles[name].font.color.rgb=RGBColor(0,0,0)
    styles['Normal'].font.size=Pt(10); styles['Normal'].paragraph_format.space_after=Pt(6)
    styles['Normal'].paragraph_format.line_spacing=1.08
    styles['Title'].font.size=Pt(23); styles['Heading 1'].font.size=Pt(15); styles['Heading 2'].font.size=Pt(12)
    doc.core_properties.title=blocks[0]['text']; doc.core_properties.author=''; doc.core_properties.subject='Retail sales forecasting research'
    pdfstyles=getSampleStyleSheet()
    pdfstyles['Normal'].fontSize=9; pdfstyles['Normal'].leading=12; pdfstyles['Normal'].spaceAfter=7
    pdfstyles['Title'].fontSize=22; pdfstyles['Title'].leading=26
    pdfstyles['Heading1'].fontSize=14; pdfstyles['Heading1'].leading=18; pdfstyles['Heading1'].spaceBefore=12
    pdfstyles['Heading2'].fontSize=11; pdfstyles['Heading2'].leading=14
    for style_name in ['Title','Heading1','Heading2']: pdfstyles[style_name].keepWithNext=True
    cellstyle=ParagraphStyle('cell',parent=pdfstyles['Normal'],fontSize=7.5,leading=10,spaceAfter=0)
    headstyle=ParagraphStyle('headcell',parent=cellstyle,textColor=colors.white)
    story=[]
    for b in blocks:
        kind=b['kind']
        if kind=='heading':
            lev=b['level']; doc.add_paragraph(b['text'],style='Title' if lev==0 else f'Heading {lev}')
            story.append(Paragraph(html.escape(b['text']),pdfstyles['Title' if lev==0 else f'Heading{lev}']))
        elif kind=='paragraph':
            doc.add_paragraph(b['text']); story.append(Paragraph(html.escape(b['text']),pdfstyles['Normal']))
        elif kind=='table':
            headers=b['headers']; rows=b['rows']; table=doc.add_table(rows=1,cols=len(headers)); table.autofit=False
            # Wider descriptive first column; evenly distribute compact numeric fields.
            available=6.87
            widths=[available*v for v in b['width_fractions']] if 'width_fractions' in b else ([available*.4]+[available*.6/(len(headers)-1)]*(len(headers)-1) if len(headers)>1 else [available])
            for col,width in zip(table.columns,widths): col.width=Inches(width)
            for cell,text,width in zip(table.rows[0].cells,headers,widths):
                cell.width=Inches(width); cell.text=text
            repeat=OxmlElement('w:tblHeader'); table.rows[0]._tr.get_or_add_trPr().append(repeat)
            for row in rows:
                for cell,text,width in zip(table.add_row().cells,row,widths): cell.width=Inches(width); cell.text=text
            for ri,row in enumerate(table.rows):
                cant_split=OxmlElement('w:cantSplit'); row._tr.get_or_add_trPr().append(cant_split)
                for cell in row.cells:
                    cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
                    props=cell._tc.get_or_add_tcPr(); shading=OxmlElement('w:shd'); shading.set(qn('w:fill'),'25374D' if ri==0 else ('F0F3F6' if ri%2==0 else 'FFFFFF')); props.append(shading)
                    borders=OxmlElement('w:tcBorders')
                    for edge in ['top','left','bottom','right']:
                        el=OxmlElement('w:'+edge); el.set(qn('w:val'),'single'); el.set(qn('w:sz'),'4'); el.set(qn('w:color'),'D9D9D9'); borders.append(el)
                    props.append(borders)
                    for p in cell.paragraphs:
                        p.paragraph_format.space_after=Pt(3); p.paragraph_format.space_before=Pt(3)
                        p.paragraph_format.keep_with_next=(ri==0 or (stem!='research_report' and len(rows)<=10 and ri<len(rows)))
                        for run in p.runs: run.font.size=Pt(8); run.font.color.rgb=RGBColor(255,255,255) if ri==0 else RGBColor(0,0,0); run.bold=ri==0
            doc.add_paragraph().paragraph_format.space_after=Pt(2)
            prows=[[Paragraph(html.escape(v),headstyle) for v in headers]]+[[Paragraph(html.escape(v),cellstyle) for v in row] for row in rows]
            t=Table(prows,colWidths=[w*72 for w in widths],repeatRows=1,hAlign='LEFT')
            t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#25374D')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F0F3F6')]),('GRID',(0,0),(-1,-1),.4,colors.HexColor('#D9D9D9')),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
            story.extend([KeepTogether([t]) if stem!='research_report' and len(rows)<=10 else t,Spacer(1,9)])
        elif kind=='figure':
            im=PILImage.open(b['path']); w,h=im.size; width=min(6.5,5.4*w/h); height=width*h/w
            if Path(b['path']).name in ['residuals.png','residual_time.png']:
                width=6.; height=width*h/w
            p=doc.add_paragraph(); p.paragraph_format.keep_with_next=True; p.add_run().add_picture(b['path'],width=Inches(width))
            # Alt text gives accessible meaning beyond the image.
            inline=p.runs[0]._r.xpath('.//wp:docPr')
            if inline: inline[0].set('descr',b['caption'])
            cap=doc.add_paragraph(b['caption']); cap.paragraph_format.space_after=Pt(12)
            for run in cap.runs: run.font.size=Pt(9); run.italic=True
            story.append(KeepTogether([Image(b['path'],width=width*72,height=height*72),Paragraph(html.escape(b['caption']),pdfstyles['Normal']),Spacer(1,10)]))
    # Remove default-template decorative paragraph borders, including Title.
    for root in [doc.element,doc.styles.element]:
        for border in list(root.xpath('.//w:pBdr')): border.getparent().remove(border)
    doc_path=ROOT/'reports'/f'{stem}.docx'
    try: doc.save(doc_path)
    except PermissionError:
        doc_path=ROOT/'reports'/f'{stem}_updated.docx'; doc.save(doc_path)
    def footer(canvas,document):
        canvas.setFont('Helvetica',8); canvas.setFillColor(colors.grey); canvas.drawRightString(A4[0]-50,28,str(document.page))
    pdf_path=ROOT/'reports'/f'{stem}.pdf'; temporary=ROOT/'reports'/f'{stem}_building.pdf'
    SimpleDocTemplate(str(temporary),pagesize=A4,rightMargin=50,leftMargin=50,topMargin=50,bottomMargin=45,title=blocks[0]['text'],author='').build(story,onFirstPage=footer,onLaterPages=footer)
    try: os.replace(temporary,pdf_path)
    except PermissionError:
        pdf_path=ROOT/'reports'/f'{stem}_updated.pdf'; os.replace(temporary,pdf_path)
    (ROOT/'outputs/logs'/f'{stem}_artifact_paths.json').write_text(json.dumps({'docx':str(doc_path),'pdf':str(pdf_path)},indent=2))
    print('Report artifacts saved',doc_path,pdf_path)

if __name__=='__main__': run()
