"""Build a typeset Chinese proposal with page-by-page fit and text checks."""
from datetime import date
import hashlib
import html
import json
from pathlib import Path
import re
import sys
import warnings

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle, Spacer
from pypdf import PdfReader

from plan_content import PAGES

ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT.parents[2]
WORKSPACE=PROJECT.parent
OUT=ROOT/'output/pdf'
PDF=OUT/'CH4_SIRSA_AB_详细研究方案_20261113_v2.pdf'
W,H=A4
M=47
BODY_W=W-2*M
BOTTOM=48
BLUE=colors.HexColor('#183b59')
TEAL=colors.HexColor('#087c83')
INK=colors.HexColor('#253447')
MUTED=colors.HexColor('#63758a')
PALE=colors.HexColor('#edf4f8')


def clean(value):
    return str(value).replace('\u2011','-').replace('\u2013','-').replace('\u2014','-').replace('\u2212','-')


def markup(value):
    s=html.escape(clean(value))
    s=re.sub(r'\*\*(.*?)\*\*',r'<b>\1</b>',s)
    return s.replace('\n','<br/>')


def styles(scale=1):
    def st(name,size,leading,**kwargs):
        return ParagraphStyle(name,fontName='YaHei',fontSize=size*scale,leading=leading*scale,
                              textColor=INK,wordWrap='CJK',splitLongWords=True,**kwargs)
    return dict(body=st('body',10.25,17.2,spaceAfter=7),
                bullet=st('bullet',10.15,16.8,leftIndent=12,firstLineIndent=-10),
                cell=st('cell',9.15,14.5),
                th=ParagraphStyle('th',fontName='YaHeiBold',fontSize=9.2*scale,leading=14.5*scale,
                                  textColor=colors.white,wordWrap='CJK'),
                callhead=st('callhead',10.3,16),
                call=st('call',10.1,16.3),
                refs=st('refs',8.5,12.4),
                code=ParagraphStyle('code',fontName='Courier',fontSize=8.1*scale,leading=12.0*scale,
                                    textColor=INK,wordWrap='CJK'),
                title=ParagraphStyle('title',fontName='YaHeiBold',fontSize=18,leading=25,textColor=BLUE,
                                     wordWrap='CJK'),
                subtitle=ParagraphStyle('subtitle',fontName='YaHei',fontSize=9.2,leading=14,textColor=MUTED,
                                        wordWrap='CJK'))


def references():
    catalog=WORKSPACE/'THESIS_LITERATURE_2026_20261113/_catalog/ch4_catalog.json'
    rows=json.loads(catalog.read_text(encoding='utf-8'))
    ids=['arxiv:2202.07013','arxiv:1702.02453','arxiv:2107.04034','arxiv:1610.01283',
         'pmlr:v306/wikman26a','arxiv:2406.08395','arxiv:2610.10905','arxiv:2611.10938']
    selected=[]
    for number,identifier in enumerate(ids,1):
        r=next(r for r in rows if r['id']==identifier)
        selected.append(dict(number=number,**r))
    return selected


REFS=references()


def flowables(block,st):
    kind=block[0]
    if kind=='p':return [Paragraph(markup(block[1]),st['body']),Spacer(1,6)]
    if kind=='bullets':
        result=[]
        for value in block[1]:result.extend([Paragraph('• '+markup(value),st['bullet']),Spacer(1,6)])
        return result
    if kind=='table':
        headers,rows,ratios=block[1:]
        widths=[BODY_W*v for v in (ratios or [1/len(headers)]*len(headers))]
        values=[[Paragraph(markup(v),st['th']) for v in headers]]
        values += [[Paragraph(markup(v),st['cell']) for v in row] for row in rows]
        table=Table(values,colWidths=widths,hAlign='LEFT')
        table.setStyle(TableStyle([
            ('BACKGROUND',(0,0),(-1,0),BLUE),('VALIGN',(0,0),(-1,-1),'TOP'),
            ('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8),
            ('TOPPADDING',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),8),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f1f5f8')]),
            ('LINEBELOW',(0,0),(-1,0),.6,BLUE),('LINEBELOW',(0,1),(-1,-1),.35,colors.HexColor('#d7e1ea'))]))
        return [table,Spacer(1,12)]
    if kind=='callout':
        table=Table([[Paragraph('<b>'+markup(block[1])+'</b>',st['callhead'])],
                     [Paragraph(markup(block[2]),st['call'])]],colWidths=[BODY_W])
        table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),PALE),
              ('LINEBEFORE',(0,0),(0,-1),3,TEAL),('LEFTPADDING',(0,0),(-1,-1),12),
              ('RIGHTPADDING',(0,0),(-1,-1),12),('TOPPADDING',(0,0),(-1,0),9),
              ('BOTTOMPADDING',(0,-1),(-1,-1),10),('TOPPADDING',(0,1),(-1,1),3)]))
        return [table,Spacer(1,12)]
    if kind=='code':
        lines=block[1].splitlines()
        cells=[[Paragraph(markup(line).replace(' ','&nbsp;'),st['code'])] for line in lines]
        table=Table(cells,colWidths=[BODY_W])
        table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),colors.HexColor('#f2f5f8')),
                  ('LEFTPADDING',(0,0),(-1,-1),10),('RIGHTPADDING',(0,0),(-1,-1),10),
                  ('TOPPADDING',(0,0),(-1,-1),2),('BOTTOMPADDING',(0,0),(-1,-1),2)]))
        return [table,Spacer(1,12)]
    if kind=='pipeline':return [('pipeline',),Spacer(1,15)]
    if kind=='references':
        result=[]
        for r in REFS:
            title=f"[R{r['number']}] {'; '.join(r['authors'])}. {r['title']}. "
            pub=r.get('publication_year');first=r.get('first_year')
            year=f"首发{first or '未核定'}；正式{pub or '本次未核定'}。"
            if r['number'] in (7,8):year+='2026预印本。'
            paragraph=markup(title+year)+'<br/><link href="'+html.escape(r['landing_url'],quote=True)+'" color="#087c83">'+markup(r['landing_url'])+'</link>'
            result.extend([Paragraph(paragraph,st['refs']),Spacer(1,8)])
        return result
    raise ValueError(kind)


def draw_pipeline(cv,x,y):
    height=82;gap=12;boxw=(BODY_W-3*gap)/4
    nodes=[('合法历史','公开状态与已发命令'),('A+B集合更新','共享结构 / 执行时序'),('SIRSA风险策略','一致场景 / 团队低尾'),('任务验证','覆盖 / 成功 / 恢复')]
    for i,(title,detail) in enumerate(nodes):
        bx=x+i*(boxw+gap)
        cv.setFillColor(BLUE if i==1 else PALE);cv.roundRect(bx,y-height,boxw,height,7,fill=1,stroke=0)
        cv.setFont('YaHeiBold',10);cv.setFillColor(colors.white if i==1 else BLUE)
        cv.drawCentredString(bx+boxw/2,y-27,title)
        cv.setFont('YaHei',7.4);cv.drawCentredString(bx+boxw/2,y-49,detail)
        if i<3:
            cv.setStrokeColor(TEAL);cv.setLineWidth(1.3)
            cv.line(bx+boxw+2,y-height/2,bx+boxw+gap-2,y-height/2)
            cv.line(bx+boxw+gap-5,y-height/2+3,bx+boxw+gap-2,y-height/2)
            cv.line(bx+boxw+gap-5,y-height/2-3,bx+boxw+gap-2,y-height/2)
    return height


def measure(blocks,st):
    elements=[];height=0
    for block in blocks:
        for f in flowables(block,st):
            if isinstance(f,tuple):size=82
            else:_,size=f.wrap(BODY_W,H)
            height+=size;elements.append((f,size))
    return elements,height


def markdown():
    lines=['# 第四章详细研究方案 v2.0','','生成日期：2026-11-13。主文为拟议设计，算法尚未按v2修改。','']
    for i,pg in enumerate(PAGES,1):
        lines += [f"## PDF第{i}页：{pg['title']}",'',pg['subtitle'],'']
        for block in pg['blocks']:
            kind=block[0]
            if kind=='p':lines += [block[1],'']
            elif kind=='bullets':lines += ['- '+s for s in block[1]]+['']
            elif kind=='callout':lines += ['**'+block[1]+'**','',block[2],'']
            elif kind=='code':lines += ['```text',block[1],'```','']
            elif kind=='table':
                lines += ['| '+' | '.join(block[1])+' |','| '+' | '.join(['---']*len(block[1]))+' |']
                lines += ['| '+' | '.join(str(s).replace('|','/') for s in row)+' |' for row in block[2]]+['']
            elif kind=='pipeline':lines += ['合法历史 → A+B条件集合 → SIRSA团队低尾策略 → 任务验证','']
            elif kind=='references':
                for r in REFS:lines += [f"[R{r['number']}] {'; '.join(r['authors'])}. {r['title']}. {r['landing_url']}",'']
    (ROOT/'CH4_SIRSA_AB_详细方案_v2.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    pdfmetrics.registerFont(TTFont('YaHei',r'C:\Windows\Fonts\msyh.ttc',subfontIndex=0))
    pdfmetrics.registerFont(TTFont('YaHeiBold',r'C:\Windows\Fonts\msyhbd.ttc',subfontIndex=0))
    pdfmetrics.registerFontFamily('YaHei',normal='YaHei',bold='YaHeiBold',italic='YaHei',boldItalic='YaHeiBold')
    cv=canvas.Canvas(str(PDF),pagesize=A4,pageCompression=1)
    cv.setTitle('第四章详细研究方案：基于SIRSA的A+B改进 v2.0')
    cv.setAuthor('AUV Thesis research planning')
    cv.setSubject('Proposed design; not implemented v2 or validated performance')
    reports=[]
    for number,pg in enumerate(PAGES,1):
        cv.setFillColor(BLUE);cv.rect(0,H-10,W,10,fill=1,stroke=0)
        cv.setFont('YaHei',8);cv.setFillColor(MUTED)
        cv.drawString(M,H-32,'CH4  /  SIRSA + A+B')
        cv.drawRightString(W-M,H-32,'详细研究方案 v2.0  ·  2026-11-13')
        cv.bookmarkPage('p'+str(number));cv.addOutlineEntry(pg['title'],'p'+str(number),level=0)
        st=styles();title=Paragraph(markup(pg['title']),st['title']);_,th=title.wrap(BODY_W,H)
        y=H-58;title.drawOn(cv,M,y-th);y-=th+8
        sub=Paragraph(markup(pg['subtitle']),st['subtitle']);_,sh=sub.wrap(BODY_W,H)
        sub.drawOn(cv,M,y-sh);y-=sh+17
        available=y-BOTTOM
        chosen=None
        for scale in (1,.98,.96,.94,.92):
            elements,height=measure(pg['blocks'],styles(scale))
            if height<=available:
                chosen=(scale,elements,height);break
        if chosen is None:raise RuntimeError(f"Page {number} overflow: {height:.1f}>{available:.1f}; edit content, do not clip")
        scale,elements,height=chosen
        for f,size in elements:
            if isinstance(f,tuple):draw_pipeline(cv,M,y)
            else:f.drawOn(cv,M,y-size)
            y-=size
        cv.setStrokeColor(colors.HexColor('#d7e1ea'));cv.setLineWidth(.5);cv.line(M,35,W-M,35)
        cv.setFont('YaHei',7.5);cv.setFillColor(MUTED)
        cv.drawString(M,22,'拟议设计；代码与性能尚待验收')
        cv.drawRightString(W-M,22,f'{number:02d} / {len(PAGES):02d}')
        reports.append(dict(page=number,title=pg['title'],scale=scale,body_height=round(height,2),
                            available_height=round(available,2),bottom=round(y,2),overflow=False))
        cv.showPage()
    cv.save();markdown()
    pdf=PdfReader(str(PDF));assert len(pdf.pages)==len(PAGES)
    texts=[pg.extract_text() or '' for pg in pdf.pages]
    for i,(pg,txt) in enumerate(zip(PAGES,texts),1):
        assert re.sub(r'\s+','',pg['title']) in re.sub(r'\s+','',txt), (i,'missing title')
        assert '\ufffd' not in txt,(i,'replacement character')
        assert len(txt)>250,(i,'suspiciously empty page')
    combined='\n'.join(texts)
    for token in ['21D','165D','705D','S0T0','S1T1','Width-only','REVIEW_REQUIRED','sigma','GitHub']:
        assert token in combined,token
    (ROOT/'support/extracted_text.txt').write_text('\n\n'.join(f'=== PDF PAGE {i+1} ===\n{txt}' for i,txt in enumerate(texts)),encoding='utf-8')
    (ROOT/'support/document_structure.json').write_text(json.dumps(PAGES,ensure_ascii=False,indent=2),encoding='utf-8')
    report=dict(pages=len(pdf.pages),pdf=str(PDF.relative_to(ROOT)),pdf_sha256=hashlib.sha256(PDF.read_bytes()).hexdigest(),
                text_checks_passed=True,embedded_fonts=['Microsoft YaHei','Microsoft YaHei Bold'],page_fits=reports,
                visual_review_status='PENDING',note='Render every page before final delivery.')
    (ROOT/'support/layout_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    (ROOT/'support/references.json').write_text(json.dumps(REFS,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'pdf':str(PDF),'pages':len(pdf.pages),'min_scale':min(r['scale'] for r in reports),
                      'bytes':PDF.stat().st_size,'text_checks_passed':True},ensure_ascii=False))


if __name__=='__main__':main()
