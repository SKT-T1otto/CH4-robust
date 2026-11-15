"""Content/geometry validation and small contact sheets for the delivered PDF."""
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw, ImageFont
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parents[1]


def main():
    report=json.loads((ROOT/'support/layout_report.json').read_text(encoding='utf-8'))
    pdf=PdfReader(str(ROOT/report['pdf']))
    assert len(pdf.pages)==34
    annotation_count=sum(len(p.get('/Annots',[])) for p in pdf.pages)
    assert annotation_count>=8
    images=sorted((ROOT/'renders').glob('page-*.png'))
    assert len(images)==34
    contacts=[]
    label_font=ImageFont.truetype(r'C:\Windows\Fonts\msyh.ttc',16)
    for start in range(0,len(images),6):
        sheet=Image.new('RGB',(1140,1130),'#e7edf2')
        draw=ImageDraw.Draw(sheet)
        for j,path in enumerate(images[start:start+6]):
            im=Image.open(path).convert('RGB')
            im.thumbnail((360,510),Image.Resampling.LANCZOS)
            x=15+(j%3)*380;y=35+(j//3)*560
            sheet.paste(im,(x,y))
            draw.text((x,y-25),f'PDF {start+j+1:02d} / 34',fill='#183b59',font=label_font)
        target=ROOT/'renders'/f'contact_{start//6+1:02d}.jpg'
        sheet.save(target,quality=90)
        contacts.append(target.relative_to(ROOT).as_posix())
    bbox=ROOT/'support/layout_bbox.xhtml'
    outliers=[]
    if bbox.exists():
        document=ET.parse(bbox)
        ns={'h':'http://www.w3.org/1999/xhtml'}
        for number,page in enumerate(document.findall('.//h:page',ns),1):
            width=float(page.attrib['width']);height=float(page.attrib['height'])
            for word in page.findall('.//h:word',ns):
                box={key:float(word.attrib[key]) for key in ('xMin','xMax','yMin','yMax')}
                if box['xMin']<0 or box['xMax']>width+.5 or box['yMin']<0 or box['yMax']>height+.5:
                    outliers.append({'page':number,'text':word.text,'box':box})
        assert not outliers,outliers[:10]
    else:raise AssertionError('PDF word-bounding-box file missing')
    report.update(rendered_pages=len(images),contact_sheets=contacts,bounding_box_outliers=outliers,
                  hyperlinks=annotation_count,bounding_box_check_passed=True,
                  renderer_warnings=['Unused fallback-font messages for Symbol and ArialUnicode; embedded YaHei is visually checked.'])
    (ROOT/'support/layout_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'pages':len(pdf.pages),'rendered':len(images),'contacts':contacts,
                      'hyperlinks':annotation_count,'geometry_passed':not outliers},ensure_ascii=False))


if __name__=='__main__':main()
