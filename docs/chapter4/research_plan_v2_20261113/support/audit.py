"""Read-only protected-file audit; all reports stay in this document delivery folder."""
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT.parents[2]
WORKSPACE=PROJECT.parent


def sha(path):
    h=hashlib.sha256()
    absolute=str(path.resolve())
    if os.name=='nt': absolute='\\\\?\\'+absolute
    with open(absolute,'rb') as f:
        for part in iter(lambda:f.read(4*1024*1024),b''): h.update(part)
    return h.hexdigest()


def inventory():
    result={}
    roots=[WORKSPACE/name for name in ('CH3','CH4','CH5')]+[PROJECT]
    active=WORKSPACE/'AUV-Thesis-Active'
    roots.extend(active/name for name in ('core','chapter3_bser','configs','checks','tools','scripts','provenance'))
    for root in roots:
        for base,dirs,files in os.walk(root):
            dirs[:]=sorted(d for d in dirs if not (Path(base)/d).resolve().is_relative_to(ROOT.resolve()))
            for name in sorted(files):
                file=Path(base)/name
                if file.resolve().is_relative_to(ROOT.resolve()):continue
                result[file.relative_to(WORKSPACE).as_posix()]=sha(file)
    return result


def main():
    mode=sys.argv[1]
    if mode in ('before','after'):
        now=inventory();report={'files':now,'file_count':len(now)}
        if mode=='after':
            old=json.loads((ROOT/'support/integrity_before.json').read_text(encoding='utf-8'))['files']
            changes=[k for k in sorted(now.keys()|old.keys()) if now.get(k)!=old.get(k)]
            report.update(unchanged=not changes,changed=changes)
        target=ROOT/'support'/('integrity_'+mode+'.json')
        with target.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
        print(json.dumps({k:v for k,v in report.items() if k!='files'},ensure_ascii=False))
        if mode=='after' and changes:raise SystemExit('Protected files changed; inspect instead of reverting')
    elif mode=='manifest':
        target=ROOT/'support/created_files.json'
        rows=[{'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)}
              for p in sorted(ROOT.rglob('*')) if p.is_file() and p!=target]
        rows.append({'path':'support/created_files.json','bytes':None,'sha256':None,'note':'self inventory'})
        with target.open('x',encoding='utf-8') as f:json.dump({'files':rows,'file_count':len(rows)},f,ensure_ascii=False,indent=2)
        print(json.dumps({'created_files':len(rows)}))
    else:raise SystemExit('mode must be before, after or manifest')


if __name__=='__main__':main()
