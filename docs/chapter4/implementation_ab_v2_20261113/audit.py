"""Fresh before/after inventories; never change or repair protected files."""
import hashlib,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
def sha(p):
    h=hashlib.sha256()
    with open(('\\\\?\\'+str(p.resolve())) if os.name=='nt' else p,'rb') as f:
        for b in iter(lambda:f.read(4194304),b''):h.update(b)
    return h.hexdigest()
def scan(root,skip=()):
    rows={}
    for base,ds,fs in os.walk(root):
        ds[:]=sorted(d for d in ds if d not in skip)
        for name in sorted(fs):
            p=Path(base)/name;rows[p.relative_to(root).as_posix()]=sha(p)
    return rows
def protected():
    r={n:scan(ROOT.parent/n) for n in ('CH3','CH4','CH5')}
    r['Active']=scan(ROOT.parent/'AUV-Thesis-Active',('.git','work','3090结果','__pycache__','.pytest_cache'))
    return r
def main():
    mode=sys.argv[1]
    data={'protected':protected(),'project':scan(ROOT,('__pycache__','.pytest_cache'))}
    if mode=='after':
        old=json.loads((HERE/'before.json').read_text(encoding='utf-8'))
        data['protected_changes']={k:[n for n in sorted(v.keys()|data['protected'][k].keys()) if v.get(n)!=data['protected'][k].get(n)] for k,v in old['protected'].items()}
        data['modified']=[n for n,h in old['project'].items() if data['project'].get(n)!=h]
        data['deleted']=[n for n in old['project'] if n not in data['project']]
        data['created']=sorted(set(data['project'])-set(old['project']))+[str((HERE/'after.json').relative_to(ROOT)).replace('\\','/')]
    with (HERE/(mode+'.json')).open('x',encoding='utf-8') as f:json.dump(data,f,ensure_ascii=False,indent=2)
    print(json.dumps({'protected_counts':{k:len(v) for k,v in data['protected'].items()},**{k:data[k] for k in ('protected_changes','modified','deleted') if k in data}},ensure_ascii=False))
if __name__=='__main__':main()
