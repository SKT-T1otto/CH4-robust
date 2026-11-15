"""Release-only audit. Never modify other repositories or runtime files."""
import hashlib,importlib.util,json,os,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(4194304),b''):h.update(block)
    return h.hexdigest()
def scan(root,skip=()):
    result={}
    for base,ds,fs in os.walk(root):
        ds[:]=[d for d in sorted(ds) if d not in skip]
        for name in sorted(fs):
            p=Path(base)/name;result[p.relative_to(root).as_posix()]=sha(p)
    return result
def protected():
    r={n:scan(ROOT.parent/n) for n in ('CH3','CH4','CH5')}
    r['Active']=scan(ROOT.parent/'AUV-Thesis-Active',('.git','work','3090结果','__pycache__','.pytest_cache'))
    r['CH4_frozen_runtime']={n:scan(ROOT/n) for n in ('core','chapter3_bser','chapter4_interval_robust','configs','checks','provenance','_snapshot','verification')}
    return r
def main():
    mode=sys.argv[1];r={'protected':protected()}
    if mode=='after':
        before=json.loads((HERE/'before.json').read_text(encoding='utf-8'))['protected']
        r['unchanged']=r['protected']==before
        if not r['unchanged']:raise AssertionError('Protected files changed; do not revert user edits')
    with (HERE/(mode+'.json')).open('x',encoding='utf-8') as f:json.dump(r,f,ensure_ascii=False,indent=2)
    print(json.dumps({'recorded':mode,'unchanged':r.get('unchanged'),'legacy_counts':{n:len(r['protected'][n]) for n in ('CH3','CH4','CH5','Active')}}))
if __name__=='__main__':main()
