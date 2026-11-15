"""Append-only CH4 source records; inherited evidence and failures stay immutable."""
import hashlib
import json
from pathlib import Path
from .configuration import ROOT


def digest(data):
    return hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='\n') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)


def sources():
    result={}
    for directory in ('chapter4_interval_robust','configs/ch4','checks/chapter4'):
        for p in (ROOT/directory).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.json'):
                result[p.relative_to(ROOT).as_posix()]=sha(p)
    for name in ('ch4_cli.py','requirements-ch4.txt','provenance/ch4/upstream_sirsa.json'):
        if (ROOT/name).is_file():result[name]=sha(ROOT/name)
    return dict(sorted(result.items()))


def parent_files():
    snapshot=json.loads((ROOT/'_snapshot/copied_files.json').read_text(encoding='utf-8'))['files']
    # Runtime inputs only; source references/results retain their original records.
    names={r['destination']:r['sha256'] for r in snapshot
           if r['role']=='ACTIVE_WORKTREE_SNAPSHOT' and
           (r['destination'].startswith(('core/','chapter3_bser/','configs/')) or
            r['destination'] in ('project_paths.py','requirements.txt','AGENTS.md'))}
    changed=[n for n,h in names.items() if not (ROOT/n).is_file() or sha(ROOT/n)!=h]
    if changed:raise ValueError('frozen parent runtime changed: '+repr(changed))
    return names


def record(reason):
    if not reason or not reason.strip():raise ValueError('source evolution reason required')
    folder=ROOT/'provenance/ch4';previous=sorted(folder.glob('source_v[0-9][0-9][0-9][0-9].json'))
    old=json.loads(previous[-1].read_text(encoding='utf-8')) if previous else None
    actual=sources();parent=parent_files();version=len(previous)+1
    prior={} if old is None else old['files']
    changes=[dict(path=n,before=prior.get(n),after=actual.get(n),reason=reason)
             for n in sorted(prior.keys()|actual.keys()) if prior.get(n)!=actual.get(n)]
    if old is not None and not changes:raise ValueError('no source evolution to record')
    data=dict(schema='auv.ch4.source.v1',version=version,files=actual,frozen_parent=parent,
              snapshot_manifest_sha256=sha(ROOT/'_snapshot/copied_files.json'),
              inherited_validation_sha256=sha(ROOT/'_snapshot/validation_report.json'),
              parent_source=None if old is None else old['sha256'],changes=changes,
              inherited_source_checks_passed=False,original_sirsa_reproduced=False,
              native_linux_accepted=False,performance_passed=False,record_type='machine_inventory_not_human_or_performance_acceptance')
    data['sha256']=digest(data)
    write_new(folder/f'source_v{version:04d}.json',data)
    return identity()


def identity():
    records=sorted((ROOT/'provenance/ch4').glob('source_v[0-9][0-9][0-9][0-9].json'))
    if not records:raise ValueError('CH4 source inventory not recorded')
    previous=None
    for path in records:
        data=json.loads(path.read_text(encoding='utf-8'))
        if digest({k:v for k,v in data.items() if k!='sha256'})!=data['sha256'] or data['parent_source']!=previous:
            raise ValueError('invalid CH4 source evolution chain')
        previous=data['sha256']
    if sources()!=data['files'] or parent_files()!=data['frozen_parent']:
        raise ValueError('CH4 source changed after source record; append a justified evolution')
    if sha(ROOT/'_snapshot/copied_files.json')!=data['snapshot_manifest_sha256'] or sha(ROOT/'_snapshot/validation_report.json')!=data['inherited_validation_sha256']:
        raise ValueError('original snapshot evidence changed')
    return dict(schema='auv.ch4.source_identity.v1',sha256=data['sha256'],version=data['version'],
                frozen_parent_sha256=digest(data['frozen_parent']),inherited_source_checks_passed=False,
                original_sirsa_reproduced=False,performance_passed=False)
