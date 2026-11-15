"""Read-only protected-tree audit; writes only new CH4 development reports."""
import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'docs/chapter4/development_audit'


def sha(path):
    value = str(path.resolve())
    if os.name == 'nt':
        value = '\\\\?\\' + value
    h = hashlib.sha256()
    with open(value, 'rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def files(root, skip=()):
    result = {}
    for base, dirs, names in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in skip)
        for name in sorted(names):
            path = Path(base) / name
            result[path.relative_to(root).as_posix()] = sha(path)
    return result


def protected():
    result = {name: files(ROOT.parent/name) for name in ('CH3', 'CH4', 'CH5')}
    source = ROOT.parent/'AUV-Thesis-Active'
    result['AUV-Thesis-Active'] = files(source, ('.git', 'work', '3090结果', '__pycache__', '.pytest_cache'))
    return result


def write_new(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=('before', 'after'))
    args = p.parse_args()
    if args.phase == 'before':
        data = dict(time=datetime.now().astimezone().isoformat(), protected=protected(),
                    project=files(ROOT, ('__pycache__', '.pytest_cache')))
        write_new(AUDIT/'before.json', data)
        print(json.dumps({'before_recorded': True, 'protected_files': {k: len(v) for k,v in data['protected'].items()}}))
    else:
        before = json.loads((AUDIT/'before.json').read_text(encoding='utf-8'))
        current = protected()
        changes = {k: sorted(n for n in current[k].keys()|v.keys() if current[k].get(n) != v.get(n))
                   for k,v in before['protected'].items()}
        project = files(ROOT, ('__pycache__', '.pytest_cache'))
        report = dict(time=datetime.now().astimezone().isoformat(), protected_changes=changes,
                      legacy_unchanged=all(not changes[k] for k in ('CH3','CH4','CH5')),
                      active_source_unchanged=not changes['AUV-Thesis-Active'],
                      modified=[n for n,h in before['project'].items() if project.get(n)!=h],
                      deleted=[n for n in before['project'] if n not in project],
                      created=[n for n in project if n not in before['project']],
                      hashes=project, git_writes=False, training_runs=False)
        # Includes the seed audit script and the final report's own path explicitly.
        report['created'].extend(['chapter4_interval_robust/dev_audit.py', 'docs/chapter4/development_audit/after.json'])
        report['created'] = sorted(set(report['created']))
        write_new(AUDIT/'after.json', report)
        print(json.dumps({k:v for k,v in report.items() if k not in ('hashes','created','modified')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
