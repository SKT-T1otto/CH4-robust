"""Audit attempts and distinguish complete missions from uncommitted prefixes."""
import argparse
import json
from pathlib import Path

from .config import read_json, file_sha


def summarize_episodes(rows):
    complete=[r for r in rows if r.get('completed') is True and r.get('eligible_for_task_statistics') is True]
    for row in rows:
        if row not in complete and (row.get('success') is not None or row.get('penalized_completion_steps') is not None):
            raise ValueError('incomplete episode carries a task outcome')
    n=len(complete)
    return dict(attempted=len(rows),completed=n,incomplete=len(rows)-n,
        success_rate=None if not n else sum(bool(r['success']) for r in complete)/n,
        mean_penalized_completion_steps=None if not n else sum(r['penalized_completion_steps'] for r in complete)/n,
        denominator='completed_native_missions_only',complete_results_are_not_nominal_admission=True)


def choose_validation(candidates):
    """Compare only full identical validation cohorts; earlier committed boundary wins ties."""
    if not candidates: raise ValueError('no validation candidates')
    catalog=candidates[0]['evaluation']['catalog_sha256']
    for item in candidates:
        evaluation=item['evaluation']; rows=evaluation['episodes']
        if (not evaluation['selection_eligible'] or evaluation['group']!='validation'
                or evaluation['catalog_sha256']!=catalog or sorted(evaluation['indices'])!=list(range(32))
                or len(rows)!=32 or sorted(r['validation_index'] for r in rows)!=list(range(32))
                or len({r['scenario_id'] for r in rows})!=32
                or not all(r['completed'] and r['eligible_for_task_statistics'] for r in rows)):
            raise ValueError('checkpoint selection requires all 32 complete validation scenes')
    return min(candidates,key=lambda item:(summarize_episodes(item['evaluation']['episodes'])['mean_penalized_completion_steps'],
                                          item['completed_episodes']))


def audit_run(folder):
    folder=Path(folder); run=read_json(folder/'run.json')
    paths=sorted((folder/'events').glob('*.json')); events=[]; torn_tail=None
    for index,path in enumerate(paths):
        try: events.append(read_json(path))
        except (json.JSONDecodeError,UnicodeDecodeError):
            if index!=len(paths)-1: raise ValueError('corrupt audit event before journal tail')
            torn_tail=path.name
    if [e['sequence'] for e in events]!=list(range(len(events))): raise ValueError('audit event sequence has a gap')
    committed=[]; partial=[]
    for path in sorted((folder/'boundaries').glob('*')):
        marker=path/'committed.json'
        if not marker.is_file(): partial.append(path.name); continue
        try: item=read_json(marker)
        except (json.JSONDecodeError,UnicodeDecodeError):
            partial.append(path.name); continue
        if item['identity']!=run['identity'] or file_sha(path/'state.pt')!=item['file_sha256']:
            raise ValueError('committed boundary identity or bytes differ')
        committed.append(item)
    restored=[e for e in events if e['kind']=='resumed_boundary']
    initial_n=0 if not restored else read_json(Path(restored[0]['path'])/'committed.json')['completed_episodes']
    last_n=max([initial_n]+[c['completed_episodes'] for c in committed])
    starts={e['episode_id']:e for e in events if e['kind']=='episode_started'}
    finishes={e['result']['episode_id']:e['result'] for e in events if e['kind']=='episode_finished'}
    interrupts={e['episode_id']:e for e in events if e['kind']=='episode_interrupted'}
    rows=[]
    for index in starts:
        if index<last_n and index in finishes: rows.append(finishes[index])
        else:
            row=interrupts.get(index,dict(reason='process_interrupted_before_boundary_commit',steps=None))
            rows.append(dict(completed=False,eligible_for_task_statistics=False,success=None,
                penalized_completion_steps=None,steps=row['steps'],reason=row['reason'],episode_id=index))
    updates=[u for e in events if e['kind']=='training_step' for u in e['updates']]
    return dict(identity=run['identity'],statistics=summarize_episodes(rows),attempt_episodes=rows,
        restored_completed_episodes=initial_n,last_committed_episodes=last_n,
        torn_event_tail=torn_tail,
        uncommitted_checkpoint_directories=partial,attempt_logged_update_rounds=len(updates),
        last_logged_counters=None if not updates else updates[-1]['result']['counters'],
        interrupted_work_is_rolled_back=True,active_environment_recovery_supported=False,
        nominal_performance_verified=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('run')
    args=parser.parse_args(); print(json.dumps(audit_run(args.run),indent=2))


if __name__=='__main__': main()
