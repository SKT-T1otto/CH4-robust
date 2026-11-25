"""Linux entry point for the registered nominal development study."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys
import uuid

import numpy as np
import torch

from .checkpoint import durable_json
from .config import ROOT, file_sha, read_json
from .development_metrics import EpisodeMetrics, summarize
from .evaluate import evaluate_d2, evaluate_training
from .execution import build_identity, load_execution
from .report import choose_validation
from .trainer import Trainer

PLAN = ROOT / 'configs/ch4/nominal_maddpg_development.json'


def run_spec(profile, seed):
    plan = read_json(PLAN)
    search = plan['optional_search']
    if profile == 'pilot':
        spec = plan['pilot']
        if seed != spec['seed']:
            raise ValueError('unregistered pilot seed')
        settings = search['configurations_in_tie_order'][0]
    else:
        lookup = {item['id']: item for item in search['configurations_in_tie_order']}
        if profile not in lookup or seed not in search['seeds']:
            raise ValueError('unregistered search configuration or development seed')
        spec, settings = search, lookup[profile]
    return dict(profile=profile, run_seed=seed, episodes=spec['episodes'], checkpoints=spec['checkpoints'],
                actor_lr=settings['actor_lr'], decay_per_environment_step=settings['decay_per_environment_step'])


class DevelopmentTrainer(Trainer):
    diagnostics_factory = EpisodeMetrics

    @staticmethod
    def settings(run_seed, profile):
        spec = run_spec(profile, run_seed)
        schedule = load_execution()['schedule']
        schedule['actor_lr'] = spec['actor_lr']
        identity = dict(build_identity(), development_sha256=file_sha(PLAN), profile=profile, run_seed=run_seed)
        return dict(stage=7, schedule=schedule, **spec), identity

    def configure_policy(self):
        for opt in self.learner.actor_optimizers:
            for group in opt.param_groups:
                group['lr'] = self.execution['actor_lr']
        self.policy.noise.settings['decay_per_environment_step'] = self.execution['decay_per_environment_step']

    def run_episode(self, **limits):
        if self.completed_episodes >= self.execution['episodes']:
            raise RuntimeError('registered development episode budget exhausted')
        return super().run_episode(**limits)


@contextmanager
def study_lock(directory):
    import fcntl
    directory.mkdir(exist_ok=True, parents=True)
    with (directory/'study.lock').open('a+b') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('another writer is running this development study') from error
        yield


def attempt(directory):
    directory.mkdir(parents=True, exist_ok=True)
    return directory / ('attempt_' + uuid.uuid4().hex)


def publish_result(path, value):
    # All study writers hold study_lock. Expose only a completely written file.
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex)
    durable_json(temporary, value)
    temporary.replace(path)


def validation(session, directory, kind):
    """Reuse only a fully sealed result from the same policy, cohort and execution."""
    indices = read_json(PLAN)['validation_indices']
    if indices != list(range(32)):
        raise ValueError('expected the full fixed validation cohort')
    expected = dict(kind=kind, identity=session.identity, boundary=session.completed_episodes,
                    seed=session.run_seed, group='validation', indices=indices,
                    weights=None if kind=='D2' else session.policy.evaluation_snapshot().initial_sha256)
    def verify(result):
        rows = result['episodes']
        if (result['group'] != 'validation' or result['catalog_sha256'] != session.catalog.sha256
                or result['indices'] != indices or result['weights_sha256'] != expected['weights']
                or len(rows) != len(indices) or not all(r['completed'] for r in rows)):
            raise ValueError('validation receipt does not match the registered policy and cohort')
        for i,row in enumerate(rows):
            origin=session.catalog.origin(session.catalog.scene('validation',i),i,allowed_groups=('validation',))
            if row['validation_index']!=i or row['source']!=origin.as_dict():
                raise ValueError('validation scene mismatch')
    for seal in sorted(directory.glob('attempt_*/complete.json')):
        receipt = read_json(seal)
        if receipt['registration'] != expected or receipt['evaluation_sha256'] != file_sha(seal.parent/'evaluation.json'):
            raise ValueError('sealed validation identity or content changed')
        result = read_json(seal.parent/'evaluation.json')
        verify(result)
        return result
    output = attempt(directory)
    if kind=='D2':
        result=evaluate_d2(session.config,session.catalog,indices=indices,output=output)
    else:
        result=evaluate_training(session,indices=indices,output=output,diagnostics=True)
    verify(result)
    publish_result(output/'complete.json',dict(registration=expected,evaluation_sha256=file_sha(output/'evaluation.json')))
    return result


def latest_boundary(directory):
    markers = sorted((p for p in directory.glob('attempt_*/boundaries/*/committed.json')
                      if p.parent.name.isdigit()), key=lambda p:int(p.parent.name), reverse=True)
    for marker in markers:
        try:
            meta = read_json(marker)
        except (UnicodeDecodeError,json.JSONDecodeError):
            # A crash can tear the final marker. The previous committed boundary owns recovery.
            continue
        if meta['completed_episodes'] != int(marker.parent.name):
            raise ValueError('boundary number differs from the commit marker')
        return marker.parent
    return None


def reference_receipt(directory, label, identity, completed_episodes, kind):
    seals = list((directory/label).glob('attempt_*/complete.json'))
    if len(seals)!=1:
        raise ValueError('expected one completed reference for '+label)
    seal=read_json(seals[0]); path=seals[0].parent/'evaluation.json'
    registration=seal['registration']
    if (file_sha(path)!=seal['evaluation_sha256'] or registration['identity']!=identity
            or registration['boundary']!=completed_episodes or registration['kind']!=kind):
        raise ValueError('reference evaluation changed')
    return read_json(path)


def run(study, profile, seed):
    options = run_spec(profile,seed)
    directory=study / f'{profile}_{seed}'
    identity=DevelopmentTrainer.settings(seed,profile)[1]
    if profile!='pilot':
        pilot=read_json(study / f"pilot_{read_json(PLAN)['pilot']['seed']}" / 'summary.json')
        if pilot['identity']!=DevelopmentTrainer.settings(read_json(PLAN)['pilot']['seed'],'pilot')[1]:
            raise ValueError('optional search requires a completed pilot with matching source/configuration')
    if (directory/'summary.json').is_file():
        result=read_json(directory/'summary.json')
        if result['identity']!=identity: raise ValueError('completed run identity differs')
        return result
    boundary=latest_boundary(directory)
    work=attempt(directory)
    try:
        if boundary is None:
            session=DevelopmentTrainer(seed,work,profile=profile)
            session.commit_boundary()
        else:
            session=DevelopmentTrainer.resume(boundary,work,run_seed=seed,profile=profile)
        if profile=='pilot':
            if session.completed_episodes==0:
                for label in ('D2','frozen_initialization'):
                    validation(session,directory/label,label)
            references={label:reference_receipt(directory,label,identity,0,label)['diagnostics']
                        for label in ('D2','frozen_initialization')}
        else:
            references={}
        checkpoints=options['checkpoints']
        while True:
            episode=session.completed_episodes
            if episode in checkpoints:
                validation(session,directory/f'validation_{episode:04d}','trained')
            if episode>=options['episodes']:
                break
            result=session.run_episode()
            if not result['completed']:
                raise RuntimeError('native training episode was interrupted')
            print(f"{profile}/{seed}: {episode+1}/{options['episodes']} complete, "
                  f"team steps {session.team_steps}, updates {session.learner.rounds}", flush=True)
        candidates=[]
        trend=[]
        for epoch in checkpoints:
            evaluation=reference_receipt(directory,f'validation_{epoch:04d}',identity,epoch,'trained')
            candidates.append(dict(completed_episodes=epoch,evaluation=evaluation))
            trend.append(dict(episode=epoch,validation=evaluation['diagnostics']))
        selected=choose_validation(candidates)['completed_episodes']
        matches=list(directory.glob(f'attempt_*/boundaries/{selected:06d}/committed.json'))
        if len(matches)!=1: raise ValueError('selected checkpoint missing or ambiguous')
        windows=[]
        start=0
        for end in checkpoints:
            windows.append(dict(first_episode=start+1,last_episode=end,**summarize(session.episodes[start:end])))
            start=end
        all_stats=summarize(session.episodes)
        health=dict(updates_in_each_interval=all(w['update_rounds']>0 for w in windows),
                    successful_command_changes_by_vehicle=all_stats['nonzero_command_steps'],
                    executor_open_steps_after_handoff=all_stats['executor_post_handoff_open_steps'],
                    executor_actor_rows_after_handoff=all_stats['executor_post_handoff_effective_rows'],
                    prior_attempt_errors=[read_json(p) for p in sorted(directory.glob('attempt_*/failure.json'))])
        summary=dict(identity=identity,specification=options,training=all_stats,training_intervals=windows,
                     references=references,validation_trend=trend,health=health,
                     selected_episode=selected,selected_checkpoint=matches[0].parent.relative_to(study).as_posix(),
                     counters=session.learner.counters(),nominal_admission_evaluated=False)
        publish_result(directory/'summary.json',summary)
        return summary
    except BaseException as error:
        if work.is_dir():
            durable_json(work/'failure.json',dict(exception=type(error).__name__,message=str(error)))
        raise


def select_config(summaries, search):
    scores=[]
    for order,config in enumerate(search['configurations_in_tie_order']):
        for episode in search['checkpoints']:
            costs=[]
            for seed in search['seeds']:
                run=summaries[config['id'],seed]
                row=next(v for v in run['validation_trend'] if v['episode']==episode)
                if row['validation']['episodes']!=32: raise ValueError('incomplete validation cohort')
                costs.append(row['validation']['mean_penalized_completion_steps'])
            scores.append(dict(id=config['id'],order=order,episode=episode,seed_costs=costs,
                               mean_cost=float(np.mean(costs))))
    winner=min(scores,key=lambda item:(item['mean_cost'],item['order'],item['episode']))
    return dict(selected=winner,scores=scores,selection=search['selection'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['pilot','search'],nargs='?',default='pilot')
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    plan=read_json(PLAN)
    if not args.execute:
        print(json.dumps(plan,indent=2))
        return
    if sys.platform!='linux' or args.output is None:
        parser.error('full development execution requires Linux and --output STUDY_DIRECTORY')
    study=args.output.resolve()
    if study.is_relative_to(ROOT) and not study.is_relative_to(ROOT/'verification'):
        parser.error('use an external study directory or a subdirectory of verification/')
    torch.set_num_threads(1)
    with study_lock(study):
        if args.mode=='pilot':
            report=run(study,'pilot',plan['pilot']['seed'])
            print(json.dumps(dict(selected_checkpoint=report['selected_checkpoint'],health=report['health'])))
        else:
            search=plan['optional_search']
            runs={(c['id'],s):run(study,c['id'],s) for c in search['configurations_in_tie_order']
                  for s in search['seeds']}
            report=select_config(runs,search)
            path=study/'search_selection.json'
            if not path.exists(): publish_result(path,report)
            elif read_json(path)!=report: raise ValueError('stored search selection differs')
            print(json.dumps(report['selected']))


if __name__=='__main__':
    main()
