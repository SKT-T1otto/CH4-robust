"""Manual research collector. Evaluation never updates learned weights."""
import copy
import json
import platform
import time
from pathlib import Path
import numpy as np
from .configuration import new_output,evaluation_contract,comparison_contract
from .runtime import ReferenceRuntime,public_nominal
from .learning import MultiAgentSAC
from .scenarios import make_scene
from .metrics import EpisodeMetrics,aggregate
from .provenance import identity,digest,write_new,sha
from . import checkpoint


def serializable(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {str(k):serializable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [serializable(x) for x in value]
    return value


def require_validation(path):
    if path is None:raise ValueError('explicit successful CH4 native validation receipt required')
    p=Path(path).resolve();report=json.loads(p.read_text(encoding='utf-8'))
    if (not report.get('passed') or report.get('source')!=identity() or report.get('platform')!=platform.system()
        or not report.get('native_environment_checked') or report.get('long_training_started')):
        raise ValueError('validation must cover current source on this native platform')
    for name,expected in report['artifacts'].items():
        artifact=(p.parent/name).resolve()
        if not artifact.is_relative_to(p.parent) or sha(artifact)!=expected:raise ValueError('validation artifacts changed')
    return report


def run(c,output,*,mode,split='dev',validation=None,resume=None,weights=None,run_label=None):
    source=identity()
    if mode not in ('train','evaluate'):raise ValueError('invalid run mode')
    if mode=='train' and split!='train':raise ValueError('training cannot consume development/final test scenes')
    if mode=='train' and c['domain']!='in_domain':raise ValueError('held-out physical domains are evaluation only')
    if mode=='evaluate' and split=='train':raise ValueError('evaluation must use isolated dev or test split')
    if mode=='train' and not c.get('mechanism'):require_validation(validation)
    if mode=='evaluate' and resume is not None:raise ValueError('evaluation cannot resume training')
    output=new_output(output);write_new(output/'config.json',c)
    weights_hash=None if weights is None else sha(Path(weights).resolve())
    manifest=dict(schema='auv.ch4.run.v1',source=source,config_sha256=digest(c),mode=mode,split=split,
                  method=c['method'],training_seed=c['seed'],platform=platform.system(),
                  disturbance_protocol=c['disturbance']['protocol'],prior_domain_sha256=digest(c['disturbance']['bounds']),
                  domain=c['domain'],sampling_domain_sha256=digest(c['disturbance']['sampling_bounds']),
                  action_schema='ch4.reference_shifted_absolute_gaussian.v1',protection=c['protection'],
                  mechanism=bool(c.get('mechanism')),performance_passed=False,
                  weights_sha256=weights_hash,comparison_label=run_label or 'development',
                  original_sirsa_reproduction=False)
    learner=MultiAgentSAC(c) if c['method_options']['learn'] else None
    summaries=[];start_episode=0
    if resume is not None:
        if learner is None:raise ValueError('nominal reference has no learned weights or training resume')
        payload=checkpoint.load(resume,c);checkpoint.restore(payload,learner)
        summaries=payload['summaries'];start_episode=payload['next_episode']
        if len(summaries)!=start_episode:raise ValueError('incomplete checkpoint episode accounting')
        # New files, never overwrite the previous run. Both the resumed run's
        # summary.json and the external summarizer now see the same full history.
        for i,summary in enumerate(summaries):write_new(output/f'episode_{i:05d}.json',summary)
    if weights is not None:
        if mode!='evaluate' or learner is None:raise ValueError('weights are only for learned-policy evaluation')
        payload=checkpoint.load(weights)
        if payload['method']!=c['method']:raise ValueError('evaluation method does not match checkpoint')
        # Evaluation may change disturbance protocol/split/count, but input/model
        # configuration and action/protection rights must match the training source.
        evaluation_contract(payload['config'],c)
        learner.actor.load_state_dict(payload['learner']['actor'])
        learner.identifier.load_state_dict(payload['learner']['identifier'])
        learner.identifier.updates=payload['learner']['identifier_updates']
    if mode=='evaluate' and learner is not None and weights is None:
        raise ValueError('learned-policy evaluation requires explicit matching weights')
    manifest['comparison_contract_sha256']=digest(comparison_contract(c,payload['config'] if weights is not None else c))
    manifest['checkpoint_training_episodes']=payload['next_episode'] if weights is not None else c['episodes']
    write_new(output/'run.json',manifest)
    count=c['episodes'] if mode=='train' else c['evaluation_episodes']
    if start_episode>=count:raise ValueError('checkpoint already completed requested episode budget')
    inherited_steps=sum(r['steps'] for r in summaries);initial_updates=0 if learner is None else learner.updates
    all_updates=[]
    with (output/'transitions.jsonl').open('x',encoding='utf-8') as trace, (output/'updates.jsonl').open('x',encoding='utf-8') as updates:
        for episode in range(start_episode,count):
            scene=make_scene(c,split,episode)
            write_new(output/f'scene_{episode:05d}.json',scene)
            # Scene innovations are paired across methods; training seed affects
            # policy/replay only. It does not silently change the evaluation world.
            runtime=ReferenceRuntime(c,scene,seed=int(scene['scenario_seed']),episode_id=episode,
                                     identifier=None if learner is None else learner.identifier)
            metric=EpisodeMetrics(c);pending=None;started=time.perf_counter()
            try:
                while not runtime.terminal:
                    public=runtime.policy_input()
                    if learner is None:
                        proposal=public_nominal(runtime.env)
                    elif mode=='train' and learner.environment_steps<c['sac']['warmup_steps']:
                        proposal=learner.rng.uniform(-1,1,(4,3)).astype(np.float32)*public.active[:,None]
                    else:proposal=learner.act(public,deterministic=mode=='evaluate')
                    row=runtime.transition(proposal)
                    if learner is not None and mode=='train':
                        # Store the next factual context when the next transition
                        # actually occurs. No arbitrary context relabelling.
                        if pending is not None:
                            pending['next_context']=row['context'].copy();learner.remember(pending)
                        pending=row
                        learner.environment_steps+=1
                        if runtime.terminal:
                            pending['next_context']=pending['context'].copy();learner.remember(pending);pending=None
                        allowed=(not c.get('mechanism') or learner.updates<c['maximum_updates'])
                        if (allowed and len(learner.replay.rows)>=c['sac']['batch_size']
                            and learner.environment_steps>=c['sac']['warmup_steps']
                            and learner.environment_steps%c['sac']['update_frequency']==0):
                            update=learner.update();all_updates.append(update)
                            updates.write(json.dumps(update,allow_nan=False)+'\n');updates.flush()
                    metric.add(row)
                    trace.write(json.dumps(serializable(dict(episode=episode,**row)),allow_nan=False)+'\n');trace.flush()
                summary=metric.summary(runtime,time.perf_counter()-started)
                summary.update(training_seed=c['seed'],scene_sha256=digest(scene),disturbance_seed=scene['disturbance_seed'])
                summary.update(weights_sha256=weights_hash,comparison_label=run_label or 'development')
                summaries.append(summary);write_new(output/f'episode_{episode:05d}.json',summary)
            finally:runtime.close()
            if mode=='train' and learner is not None and ((episode+1)%c['checkpoint_interval']==0 or episode==count-1):
                checkpoint.save(output/f'checkpoint_{episode+1:05d}.pt',learner,c,episode+1,summaries)
    result=aggregate(summaries,c['risk']['alpha'])
    result.update(source=source,method=c['method'],split=split,mode=mode,
                  total_updates=0 if learner is None else learner.updates,
                  total_physical_steps=sum(r['steps'] for r in summaries),
                  physical_steps_executed_this_run=sum(r['steps'] for r in summaries)-inherited_steps,
                  inherited_physical_steps=inherited_steps,updates_executed_this_run=(0 if learner is None else learner.updates)-initial_updates,
                  risk_supported_updates=sum(r['risk_supported_rows']>0 for r in all_updates),
                  original_sirsa_reproduced=False,native_linux=platform.system()=='Linux',
                  mechanism=bool(c.get('mechanism')),performance_passed=False)
    write_new(output/'summary.json',result)
    return dict(output=str(output),**result)


def summarize(paths,output,alpha=.25):
    if len(set(str(Path(p).resolve()) for p in paths))!=len(paths):raise ValueError('duplicate run input')
    episodes=[];runs=[]
    for path in paths:
        folder=Path(path).resolve()
        run=json.loads((folder/'run.json').read_text(encoding='utf-8'))
        rows=[json.loads(p.read_text(encoding='utf-8')) for p in sorted(folder.glob('episode_*.json'))]
        if not rows:raise ValueError('run has no completed episodes: '+str(folder))
        runs.append(run);episodes.extend(rows)
    # Do not pool different methods/splits into an apparently improved result.
    groups={}
    for run,path in zip(runs,paths):
        key=run['method']+':'+run['split']+':'+run['disturbance_protocol']+':'+run['domain']
        groups.setdefault(key,[]).extend(json.loads(p.read_text(encoding='utf-8')) for p in sorted(Path(path).glob('episode_*.json')))
    # Report across training seeds separately from within-seed scene variation.
    # A group must share implementation, prior domain, control permissions and
    # an explicitly frozen comparison label; repeated scenes cannot be hidden.
    contracts={}
    for run in runs:
        key=run['method']+':'+run['split']+':'+run['disturbance_protocol']+':'+run['domain']
        contract=(run['source']['sha256'],run['comparison_contract_sha256'],run['checkpoint_training_episodes'],run['prior_domain_sha256'],run['sampling_domain_sha256'],run['action_schema'],digest(run['protection']),run['comparison_label'])
        if key in contracts and contracts[key]!=contract:raise ValueError('incompatible comparison runs: '+key)
        contracts[key]=contract
    result={};nominal={}
    for key,rows in groups.items():
        method,split,protocol,domain=key.split(':')
        if protocol=='nominal':
            for row in rows:nominal[(method,split,domain,row['training_seed'],row['scenario_id'],row['weights_sha256'])]=row['team_return']
    for key,rows in groups.items():
        value=aggregate(rows,alpha);per_seed={}
        identifiers=[(r['training_seed'],r['scenario_id']) for r in rows]
        if len(set(identifiers))!=len(identifiers):raise ValueError('duplicate seed/scene in comparison group: '+key)
        for row in rows:per_seed.setdefault(row['training_seed'],[]).append(row['team_return'])
        means=[float(np.mean(v)) for v in per_seed.values()]
        value['per_training_seed_mean_return']={str(k):float(np.mean(v)) for k,v in per_seed.items()}
        value['std_of_training_seed_means']=None if len(means)<2 else float(np.std(means,ddof=1))
        deltas=[]
        method,split,protocol,domain=key.split(':')
        for row in rows:
            match=nominal.get((method,split,domain,row['training_seed'],row['scenario_id'],row['weights_sha256']))
            if match is not None and protocol!='nominal':deltas.append(row['team_return']-match)
        value['paired_nominal_return_delta']=float(np.mean(deltas)) if deltas else None
        value['paired_nominal_scene_count']=len(deltas)
        value['training_seed_uncertainty_established']=False
        result[key]=value
    folder=new_output(output);write_new(folder/'summary.json',result)
    return dict(output=str(folder),groups=result)
