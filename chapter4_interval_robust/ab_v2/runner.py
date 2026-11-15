"""Explicit manual runs; small mechanism mode, independent outputs and schemas."""
import copy,json,platform,time
from pathlib import Path
import numpy as np
from ..provenance import identity,digest,sha,write_new
from ..runner import serializable
from ..configuration import new_output
from ..runtime import public_nominal
from .configuration import evaluation_contract,frozen_contract
from .parameters import SCHEMAS
from .runtime import ReferenceRuntime,make_scene
from .learning import MultiAgentSAC
from .metrics import EpisodeMetrics,aggregate
from . import checkpoint

def require_validation(path,c):
    if path is None:raise ValueError('current native v2 validation receipt required')
    p=Path(path).resolve();v=json.loads(p.read_text(encoding='utf-8'))
    if v.get('schema')!='auv.ch4.ab_validation.v2' or not v.get('passed') or v['source']!=identity() or v['platform']!=platform.system() or not v['native_environment_checked']:raise ValueError('matching native v2 receipt required')
    if v['noise_protocol']!=c['noise_protocol']:raise ValueError('noise protocol needs its own validation')
    if v['configuration_contract_sha256']!=digest(frozen_contract(c,False)):raise ValueError('validation does not cover this v2 configuration')
    for name,h in v['artifacts'].items():
        file=(p.parent/name).resolve()
        if not file.is_relative_to(p.parent) or sha(file)!=h:raise ValueError('validation artifacts changed')
    return v

def run(c,output,*,mode,split,validation=None,weights=None,resume=None,run_label=None):
    source=identity()
    if mode not in ('train','evaluate'):raise ValueError('invalid run mode')
    if mode=='train' and (split!='train' or c['domain']!='in_domain'):raise ValueError('training must use in-domain training scenes')
    if mode=='evaluate' and split not in ('dev','test'):raise ValueError('evaluation uses dev/test only')
    if mode=='train' and not c.get('mechanism'):require_validation(validation,c)
    if mode=='evaluate' and resume:raise ValueError('evaluation cannot resume training')
    if split=='test' and not run_label:raise ValueError('frozen final-test label required')
    learner=MultiAgentSAC(c) if c['method_options']['learn'] else None
    summaries=[];start=0;payload=None
    if resume:
        if learner is None or mode!='train':raise ValueError('only learned training resumes')
        payload=checkpoint.load(resume,c);checkpoint.restore(payload,learner);summaries=payload['summaries'];start=payload['next_episode']
    if weights:
        if learner is None or mode!='evaluate':raise ValueError('weights for learned evaluation only')
        payload=checkpoint.load(weights);evaluation_contract(payload['config'],c)
        if payload['method']!=c['method']:raise ValueError('checkpoint method mismatch')
        learner.actor.load_state_dict(payload['learner']['actor']);learner.identifier.load_state_dict(payload['learner']['identifier'])
        learner.identifier.updates=payload['learner']['identifier_updates']
    if mode=='evaluate' and learner is not None and weights is None:raise ValueError('learned evaluation requires matching explicit v2 weights')
    count=c['episodes'] if mode=='train' else c['evaluation_episodes']
    if start>=count:raise ValueError('completed checkpoint has no remaining budget')
    folder=new_output(output);write_new(folder/'config.json',c)
    metadata=dict(schema='auv.ch4.ab_run.v2',schemas=SCHEMAS,source=source,method=c['method'],mode=mode,split=split,
        training_seed=c['seed'],noise_protocol=c['noise_protocol'],protocol=c['disturbance']['protocol'],domain=c['domain'],
        label=run_label or 'development',comparison_hash=digest(frozen_contract(c,False)),
        training_protocol=(payload['config'] if weights else c)['disturbance']['protocol'],
        weights_sha256=None if weights is None else sha(Path(weights)),mechanism=bool(c.get('mechanism')),performance_passed=False,
        checkpoint_training_episodes=payload['next_episode'] if weights else c['episodes'],query_measure='frozen_conditional_monte_carlo_bank')
    write_new(folder/'run.json',metadata)
    for i,r in enumerate(summaries):write_new(folder/f'episode_{i:05d}.json',r)
    inherited=sum(r['steps'] for r in summaries);initial_updates=0 if learner is None else learner.updates;updates=[]
    with (folder/'transitions.jsonl').open('x',encoding='utf-8') as trace,(folder/'updates.jsonl').open('x',encoding='utf-8') as log:
        for ep in range(start,count):
            scene=make_scene(c,split,ep);write_new(folder/f'scene_{ep:05d}.json',scene)
            runtime=ReferenceRuntime(c,scene,seed=scene['scenario_seed'],episode_id=ep,identifier=None if learner is None else learner.identifier)
            metrics=EpisodeMetrics(c);pending=None;began=time.perf_counter()
            try:
                while not runtime.terminal:
                    public=runtime.policy_input()
                    if learner is None:proposal=public_nominal(runtime.env)
                    elif mode=='train' and learner.environment_steps<c['sac']['warmup_steps']:proposal=learner.rng.uniform(-1,1,(4,3)).astype(np.float32)*public.active[:,None]
                    else:proposal=learner.act(public,deterministic=mode=='evaluate')
                    row=runtime.transition(proposal)
                    if learner is not None and mode=='train':
                        if pending is not None:pending['next_context']=row['context'].copy();learner.remember(pending)
                        pending=row;learner.environment_steps+=1
                        if runtime.terminal:
                            if not row['terminated']:raise ValueError('collector-only cut requires an observed successor context; no fabricated bootstrap')
                            pending['next_context']=pending['context'].copy();learner.remember(pending);pending=None
                        allowed=not c.get('mechanism') or learner.updates<c['maximum_updates']
                        if allowed and len(learner.replay.rows)>=c['sac']['batch_size'] and learner.environment_steps>=c['sac']['warmup_steps'] and learner.environment_steps%c['sac']['update_frequency']==0:
                            u=learner.update();updates.append(u);log.write(json.dumps(u,allow_nan=False)+'\n');log.flush()
                    metrics.add(row);trace.write(json.dumps(serializable(dict(episode=ep,**row)),allow_nan=False)+'\n');trace.flush()
                value=metrics.summary(runtime,time.perf_counter()-began)
                value.update(training_seed=c['seed'],scenario_id=scene['scenario_id'],scene_sha256=digest(scene),weights_sha256=metadata['weights_sha256'])
                summaries.append(value);write_new(folder/f'episode_{ep:05d}.json',value)
            finally:runtime.close()
            if learner is not None and mode=='train' and ((ep+1)%c['checkpoint_interval']==0 or ep==count-1):checkpoint.save(folder/f'checkpoint_{ep+1:05d}.pt',learner,c,ep+1,summaries)
    result=aggregate(summaries)
    result.update(source=source,total_physical_steps=sum(r['steps'] for r in summaries),physical_steps_executed_this_run=sum(r['steps'] for r in summaries)-inherited,
        inherited_steps=inherited,total_updates=0 if learner is None else learner.updates,updates_executed_this_run=(0 if learner is None else learner.updates)-initial_updates,
        risk_supported_rows=sum(u['risk_supported_rows'] for u in updates),risk_fallback_rows=sum(u['risk_fallback_rows'] for u in updates),
        mechanism=bool(c.get('mechanism')),original_sirsa_reproduced=False,performance_passed=False)
    write_new(folder/'summary.json',result);return dict(output=str(folder),**result)

def summarize(paths,output):
    if not paths or len(set(str(Path(p).resolve()) for p in paths))!=len(paths):raise ValueError('distinct runs required')
    groups={};contracts={}
    for path in paths:
        folder=Path(path);m=json.loads((folder/'run.json').read_text(encoding='utf-8'))
        if m['schema']!='auv.ch4.ab_run.v2':raise ValueError('v1/foreign runs cannot be mixed')
        key=':'.join(m[k] for k in ('method','split','noise_protocol','protocol','domain'))
        fingerprint=(m['source']['sha256'],m['comparison_hash'],m['label'],m['checkpoint_training_episodes'],m['training_protocol'])
        if key in contracts and contracts[key]!=fingerprint:raise ValueError('incompatible comparison runs')
        contracts[key]=fingerprint;groups.setdefault(key,[]).extend(json.loads(p.read_text(encoding='utf-8')) for p in sorted(folder.glob('episode_*.json')))
    result={}
    for key,rows in groups.items():
        ids=[(r['training_seed'],r['scenario_id']) for r in rows]
        if len(set(ids))!=len(ids):raise ValueError('duplicate training seed/scenario')
        result[key]=aggregate(rows)
        method,split,noise,protocol,domain=key.split(':')
        nominal=groups.get(':'.join((method,split,noise,'nominal',domain)),[])
        references={(r['training_seed'],r['scenario_id'],r['weights_sha256']):r['team_return'] for r in nominal}
        differences=[r['team_return']-references[(r['training_seed'],r['scenario_id'],r['weights_sha256'])] for r in rows
                     if protocol!='nominal' and (r['training_seed'],r['scenario_id'],r['weights_sha256']) in references]
        result[key].update(paired_nominal_scene_count=len(differences),
                           paired_nominal_return_delta=None if not differences else float(np.mean(differences)))
    folder=new_output(output);write_new(folder/'summary.json',result);return dict(output=str(folder),groups=result)

def identify(c,output,split):
    """Same native public trajectory for all four physics-matched estimators."""
    if split not in ('calib','dev','test'):raise ValueError('identification data uses explicit held-out split')
    from .estimation import ConditionalEstimator,diagnostics
    folder=new_output(output);write_new(folder/'config.json',c)
    records=[];calls={};padding={};count=c['evaluation_episodes']
    base=copy.deepcopy(c);base['method']='nominal';base['method_options']=c['methods']['nominal']
    with (folder/'identification.jsonl').open('x',encoding='utf-8') as log:
        for ep in range(count):
            scene=make_scene(c,split,ep);runtime=ReferenceRuntime(base,scene,seed=scene['scenario_seed'],episode_id=ep)
            estimators={name:ConditionalEstimator(c['prior'],runtime.specs,runtime.contract['dt'],c['estimation'],scene['scenario_seed'],shared=s,aligned=t)
                        for name,s,t in [('s0t0',False,False),('s1t0',True,False),('s0t1',False,True),('s1t1',True,True)]}
            try:
                while not runtime.terminal:
                    row=runtime.transition(public_nominal(runtime.env));frame=row['next_history'][-1]
                    for name,e in estimators.items():
                        score=diagnostics(e.estimate,row['privileged_theta'],c['estimation'])
                        e.update(frame[:,:3]*20,frame[:,3:6]*3,frame[:,6:9],frame[:,9:12]*3,frame[:,12]==1)
                        record=dict(method=name,episode=ep,step=row['physical_step'],preaction=score,postfit=diagnostics(e.estimate,row['privileged_theta'],c['estimation']),prediction_calls=e.prediction_calls,padding_prediction_calls=e.padding_prediction_calls)
                        log.write(json.dumps(serializable(record),allow_nan=False)+'\n');records.append(record)
                for name,e in estimators.items():
                    calls[name]=calls.get(name,0)+e.prediction_calls
                    padding[name]=padding.get(name,0)+e.padding_prediction_calls
            finally:runtime.close()
    result={name:dict(joint_box_coverage=float(np.mean([r['preaction']['joint_box'] for r in records if r['method']==name])),prediction_calls=calls[name],padding_prediction_calls=padding[name],evidence_prediction_calls=calls[name]-padding[name]) for name in calls}
    write_new(folder/'summary.json',dict(groups=result,thresholds_calibrated=False,performance_passed=False,source=identity()))
    return dict(output=str(folder),groups=result,performance_passed=False)
