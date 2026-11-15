"""Explicit v2 configuration; no relabelling of v1 settings or checkpoints."""
import copy,json
from pathlib import Path
import numpy as np
from ..configuration import ROOT,new_output
from . import SCHEMA
from .parameters import bounds,SCHEMAS,legacy_bounds,NOMINAL

def mechanism_config(config):
    c=copy.deepcopy(config)
    c.update(mechanism=True,episodes=2,evaluation_episodes=2,max_steps=16,checkpoint_interval=1,maximum_updates=4)
    c['sac'].update(hidden=32,batch_size=4,replay_size=256,warmup_steps=4,update_frequency=4)
    c['estimation'].update(shared_candidates=5,per_delay=4,ensemble_hidden=32)
    c['disturbance']['change_windows']=[[7,9]]
    return c

def load(path,method=None,mechanism=False):
    c=json.loads(Path(path).read_text(encoding='utf-8'))
    if method:c['method']=method
    return validate(mechanism_config(c) if mechanism else c)

def validate(c):
    c=copy.deepcopy(c)
    if c['schema']!=SCHEMA or c['method'] not in c['methods']:raise ValueError('unknown v2 configuration/method')
    b=bounds(c['prior']);bounds(c['domains'][c['domain']] or b)
    if np.any(NOMINAL<b[:,0]) or np.any(NOMINAL>b[:,1]):raise ValueError('public prior must contain nominal process')
    if c['disturbance']['protocol'] not in ('nominal','episode','piecewise','drift'):raise ValueError('unknown v2 protocol')
    if c['noise_protocol'] not in ('N0','N1') or c['noise_std']!=dict(N0=0.,N1=.01)[c['noise_protocol']]:raise ValueError('fixed known noise protocol required')
    if c['protection']['mode']!='off':raise ValueError('v2 finite protection not accepted; all comparisons use off')
    if c['estimation']['history']!=8 or c['estimation']['shared_candidates']<2 or c['estimation']['per_delay']<2:raise ValueError('history8 and stratified candidate pools required')
    e=c['estimation']
    if not 1<=e['minimum_candidates']<=3*e['per_delay'] or not 1<=e['minimum_data']<=8 or e['velocity_tolerance']<=0:raise ValueError('invalid empirical estimator settings')
    r=c['risk']
    if not 0<r['alpha']<=1 or int(r['alpha']*r['samples'])<r['minimum_tail_samples']:raise ValueError('tail support too small')
    if (set(r['radii'])!={'theta','memory','state','history','action','set'} or r['minimum_facts']<2
        or r['maximum_facts']<r['minimum_facts'] or any(not np.isfinite(v) or v<=0 for v in r['radii'].values())
        or not np.isfinite(r['maximum_q_disagreement']) or r['maximum_q_disagreement']<=0):raise ValueError('invalid risk support gate')
    for name in ('episodes','evaluation_episodes','max_steps','checkpoint_interval'):
        if type(c[name]) is not int or c[name]<1:raise ValueError('positive integer '+name+' required')
    if c['sac']['gamma']!=.95 or not 0<c['sac']['tau']<=1:raise ValueError('preserve gamma .95 and target rate')
    if c['sac']['batch_size']<1 or c['sac']['replay_size']<c['sac']['batch_size'] or c['sac']['lr']<=0:raise ValueError('invalid learning budget')
    if set(c['splits'])!={'train','calib','dev','test'}:raise ValueError('four independent scene blocks required')
    values=list(c['splits'].values())
    if min(abs(a-b) for i,a in enumerate(values) for b in values[i+1:])<=max(c['episodes'],c['evaluation_episodes']):raise ValueError('scene blocks overlap')
    if type(c['seed']) is not int or not 0<=c['seed']<2**31:raise ValueError('31-bit training seed required')
    if c.get('mechanism') and (c['episodes']>8 or c['max_steps']>32 or c['maximum_updates']>4):raise ValueError('mechanism budget exceeded')
    drift=np.asarray(c['disturbance']['drift_rate'])
    if drift.shape!=(6,) or np.any(drift<0) or drift[5]!=0:raise ValueError('integer delay cannot drift')
    for lo,hi in c['disturbance']['change_windows']:
        if not 0<lo<=hi<c['max_steps']:raise ValueError('invalid random change window')
    c['schemas']=SCHEMAS.copy();c['method_options']=copy.deepcopy(c['methods'][c['method']])
    return c

def legacy_reference(c):
    # Only the frozen parent task adapter consumes this v1-shaped reference.
    return dict(reference_config=c['reference_config'],max_steps=c['max_steps'])

def physical_options(c,scene):
    main=np.asarray(c['prior']);sample=np.asarray(c['domains'][c['domain']] or main)
    rates=np.zeros(9);rates[[0,2,4,3,6,7]]=c['disturbance']['drift_rate']
    return dict(protocol=c['disturbance']['protocol'],bounds=legacy_bounds(main,c['noise_std']).tolist(),
                sampling_bounds=legacy_bounds(sample,c['noise_std']).tolist(),change_steps=scene['change_steps'],drift_rate=rates.tolist())

def frozen_contract(c,include_method=True):
    excluded={'method_options','schemas','method'} if not include_method else set()
    return {k:copy.deepcopy(v) for k,v in c.items() if k not in {'domain','evaluation_episodes','seed'}|excluded and k!='disturbance'}|{'disturbance':{k:v for k,v in c['disturbance'].items() if k!='protocol'}}

def evaluation_contract(train,evaluation):
    if frozen_contract(train)!=frozen_contract(evaluation) or train['seed']!=evaluation['seed']:raise ValueError('evaluation changed frozen v2 contract')
