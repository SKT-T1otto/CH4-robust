"""Fail-closed configuration and local output contracts."""
import copy
import json
from pathlib import Path
import numpy as np
from .parameters import validate_bounds

ROOT=Path(__file__).resolve().parents[1]


def load(path=None,method=None,mechanism=False):
    path=ROOT/'configs/ch4/experiment.json' if path is None else Path(path).resolve()
    c=json.loads(path.read_text(encoding='utf-8'))
    if method is not None:c['method']=method
    if mechanism:
        c.update(episodes=2,evaluation_episodes=2,max_steps=16,checkpoint_interval=1)
        c['sac'].update(batch_size=4,warmup_steps=4,update_frequency=4,hidden=32,replay_size=256)
        c['estimation'].update(particles=64,hidden=32,minimum_particles=4)
        c['disturbance']['change_steps']=[8]
        c['mechanism']=True;c['maximum_updates']=4
    return validate(c)


def validate(config):
    c=copy.deepcopy(config)
    if c['schema']!='auv.ch4.interval_robust.v1' or c['method'] not in c['methods']:
        raise ValueError('unknown CH4 schema or method')
    if type(c['seed']) is not int or not 0<=c['seed']<2**31:raise ValueError('training seed must be a nonnegative 31-bit integer')
    if c['disturbance']['protocol'] not in ('nominal','episode','piecewise','drift'):
        raise ValueError('unknown disturbance protocol')
    validate_bounds(c['disturbance']['bounds'])
    domain=c.get('domain','in_domain')
    if domain not in c['domains'] or ':' in domain:raise ValueError('unknown physical evaluation domain')
    physical=c['domains'][domain] or c['disturbance']['bounds']
    c['disturbance']['sampling_bounds']=validate_bounds(physical,require_nominal=False).tolist()
    drift=np.asarray(c['disturbance']['drift_rate'],float)
    if drift.shape!=(9,) or not np.isfinite(drift).all() or np.any(drift<0):raise ValueError('invalid drift rates')
    for key in ('episodes','evaluation_episodes','max_steps','checkpoint_interval'):
        if type(c[key]) is not int or c[key]<1:raise ValueError('positive integer '+key+' required')
    seeds=c['splits']
    if set(seeds)!= {'train','dev','test'} or len(set(seeds.values()))!=3:
        raise ValueError('three isolated scene splits required')
    # Disjoint seed blocks, including future episode-index offsets.
    if any(abs(a-b)<=max(c['episodes'],c['evaluation_episodes']) for a in seeds.values() for b in seeds.values() if a!=b):
        raise ValueError('scene seed blocks overlap')
    if c['sac']['gamma']!=.95 or not 0<c['sac']['tau']<=1:
        raise ValueError('gamma=.95 and a valid target update rate required')
    if c['sac']['batch_size']<1 or c['sac']['replay_size']<c['sac']['batch_size']:
        raise ValueError('replay too small')
    risk=c['risk']
    if not 0<risk['alpha']<=1 or risk['samples']<1 or int(risk['alpha']*risk['samples'])<risk['minimum_tail_samples']:
        raise ValueError('insufficient declared tail samples')
    if risk['minimum_contexts']<2 or risk['support_radius']<=0:raise ValueError('invalid support gate')
    if c['protection']['mode'] not in ('off','finite_probe'):raise ValueError('invalid protection mode')
    e=c['estimation']
    if not 1<=e['minimum_particles']<=e['particles'] or e['history']<1 or e['ensemble']<2 or e['velocity_tolerance']<=0:
        raise ValueError('invalid estimator settings')
    if c.get('mechanism') and (c['episodes']>8 or c['max_steps']>32 or c['maximum_updates']>4):
        raise ValueError('mechanism budget exceeded')
    c['method_options']=copy.deepcopy(c['methods'][c['method']])
    return c


EVALUATION_CONTRACT_KEYS=('schema','reference_config','max_steps','splits','domains','seed','episodes',
                          'method','method_options','estimation','sac','risk','protection','recovery')


def evaluation_contract(training,evaluation):
    for key in EVALUATION_CONTRACT_KEYS:
        if training[key]!=evaluation[key]:raise ValueError('evaluation changed frozen contract: '+key)
    if training['disturbance']['bounds']!=evaluation['disturbance']['bounds']:
        raise ValueError('evaluation changed public prior')
    for key in ('change_steps','drift_rate'):
        if training['disturbance'][key]!=evaluation['disturbance'][key]:
            raise ValueError('evaluation changed frozen disturbance schedule: '+key)


def comparison_contract(config,training=None):
    training=config if training is None else training
    keys=tuple(k for k in EVALUATION_CONTRACT_KEYS if k!='seed')
    return dict(settings={k:config[k] for k in keys},public_prior=config['disturbance']['bounds'],
                disturbance_process={k:config['disturbance'][k] for k in ('change_steps','drift_rate')},
                training_protocol=training['disturbance']['protocol'])


def new_output(path):
    p=Path(path).resolve()
    if not p.is_relative_to((ROOT/'work/ch4').resolve()):
        raise ValueError('new CH4 outputs must be inside this project work/ch4')
    if p.exists():raise FileExistsError(p)
    p.mkdir(parents=True)
    return p


def reference_config(c):
    from chapter3_bser.learning.config import validate as validate_parent
    path=(ROOT/c['reference_config']).resolve()
    if not path.is_relative_to(ROOT) or path.name!='b2.json':raise ValueError('frozen OLD/D2 reference required')
    ref=json.loads(path.read_text(encoding='utf-8'))
    ref['max_steps']=c['max_steps']
    # Absence is the original default-off OLD/D2 identity; no incomplete map record.
    ref.pop('map_experiment',None)
    return validate_parent(ref)
