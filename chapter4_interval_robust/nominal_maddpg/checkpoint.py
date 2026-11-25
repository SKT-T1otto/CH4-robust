"""CPU episode-boundary snapshots with a durable commit marker and strict identity."""
import copy
import os
from pathlib import Path
import random

import numpy as np
import torch

from chapter3_bser.learning.checkpoint import state_digest
from .config import file_sha, read_json

SCHEMA = 'auv.ch4.nominal_maddpg.boundary.v1'


def global_rng_state():
    n = np.random.get_state()
    return dict(python=random.getstate(), numpy=[n[0],n[1].tolist(),n[2],n[3],n[4]],
                torch=torch.get_rng_state().clone())


def restore_global_rng(state):
    # Probe every stream before changing the process-global state.
    p = random.Random(); p.setstate(state['python'])
    n = state['numpy']; numpy_state=(n[0],np.asarray(n[1],dtype=np.uint32),n[2],n[3],n[4])
    probe = np.random.RandomState(0); probe.set_state(numpy_state)
    t = torch.Generator(); t.set_state(state['torch'])
    random.setstate(p.getstate()); np.random.set_state(probe.get_state()); torch.set_rng_state(t.get_state())


def durable_json(path, value):
    import json
    path=Path(path)
    with path.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.flush(); os.fsync(f.fileno())


def learner_state(learner):
    return dict(networks=copy.deepcopy(learner.networks.state_dict()),
        actors=[copy.deepcopy(o.state_dict()) for o in learner.actor_optimizers],
        critics=[copy.deepcopy(o.state_dict()) for o in learner.critic_optimizers], counters=learner.counters())


def restore_learner(learner, state):
    expected=learner.networks.state_dict(); loaded=state['networks']
    if expected.keys()!=loaded.keys(): raise ValueError('checkpoint network keys differ')
    for key,value in loaded.items():
        if (not torch.is_tensor(value) or value.shape!=expected[key].shape or value.dtype!=expected[key].dtype
                or value.device.type!='cpu' or not bool(torch.isfinite(value).all())):
            raise ValueError('invalid checkpoint network tensor: '+key)
    counts=state['counters']; rounds=counts.get('rounds')
    if type(rounds) is not int or rounds<0 or counts.get('target_updates')!=rounds:
        raise ValueError('invalid team update counters')
    for key in ('actor_steps','actor_skips','critic_steps'):
        if not isinstance(counts.get(key),list) or len(counts[key])!=4 or any(type(n) is not int or n<0 for n in counts[key]):
            raise ValueError('invalid per-agent update counters')
    if counts['critic_steps']!=[rounds]*4 or any(a+s!=rounds for a,s in zip(counts['actor_steps'],counts['actor_skips'])):
        raise ValueError('round/critic/actor/skip counters disagree')
    for kind,optimizers,counters in (('actors',learner.actor_optimizers,counts['actor_steps']),
                                    ('critics',learner.critic_optimizers,counts['critic_steps'])):
        if len(state[kind])!=4: raise ValueError('missing optimizer state')
        for opt,stored,steps in zip(optimizers,state[kind],counters):
            if stored['param_groups']!=opt.state_dict()['param_groups']:
                raise ValueError('optimizer configuration or parameter ownership differs')
            ids=stored['param_groups'][0]['params']; params=opt.param_groups[0]['params']
            if set(stored['state'])!=(set(ids) if steps else set()):
                raise ValueError('optimizer state does not match actual steps')
            for index,param in zip(ids,params):
                if not steps: continue
                item=stored['state'][index]
                if set(item)!={'step','exp_avg','exp_avg_sq'} or float(item['step'])!=steps:
                    raise ValueError('Adam clock differs from per-agent steps')
                for key in ('exp_avg','exp_avg_sq'):
                    v=item[key]
                    if not torch.is_tensor(v) or v.dtype!=param.dtype or v.shape!=param.shape or not bool(torch.isfinite(v).all()):
                        raise ValueError('invalid Adam moments')
    learner.networks.load_state_dict(loaded,strict=True)
    for kind,optimizers in (('actors',learner.actor_optimizers),('critics',learner.critic_optimizers)):
        for opt,stored in zip(optimizers,state[kind]): opt.load_state_dict(copy.deepcopy(stored))
    for key,value in counts.items(): setattr(learner,key,copy.deepcopy(value))
    learner.networks.zero_grad(set_to_none=True)


def save_boundary(session, path):
    if session.phase!='boundary' or session.active is not None or session.policy._active_episode:
        raise RuntimeError('checkpoint requires a committed-capable episode boundary; active/truncated episodes cannot be saved')
    state=session.state_dict()
    session.validate_counters(state)
    payload=dict(schema=SCHEMA, identity=copy.deepcopy(session.identity), state=state)
    payload['state_sha256']=state_digest(state)
    path=Path(path); path.mkdir(parents=True,exist_ok=False)
    # The state file alone is not a checkpoint. A final marker is the commit point.
    with (path/'state.pt').open('xb') as f:
        torch.save(payload,f); f.flush(); os.fsync(f.fileno())
    marker=dict(schema=SCHEMA, file='state.pt', file_sha256=file_sha(path/'state.pt'),
        state_sha256=payload['state_sha256'], identity=payload['identity'],
        completed_episodes=session.completed_episodes, team_steps=session.team_steps)
    durable_json(path/'committed.json',marker)
    return marker


def read_boundary(path, expected_identity):
    path=Path(path); marker_path=path/'committed.json'
    if not marker_path.is_file(): raise ValueError('incomplete checkpoint: no committed boundary marker')
    marker=read_json(marker_path)
    if marker.get('schema')!=SCHEMA or marker.get('file')!='state.pt': raise ValueError('checkpoint schema differs')
    for key in set(expected_identity)|set(marker.get('identity',{})):
        if expected_identity.get(key)!=marker['identity'].get(key):
            raise ValueError('checkpoint identity mismatch: '+key)
    if file_sha(path/'state.pt')!=marker['file_sha256']: raise ValueError('checkpoint bytes differ from commit marker')
    payload=torch.load(path/'state.pt',map_location='cpu',weights_only=True)
    if (payload.get('schema')!=SCHEMA or payload.get('identity')!=expected_identity
            or payload.get('state_sha256')!=marker['state_sha256']
            or state_digest(payload['state'])!=marker['state_sha256']
            or payload['state']['completed_episodes']!=marker['completed_episodes']
            or payload['state']['team_steps']!=marker['team_steps']):
        raise ValueError('checkpoint payload/marker integrity differs')
    return payload['state']
