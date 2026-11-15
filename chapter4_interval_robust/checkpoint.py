"""Local trusted checkpoints. Training resume is supported only at episode boundaries."""
import copy
import json
import random
from pathlib import Path
import numpy as np
import torch
from .configuration import ROOT
from .provenance import identity,digest,sha,write_new


def save(path,learner,config,next_episode,episode_summaries):
    path=Path(path).resolve()
    if path.exists() or not path.is_relative_to((ROOT/'work/ch4').resolve()):raise ValueError('fresh local CH4 checkpoint required')
    payload=dict(schema='auv.ch4.episode_boundary_checkpoint.v1',source=identity(),config=copy.deepcopy(config),
                 config_sha256=digest(config),method=config['method'],next_episode=int(next_episode),
                 summaries=copy.deepcopy(episode_summaries),learner=learner.state_dict(),
                 numpy_global_rng=np.random.get_state(),python_global_rng=random.getstate(),
                 episode_state=dict(active_environment=False,estimator_history=[],actuator_queue=[],
                                    lag_state='reset_on_next_episode',seed_rule='split_scene_and_episode_explicit'),
                 resume_scope='episode_boundary_only',performance_passed=False)
    with path.open('xb') as f:torch.save(payload,f)
    write_new(path.with_suffix('.json'),dict(schema=payload['schema'],sha256=sha(path),source=payload['source'],
              config_sha256=payload['config_sha256'],method=config['method'],next_episode=next_episode,
              resume_scope=payload['resume_scope'],trusted_local_pickle_only=True))


def load(path,config=None):
    path=Path(path).resolve()
    if not path.is_relative_to((ROOT/'work/ch4').resolve()) or not path.is_file():
        raise ValueError('only explicitly selected trusted local CH4 checkpoints are accepted')
    metadata=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    current=identity()
    if metadata['source']!=current or metadata['sha256']!=sha(path):raise ValueError('checkpoint source/hash mismatch')
    # Local outputs only; never load downloaded or historical arbitrary pickle files.
    payload=torch.load(path,map_location='cpu',weights_only=False)
    if (payload['schema']!='auv.ch4.episode_boundary_checkpoint.v1' or payload['source']!=current
        or payload['config_sha256']!=digest(payload['config'])
        or payload['method']!=metadata['method'] or payload['config_sha256']!=metadata['config_sha256']
        or payload['episode_state']['active_environment'] or payload['resume_scope']!='episode_boundary_only'):
        raise ValueError('invalid checkpoint schema, method or boundary')
    if config is not None and digest(config)!=payload['config_sha256']:raise ValueError('resume configuration differs')
    return payload


def restore(payload,learner):
    learner.load_state_dict(payload['learner'])
    np.random.set_state(payload['numpy_global_rng']);random.setstate(payload['python_global_rng'])
