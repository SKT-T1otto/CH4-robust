"""New local checkpoint identity; complete episode-boundary resume only."""
import copy,json,random
from pathlib import Path
import numpy as np
import torch
from ..configuration import ROOT
from ..provenance import identity,digest,sha,write_new
from .parameters import SCHEMAS
SCHEMA='auv.ch4.ab_episode_checkpoint.v2'
def save(path,model,c,next_episode,summaries):
    p=Path(path).resolve()
    if p.exists() or not p.is_relative_to((ROOT/'work/ch4').resolve()):raise ValueError('fresh local checkpoint required')
    value=dict(schema=SCHEMA,schemas=SCHEMAS,source=identity(),config=copy.deepcopy(c),config_sha256=digest(c),
        method=c['method'],next_episode=next_episode,summaries=copy.deepcopy(summaries),learner=model.state_dict(),
        numpy_global_rng=np.random.get_state(),python_global_rng=random.getstate(),resume_scope='episode_boundary_only',
        environment_state='reset_before_next_episode',performance_passed=False)
    with p.open('xb') as f:torch.save(value,f)
    write_new(p.with_suffix('.json'),dict(schema=SCHEMA,schemas=SCHEMAS,sha256=sha(p),source=value['source'],config_sha256=value['config_sha256'],method=c['method']))
def load(path,c=None):
    p=Path(path).resolve()
    if not p.is_relative_to((ROOT/'work/ch4').resolve()):raise ValueError('only trusted local CH4 outputs accepted')
    meta=json.loads(p.with_suffix('.json').read_text(encoding='utf-8'))
    # Reject legacy metadata before deserializing any pickle.
    if meta.get('schema')!=SCHEMA or meta.get('schemas')!=SCHEMAS:raise ValueError('v1/foreign checkpoint is incompatible with v2')
    if meta['source']!=identity() or meta['sha256']!=sha(p):raise ValueError('checkpoint source/hash mismatch')
    value=torch.load(p,map_location='cpu',weights_only=False)
    if (value['schema']!=SCHEMA or value['schemas']!=SCHEMAS or value['source']!=identity()
        or value['config_sha256']!=digest(value['config']) or value['config_sha256']!=meta['config_sha256']
        or value['method']!=meta['method'] or value['method']!=value['config']['method']
        or value['resume_scope']!='episode_boundary_only' or value['environment_state']!='reset_before_next_episode'
        or value['learner']['boundary']!='episode_complete_reset_before_next'
        or type(value['next_episode']) is not int or value['next_episode']<0
        or len(value['summaries'])!=value['next_episode']):raise ValueError('invalid v2 checkpoint identity/boundary')
    if c is not None and digest(c)!=value['config_sha256']:raise ValueError('resume contract differs')
    return value
def restore(value,model):
    if value['resume_scope']!='episode_boundary_only':raise ValueError('unsupported resume boundary')
    model.load_state_dict(value['learner']);np.random.set_state(value['numpy_global_rng']);random.setstate(value['python_global_rng'])
