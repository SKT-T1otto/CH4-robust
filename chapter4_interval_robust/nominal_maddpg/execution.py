"""Stage-6 execution identity; earlier registrations retain their old meanings."""
import platform
import sys
import numpy as np
import torch
from .config import ROOT, DEFAULT_CONFIG, file_sha, read_json, load
from .train_config import TRAIN_PROTOCOL, load_training_protocol
from .provenance import verify_source

EXECUTION_CONFIG = ROOT/'configs/ch4/nominal_maddpg_execution.json'


def load_execution():
    p, base = read_json(EXECUTION_CONFIG), load()
    load_training_protocol()
    if (p['schema']!='auv.ch4.nominal_maddpg.execution.v1' or p['stage']!=6
            or p['formal_training_authorized'] is not False
            or p['algorithm']!=base['algorithm']['revision']
            or p['base_config_sha256']!=file_sha(DEFAULT_CONFIG)
            or p['scene_manifest_sha256']!=file_sha(ROOT/base['scene_manifest'])
            or p['schedule']!={key:base['learning'][key] for key in p['schedule']}
            or set(p['schedule'])!={'batch_size','replay_capacity','warmup_steps','update_frequency',
                'updates_per_train','actor_lr','critic_lr','optimizer','gamma','tau'}
            or p['evaluation_groups']!=['validation']
            or p['mechanism_limits']!={'episodes':8,'steps_per_prefix':32,'rounds_per_model':4}
            or p['nominal_admission_opened'] is not False or p['mismatch_sets_opened'] is not False):
        raise ValueError('stage-6 execution registration differs')
    return p


def build_identity():
    p = load_execution()
    source = verify_source()
    if torch.get_num_threads()!=1 or torch.get_default_dtype()!=torch.float32:
        raise ValueError('registered CPU resume requires one Torch thread and float32')
    return dict(algorithm=p['algorithm'], source_sha256=source['sha256'],
        config_sha256=file_sha(DEFAULT_CONFIG), execution_sha256=file_sha(EXECUTION_CONFIG),
        collection_sha256=file_sha(TRAIN_PROTOCOL), scenes_sha256=p['scene_manifest_sha256'],
        environment=dict(python=sys.version, torch=str(torch.__version__), numpy=np.__version__,
            platform=platform.platform(), machine=platform.machine(), torch_threads=1,
            device='cpu', default_dtype='float32'))
