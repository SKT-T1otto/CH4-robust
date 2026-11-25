"""Bounded real-environment complement to synthetic post-update boundary tests."""
import argparse
import copy
import json
from pathlib import Path

import torch

from chapter3_bser.learning.checkpoint import state_digest
from .checkpoint import durable_json, read_boundary
from .config import DEFAULT_CONFIG
from .execution import load_execution
from .evaluate import evaluate_training
from .provenance import append_source, verify_source, imported_local_files
from .report import audit_run
from .trainer import Trainer
from .train_config import SceneCatalog


def training_rows(folder):
    result=[]
    for p in sorted((folder/'events').glob('*.json')):
        e=json.loads(p.read_text(encoding='utf-8'))
        if e['kind']=='training_step':
            e.pop('sequence'); result.append(e)
    return result


def execute(output):
    torch.set_num_threads(1); source=verify_source(); catalog=SceneCatalog()
    folder=Path(output); folder.mkdir(parents=True,exist_ok=False)
    durable_json(folder/'execution_registration.json',dict(source_sha256=source['sha256'],
        protocol=load_execution(),real_prefixes=3,total_real_steps_limit=20,maximum_updates=0,
        restore_boundary='initial_episode_boundary',
        nonzero_optimizer_resume_evidence='separate_synthetic_scheduler_test_not_native_AUV_episodes'))
    try:
        a=Trainer(2729,folder/'continuous',catalog=catalog)
        boundary=a.commit_boundary()
        result_a=a.run_episode(max_steps=8,round_budget=4)
        state_a=state_digest(a.state_dict()); rows_a=training_rows(a.output)
        b=Trainer.resume(boundary,folder/'restored',run_seed=2729,catalog=catalog)
        evaluation=evaluate_training(b,indices=[0],output=folder/'evaluation',max_steps=4)
        result_b=b.run_episode(max_steps=8,round_budget=4)
        state_b=state_digest(b.state_dict()); rows_b=training_rows(b.output)
        if rows_a!=rows_b or state_a!=state_b:
            raise AssertionError('fresh real task differs after initial-boundary restoration/evaluation')
        if any(r['completed'] or r['success'] is not None for r in (result_a,result_b)):
            raise AssertionError('real prefix unexpectedly became a native complete mission')
        rejects=[]
        for key in ('source_sha256','config_sha256','execution_sha256','scenes_sha256','algorithm','environment'):
            bad=copy.deepcopy(a.identity); bad[key]='mismatched_test_identity'
            try: read_boundary(boundary,bad)
            except ValueError as exc:
                if key not in str(exc): raise
                rejects.append(key)
            else: raise AssertionError('identity mismatch was accepted: '+key)
        for session in (a,b): durable_json(session.output/'audit.json',audit_run(session.output))
        report=dict(passed=True,source_sha256=source['sha256'],real_prefixes=3,real_team_steps=20,
            training_prefix_steps=16,evaluation_prefix_steps=4,native_completed_episodes=0,
            learner_rounds=0,identity_mismatches_rejected=rejects,
            initial_boundary_resume_exact=True,training_state_sha256=state_a,
            real_action_reward_gate_and_done_rows_exact=rows_a==rows_b,
            evaluation_isolation=evaluation['isolation'],task_success_rate=None,
            nonzero_optimizer_resume_test_scope='synthetic_scheduler_fixture_with_real_learner_Replay_and_noise',
            active_environment_restore_claimed=False,formal_training_executed=False,
            imported_files=imported_local_files(source))
        durable_json(folder/'acceptance.json',report)
        return report
    except BaseException as exc:
        durable_json(folder/'failure.json',dict(error=repr(exc)))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true'); parser.add_argument('--register-source',action='store_true')
    parser.add_argument('--output',default='verification/nominal_maddpg_training_state_20261125/real_environment')
    args=parser.parse_args()
    if args.register_source:
        source=append_source(DEFAULT_CONFIG,stage=6,reason='Add audited team-step scheduling, boundary checkpoints, isolated evaluation and resume tests')
        print(json.dumps(dict(registered_source=source['sha256'])))
    if args.execute:
        result=execute(args.output)
        print(json.dumps({k:v for k,v in result.items() if k!='imported_files'}))
    else: print(json.dumps(dict(executed=False,protocol=load_execution())))


if __name__=='__main__': main()
