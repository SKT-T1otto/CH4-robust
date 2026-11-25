"""Bounded real-environment checks for live policies and isolated exploration."""
import argparse
import copy
import gzip
import json
from datetime import datetime, timezone
from unittest.mock import patch

import numpy as np
import torch

from chapter3_bser.learning.checkpoint import digest, state_digest
from chapter4_interval_robust.provenance import write_new
from .config import DEFAULT_CONFIG, ROOT, file_sha, load, local_path
from .learner import SingleCriticLearner
from .policy import TrainingPolicy
from .provenance import append_source, imported_local_files, verify_source
from .replay import NominalReplay, TrainingReplay
from .runtime import NominalRuntime, FrozenInitialPolicy, json_transition
from .train_config import TRAIN_PROTOCOL, SceneCatalog, SceneOrder, load_training_protocol, stream_seeds
from .validate_runtime import global_rng, replay_digest


def learning_state(policy, replay, order, runtime):
    return dict(networks=state_digest(policy.learner.networks.state_dict()),
        optimizers=state_digest([o.state_dict() for o in policy.learner.actor_optimizers+policy.learner.critic_optimizers]),
        counters=policy.learner.counters(),noise=state_digest(policy.noise.state_dict()),
        replay_data=replay_digest(replay),replay_rng=state_digest(dict(rng=replay.sampling_rng)),
        sources=digest(replay.slot_sources[:len(replay)]),sample_calls=replay.sample_calls,
        order=digest(order.state_dict()),world=runtime.world.identity(),global_rng=global_rng())


def evaluate_prefix(config, catalog, policy, protocol, folder):
    p=protocol['mechanism'];snapshot=policy.evaluation_snapshot()
    replay=NominalReplay(32,stream_seeds(p['run_seed'])['replay']+1)
    rows=[]
    with NominalRuntime(config,catalog.scene('validation',p['evaluation_scene_index']),snapshot,replay,
            episode_id=0,mode='evaluation',catalog=catalog) as runtime:
        for _ in range(p['evaluation_steps']):
            if runtime.terminal:raise AssertionError('mechanism evaluation ended before its registered prefix')
            row=runtime.advance();rows.append(dict(transition=json_transition(row),record=runtime.last_record))
            replay.sample(min(4,len(replay)))
        result=runtime.truncate()
        if row['dones'] != [False]*4 or result['success'] is not None or result['eligible_for_task_statistics']:
            raise AssertionError('external evaluation prefix was relabeled as a mission outcome')
    snapshot.verify_frozen()
    write_log(folder/'evaluation_prefix.jsonl.gz',rows)
    return dict(prefix=result,weights_sha256=snapshot.initial_sha256,rows=len(rows))


def write_log(path, rows):
    with gzip.open(path,'xt',encoding='utf-8',compresslevel=3) as f:
        for row in rows:f.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')


def collect(config,catalog,protocol,folder,*,insert_evaluation):
    p=protocol['mechanism'];seeds=stream_seeds(p['run_seed'])
    learner=SingleCriticLearner(config,seeds['initialization'])
    policy=TrainingPolicy(learner,seeds['exploration'])
    replay=TrainingReplay(64,seeds['replay'],catalog)
    order=SceneOrder(catalog,seeds['scene_order']);order.next_scene()
    rows=[];evaluation=None;update_evidence=None;expected_next_actor=None;old_actor=None
    with NominalRuntime(config,catalog.scene('train',p['train_scene_index']),policy,replay,
            episode_id=0,mode='train',explore=True) as runtime:
        initial=runtime.world.identity()
        for step in range(p['prefix_steps']):
            if runtime.terminal:raise AssertionError('training mechanism fixture ended before registered prefix')
            if step==p['update_after_steps']:
                obs=torch.as_tensor(np.asarray(runtime.rt.observations),dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():old_actor=learner.networks.actions(obs)[0].clone()
                before_hash=state_digest(learner.networks.actors.state_dict())
                sample=replay.sample(len(replay))
                # Intentionally outside NominalRuntime.advance() and its no_grad.
                update=learner.update(sample)
                with torch.no_grad():expected_next_actor=learner.networks.actions(obs)[0].clone()
                update_evidence=dict(update=update,sample_indices=sample[6].tolist(),sources=sample.sources,
                    before_actor_sha256=before_hash,after_actor_sha256=state_digest(learner.networks.actors.state_dict()),
                    max_raw_actor_change=float((expected_next_actor-old_actor).abs().max()))
                if insert_evaluation:
                    before=learning_state(policy,replay,order,runtime)
                    evaluation=evaluate_prefix(config,catalog,policy,protocol,folder)
                    after=learning_state(policy,replay,order,runtime)
                    if before!=after:raise AssertionError('inserted evaluation changed the active training state')
                    evaluation.update(training_state_before=before,training_state_after=after)
            transition=runtime.advance()
            rows.append(dict(transition=json_transition(transition),record=copy.deepcopy(runtime.last_record)))
            if step==p['update_after_steps']:
                decision=policy.last_decision
                actual=torch.tensor(decision['actor'],dtype=torch.float32)
                if not torch.equal(actual,expected_next_actor):
                    raise AssertionError('next real decision used stale Actor weights')
                noise=torch.tensor(decision['noise'],dtype=torch.float32)
                gates=torch.as_tensor(transition['action_enabled'])[:,None]
                old_applied=(old_actor+noise).clamp(-1,1)*gates
                actual_applied=torch.as_tensor(transition['actions'])
                delta=float((actual_applied-old_applied).abs().max())
                if delta<=1e-9:raise AssertionError('Actor update did not change an enabled real control action')
                update_evidence.update(next_actor_exact=True,max_applied_action_change=delta,
                    actual_next_actor_sha256=decision['actor_sha256'])
        final_world=runtime.world.identity()
        suppressed=runtime.suppressed_nonzero_boat_steps
        if rows[-1]['transition']['dones'] != [False]*4:raise AssertionError('prefix accidentally reached a native terminal')
        prefix=runtime.truncate()
        try:runtime.finalize()
        except RuntimeError:pass
        else:raise AssertionError('partial collection was accepted as a full mission')
    samples=[]
    for _ in range(3):
        s=replay.sample(8);samples.append(dict(indices=s[6].tolist(),sources=s.sources))
    result=dict(initial_state_sha256=initial,final_state_sha256=final_world,rows=len(rows),
        transition_sequence_sha256=digest(rows),replay_sha256=replay_digest(replay),
        replay_sources_sha256=digest(replay.slot_sources[:len(replay)]),next_samples=samples,
        next_scene_ids=[order.next_scene()['scenario_id'] for _ in range(3)],
        final_noise_sha256=state_digest(policy.noise.state_dict()),final_weights_sha256=state_digest(learner.networks.state_dict()),
        counters=learner.counters(),suppressed_nonzero_boat_steps=suppressed,prefix=prefix,
        update_evidence=update_evidence,evaluation=evaluation,noise_steps=policy.noise.steps,noise_draws=policy.noise.draws)
    name='with_evaluation' if insert_evaluation else 'without_evaluation'
    write_log(folder/(name+'.jsonl.gz'),rows);write_new(folder/(name+'.json'),result)
    return result


def zero_exploration(config,catalog,protocol,folder):
    seeds=stream_seeds(protocol['mechanism']['run_seed']);scene=catalog.scene('train',0)
    traces=[]
    for mode in ('regression','train'):
        if mode=='regression':
            policy=FrozenInitialPolicy(config,seeds['initialization']);replay=NominalReplay(16,seeds['replay'])
        else:
            policy=TrainingPolicy(SingleCriticLearner(config,seeds['initialization']),seeds['exploration'])
            replay=TrainingReplay(16,seeds['replay'],catalog)
        rows=[]
        with NominalRuntime(config,scene,policy,replay,episode_id=2,mode=mode,explore=False) as runtime:
            for _ in range(8):
                row=runtime.advance()
                rows.append(dict(transition=json_transition(row),legacy=copy.deepcopy(runtime.last_record['legacy'])))
            runtime.truncate()
        traces.append(rows)
    if traces[0]!=traces[1]:raise AssertionError('zero-exploration live policy changed frozen initialization behavior')
    result=dict(passed=True,steps=8,trace_sha256=digest(traces[0]),optimizer_rounds=0)
    write_new(folder/'zero_exploration.json',result)
    return result


def closed_gate_fixture(config,catalog,protocol,folder):
    """Explicit gate intervention for interface coverage; not a natural task outcome."""
    seeds=stream_seeds(protocol['mechanism']['run_seed'])
    policy=TrainingPolicy(SingleCriticLearner(config,seeds['initialization']),seeds['exploration'])
    replay=TrainingReplay(8,seeds['replay'],catalog)
    rows=[]
    with NominalRuntime(config,catalog.scene('train',0),policy,replay,episode_id=3,
            mode='train',explore=True) as runtime, patch.object(runtime.rt,'_gates',
                return_value=np.asarray([True,False,True,False])):
        for _ in range(4):
            row=runtime.advance()
            if np.any(row['actions'][[1,3]]) or not np.any(row['proposed_actions'][[1,3]]):
                raise AssertionError('noise escaped the closed execution gate')
            rows.append(dict(transition=json_transition(row),record=copy.deepcopy(runtime.last_record)))
        suppressed=runtime.suppressed_nonzero_boat_steps
        prefix=runtime.truncate()
    write_log(folder/'closed_gate_fixture.jsonl.gz',rows)
    return dict(passed=True,explicit_gate_intervention=True,steps=4,
        suppressed_nonzero_boat_steps=suppressed,prefix=prefix,optimizer_rounds=0)


def execute(output):
    protocol=load_training_protocol();p=protocol['mechanism']
    if (p['prefix_steps']>32 or p['evaluation_steps']>32 or not 0<p['update_after_steps']<p['prefix_steps']
            or p['prefix_steps']>64 or protocol['maximum_prefix_episodes']<6):
        raise ValueError('mechanism prefix budget differs')
    source=verify_source();folder=local_path(output)
    if folder.exists():raise FileExistsError(folder)
    folder.mkdir(parents=True)
    write_new(folder/'execution_registration.json',dict(source_sha256=source['sha256'],
        protocol_sha256=file_sha(TRAIN_PROTOCOL),utc=datetime.now(timezone.utc).isoformat(),
        prefix_episodes=6,maximum_steps_per_prefix=32,maximum_updates_per_model=1,checkpoints_written=0))
    torch.set_num_threads(1);before=global_rng();config=load();catalog=SceneCatalog()
    try:
        a=collect(config,catalog,protocol,folder,insert_evaluation=False)
        b=collect(config,catalog,protocol,folder,insert_evaluation=True)
        keys=('initial_state_sha256','final_state_sha256','transition_sequence_sha256','replay_sha256',
              'replay_sources_sha256','next_samples','next_scene_ids','final_noise_sha256','final_weights_sha256',
              'counters','prefix','update_evidence','noise_steps','noise_draws')
        for key in keys:
            if a[key]!=b[key]:raise AssertionError('repeated collection differs after evaluation: '+key)
        zero=zero_exploration(config,catalog,protocol,folder)
        gate=closed_gate_fixture(config,catalog,protocol,folder)
        if before!=global_rng():raise AssertionError('bounded collection consumed external RNG state')
        result=dict(schema='auv.ch4.nominal_maddpg.collection_acceptance.v1',passed=True,
            source_sha256=source['sha256'],protocol_sha256=file_sha(TRAIN_PROTOCOL),
            prefix_episodes=6,environment_steps=2*p['prefix_steps']+p['evaluation_steps']+20,
            optimizer_rounds_per_training_fixture=[a['counters']['rounds'],b['counters']['rounds']],
            repeat_and_evaluation_insertion_exact=True,zero_exploration_regression=zero,
            max_applied_action_change_after_update=a['update_evidence']['max_applied_action_change'],
            suppressed_nonzero_boat_steps=a['suppressed_nonzero_boat_steps']+b['suppressed_nonzero_boat_steps'],
            closed_gate_fixture=gate,
            global_rng_unchanged=True,task_training_started=False,training_checkpoints_written=0,
            nominal_performance_verified=False,nominal_admission_rollouts=0,mismatch_rollouts=0,
            local_imports=imported_local_files(source))
        write_new(folder/'collection_acceptance.json',result)
        print(json.dumps({k:v for k,v in result.items() if k!='local_imports'},indent=2))
    except BaseException as exc:
        write_new(folder/'failure.json',dict(passed=False,error=repr(exc),source_sha256=source['sha256']))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--register-source',action='store_true')
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--output',default='verification/nominal_maddpg_exploration_20261125/collection')
    args=parser.parse_args();load_training_protocol()
    if args.register_source:
        append_source(DEFAULT_CONFIG,stage=5,reason='Add live learner policies, isolated OU exploration, registered training Replay sources and bounded collection checks')
    if args.execute:execute(args.output)
    else:print(json.dumps(dict(stage=5,execute=False,maximum_prefix_episodes=8,maximum_steps_per_prefix=32)))


if __name__=='__main__':main()
