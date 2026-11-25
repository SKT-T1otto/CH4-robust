"""Frozen independent validation with complete training-state side-effect checks."""
import argparse
import copy
import json
from pathlib import Path
from time import perf_counter

import torch

from chapter3_bser.learning.checkpoint import state_digest
from chapter3_bser.learning.evaluate import preserved_rng
from chapter3_bser.learning.reference import ZeroResidualPolicy
from .checkpoint import durable_json, read_boundary, restore_learner
from .config import load
from .execution import build_identity, load_execution
from .learner import SingleCriticLearner
from .policy import FrozenEvaluationPolicy
from .replay import NominalReplay
from .runtime import NominalRuntime
from .train_config import SceneCatalog
from .report import summarize_episodes
from .development_metrics import EpisodeMetrics, summarize


@preserved_rng()
def evaluate_snapshot(config, catalog, snapshot, *, indices, output, max_steps=400, diagnostics=False):
    if not isinstance(snapshot,FrozenEvaluationPolicy): raise TypeError('evaluation requires frozen weights')
    return _validation(config, catalog, snapshot, indices, output, max_steps, diagnostics)


@preserved_rng()
def evaluate_d2(config, catalog, *, indices, output, max_steps=400):
    return _validation(config, catalog, ZeroResidualPolicy(), indices, output, max_steps, True)


def _validation(config, catalog, snapshot, indices, output, max_steps, diagnostics):
    d2 = type(snapshot) is ZeroResidualPolicy
    if not indices or len(indices)!=len(set(indices)) or any(type(i) is not int or not 0<=i<32 for i in indices):
        raise ValueError('evaluation requires unique registered validation indices')
    if type(max_steps) is not int or not 1<=max_steps<=400: raise ValueError('invalid evaluation horizon')
    folder=Path(output); folder.mkdir(parents=True,exist_ok=False)
    results=[]
    for index in indices:
        # This buffer cannot accept or be confused with training provenance.
        replay=NominalReplay(400,9100000+index)
        scene=catalog.scene('validation',index)
        origin=catalog.origin(scene,index,allowed_groups=('validation',))
        started=perf_counter()
        with NominalRuntime(config,scene,snapshot,replay,episode_id=index,mode='regression' if d2 else 'evaluation',
                            catalog=catalog,explore=False) as runtime:
            telemetry=EpisodeMetrics(perf_counter()-started) if diagnostics else None
            for _ in range(max_steps):
                started=perf_counter()
                row=runtime.advance()
                if telemetry is not None:
                    telemetry.observe(runtime,row,perf_counter()-started)
                if runtime.terminal: break
            if runtime.terminal:
                result=runtime.finalize()
                result.update(completed=True,terminated=True,truncated=False,eligible_for_task_statistics=True)
            else: result=runtime.truncate('external_evaluation_limit')
            result.update(scenario_id=scene['scenario_id'],validation_index=index,
                          source=origin.as_dict())
            if telemetry is not None:
                result['diagnostics']=telemetry.report()
            results.append(result)
            durable_json(folder/f'scene_{index:03d}.json',result)
    if not d2:
        snapshot.verify_frozen()
    result=dict(weights_sha256=None if d2 else snapshot.initial_sha256,
        policy_kind='D2' if d2 else 'frozen_weights', group='validation',indices=list(indices),
        catalog_sha256=catalog.sha256,episodes=results,statistics=summarize_episodes(results),
        selection_eligible=not d2 and len(indices)==32 and all(r['completed'] for r in results))
    if diagnostics and all(r['completed'] for r in results):
        result['diagnostics']=summarize(results)
    durable_json(folder/'evaluation.json',result)
    return result


def evaluate_training(session, *, indices, output, max_steps=400, diagnostics=False):
    before=state_digest(session.state_dict())
    world_before=None if session.active is None else session.active.world.identity()
    try:
        with preserved_rng():
            result=evaluate_snapshot(session.config,session.catalog,session.policy.evaluation_snapshot(),
                                     indices=indices,output=output,max_steps=max_steps,diagnostics=diagnostics)
    finally:
        after=state_digest(session.state_dict())
        world_after=None if session.active is None else session.active.world.identity()
        if before!=after or world_before!=world_after:
            raise AssertionError('evaluation changed training networks/optimizers/Replay/RNG/order/live environment')
    receipt=dict(unchanged=True,before=before,after=after,world_before=world_before,world_after=world_after)
    durable_json(Path(output)/'isolation.json',receipt)
    result['isolation']=copy.deepcopy(receipt)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true'); parser.add_argument('--checkpoint')
    parser.add_argument('--output'); parser.add_argument('--steps',type=int,default=32)
    args=parser.parse_args()
    if not args.execute:
        print(json.dumps(dict(executed=False,protocol=load_execution()))); return
    if not args.checkpoint or not args.output or not 1<=args.steps<=32:
        parser.error('current registration permits bounded validation of a committed checkpoint only')
    torch.set_num_threads(1)
    with preserved_rng():
        state=read_boundary(args.checkpoint,build_identity())
        learner=SingleCriticLearner(load(),state['seeds']['initialization'])
        restore_learner(learner,state['learner'])
        result=evaluate_snapshot(load(),SceneCatalog(),FrozenEvaluationPolicy(learner),
                                 indices=[0],output=args.output,max_steps=args.steps)
    print(json.dumps(result['statistics']))


if __name__=='__main__': main()
