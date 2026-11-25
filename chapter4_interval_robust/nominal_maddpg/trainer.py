"""Auditable scheduling of real transitions, with episode-boundary recovery only."""
import argparse
import copy
import json
from pathlib import Path
from time import perf_counter

import torch

from .config import load
from .execution import build_identity, load_execution
from .checkpoint import (durable_json, global_rng_state, restore_global_rng, learner_state,
                         restore_learner, save_boundary, read_boundary)
from .learner import SingleCriticLearner
from .policy import TrainingPolicy
from .replay import TrainingReplay
from .runtime import NominalRuntime
from .train_config import SceneCatalog, SceneOrder, stream_seeds


def rounds_due(team_steps, learning):
    if type(team_steps) is not int or team_steps<0: raise ValueError('invalid team clock')
    return (learning['updates_per_train'] if team_steps>=learning['warmup_steps']
            and team_steps%learning['update_frequency']==0 else 0)


class Trainer:
    diagnostics_factory = None

    @staticmethod
    def settings(run_seed, profile):
        if profile is not None:
            raise ValueError('stage 6 uses its registered schedule')
        return load_execution(), build_identity()

    def configure_policy(self):
        """Stage 6 already has the registered optimizer and exploration settings."""

    def __init__(self, run_seed, output, *, catalog=None, profile=None):
        self.config = load()
        self.profile = profile
        self.execution, self.identity = self.settings(run_seed, profile)
        self.run_seed=run_seed; self.seeds=stream_seeds(run_seed)
        self.catalog=catalog if catalog is not None else SceneCatalog()
        self.learner=SingleCriticLearner(self.config,self.seeds['initialization'])
        self.policy=TrainingPolicy(self.learner,self.seeds['exploration'])
        self.configure_policy()
        self.replay=TrainingReplay(self.config['learning']['replay_capacity'],self.seeds['replay'],self.catalog)
        self.order=SceneOrder(self.catalog,self.seeds['scene_order'])
        self.team_steps=0; self.completed_episodes=0; self.episodes=[]
        self.active=None; self.phase='boundary'; self.last_boundary=None
        self.output=Path(output); self.output.mkdir(parents=True,exist_ok=False)
        (self.output/'events').mkdir(); self.event_index=0
        durable_json(self.output/'run.json',dict(identity=self.identity,run_seed=run_seed,streams=self.seeds,
            execution=self.execution,task_performance_verified=False))

    def event(self, kind, **values):
        durable_json(self.output/'events'/f'{self.event_index:08d}.json',
                     dict(sequence=self.event_index,kind=kind,**values))
        self.event_index+=1

    def state_dict(self):
        return dict(run_seed=self.run_seed,seeds=copy.deepcopy(self.seeds),phase=self.phase,
            learner=learner_state(self.learner),replay=self.replay.state_dict(),order=self.order.state_dict(),
            policy=dict(policy_id=self.policy.policy_id,noise=self.policy.noise.state_dict(),
                last_decision=copy.deepcopy(self.policy.last_decision),last_input_sha256=self.policy.last_input_sha256),
            global_rng=global_rng_state(),team_steps=self.team_steps,completed_episodes=self.completed_episodes,
            episodes=copy.deepcopy(self.episodes),
            environment=dict(active_environment_saved=False,stream_rule='step_stream(scene_seed, t, 49031)',
                             next_episode_id=self.completed_episodes))

    def validate_counters(self, state):
        n,steps=state['completed_episodes'],state['team_steps']; p=self.config['learning']
        if (state['phase']!='boundary' or type(n) is not int or n<0 or type(steps) is not int or steps<0
                or len(state['episodes'])!=n or sum(row['steps'] for row in state['episodes'])!=steps
                or any(row.get('completed') is not True or row.get('eligible_for_task_statistics') is not True
                       or not 1<=row['steps']<=self.config['task']['max_steps'] for row in state['episodes'])
                or state['replay']['insertions']!=steps or state['policy']['noise']['steps']!=steps
                or state['policy']['noise']['draws']!=steps
                or state['order']['epoch']*len(state['order']['order'])+state['order']['cursor']!=n
                or state['environment']!=dict(active_environment_saved=False,
                    stream_rule='step_stream(scene_seed, t, 49031)',next_episode_id=n)):
            raise ValueError('boundary episode/Replay/noise/scene counters disagree')
        expected=0 if steps<p['warmup_steps'] else ((steps-p['warmup_steps'])//p['update_frequency']+1)*p['updates_per_train']
        if (state['learner']['counters']['rounds']!=expected or state['replay']['sample_calls']!=expected
                or state['replay']['sampled_rows']!=expected*p['batch_size']):
            raise ValueError('scheduled updates and Replay sampling counters disagree')
        for record in state['replay']['slot_sources']:
            episode=record['episode_id']
            if (not 0<=episode<n or state['episodes'][episode]['episode_id']!=episode
                    or state['episodes'][episode]['scenario_id']!=record['scenario_id']
                    or record['transition_step']>=state['episodes'][episode]['steps']):
                raise ValueError('Replay source does not belong to a committed episode')

    def commit_boundary(self):
        if self.settings(self.run_seed, self.profile)[1]!=self.identity:
            raise ValueError('source/config/execution identity changed during this attempt')
        path=self.output/'boundaries'/f'{self.completed_episodes:06d}'
        marker=save_boundary(self,path)
        self.last_boundary=path
        self.event('boundary_committed',path=str(path.resolve()),marker=marker)
        return path

    @classmethod
    def resume(cls, boundary, output, *, run_seed, catalog=None, profile=None):
        # Reject identities before creating output, allocating Replay or loading weights.
        state=read_boundary(boundary,cls.settings(run_seed, profile)[1])
        if state['run_seed']!=run_seed or state['seeds']!=stream_seeds(run_seed):
            raise ValueError('checkpoint run seed or private stream identity differs')
        before=global_rng_state()
        try:
            session=cls(run_seed,output,catalog=catalog,profile=profile)
            session.validate_counters(state)
            if state['policy']['policy_id']!=session.policy.policy_id:
                raise ValueError('checkpoint initial policy identity differs')
            restore_learner(session.learner,state['learner'])
            session.replay.load_state_dict(state['replay'])
            session.order.load_state_dict(state['order'])
            session.policy.noise.load_state_dict(state['policy']['noise'])
            session.policy.policy_id=state['policy']['policy_id']
            session.policy.last_decision=copy.deepcopy(state['policy']['last_decision'])
            session.policy.last_input_sha256=state['policy']['last_input_sha256']
            session.team_steps=state['team_steps']; session.completed_episodes=state['completed_episodes']
            session.episodes=copy.deepcopy(state['episodes']); session.last_boundary=Path(boundary).resolve()
            session.event('resumed_boundary',path=str(session.last_boundary),team_steps=session.team_steps,
                          abandoned_uncommitted_episode_rolled_back=True,active_environment_restored=False)
            restore_global_rng(state['global_rng'])
            return session
        except BaseException:
            restore_global_rng(before)
            raise

    def _interrupt(self, reason, error=None, *, episode_id=None, observed_steps=None):
        steps=(0 if self.active is None else self.active.step) if observed_steps is None else observed_steps
        episode_id=self.completed_episodes if episode_id is None else episode_id
        if self.active is not None: self.active.close()
        self.active=None; self.phase='interrupted'
        result=dict(completed=False,terminated=False,truncated=True,success=None,
            penalized_completion_steps=None,eligible_for_task_statistics=False,steps=steps,
            reason=reason,error=error,rollback_boundary=str(self.last_boundary),
            active_environment_restored=False)
        self.event('episode_interrupted',episode_id=episode_id,**result)
        return result

    def run_episode(self, *, max_steps=400, round_budget=None):
        if self.phase!='boundary' or self.active is not None:
            raise RuntimeError('interrupted sessions must resume from their last committed boundary')
        if type(max_steps) is not int or not 1<=max_steps<=400:
            raise ValueError('invalid episode collection limit')
        if round_budget is not None and (type(round_budget) is not int or round_budget<0):
            raise ValueError('invalid cumulative round budget')
        if self.last_boundary is None: self.commit_boundary()
        scene=self.order.next_scene(); self.phase='active'
        episode_id=self.completed_episodes; runtime=None
        self.event('episode_started',episode_id=self.completed_episodes,scene=scene['scenario_id'],
                   previous_boundary=str(self.last_boundary))
        try:
            started = perf_counter()
            self.active=NominalRuntime(self.config,scene,self.policy,self.replay,
                episode_id=self.completed_episodes,mode='train',explore=True)
            runtime=self.active
            telemetry = self.diagnostics_factory(perf_counter() - started) if self.diagnostics_factory else None
            for _ in range(max_steps):
                due=rounds_due(self.team_steps+1,self.config['learning'])
                if round_budget is not None and self.learner.rounds+due>round_budget:
                    return self._interrupt('mechanism_update_budget_before_next_decision')
                started = perf_counter()
                row=runtime.advance(); self.team_steps+=1
                if telemetry is not None:
                    post_handoff = telemetry.observe(runtime, row, perf_counter() - started)
                    self.replay.slot_sources[runtime.last_record['replay_index']]['executor_post_handoff'] = post_handoff
                updates=[]
                for _ in range(due):
                    if len(self.replay)<self.config['learning']['batch_size']:
                        raise RuntimeError('registered warmup did not produce a full batch')
                    started = perf_counter()
                    sample=self.replay.sample(self.config['learning']['batch_size'])
                    # Runtime.advance() owns no_grad; this call is outside that scope.
                    update=self.learner.update(sample)
                    if telemetry is not None:
                        telemetry.update(sample, update, perf_counter() - started)
                    updates.append(dict(indices=sample[6].tolist(),result=update))
                self.event('training_step',episode_id=self.completed_episodes,team_step=self.team_steps,
                    episode_step=runtime.step,actions=row['actions'].tolist(),
                    gates=row['action_enabled'].tolist(),team_reward=float(row['rewards'][0]),
                    terminal=bool(row['dones'][0]),updates=updates,counters=self.learner.counters())
                if runtime.terminal:
                    result=runtime.finalize()
                    if telemetry is not None:
                        result['diagnostics'] = telemetry.report()
                    result.update(completed=True,terminated=True,truncated=False,eligible_for_task_statistics=True,
                        episode_id=self.completed_episodes,scenario_id=scene['scenario_id'])
                    runtime.close(); self.active=None
                    self.completed_episodes+=1; self.episodes.append(copy.deepcopy(result)); self.phase='boundary'
                    self.event('episode_finished',result=result)
                    self.commit_boundary()
                    return result
            return self._interrupt('external_mechanism_limit')
        except BaseException as exc:
            self._interrupt('exception_before_boundary_commit',repr(exc),episode_id=episode_id,
                            observed_steps=0 if runtime is None else runtime.step)
            raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true'); parser.add_argument('--output')
    parser.add_argument('--seed',type=int,default=2729); parser.add_argument('--resume')
    parser.add_argument('--episodes',type=int,default=1); parser.add_argument('--step-limit',type=int,default=32)
    args=parser.parse_args(); protocol=load_execution(); limits=protocol['mechanism_limits']
    if not args.execute:
        print(json.dumps(dict(executed=False,protocol=protocol),indent=2)); return
    if not args.output or not 1<=args.episodes<=limits['episodes'] or not 1<=args.step_limit<=limits['steps_per_prefix']:
        parser.error('current registration permits only a fresh output and bounded mechanism execution')
    torch.set_num_threads(1)
    session=(Trainer.resume(args.resume,args.output,run_seed=args.seed) if args.resume
             else Trainer(args.seed,args.output))
    for _ in range(args.episodes):
        result=session.run_episode(max_steps=args.step_limit,round_budget=limits['rounds_per_model'])
        if not result['completed']: break
    from .report import audit_run
    durable_json(session.output/'report.json',audit_run(session.output))
    print(json.dumps(dict(output=str(session.output),team_steps=session.team_steps,
                         counters=session.learner.counters(),phase=session.phase)))


if __name__=='__main__': main()
