"""Temporary checkpoints only. Synthetic task fixtures never certify mission performance."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from chapter3_bser.learning.checkpoint import state_digest
from chapter4_interval_robust.nominal_maddpg.config import load
from chapter4_interval_robust.nominal_maddpg.checkpoint import read_boundary, save_boundary, global_rng_state, durable_json
from chapter4_interval_robust.nominal_maddpg.execution import load_execution
from chapter4_interval_robust.nominal_maddpg.replay import TrainingReplay
from chapter4_interval_robust.nominal_maddpg.train_config import SceneCatalog, SceneOrder
from chapter4_interval_robust.nominal_maddpg.trainer import Trainer, rounds_due
from chapter4_interval_robust.nominal_maddpg.report import audit_run, summarize_episodes, choose_validation
from checks.chapter4.test_nominal_exploration import synthetic_row

UNIT_IDENTITY=dict(algorithm='ch4.maddpg.single_critic.v1',source_sha256='UNIT_FIXTURE_NOT_RESEARCH_SOURCE',
                  config_sha256='unit',execution_sha256='unit',scenes_sha256='unit',environment={'cpu':'unit'})


class SyntheticEpisode:
    """A scheduler test double, NOT AUV simulation; real networks/Adam/Replay/OU remain in use."""
    def __init__(self, config, scene, policy, replay, episode_id=0, **kwargs):
        self.policy,self.replay=policy,replay; self.episode_id=episode_id
        self.source=replay.bind_scene(scene,episode_id); self.step=0; self.terminal=False
        self.horizon=256 if episode_id==0 else 4
        self.policy.begin_episode()

    def advance(self):
        obs=np.random.default_rng([871,self.episode_id,self.step]).normal(0,.1,(4,28)).astype(np.float32)
        actions=np.concatenate([a.numpy() for a in self.policy.step(obs,explore=True)])
        gates=np.array([True, self.step%2==0, True, False])
        actions*=gates[:,None]
        t=self.step; self.step+=1; self.terminal=self.step==self.horizon
        row=dict(t=t,observations=obs,next_observations=obs+.01,actions=actions,
            action_enabled=gates,next_action_enabled=np.zeros(4,bool) if self.terminal else gates,
            rewards=np.full(4,.1+float(actions[0].mean()),np.float32),
            dones=[self.terminal]*4,success_flags=[False]*4)
        self.replay.push_transition(row,source=self.source)
        return row

    def finalize(self):
        if not self.terminal: raise RuntimeError('fixture has not ended')
        return dict(steps=self.step,success=False,penalized_completion_steps=500,
                    termination='synthetic_unit_fixture',not_an_auv_mission=True)

    def close(self): self.policy.end_episode()


class NominalTrainingStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1); cls.catalog=SceneCatalog(); cls.config=load()

    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.folder=Path(self.tmp.name)
        self.identity_patch=patch('chapter4_interval_robust.nominal_maddpg.trainer.build_identity',return_value=UNIT_IDENTITY)
        self.identity_patch.start()

    def tearDown(self):
        self.identity_patch.stop(); self.tmp.cleanup()

    def trainer(self,name): return Trainer(2729,self.folder/name,catalog=self.catalog)

    def replay(self,n=4): return TrainingReplay(n,87,self.catalog)

    def test_replay_valid_contents_ring_gates_sources_and_rng_roundtrip(self):
        for count in (0,3,11):
            with self.subTest(count=count):
                a,b=self.replay(),self.replay()
                origin=a.bind_scene(self.catalog.scene('train',0),0)
                for t in range(count): a.push_transition(synthetic_row(t),source=origin)
                if count: a.sample(2)
                state=a.state_dict(); b.load_state_dict(state)
                self.assertEqual(state_digest(state),state_digest(b.state_dict()))
                if count:
                    for _ in range(3):
                        x,y=a.sample(3),b.sample(3)
                        self.assertTrue(torch.equal(x[6],y[6])); self.assertEqual(x.sources,y.sources)
                    state['obs_buffs'][0].add_(99)
                    self.assertFalse(torch.equal(state['obs_buffs'][0],b.obs_buffs[0][:len(b)]))

    def test_replay_rejects_bad_source_gate_pointer_before_mutation(self):
        replay=self.replay(); source=replay.bind_scene(self.catalog.scene('train',0),0)
        replay.push_transition(synthetic_row(),source=source); original=replay.state_dict()
        cases=[]
        bad=copy.deepcopy(original); bad['slot_sources'][0]['group']='validation'; cases.append(bad)
        bad=copy.deepcopy(original); bad['next_idx']=3; cases.append(bad)
        bad=copy.deepcopy(original); bad['enabled'].zero_(); cases.append(bad)
        bad=copy.deepcopy(original); bad['sampling_rng']=torch.zeros(3,dtype=torch.uint8); cases.append(bad)
        for bad in cases:
            with self.assertRaises((ValueError,RuntimeError)): replay.load_state_dict(bad)
            self.assertEqual(state_digest(original),state_digest(replay.state_dict()))

    def test_scene_order_rollover_restore_and_invalid_permutation(self):
        a,b=SceneOrder(self.catalog,99),SceneOrder(self.catalog,99)
        for _ in range(998): a.next_scene()
        b.load_state_dict(a.state_dict())
        self.assertEqual([a.next_scene() for _ in range(5)],[b.next_scene() for _ in range(5)])
        before=b.state_dict(); bad=copy.deepcopy(before); bad['order'][0]=bad['order'][1]
        with self.assertRaises(ValueError): b.load_state_dict(bad)
        self.assertEqual(before,b.state_dict())

    def test_registered_schedule_first_256_then_260_and_exact_settings(self):
        p=load_execution()['schedule']; self.assertEqual(p,{k:self.config['learning'][k] for k in p})
        self.assertEqual([(t,rounds_due(t,p)) for t in range(1,261) if rounds_due(t,p)],[(256,2),(260,2)])

    @patch('chapter4_interval_robust.nominal_maddpg.trainer.NominalRuntime',SyntheticEpisode)
    def test_continuous_vs_boundary_restore_exact_actions_batches_losses_parameters(self):
        a=self.trainer('continuous'); a.run_episode()
        boundary=a.last_boundary
        self.assertEqual(a.learner.counters(),dict(rounds=2,target_updates=2,actor_steps=[2,2,2,0],critic_steps=[2]*4,actor_skips=[0,0,0,2]))
        saved_digest=state_digest(a.state_dict())
        a.run_episode(); final=state_digest(a.state_dict())
        b=Trainer.resume(boundary,self.folder/'resumed',run_seed=2729,catalog=self.catalog)
        self.assertEqual(saved_digest,state_digest(b.state_dict()))
        b.run_episode(); self.assertEqual(final,state_digest(b.state_dict()))
        def rows(t):
            events=[json.loads(p.read_text()) for p in sorted((t.output/'events').glob('*.json'))]
            return [e for e in events if e['kind']=='training_step' and e['episode_id']==1]
        x,y=rows(a),rows(b)
        for values in (x,y):
            for v in values: v.pop('sequence')
        self.assertEqual(x,y)
        self.assertEqual(a.learner.rounds,4)
        self.assertTrue(torch.equal(a.replay.sample(128)[6],b.replay.sample(128)[6]))

    def test_source_config_scenes_algorithm_and_execution_mismatch_rejected(self):
        a=self.trainer('a'); boundary=a.commit_boundary()
        before=state_digest(a.state_dict())
        for key in ('source_sha256','config_sha256','scenes_sha256','algorithm','execution_sha256','environment'):
            with self.subTest(key=key):
                expected=copy.deepcopy(UNIT_IDENTITY); expected[key]='different'
                with self.assertRaisesRegex(ValueError,key): read_boundary(boundary,expected)
        with self.assertRaisesRegex(ValueError,'run seed'):
            Trainer.resume(boundary,self.folder/'rejected',run_seed=3733,catalog=self.catalog)
        self.assertFalse((self.folder/'rejected').exists())
        self.assertEqual(before,state_digest(a.state_dict()))

    def test_missing_commit_marker_and_corrupted_file_rejected(self):
        a=self.trainer('a'); boundary=a.commit_boundary()
        incomplete=self.folder/'incomplete'; incomplete.mkdir()
        (incomplete/'state.pt').write_bytes((boundary/'state.pt').read_bytes())
        with self.assertRaisesRegex(ValueError,'no committed'): read_boundary(incomplete,UNIT_IDENTITY)
        # Only this test's freshly created checkpoint is corrupted.
        with (boundary/'state.pt').open('ab') as f:f.write(b'corruption')
        with self.assertRaisesRegex(ValueError,'bytes differ'): read_boundary(boundary,UNIT_IDENTITY)

    @patch('chapter4_interval_robust.nominal_maddpg.trainer.NominalRuntime',SyntheticEpisode)
    def test_truncation_cannot_be_checkpointed_or_counted_as_task_failure(self):
        a=self.trainer('a'); result=a.run_episode(max_steps=3)
        self.assertIsNone(result['success']); self.assertEqual(a.learner.rounds,0)
        with self.assertRaisesRegex(RuntimeError,'boundary'): save_boundary(a,self.folder/'bad')
        with self.assertRaises(RuntimeError): a.run_episode()
        report=audit_run(a.output)
        self.assertEqual(report['statistics']['completed'],0); self.assertIsNone(report['statistics']['success_rate'])
        b=Trainer.resume(a.last_boundary,self.folder/'resumed',run_seed=2729,catalog=self.catalog)
        self.assertEqual((b.team_steps,len(b.replay),b.order.cursor,b.policy.noise.steps),(0,0,0,0))

    def test_unclosed_journal_episode_is_reported_as_process_interruption(self):
        a=self.trainer('a'); a.commit_boundary()
        a.event('episode_started',episode_id=0,scene='unit',previous_boundary=str(a.last_boundary))
        report=audit_run(a.output)
        self.assertEqual(report['attempt_episodes'][0]['reason'],'process_interrupted_before_boundary_commit')
        self.assertIsNone(report['statistics']['success_rate'])

    @patch('chapter4_interval_robust.nominal_maddpg.trainer.NominalRuntime',SyntheticEpisode)
    def test_update_error_requires_rollback_and_never_commits_partial_optimizer_round(self):
        a=self.trainer('a')
        with patch.object(a.learner,'update',side_effect=RuntimeError('injected optimizer failure')):
            with self.assertRaisesRegex(RuntimeError,'injected'): a.run_episode()
        self.assertEqual(a.phase,'interrupted')
        self.assertEqual(len(list((a.output/'boundaries').glob('*'))),1)
        b=Trainer.resume(a.last_boundary,self.folder/'resumed',run_seed=2729,catalog=self.catalog)
        self.assertEqual((b.team_steps,b.learner.rounds,len(b.replay)),(0,0,0))

    @patch('chapter4_interval_robust.nominal_maddpg.trainer.NominalRuntime',SyntheticEpisode)
    def test_terminal_before_failed_checkpoint_commit_still_rolls_back_current_episode(self):
        a=self.trainer('a'); boundary=a.commit_boundary()
        with patch.object(a,'commit_boundary',side_effect=OSError('injected disk write failure')):
            with self.assertRaisesRegex(OSError,'injected'):a.run_episode()
        report=audit_run(a.output)
        self.assertEqual(report['statistics']['completed'],0)
        self.assertEqual(report['attempt_episodes'][0]['episode_id'],0)
        self.assertEqual(report['attempt_episodes'][0]['steps'],256)
        self.assertEqual(report['last_committed_episodes'],0)
        b=Trainer.resume(boundary,self.folder/'restored',run_seed=2729,catalog=self.catalog)
        self.assertEqual((b.completed_episodes,b.team_steps,b.learner.rounds),(0,0,0))

    def test_evaluation_exception_preserves_full_training_state(self):
        from chapter4_interval_robust.nominal_maddpg.evaluate import evaluate_training
        a=self.trainer('a'); before=state_digest(a.state_dict())
        def fail(*args,**kwargs):
            torch.rand(5); np.random.rand(5)
            raise RuntimeError('injected evaluation error')
        with patch('chapter4_interval_robust.nominal_maddpg.evaluate.evaluate_snapshot',side_effect=fail):
            with self.assertRaisesRegex(RuntimeError,'evaluation error'):
                evaluate_training(a,indices=[0],output=self.folder/'eval')
        self.assertEqual(before,state_digest(a.state_dict()))

    def test_real_evaluation_preserves_nonempty_replay_adam_and_exploration(self):
        from chapter4_interval_robust.nominal_maddpg.evaluate import evaluate_training
        a=self.trainer('primed')
        with patch('chapter4_interval_robust.nominal_maddpg.trainer.NominalRuntime',SyntheticEpisode):
            a.run_episode()
        self.assertEqual(a.learner.rounds,2); self.assertEqual(len(a.replay),256)
        before=state_digest(a.state_dict())
        result=evaluate_training(a,indices=[0],output=self.folder/'real_validation',max_steps=2)
        self.assertTrue(result['isolation']['unchanged'])
        self.assertEqual(before,state_digest(a.state_dict()))
        self.assertIsNone(result['statistics']['success_rate'])
        b=Trainer.resume(a.last_boundary,self.folder/'restored',run_seed=2729,catalog=self.catalog)
        self.assertTrue(torch.equal(a.replay.sample(128)[6],b.replay.sample(128)[6]))
        self.assertTrue(torch.equal(a.policy.noise.sample()[0],b.policy.noise.sample()[0]))

    def test_torn_last_event_is_interruption_not_completed_failure(self):
        a=self.trainer('a'); a.commit_boundary()
        a.event('episode_started',episode_id=0,scene='unit',previous_boundary=str(a.last_boundary))
        (a.output/'events'/f'{a.event_index:08d}.json').write_text('{"sequence":',encoding='utf-8')
        report=audit_run(a.output)
        self.assertIsNotNone(report['torn_event_tail'])
        self.assertEqual(report['statistics']['incomplete'],1)
        self.assertIsNone(report['statistics']['success_rate'])

    def test_reports_exclude_prefixes_and_selection_requires_complete_cohort(self):
        partial=dict(completed=False,eligible_for_task_statistics=False,success=None,penalized_completion_steps=None)
        self.assertIsNone(summarize_episodes([partial])['success_rate'])
        malformed=dict(partial,success=False)
        with self.assertRaises(ValueError):summarize_episodes([malformed])
        candidate=dict(completed_episodes=1,evaluation=dict(catalog_sha256='unit',selection_eligible=False,
            group='validation',episodes=[partial],indices=[0]))
        with self.assertRaises(ValueError):choose_validation([candidate])
        rows=[dict(completed=True,eligible_for_task_statistics=True,success=True,penalized_completion_steps=100,
                   scenario_id=str(i),validation_index=i) for i in range(32)]
        one=dict(completed_episodes=2,evaluation=dict(catalog_sha256='unit',selection_eligible=True,
            group='validation',episodes=rows,indices=list(range(32))))
        two=copy.deepcopy(one); two['completed_episodes']=1
        self.assertIs(choose_validation([one,two]),two)


if __name__=='__main__': unittest.main()
