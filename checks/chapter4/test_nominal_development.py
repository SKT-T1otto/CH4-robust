"""Bounded development checks. No pilot or search training is run by this suite."""
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from chapter4_interval_robust.nominal_maddpg import development as dev
from chapter4_interval_robust.nominal_maddpg.checkpoint import read_boundary
from chapter4_interval_robust.nominal_maddpg.development_metrics import EpisodeMetrics, summarize
from chapter4_interval_robust.nominal_maddpg.evaluate import evaluate_d2, evaluate_training
from chapter4_interval_robust.nominal_maddpg.train_config import SceneCatalog
from chapter4_interval_robust.nominal_maddpg.learner import prepare_batch


class DevelopmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.catalog=SceneCatalog()

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)

    def test_paired_real_reference_prefixes_and_direct_physical_effect(self):
        trainer=dev.DevelopmentTrainer(2729,self.path/'trainer',profile='pilot',catalog=self.catalog)
        d2=evaluate_d2(trainer.config,self.catalog,indices=[0],output=self.path/'d2',max_steps=3)
        frozen=evaluate_training(trainer,indices=[0],output=self.path/'frozen',max_steps=3,diagnostics=True)
        self.assertTrue(frozen['isolation']['unchanged'])
        a,b=d2['episodes'][0],frozen['episodes'][0]
        self.assertEqual(a['source'],b['source'])
        self.assertEqual(a['diagnostics']['nonzero_command_steps'],[0]*4)
        np.testing.assert_allclose(a['diagnostics']['thrust_delta_l2_sum_mps2'],0,atol=1e-6)
        self.assertGreater(sum(b['diagnostics']['nonzero_command_steps']),0)
        self.assertFalse(d2['selection_eligible'] or frozen['selection_eligible'])
        self.assertEqual(len(trainer.replay),0)
        with self.assertRaises(ValueError):
            summarize([a])

    def test_registered_search_settings_and_source_checked_resume(self):
        for name,lr,decay in [('A',.001,.00002),('B',.001,.000002),('C',.0003,.00002),('D',.0003,.000002)]:
            spec=dev.run_spec(name,17239)
            self.assertEqual((spec['actor_lr'],spec['decay_per_environment_step'],spec['episodes']),(lr,decay,200))
        with self.assertRaises(ValueError):
            dev.run_spec('D',2729)
        a=dev.DevelopmentTrainer(17239,self.path/'train',profile='D',catalog=self.catalog)
        self.assertTrue(all(o.param_groups[0]['lr']==.0003 for o in a.learner.actor_optimizers))
        self.assertTrue(all(o.param_groups[0]['lr']==.0005 for o in a.learner.critic_optimizers))
        self.assertEqual(a.policy.noise.settings['decay_per_environment_step'],.000002)
        path=a.commit_boundary()
        b=dev.DevelopmentTrainer.resume(path,self.path/'resume',profile='D',run_seed=17239,catalog=self.catalog)
        self.assertEqual(a.identity,b.identity)
        with self.assertRaisesRegex(ValueError,'profile'):
            read_boundary(path,dev.DevelopmentTrainer.settings(17239,'C')[1])
        with self.assertRaisesRegex(ValueError,'scene group'):
            b.replay.bind_scene(self.catalog.scene('validation',0),0)

    def test_real_sample_q_and_stage_labels(self):
        a=dev.DevelopmentTrainer(2729,self.path/'pilot',profile='pilot',catalog=self.catalog)
        result=a.run_episode(max_steps=4,round_budget=0)
        self.assertFalse(result['completed'])
        sample=a.replay.sample(4)
        self.assertEqual([s['executor_post_handoff'] for s in sample.sources],[False]*4)
        batch=prepare_batch(sample)
        with torch.no_grad():
            joint=a.learner._joint(batch.observations,batch.actions)
            expected=[q(joint).mean().item() for q in a.learner.networks.critics]
        update=a.learner.update(sample)
        np.testing.assert_allclose([c['q_mean'] for c in update['critics']],expected,rtol=0,atol=0)
        metrics=EpisodeMetrics(0.)
        # Inject phase metadata only in this detached sample to test the positive case.
        for j,s in enumerate(sample.sources):
            s['executor_post_handoff']=j%2==0
        metrics.update(sample,update,.125)
        d=metrics.report()
        self.assertEqual(d['executor_post_handoff_effective_rows'],int(sample.enabled[::2,3].sum()))
        self.assertEqual(d['actor_effective_rows'],sample.enabled.sum(dim=0).tolist())
        self.assertEqual(d['rounds'],1)
        self.assertEqual(d['seconds']['updates'],.125)
        self.assertEqual(set(d['ranges']),{'q','td_target','critic_gradient_norm','actor_gradient_norm'})

    def test_common_checkpoint_and_equal_seed_config_selection(self):
        search=dev.read_json(dev.PLAN)['optional_search']
        runs={}
        for cfg in 'ABCD':
            for seed in search['seeds']:
                trend=[]
                for e in search['checkpoints']:
                    cost=200
                    if cfg=='A': cost=20 if e==(16 if seed==17239 else 32) else 400
                    trend.append(dict(episode=e,validation=dict(episodes=32,mean_penalized_completion_steps=cost)))
                runs[cfg,seed]=dict(validation_trend=trend)
        result=dev.select_config(runs,search)
        self.assertEqual((result['selected']['id'],result['selected']['episode']),('B',16))
        runs.pop(('C',17239))
        with self.assertRaises(KeyError):dev.select_config(runs,search)

    def test_linux_guard_dry_run_and_partial_commit(self):
        import contextlib,io
        with patch('sys.argv',['development','pilot']),contextlib.redirect_stdout(io.StringIO()) as output:
            dev.main()
        self.assertEqual(json.loads(output.getvalue())['pilot']['episodes'],64)
        with patch('sys.platform','win32'),patch('sys.argv',['development','search','--execute','--output',str(self.path)]):
            with self.assertRaises(SystemExit):dev.main()
        a=self.path/'attempt_1/boundaries/000003'; a.mkdir(parents=True)
        (a/'committed.json').write_text('{"completed_episodes": 3}',encoding='utf-8')
        b=self.path/'attempt_2/boundaries/000004'; b.mkdir(parents=True)
        (b/'committed.json').write_text('{',encoding='utf-8')
        self.assertEqual(dev.latest_boundary(self.path),a)
        path=self.path/'result.json'; dev.publish_result(path,{'v':1})
        with self.assertRaises(FileExistsError):dev.publish_result(path,{'v':2})
        self.assertEqual(dev.read_json(path),{'v':1})

    def test_driver_orders_references_and_restarts_after_failed_validation(self):
        # Drive the actual orchestration, seals and selection with synthetic metrics.
        # The fixture advances episode numbers; no optimizer or mission is run here.
        from types import SimpleNamespace
        from chapter4_interval_robust.nominal_maddpg.checkpoint import durable_json
        identity={'test':'workflow'}
        calls=[]
        rows=[]
        class Session:
            @staticmethod
            def settings(seed,profile): return {},identity

            def __init__(self,seed,work,profile):
                self.identity=identity
                self.work=work
                self.completed_episodes=0
                self.team_steps=0
                self.episodes=[]
                self.learner=SimpleNamespace(rounds=0,counters=lambda:{})

            def commit_boundary(self):
                path=self.work/'boundaries'/f'{self.completed_episodes:06d}'
                path.mkdir(parents=True)
                durable_json(path/'committed.json',dict(completed_episodes=self.completed_episodes))

            @classmethod
            def resume(cls,path,work,run_seed,profile):
                session=cls(run_seed,work,profile)
                session.completed_episodes=int(path.name)
                session.episodes=rows.copy()
                return session

            def run_episode(self):
                self.completed_episodes+=1
                self.commit_boundary()
                self.episodes.append(training_row)
                rows.append(training_row)
                calls.append('episode')
                return training_row

        diagnostics=EpisodeMetrics(0)
        diagnostics.steps=1
        training_row=dict(completed=True,success=False,termination='fixture',penalized_completion_steps=500,
                          team_undiscounted_return=0.,team_discounted_return=0.,diagnostics=diagnostics.report())
        def validation(session,folder,kind):
            calls.append((kind,session.completed_episodes))
            if calls.count(('trained',16))==1 and kind=='trained' and session.completed_episodes==16:
                raise RuntimeError('injected validation interruption')
            path=dev.attempt(folder); path.mkdir()
            value=dict(catalog_sha256='catalog',selection_eligible=True,group='validation',indices=list(range(32)),
                       episodes=[dict(training_row,validation_index=i,scenario_id=str(i),eligible_for_task_statistics=True)
                                 for i in range(32)], diagnostics=dict(episodes=32))
            durable_json(path/'evaluation.json',value)
            dev.publish_result(path/'complete.json',dict(registration=dict(identity=identity,kind=kind,
                boundary=session.completed_episodes),evaluation_sha256=dev.file_sha(path/'evaluation.json')))
            return value

        import contextlib,io
        with patch.object(dev,'DevelopmentTrainer',Session),patch.object(dev,'validation',validation),contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError,'injected validation'):
                dev.run(self.path,'pilot',2729)
            result=dev.run(self.path,'pilot',2729)
            self.assertEqual(dev.run(self.path,'pilot',2729),result)
        self.assertEqual(calls[:2],[('D2',0),('frozen_initialization',0)])
        self.assertEqual(calls.count('episode'),64)
        self.assertEqual(result['selected_episode'],16)
        self.assertTrue((self.path/result['selected_checkpoint']/'committed.json').is_file())
        self.assertEqual([v['episode'] for v in result['validation_trend']],[16,32,48,64])
        self.assertEqual(len(result['health']['prior_attempt_errors']),1)


if __name__=='__main__':
    unittest.main()
