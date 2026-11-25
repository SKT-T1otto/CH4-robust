"""No-mission unit checks for policy, private randomness and Replay source guards."""
import copy
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from chapter3_bser.learning.checkpoint import state_digest
from chapter4_interval_robust.nominal_maddpg.config import load
from chapter4_interval_robust.nominal_maddpg.learner import SingleCriticLearner
from chapter4_interval_robust.nominal_maddpg.noise import OUNoise
from chapter4_interval_robust.nominal_maddpg.policy import TrainingPolicy
from chapter4_interval_robust.nominal_maddpg.replay import NominalReplay, TrainingReplay
from chapter4_interval_robust.nominal_maddpg.runtime import NominalRuntime, FrozenInitialPolicy
from chapter4_interval_robust.nominal_maddpg.train_config import SceneCatalog, SceneOrder, stream_seeds, load_training_protocol
from chapter4_interval_robust.nominal_maddpg.validate_runtime import global_rng
from checks.chapter4.test_nominal_learner import synthetic_sample


def synthetic_row(t=0):
    s = synthetic_sample()
    return dict(t=t,observations=np.stack([v[1].numpy() for v in s[0]]),
        actions=np.stack([v[1].numpy() for v in s[1]]),rewards=np.stack([v[1].numpy() for v in s[2]]),
        next_observations=np.stack([v[1].numpy() for v in s[3]]),dones=[False]*4,
        action_enabled=s.enabled[1].numpy(),next_action_enabled=s.next_enabled[1].numpy(),success_flags=[False]*4)


class NominalExplorationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.config,cls.catalog=load(),SceneCatalog()

    def policy(self):
        return TrainingPolicy(SingleCriticLearner(self.config,2729),8137)

    def test_noise_matches_independent_ou_recurrence_without_global_rng_changes(self):
        noise=OUNoise(731)
        rng=torch.Generator().manual_seed(731)
        before=global_rng()
        expected=np.zeros((4,3))
        for step in range(4):
            innovation=torch.randn((4,3),generator=rng).numpy()
            expected=expected+.15*(0-expected)+(.20-step*.00002)*innovation
            actual,sigma=noise.sample()
            np.testing.assert_allclose(actual.numpy(),expected,atol=2e-7,rtol=2e-6)
            self.assertAlmostEqual(sigma,.20-step*.00002)
        self.assertEqual(before,global_rng())

    def test_noise_floor_is_9000_team_steps_and_does_not_restart_each_episode(self):
        noise=OUNoise(1)
        for _ in range(9000):noise.sample(enabled=False)
        self.assertAlmostEqual(noise.sigma,.02,places=14)
        self.assertEqual(noise.draws,0)
        before=noise.generator.get_state().clone()
        noise.reset_episode()
        self.assertEqual(noise.steps,9000)
        self.assertTrue(torch.equal(noise.generator.get_state(),before))
        noise.sample()
        self.assertEqual(noise.sigma,.02)

    def test_noise_state_roundtrip_and_bad_restore_are_isolated(self):
        a,b=OUNoise(7),OUNoise(7)
        a.sample();saved=a.state_dict();b.load_state_dict(saved)
        saved['memory'].add_(100)
        for _ in range(3):self.assertTrue(torch.equal(a.sample()[0],b.sample()[0]))
        before=state_digest(b.state_dict());bad=b.state_dict();bad['memory'][0,0]=float('nan')
        with self.assertRaises(ValueError):b.load_state_dict(bad)
        self.assertEqual(before,state_digest(b.state_dict()))

    def test_disabled_exploration_draws_nothing_and_episode_reset_only_clears_memory(self):
        noise=OUNoise(1);noise.sample();before=noise.state_dict()
        result,_=noise.sample(enabled=False)
        self.assertFalse(bool(result.any()))
        self.assertTrue(torch.equal(before['rng'],noise.state_dict()['rng']))
        self.assertTrue(torch.equal(before['memory'],noise.memory))
        self.assertEqual(noise.steps,2)
        noise.reset_episode()
        self.assertFalse(bool(noise.memory.any()))
        self.assertEqual((noise.steps,noise.draws),(2,1))

    def test_scenario_order_and_stream_derivation_are_private_and_repeatable(self):
        before=global_rng();seeds=stream_seeds(81729)
        self.assertEqual(len(set(seeds.values())),4)
        a,b=SceneOrder(self.catalog,seeds['scene_order']),SceneOrder(self.catalog,seeds['scene_order'])
        order=[a.next_scene()['scenario_id'] for _ in range(20)]
        self.assertEqual(order,[b.next_scene()['scenario_id'] for _ in range(20)])
        self.assertEqual(len(set(order)),20)
        self.assertEqual(before,global_rng())

    def test_policy_clips_actor_plus_noise_before_native_execution_gate(self):
        policy=self.policy();policy.begin_episode()
        policy.noise.memory.fill_(4.)
        with torch.no_grad():actor=policy.learner.networks.actions(torch.zeros(1,4,28))[0]
        actions=torch.cat(policy.step(np.zeros((4,28)),explore=True))
        noise=torch.tensor(policy.last_decision['noise'])
        self.assertTrue(torch.equal(actions,(actor+noise).clamp(-1,1)))
        self.assertTrue(bool((actions==1).all()))
        self.assertEqual(policy.noise.steps,1)
        policy.end_episode()

    def test_live_policy_reads_updated_learner_and_frozen_snapshot_remains_unchanged(self):
        policy=self.policy();policy.begin_episode();frozen=policy.evaluation_snapshot()
        obs=np.ones((4,28),dtype=np.float32)
        frozen_before=frozen.verify_frozen()
        policy.learner.update_actor(synthetic_sample(),0)
        actions=torch.cat(policy.step(obs,explore=False))
        with torch.no_grad():expected=policy.learner.networks.actions(torch.from_numpy(obs).unsqueeze(0))[0]
        self.assertTrue(torch.equal(actions,expected))
        self.assertEqual(frozen_before,frozen.verify_frozen())
        self.assertNotEqual(policy.learner.networks.actors[0].fc1.weight.data_ptr(),frozen.networks.actors[0].fc1.weight.data_ptr())
        self.assertFalse(torch.equal(actions,torch.cat(frozen.step(obs))))
        with self.assertRaises(ValueError):frozen.step(obs,explore=True)
        policy.end_episode()

    def test_original_frozen_policy_protection_is_retained(self):
        policy=FrozenInitialPolicy(self.config,2729)
        before=policy.verify_frozen()
        with self.assertRaises(ValueError):policy.step(np.zeros((4,28)),explore=True)
        self.assertEqual(before,policy.verify_frozen())

    def test_scene_catalog_rejects_relabeling_and_foreign_physical_geometry(self):
        scene=self.catalog.scene('validation',0)
        with self.assertRaises(ValueError):self.catalog.origin(scene,0,allowed_groups=('train',))
        train=self.catalog.scene('train',0)
        for name in ('scenario_id','scenario_seed','scenario_role','scenario_split','ch4_nominal_group'):
            scene[name]=train[name]
        with self.assertRaises(ValueError):self.catalog.origin(scene,0,allowed_groups=('train',))
        train['initial_agent_positions'][0][0]+=.1
        with self.assertRaises(ValueError):self.catalog.origin(train,0,allowed_groups=('train',))

    def test_training_replay_rejects_evaluation_missing_and_forged_origins_before_write(self):
        replay=TrainingReplay(4,1,self.catalog);row=synthetic_row()
        eval_origin=self.catalog.origin(self.catalog.scene('validation',0),0,allowed_groups=('validation',))
        good=replay.bind_scene(self.catalog.scene('train',0),0)
        for source in (None,eval_origin,replace(good,physical_fingerprint='0'*64)):
            with self.assertRaises(ValueError):replay.push_transition(row,source=source)
            self.assertEqual((len(replay),replay.next_idx,replay.insertions),(0,0,0))
        replay.push_transition(row,source=good)
        replay.verify_slot(0,row,source=good)

    def test_replay_ring_sources_follow_replaced_slots_and_do_not_alias_sample_records(self):
        replay=TrainingReplay(2,123,self.catalog)
        for index in range(3):
            source=replay.bind_scene(self.catalog.scene('train',index),index)
            replay.push_transition(synthetic_row(index),source=source)
        sample=replay.sample(2)
        self.assertEqual({s['insertion'] for s in sample.sources},{1,2})
        self.assertEqual({s['transition_step'] for s in sample.sources},{1,2})
        for idx,record in zip(sample[6].tolist(),sample.sources):self.assertEqual(record,replay.slot_sources[idx])
        sample.sources[0]['group']='evaluation'
        self.assertTrue(all(x['group']=='train' for x in replay.sample(2).sources))

    def test_runtime_rejects_wrong_mode_or_evaluation_into_training_before_reset(self):
        policy=self.policy();replay=TrainingReplay(4,1,self.catalog)
        with patch('chapter4_interval_robust.nominal_maddpg.runtime.records.Branch') as constructor:
            with self.assertRaises(ValueError):NominalRuntime(self.config,self.catalog.scene('validation',0),policy,replay,mode='train')
            with self.assertRaises(ValueError):NominalRuntime(self.config,self.catalog.scene('train',0),policy,replay)
            with self.assertRaises(TypeError):NominalRuntime(self.config,self.catalog.scene('validation',0),policy.evaluation_snapshot(),replay,mode='evaluation',catalog=self.catalog)
            with self.assertRaises(ValueError):NominalRuntime(self.config,self.catalog.scene('nominal_admission',0),policy.evaluation_snapshot(),NominalReplay(4,2),mode='evaluation',catalog=self.catalog)
            constructor.assert_not_called()

    def test_live_policy_episode_ownership_and_local_observation_contract(self):
        policy=self.policy()
        with self.assertRaises(RuntimeError):policy.step(np.zeros((4,28)))
        policy.begin_episode()
        with self.assertRaises(RuntimeError):policy.begin_episode()
        with self.assertRaises(ValueError):policy.step(np.zeros((4,29)),explore=True)
        self.assertEqual(policy.noise.steps,0)
        policy.end_episode();policy.begin_episode();policy.end_episode()

    def test_collection_protocol_preserves_original_decay_and_unopened_sets(self):
        p=load_training_protocol()
        self.assertEqual(p['exploration']['decay_per_environment_step'],.00002)
        self.assertFalse(p['nominal_admission_opened'])
        self.assertFalse(p['mismatch_sets_opened'])
        self.assertEqual(p['maximum_steps_per_prefix'],32)


if __name__=='__main__':unittest.main()
