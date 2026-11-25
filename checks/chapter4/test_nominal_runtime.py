"""Bounded real-transition tests for the read-only nominal runtime."""
import copy
import gzip
import json
import unittest
from unittest.mock import patch

import numpy as np
import torch

from chapter4_interval_robust.nominal_maddpg.config import ROOT, load, read_json
from chapter4_interval_robust.nominal_maddpg.replay import NominalReplay
from chapter4_interval_robust.nominal_maddpg.runtime import FrozenInitialPolicy, NominalRuntime, ZeroResidualPolicy, json_transition
from chapter4_interval_robust.nominal_maddpg.validate_runtime import global_rng, replay_digest, load_protocol
from experiments.ch4_work1_recoverability import records
from experiments.ch4_work1_recoverability.run import materialize


class ConstantPolicy:
    def step(self, observations, explore=False):
        return [torch.full((1, 3), 2.0) for _ in range(4)]


class NominalRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.config = load()
        protocol = read_json(ROOT/"experiments/ch4_work1_recoverability/protocol.json")
        cls.scene = materialize(protocol, "calibration", 3)

    def runtime(self, policy=None, capacity=16):
        replay = NominalReplay(capacity, 81729)
        return NominalRuntime(self.config, self.scene, policy or ZeroResidualPolicy(), replay, episode_id=3)

    def test_zero_transition_matches_historical_physics_and_guidance(self):
        with self.runtime() as rt:
            row = rt.advance()
            path = ROOT/"experiments/ch4_work1_recoverability/results/step3_d2_20261124/calibration_03.jsonl.gz"
            with gzip.open(path, "rt", encoding="utf-8") as f:
                self.assertEqual(rt.last_record["legacy"], json.loads(next(f)))
            self.assertEqual(len(rt.replay), 1)
            self.assertFalse(bool(row["actions"].any()))
            self.assertTrue(bool(rt.world.plant.command.abs().sum() > 0))
            self.assertEqual(rt.replay.verify_sample(1)["critic_shape"], [1, 124])

    def test_actual_control_gate_and_clipping_reach_replay(self):
        # An explicit unit fixture exercises both open and closed control paths.
        with self.runtime(ConstantPolicy()) as rt, patch.object(rt.rt, "_gates",
                return_value=np.asarray([True, False, True, False])):
            row = rt.advance()
            self.assertTrue(np.array_equal(row["proposed_actions"], np.full((4, 3), 2.0)))
            self.assertTrue(np.array_equal(row["actions"], np.asarray([[1, 1, 1], [0, 0, 0], [1, 1, 1], [0, 0, 0]])))
            rt.replay.verify_slot(0, row)
            self.assertEqual(rt.suppressed_nonzero_boat_steps, 2)

    def test_reward_and_refreshed_next_observation_are_native(self):
        with self.runtime() as rt:
            row = rt.advance()
            self.assertTrue(np.array_equal(row["next_observations"], rt.rt.observations))
            self.assertEqual(row["reward_transform_applied_count"], 1)
            self.assertTrue(np.array_equal(row["rewards"], np.full(4, row["team_reward"], dtype=np.float32)))
            for i in range(4):
                self.assertEqual(float(rt.replay.rew_buffs[i][0]), float(row["rewards"][i]))
            json.dumps(dict(transition=json_transition(row), record=rt.last_record), allow_nan=False)

    def test_replay_owns_transition_memory_and_uses_real_ring_slots(self):
        with self.runtime(capacity=2) as rt:
            rows = [rt.advance() for _ in range(3)]
            self.assertEqual((len(rt.replay), rt.replay.next_idx), (2, 1))
            rt.replay.verify_slot(0, rows[2])
            rt.replay.verify_slot(1, rows[1])
            before = replay_digest(rt.replay)
            rows[2]["observations"][0][:] = 19.0
            rows[2]["actions"][:] = 0.5
            self.assertEqual(replay_digest(rt.replay), before)
            self.assertEqual(rt.replay.verify_sample(2)["rows"], 2)

    def test_sampling_and_simulation_preserve_external_rng(self):
        with self.runtime() as rt:
            before = global_rng()
            rt.advance()
            self.assertEqual(global_rng(), before)
            sampling = rt.replay.sampling_rng.clone()
            rt.replay.verify_sample(1)
            self.assertEqual(global_rng(), before)
            self.assertFalse(torch.equal(sampling, rt.replay.sampling_rng))

    def test_bad_replay_actions_or_local_done_are_rejected_before_insertion(self):
        with self.runtime() as rt:
            row = rt.advance()
            for kind in ("closed_action", "local_done"):
                broken = copy.deepcopy(row)
                if kind == "closed_action":
                    broken["action_enabled"][0] = False
                    broken["actions"][0, 0] = 0.5
                else:
                    broken["dones"][0] = True
                before = replay_digest(rt.replay)
                with self.subTest(kind=kind), self.assertRaises(ValueError):
                    rt.replay.push_transition(broken)
                self.assertEqual(replay_digest(rt.replay), before)
            with self.assertRaises(ValueError):
                rt.replay.sample(1, norm_rews=True)

    def test_frozen_actor_sees_28d_inputs_and_retains_weights(self):
        policy = FrozenInitialPolicy(self.config, 2729)
        with self.runtime(policy) as rt:
            row = rt.advance()
            self.assertEqual(policy.last_input_sha256, records.stamp(np.asarray(row["observations"], dtype=np.float32)))
            self.assertEqual(policy.verify_frozen(), policy.initial_sha256)
            self.assertTrue(bool(np.any(row["proposed_actions"])))
            with self.assertRaises(ValueError):
                policy.step(row["observations"], explore=True)

    def test_partial_prefix_and_closed_runtime_are_explicit(self):
        rt = self.runtime()
        try:
            rt.advance()
            with self.assertRaisesRegex(RuntimeError, "partial"):
                rt.finalize()
        finally:
            rt.close()
        rt.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            rt.advance()

    def test_bad_task_scene_is_rejected_before_mission_creation(self):
        broken = copy.deepcopy(self.scene)
        broken["max_steps"] = 399
        with patch("chapter4_interval_robust.nominal_maddpg.runtime.records.Branch") as constructor:
            with self.assertRaises(ValueError):
                NominalRuntime(self.config, broken, ZeroResidualPolicy(), NominalReplay(16, 1))
            constructor.assert_not_called()

    def test_protocol_preserves_unopened_admission_and_mismatch_sets(self):
        p, _, _ = load_protocol()
        self.assertEqual(len(p["cases"]), 8)
        self.assertEqual({c["group"] for c in p["cases"]}, {"calibration", "validation"})
        self.assertFalse(p["nominal_admission_opened"])
        self.assertFalse(p["mismatch_sets_opened"])
        self.assertFalse(p["training_enabled"])


if __name__ == "__main__":
    unittest.main()
