"""Regression checks for the independent steps 1-2 registration, with no updates."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from chapter3_bser.learning.checkpoint import digest, state_digest
from chapter4_interval_robust.nominal_maddpg.config import DEFAULT_CONFIG, load, read_json, validate
from chapter4_interval_robust.nominal_maddpg.learner import SingleCriticNetworks
from chapter4_interval_robust.nominal_maddpg.provenance import compare_inventory, inventory, verify_record
from chapter4_interval_robust.nominal_maddpg.scenarios import validate_registry
from chapter4_interval_robust.nominal_maddpg.verify import resolved_task_contract


class NominalRegistrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load()
        cls.registry = read_json(DEFAULT_CONFIG.parent/"nominal_maddpg_scenes.json")

    def test_contract_rejects_task_plant_algorithm_or_training_drift(self):
        changes = (("task", "residual_scale_search", 0.45), ("plant", "beta_m_per_s", 0.15),
                   ("plant", "actuator_delay_steps", 1), ("algorithm", "critics_per_vehicle", 2),
                   ("algorithm", "actor_truth_parameters_visible", True),
                   ("algorithm", "critics_per_vehicle", True),
                   ("stage_status", "training_enabled", True), ("stage_status", "optimizer_updates_allowed", 1))
        for section, key, value in changes:
            c = copy.deepcopy(self.config)
            c[section][key] = value
            with self.subTest(section=section, key=key), self.assertRaises(ValueError):
                validate(c)

    def test_config_rejects_overlapping_groups_and_wrong_counts(self):
        c = copy.deepcopy(self.config)
        c["scene_groups"]["validation"]["start_seed"] = c["scene_groups"]["train"]["start_seed"] + 1
        with self.assertRaisesRegex(ValueError, "overlapping"):
            validate(c)
        c = copy.deepcopy(self.config)
        c["scene_groups"]["train"]["count"] = 999
        with self.assertRaises(ValueError):
            validate(c)

    def test_registry_matches_all_registered_groups(self):
        groups = validate_registry(self.config, self.registry, DEFAULT_CONFIG)
        self.assertEqual([groups[g]["count"] for g in self.config["scene_groups"]], [1000, 32, 100, 24, 256])
        self.assertFalse(self.registry["outcomes_generated"])

    def test_registry_rejects_physical_change(self):
        registry = copy.deepcopy(self.registry)
        registry["groups"]["validation"][0]["scene"]["initial_agent_positions"][0][0] += 0.25
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            validate_registry(self.config, registry, DEFAULT_CONFIG)

    def test_registry_rejects_cross_group_duplicate_even_with_new_labels(self):
        registry = copy.deepcopy(self.registry)
        entry = registry["groups"]["validation"][0]
        donor = registry["groups"]["train"][0]
        identity = {k:entry["scene"][k] for k in ("scenario_seed", "scenario_id", "scenario_role", "scenario_split", "ch4_nominal_group")}
        entry["scene"] = copy.deepcopy(donor["scene"])
        entry["scene"].update(identity)
        entry["physical_fingerprint"] = donor["physical_fingerprint"]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_registry(self.config, registry, DEFAULT_CONFIG)

    def test_registry_rejects_mislabeled_holdout_and_changed_config(self):
        registry = copy.deepcopy(self.registry)
        registry["groups"]["nominal_admission"][0]["scene"]["scenario_role"] = "train"
        with self.assertRaises(ValueError):
            validate_registry(self.config, registry, DEFAULT_CONFIG)
        registry = copy.deepcopy(self.registry)
        registry["config_sha256"] = "0"*64
        with self.assertRaisesRegex(ValueError, "identity"):
            validate_registry(self.config, registry, DEFAULT_CONFIG)

    def test_source_rejects_modified_missing_and_added_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/"core").mkdir()
            path = root/"core/physical.py"
            path.write_text("coefficient = 1\n", encoding="utf-8")
            files = inventory(root)
            path.write_text("coefficient = 2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "inventory"):
                compare_inventory(files, inventory(root))
            path.write_text("coefficient = 1\n", encoding="utf-8")
            (root/"core/extra.py").write_text("new = True\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "inventory"):
                compare_inventory(files, inventory(root))
            path.unlink()
            with self.assertRaisesRegex(ValueError, "inventory"):
                compare_inventory(files, inventory(root))
        record = dict(schema="auv.ch4.nominal_maddpg.source.v1", files=files, training_enabled=False)
        record["sha256"] = digest(record)
        verify_record(record, files)
        record["training_enabled"] = True
        with self.assertRaisesRegex(ValueError, "digest"):
            verify_record(record, files)

    def test_networks_are_single_critic_independent_and_deterministic(self):
        before = torch.get_rng_state().clone()
        a = SingleCriticNetworks(self.config, 2729)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        b = SingleCriticNetworks(self.config, 2729)
        c = SingleCriticNetworks(self.config, 3733)
        self.assertEqual(state_digest(a.state_dict()), state_digest(b.state_dict()))
        self.assertNotEqual(state_digest(a.state_dict()), state_digest(c.state_dict()))
        self.assertEqual((len(a.actors), len(a.critics)), (4, 4))
        self.assertEqual(a.critics[0].fc1.in_features, 124)
        self.assertEqual(a.actors[0].fc1.in_features, 28)
        self.assertEqual(a.actors[0].fc3.out_features, 3)
        self.assertNotEqual(a.actors[0].fc1.weight.data_ptr(), a.actors[1].fc1.weight.data_ptr())
        self.assertNotEqual(a.critics[0].fc1.weight.data_ptr(), a.target_critics[0].fc1.weight.data_ptr())
        self.assertFalse(any(p.requires_grad for p in a.target_critics.parameters()))
        for online, target in ((a.actors, a.target_actors), (a.critics, a.target_critics)):
            self.assertEqual(state_digest(online.state_dict()), state_digest(target.state_dict()))

    def test_static_contract_does_not_construct_a_mission(self):
        with patch("chapter4_interval_robust.nominal_maddpg.verify.MissionCoreEnv.__init__",
                   side_effect=AssertionError("mission construction is outside steps 1-2")):
            result = resolved_task_contract(self.config)
        self.assertEqual(result["static_contract"]["agents"], 4)
        self.assertEqual(result["static_contract"]["residual_scale_executor"], 0.15)
        self.assertFalse(result["runtime_executed"])

    def test_network_inference_is_finite_and_read_only(self):
        model = SingleCriticNetworks(self.config, 2729)
        before = state_digest(model.state_dict())
        with torch.no_grad():
            obs = torch.zeros(2, 4, 28)
            actions = model.actions(obs)
            self.assertEqual(actions.shape, (2, 4, 3))
            values = model.values(obs, actions)
        self.assertTrue(bool(torch.isfinite(values).all()))
        self.assertEqual(values.shape, (2, 4))
        self.assertEqual(state_digest(model.state_dict()), before)
        with self.assertRaises(ValueError):
            model.actions(torch.zeros(2, 4, 29))
        with self.assertRaises(ValueError):
            model.values(obs, torch.full((2, 4, 3), 2.0))


if __name__ == "__main__":
    unittest.main()
