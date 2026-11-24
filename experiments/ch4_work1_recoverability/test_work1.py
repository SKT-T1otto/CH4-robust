"""Run with python -B -m unittest discover -s experiments/ch4_work1_recoverability."""
import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE.parents[1]))
from experiments.ch4_work1_recoverability import run


class WorkOneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = run.read_json(HERE / "protocol.json")
        cls.native = run.validate(run.read_json(run.ROOT / cls.plan["nominal_references"]["native_ch3"]["config"]))

    def test_disjoint_banks_and_reserved_confirmation(self):
        catalog = run.audit_catalog(self.plan)
        self.assertEqual({name: group["count"] for name, group in catalog.items()}, {
            "legacy_calibration": 8, "legacy_regression": 24, "calibration": 8,
            "development": 24, "confirmation_reserve": 64})
        for field in ("start_seed", "fingerprints"):
            with self.subTest(field=field):
                plan = copy.deepcopy(self.plan)
                dev, holdout = (plan["scene_groups"][name] for name in ("development", "confirmation_reserve"))
                if field == "start_seed":
                    holdout[field] = dev[field] + 3
                else:
                    holdout[field][-1] = dev[field][0]
                with self.assertRaisesRegex(ValueError, "duplicate"):
                    run.audit_catalog(plan)
        plan = copy.deepcopy(self.plan)
        plan["scene_generation"]["replay_group"] = "confirmation_reserve"
        with self.assertRaisesRegex(ValueError, "calibration"):
            run.audit_catalog(plan)

    def test_materialized_fingerprint_ignores_labels_but_detects_physical_drift(self):
        scene = run.materialize(self.plan, "legacy_regression", 0)
        physical = self.plan["scene_groups"]["legacy_regression"]["fingerprints"][0]
        self.assertEqual(run.scenario_fingerprint(dict(scene, scenario_id="renamed", scenario_role="test")), physical)
        altered = copy.deepcopy(scene)
        altered["initial_agent_positions"][0][0] += 0.25
        with patch.object(run, "make_scene", return_value=altered):
            with self.assertRaisesRegex(ValueError, "physical scene changed"):
                run.materialize(self.plan, "legacy_regression", 0)

    def test_both_nominal_references(self):
        refs = self.plan["nominal_references"]
        self.assertFalse(refs["native_ch3"]["plant_override"])
        anchor = refs["native_ch3"]["active_reference_replay"]
        calibration = self.plan["scene_groups"]["calibration"]
        index = self.plan["scene_generation"]["replay_index"]
        self.assertEqual(anchor["development_project"], "AUV-Thesis-Active")
        self.assertEqual(anchor["calibration_seed"], calibration["start_seed"] + index)
        self.assertEqual(anchor["physical_fingerprint"], calibration["fingerprints"][index])
        self.assertEqual(refs["relative_water"]["law"], "linear")
        for name, value in {"beta_m_per_s": 0.1, "drag_multiplier": 1, "horizontal_thrust_multiplier": 1,
                            "propulsion_lag": 0.15, "actuator_delay_steps": 0, "estimator_delay_steps": 0}.items():
            self.assertEqual(refs["relative_water"][name], value)
        self.assertEqual(self.plan["future_physical_design"]["varied_continuous_coordinates"], 9)
        self.assertEqual(self.native["repairs"], self.plan["control"]["repairs"])

    def test_completion_cost_includes_failure_and_requires_terminal(self):
        task = self.plan["task"]
        for reason, length, expected in [("success", 120, 120), ("obstacle_collision", 12, 500), ("timeout", 400, 500)]:
            row = dict(termination_reason=reason, terminal_step=length, success=reason == "success",
                       terminated=True, truncated=False)
            self.assertEqual(run.completion_cost(row, length, task), expected)
            with self.assertRaises(ValueError):
                run.completion_cost(dict(row, terminated=False), length, task)
        with self.assertRaises(ValueError):
            run.completion_cost(dict(row, terminal_step=30), 30, task)

    def test_stepwise_audit_rejects_command_reward_and_terminal_drift(self):
        a = ([dict(t=0, sent_acceleration=[[0.25]], team_reward=0.1)],
             dict(nonzero_physical_command_steps=1, termination_reason="success"), {})
        self.assertTrue(run.compare_replays(a, copy.deepcopy(a))["per_step_exact"])
        for index, key, value in [(0, "sent_acceleration", [[0.251]]), (0, "team_reward", 0.1000000001),
                                  (1, "termination_reason", "timeout")]:
            with self.subTest(key=key):
                b = copy.deepcopy(a)
                record = b[0][0] if index == 0 else b[1]
                record[key] = value
                with self.assertRaisesRegex(ValueError, "differ"):
                    run.compare_replays(a, b)

    def checkpoint_payload(self):
        return dict(optimizer_updates=1, training_scenario_ids=["training_case_1"], training_scenario_seeds=[42],
                    training_scenario_fingerprints=["a" * 64], model_state_sha256=run.state_digest({}))

    def test_b2_overlap_and_task_mismatch(self):
        groups = self.plan["scene_groups"]
        payload = self.checkpoint_payload()
        run.check_b2_inputs(self.native, self.native, payload, groups)
        # Old regression-only geometry is never a B2 evaluation case in this protocol.
        historical = dict(payload, training_scenario_seeds=[groups["legacy_regression"]["start_seed"]])
        run.check_b2_inputs(self.native, self.native, historical, groups)
        for key, value in [("ids", "ch4_dev_4436000"), ("seeds", 4436000),
                           ("fingerprints", groups["confirmation_reserve"]["fingerprints"][0])]:
            with self.subTest(key=key):
                bad = dict(payload, **{"training_scenario_" + key: [value]})
                with self.assertRaisesRegex(ValueError, "overlaps"):
                    run.check_b2_inputs(self.native, self.native, bad, groups)
        for config in [dict(self.native, max_steps=32), dict(self.native, map_experiment={}),
                       dict(self.native, algorithm="sac")]:
            with self.assertRaisesRegex(ValueError, "task-matched plain B2"):
                run.check_b2_inputs(config, self.native, payload, groups)

    def test_rejected_native_loader_never_runs_b2(self):
        with patch.object(run, "load_for_evaluation", side_effect=ValueError("source mismatch")) as loader, \
                patch.object(run, "replay") as replay:
            candidate = HERE / "protocol.json"
            result = run.evaluate_b2(candidate, self.plan, self.native)
            loader.assert_called_once_with(candidate, allow_mechanism=False)
            replay.assert_not_called()
        self.assertEqual(result["status"], "not_admitted")
        self.assertEqual(result["reason"], "source mismatch")

    def test_b2_requires_measurable_performance_and_frozen_state(self):
        model = Mock(niter=1)
        model.training_state_dict.return_value = {}
        for successes, nonzero, status in [(1, 8, "nominal_performance_insufficient"),
                                           (2, 0, "nominal_performance_insufficient"), (2, 1, "admitted")]:
            with self.subTest(successes=successes, nonzero=nonzero):
                rows = [([], dict(success=i < successes, nonzero_residual_steps=int(i < nonzero)), {})
                        for i in range(8)]
                with patch.object(run, "load_for_evaluation", return_value=(model, self.native, self.checkpoint_payload())), \
                        patch.object(run, "materialize", side_effect=lambda p, g, i: {"scenario_seed": i}) as scenes, \
                        patch.object(run, "replay", side_effect=rows):
                    result = run.evaluate_b2(HERE / "protocol.json", self.plan, self.native)
                self.assertEqual(result["status"], status)
                self.assertEqual(result["evaluation_updates"], 0)
                self.assertEqual([call.args[1:] for call in scenes.call_args_list], [("calibration", i) for i in range(8)])
        model.training_state_dict.side_effect = [{}, {"changed": 1}]
        with patch.object(run, "load_for_evaluation", return_value=(model, self.native, self.checkpoint_payload())), \
                patch.object(run, "materialize", return_value={"scenario_seed": 1}), \
                patch.object(run, "replay", return_value=([], {"success": True, "nonzero_residual_steps": 1}, {})):
            with self.assertRaisesRegex(ValueError, "changed during"):
                run.evaluate_b2(HERE / "protocol.json", self.plan, self.native)


if __name__ == "__main__":
    unittest.main()
