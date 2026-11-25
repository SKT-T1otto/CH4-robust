"""Run with python -B -m unittest discover -s experiments/ch4_work1_recoverability."""
import copy
import argparse
import json
import random
import re
import sys
import unittest
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE.parents[1]))
from experiments.ch4_work1_recoverability import run
from experiments.ch4_work1_recoverability import records
import numpy as np
import torch
from experiments.ch4_work1_recoverability.plant_adapter import MIXED, RelativeWaterPlant, WaterParameters


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


# Full D2 trajectories regenerated with the archived round01 implementation. Its
# saved decision-state hashes and terminal records were checked before pinning.
LEGACY_TRACES = {
    8: ("linear", 90, "obstacle_collision", "fe531e5851e1ea092eccab9b5d00b5679df9de69bd663399b2e185e1b95ce77a",
        "a0efec9ea10c12105e1ad2dcfcc372b11d53eb3ef333ad761224e8f5b1895bd7"),
    23: (MIXED, 142, "success", "2ce568b44704812ca13a4b41a3d4aff61f0d770fdf00f3837a6e5a6a7c7582c2",
         "3c0ad130f0b9332c94a0a5dfd6e467560c685c26df7ec5ead250855cf7257d36"),
}
TOL = 2e-6


@contextmanager
def mission(config, scene, index, params):
    seed = scene["scenario_seed"]
    rt = run.RepairedMissionRuntime(config, scene, seed=seed, episode_id=index)
    core = rt.env.unwrapped
    original = core._choose_next_search_waypoint

    def paired_waypoint(agent_id, reserved_positions=None):
        with torch.random.fork_rng(devices=[]):
            key = [seed, core.step_count, int(agent_id), 49032]
            torch.random.default_generator.manual_seed(int(np.random.SeedSequence(key).generate_state(1)[0]))
            return original(agent_id, reserved_positions)

    try:
        with patch.object(core, "_choose_next_search_waypoint", paired_waypoint), RelativeWaterPlant(core, params) as plant:
            yield rt, plant
    finally:
        rt.close()


def array(value):
    return np.asarray(value, dtype=np.float64).copy()


def physical_step(rt, plant, previous_lag):
    """Independent float64 oracle, reconstructed from the original plant specs and issued commands."""
    c = rt.env.unwrapped
    seed = rt.scenario["scenario_seed"]
    innovations = np.random.SeedSequence([seed, rt.step, 49031]).generate_state(4)
    random.seed(int(innovations[0]))
    np.random.seed(int(innovations[1]))
    torch.random.default_generator.manual_seed(int(innovations[2]))
    pos, vel = array(c._agent_pos), array(c._agent_vel)
    active = ~np.asarray(c.agent_finished).copy()
    row = rt.advance(run.ZeroResidualPolicy(), explore=False)
    post_pos, post_vel = array(c._agent_pos), array(c._agent_vel)
    params = plant.parameters
    dt = c.dt
    limits = np.array([[spec["a_xy_max"], spec["a_xy_max"], spec["a_z_max"]] for spec in c.agent_specs])
    # Use the core's actual pre-filter limit precision for the normalized command.
    limits = limits.astype(np.float32).astype(np.float64)
    command = array(plant.command)
    nominal = np.asarray(c._last_prior_acc) + np.asarray(c._last_residual_acc)
    np.testing.assert_array_equal(command, np.clip(nominal, -limits, limits))
    np.testing.assert_array_equal(row["actions"], np.zeros((4, 3)))
    lag = .15 * previous_lag + .85 * command / limits
    thrust = limits * lag
    thrust[:, :2] *= np.asarray(params.thrust)[:, None]
    np.testing.assert_allclose(array(plant.lag), lag, atol=3e-7, rtol=0)
    np.testing.assert_allclose(array(plant.thrust_acc), thrust, atol=5e-7, rtol=0)
    water = params.beta * np.column_stack((np.sin(pos[:, 1] / 4.), np.cos(.22 * pos[:, 0])))
    np.testing.assert_allclose(array(plant.water), water, atol=1e-7, rtol=0)
    vplus = vel + dt * thrust
    relative = vplus[:, :2] - water
    factor = 1. if params.law == "linear" else .5 + .5 * np.linalg.norm(relative, axis=1, keepdims=True) / .6925365924835205
    base_drag = np.array([spec["drag_xy"] for spec in c.agent_specs])
    drag = -(base_drag * params.drag)[:, None] * relative * factor
    np.testing.assert_allclose(array(plant.drag_acc), drag, atol=TOL, rtol=0)
    predicted = vplus.copy()
    predicted[:, :2] += dt * drag
    predicted[:, 2] += dt * np.array([s["buoyancy_bias"] - s["drag_z"] * v for s, v in zip(c.agent_specs, vplus[:, 2])])
    interior = np.minimum(post_pos, array(c.space_size) - post_pos).min(axis=1) > 1e-5
    free = active & ~np.asarray(c.agent_finished) & ~np.asarray(c._collision_flags) & interior
    free_xy = free & (np.linalg.norm(post_vel[:, :2], axis=1) < array(c._v_xy_max) - 1e-5)
    free_z = free & (np.abs(post_vel[:, 2]) < array(c._v_z_max) - 1e-5)
    xy_error = float(np.max(np.abs(predicted[free_xy, :2] - post_vel[free_xy, :2]), initial=0.))
    z_error = float(np.max(np.abs(predicted[free_z, 2] - post_vel[free_z, 2]), initial=0.))
    if max(xy_error, z_error) > TOL:
        raise AssertionError(f"free velocity differs at step {row['t']}: xy={xy_error}, z={z_error}")
    signature = run.digest([row["t"], post_pos.tolist(), post_vel.tolist(), plant.sent.tolist(), plant.lag.tolist(),
                            row["actions"].tolist(), row["action_enabled"].tolist(), row["team_reward"],
                            row["rewards"].tolist(), row["dones"]])
    record = dict(t=row["t"], position_m=pos.tolist(), velocity_before_mps=vel.tolist(),
                  command_mps2=command.tolist(), filtered_normalized=plant.lag.tolist(),
                  thrust_mps2=plant.thrust_acc.tolist(), water_mps=plant.water.tolist(), drag_xy_mps2=plant.drag_acc.tolist(),
                  realized_acceleration_mps2=((post_vel - vel) / dt).tolist(), velocity_after_mps=post_vel.tolist(),
                  team_reward=row["team_reward"], free_xy=free_xy.tolist(), free_z=free_z.tolist(),
                  xy_error_mps=xy_error, z_error_mps=z_error)
    return record, signature, lag


def nominal_regression(protocol, config, index):
    law, steps, reason, trace_hash, terminal_hash = LEGACY_TRACES[index]
    scene = run.materialize(protocol, "legacy_regression", index)
    records, signatures = [], []
    lag = np.zeros((4, 3))
    with mission(config, scene, index, WaterParameters(law=law)) as (rt, plant):
        while not rt.terminal:
            record, signature, lag = physical_step(rt, plant, lag)
            records.append(record)
            signatures.append(signature)
        outcome = rt.env.get_episode_result()
        if (len(records), outcome["termination_reason"], run.digest(signatures), run.digest(outcome)) != (
                steps, reason, trace_hash, terminal_hash):
            raise AssertionError(f"nominal run no longer reproduces round01 scene {index}")
        if not any(any(r["free_xy"]) for r in records) or not any(any(r["free_z"]) for r in records):
            raise AssertionError("no free integration evidence")
        return dict(index=index, law=law, steps=steps, termination=reason, trace_sha256=trace_hash,
                    discounted_return=rt.env.reward_accounting.team_discounted_return, records=records)


def parameter_cases(law):
    yield "nominal", WaterParameters(law=law)
    for beta in (.05, .15):
        yield f"public_{beta}", WaterParameters(beta=beta, law=law)
    for key, levels in (("drag", (.85, 1.20)), ("thrust", (.85, 1.10))):
        for level in levels:
            yield f"{key}_{level}", WaterParameters(law=law, **{key: (level, 1., 1., 1.)})
    for beta in (.05, .15):
        for rotation in range(4):
            drag = tuple(np.roll([.85, .85, 1.20, 1.20], rotation))
            thrust = tuple(np.roll([.85, 1.10, .85, 1.10], rotation))
            yield f"joint_{beta}_{rotation}", WaterParameters(beta, drag, thrust, law)


def probe(config, scene, params, anchor=None, poison_flow=False):
    """Reconstruct a common linear-nominal prefix before changing the plant for one D2 step."""
    signatures = []
    lag = np.zeros((4, 3))
    with mission(config, scene, 0, WaterParameters()) as (rt, plant):
        while not rt.terminal and rt.step < 32:
            t = rt.step
            if t == anchor:
                plant.set_parameters(params)
                if poison_flow:
                    rt.env.unwrapped.flow_gain = 1234.
            row, signature, lag = physical_step(rt, plant, lag)
            if anchor is None:
                # Common selection rule uses only the nominal calibration trajectory.
                excited = np.linalg.norm(np.asarray(row["thrust_mps2"])[:, :2], axis=1)
                selected = t > 0 and all(row["free_xy"]) and min(excited) > .01
            else:
                selected = t == anchor
            if selected:
                if not all(row["free_xy"]):
                    raise AssertionError("probe hit a constraint before direct response could be measured")
                return run.digest(signatures), row
            signatures.append(signature)
    raise AssertionError("no fully excited four-vessel anchor in the calibration prefix")


def response_experiment(protocol, config):
    scene = run.materialize(protocol, "calibration", 0)
    initial_hash, initial = probe(config, scene, WaterParameters())
    anchor = initial["t"]
    records = []
    for law in ("linear", MIXED):
        for name, params in parameter_cases(law):
            prefix, row = probe(config, scene, params, anchor)
            if prefix != initial_hash:
                raise AssertionError("pre-intervention prefix differs")
            for key in ("position_m", "velocity_before_mps", "command_mps2", "filtered_normalized"):
                np.testing.assert_array_equal(row[key], initial[key])
            if name == "nominal":
                baseline = np.asarray(row["velocity_after_mps"])
            velocity = np.asarray(row["velocity_after_mps"])
            delta = velocity - baseline
            np.testing.assert_array_equal(delta[:, 2], np.zeros(4))
            norms = np.linalg.norm(delta[:, :2], axis=1)
            if name.startswith(("drag_", "thrust_")):
                if norms[0] <= TOL or np.any(norms[1:]):
                    raise AssertionError(f"individual intervention not isolated: {name}, {norms}")
            if name.startswith("public_") and (norms <= TOL).any():
                raise AssertionError("missing public-flow response on one or more vessels")
            if name.startswith("joint_") and norms.max() <= TOL:
                raise AssertionError("joint shift had no measurable effect")
            records.append(dict(law=law, case=name, parameters=asdict(params), delta_xy_norm_mps=norms.tolist(), **row))
    _, poisoned = probe(config, scene, WaterParameters(), anchor, poison_flow=True)
    if poisoned != initial:
        raise AssertionError("old acceleration field leaked into the relative-water model")
    return dict(seed=scene["scenario_seed"], anchor_step=anchor, prefix_sha256=initial_hash,
                focal_vehicle_index=0, legacy_flow_gain_poison_exact=True, records=records)


class PlantAdapterTests(unittest.TestCase):
    def test_truth_inputs_are_checked_and_copied(self):
        original = [1.] * 4
        params = WaterParameters(thrust=original)
        original[0] = .9
        self.assertEqual(params.thrust, (1.,) * 4)
        for bad in (dict(beta=float("nan")), dict(drag=(1.,)), dict(thrust=(1., 1., 1., 1.11)), dict(law="g")):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                WaterParameters(**bad)

    def test_joint_design_covers_private_factors_on_every_hull(self):
        for law in ("linear", MIXED):
            joint = [p for n, p in parameter_cases(law) if n.startswith("joint")]
            for beta in (.05, .15):
                for i in range(4):
                    self.assertEqual({(p.drag[i], p.thrust[i]) for p in joint if p.beta == beta},
                                     {(.85, .85), (.85, 1.10), (1.20, .85), (1.20, 1.10)})

    def test_command_lag_saturation_one_thrust_gain_and_detachment(self):
        class Core:
            step_count = _flow_phase_x = _flow_phase_y = 0
            use_residual_prior = True
            episode_outcome = object()
            _a_xy_max = _a_z_max = torch.ones(4)
            _drag_xy = torch.ones(4)

            def _record_residual_prior_diagnostics(self, *args):
                pass

            def _flow_at(self, position):
                raise AssertionError("native flow should have been replaced")

        c = Core()
        native_drag = c._drag_xy
        with RelativeWaterPlant(c, WaterParameters(thrust=(1.10, 1., 1., 1.))) as p:
            with self.assertRaises(RuntimeError):
                with RelativeWaterPlant(c, WaterParameters()):
                    pass
            for expected in (.85, .9775):
                final = torch.ones(4, 3)
                c._record_residual_prior_diagnostics(final, final, final)
                np.testing.assert_allclose(p.lag, expected, atol=1e-7)
                np.testing.assert_allclose(final[0], [expected * 1.10, expected * 1.10, expected], atol=1e-7)
            c.episode_outcome = object()
            with self.assertRaises(RuntimeError):
                c._record_residual_prior_diagnostics(final, final, final)
        self.assertIs(c._drag_xy, native_drag)
        self.assertNotIn("_flow_at", vars(c))
        self.assertNotIn("_record_residual_prior_diagnostics", vars(c))


class StageRecordsTests(unittest.TestCase):
    def test_approach_uses_tracking_geometry_and_positive_closing_speed(self):
        rule = run.read_json(HERE / "protocol.json")["task"]["approach_rule"]
        for point, velocity, expected in [([.75, 0, 0], [1, 0, 0], False),
                                          ([2.4, 0, 0], [.5, 0, 0], True),
                                          ([1, 0, 2], [1, 0, 0], False),
                                          ([1, 0, 0], [-1, 0, 0], False)]:
            self.assertEqual(records.approaching(np.zeros(3), velocity, point, 2., rule), expected)
        self.assertIsNone(records.phase_for(3, True, False))
        self.assertEqual(records.phase_for(3, True, True), records.EXECUTOR)
        self.assertIsNone(records.phase_for(0, True, True))

    def test_no_window_scenes_stay_in_denominator_and_ties_are_causal(self):
        counts = records.phase_counts()
        hulls = [dict(phase=records.SEARCH, vehicle=i, qualifies=i < 2, controllable=True) for i in (2, 1, 0)]
        self.assertEqual(records.count_before({"hulls": hulls}, counts), {records.SEARCH: 0})
        counts[records.SEARCH]["window"] = {"t": 6, "vehicle": 0}
        self.assertEqual(records.count_before({"hulls": hulls}, counts), {})
        summary = records.aggregate([{"phase_counts": counts}, {"phase_counts": records.phase_counts()}], records.SEARCH)
        self.assertEqual((summary["entered_scenes"], summary["window_scenes"], summary["no_window_scenes"]), (1, 1, 1))
        self.assertEqual((summary["controllable_boat_steps"], summary["controllable_task_steps"]), (6, 2))

    def test_receiver_and_time_restrict_real_target_receipt(self):
        receipt = dict(physical_step=8, received_step=7, payload=dict(sample_step=6, position=[1, 2, 3]))
        self.assertIsNone(records.delivered_view(receipt, 8, 10, 0, 1))
        self.assertIsNone(records.delivered_view(None, 8, 10, 0, 3))
        actual = records.delivered_view(receipt, 8, 10, 0, 3)
        self.assertEqual(actual["kind"], "target_handoff")
        self.assertEqual(actual["id"], records.delivered_view(dict(receipt, physical_step=9), 9, 10, 0, 3)["id"])
        actual["payload"]["position"][0] = -2
        self.assertEqual(receipt["payload"]["position"][0], 1)
        with self.assertRaises(ValueError):
            records.delivered_view(dict(receipt, received_step=9), 8, 10, 0, 3)


def audit_plant():
    parser = argparse.ArgumentParser(description="Run the D2 N0 parameterized-plant acceptance experiment.")
    parser.add_argument("--execute", action="store_true", required=True)
    parser.add_argument("--report-name", required=True)
    args = parser.parse_args()
    if not re.fullmatch("step2_[A-Za-z0-9_-]+", args.report_name):
        parser.error("expected a fresh step2_ report stem")
    dest = HERE / "results" / (args.report_name + ".json")
    if dest.exists():
        raise FileExistsError(dest)
    protocol = run.read_json(HERE / "protocol.json")
    source = run.ch4_source_identity()
    run.audit_catalog(protocol)
    config = run.validate(run.read_json(run.ROOT / "configs/b2.json"))
    torch.set_num_threads(1)
    with torch.no_grad(), run.preserved_rng():
        legacy = [nominal_regression(protocol, config, i) for i in LEGACY_TRACES]
        probes = response_experiment(protocol, config)
    rows = [r for s in legacy for r in s["records"]] + probes["records"]
    summary = dict(free_xy_vehicle_steps=sum(sum(r["free_xy"]) for r in rows),
                   maximum_free_xy_error_mps=max(r["xy_error_mps"] for r in rows),
                   maximum_free_z_error_mps=max(r["z_error_mps"] for r in rows),
                   parameter_cases=len(probes["records"]), tolerance_mps=TOL)
    files = {f: run.hashlib.sha256((HERE / f).read_bytes()).hexdigest()
             for f in ("protocol.json", "run.py", "plant_adapter.py", "test_work1.py")}
    report = dict(passed=True, training_updates=0, confirmation_rollouts=0, noise="N0",
                  source=source, files_sha256=files, summary=summary, nominal=legacy, interventions=probes)
    dest.parent.mkdir(exist_ok=True)
    with dest.open("x", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    print(json.dumps(summary))


if __name__ == "__main__":
    if "--execute" in sys.argv:
        audit_plant()
    else:
        unittest.main()
