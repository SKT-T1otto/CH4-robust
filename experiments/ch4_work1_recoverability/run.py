"""Audit the frozen native baseline. Run with --execute; this entry point never trains."""
import argparse
import hashlib
import json
import pickle
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from chapter3_bser.learning.checkpoint import digest, state_digest
from chapter3_bser.learning.config import scenario_fingerprint, validate
from chapter3_bser.learning.evaluate import preserved_rng
from chapter3_bser.learning.reference import ZeroResidualPolicy
from chapter3_bser.learning.runtime import RepairedMissionRuntime
from chapter3_bser.learning.scenarios import validate_manifest
from chapter3_bser.learning.train import load_for_evaluation
from chapter4_interval_robust.provenance import identity as ch4_source_identity
from chapter4_interval_robust.scenarios import make_scene


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def audit_catalog(protocol):
    groups = protocol["scene_groups"]
    if set(groups) != set(protocol["scene_generation"]["usage"]):
        raise ValueError("missing or unknown scene group")
    old = groups["legacy_regression"]
    if old["start_seed"] != 2739500 or len(old["fingerprints"]) != 24:
        raise ValueError("old D2 development bank must remain regression-only")
    used_seeds, used_physical, report = set(), set(), {}
    for name, group in groups.items():
        fingerprints = group["fingerprints"]
        start = group["start_seed"]
        if type(start) is not int or start < 0 or not fingerprints:
            raise ValueError(f"invalid seed block: {name}")
        for index, fp in enumerate(fingerprints):
            if not isinstance(fp, str) or not re.fullmatch("[0-9a-f]{64}", fp):
                raise ValueError(f"invalid physical fingerprint: {name}/{index}")
            seed = start + index
            if seed in used_seeds or fp in used_physical:
                raise ValueError(f"duplicate seed or physical fingerprint: {name}/{index}")
            used_seeds.add(seed)
            used_physical.add(fp)
        report[name] = dict(count=len(fingerprints), start_seed=start, sha256=digest(group))
    generation = protocol["scene_generation"]
    if generation["replay_group"] != "calibration" or not 0 <= generation["replay_index"] < report["calibration"]["count"]:
        raise ValueError("step 1 replay must use a calibration scene")
    if protocol["b2_admission"]["calibration_scenes"] != report["calibration"]["count"]:
        raise ValueError("B2 must use the full, predeclared calibration bank")
    return report


def materialize(protocol, name, index):
    group = protocol["scene_groups"][name]
    if not 0 <= index < len(group["fingerprints"]):
        raise ValueError(f"scene index out of range: {name}/{index}")
    seed = group["start_seed"] + index
    scene = make_scene({"splits": {"dev": seed}, "max_steps": protocol["task"]["max_steps"]}, "dev", 0)
    if scene["scenario_seed"] != seed or scenario_fingerprint(scene) != group["fingerprints"][index]:
        raise ValueError(f"generated physical scene changed: {name}/{index}")
    # The native evaluator requires an eval role. Neither role nor ID is a physical fingerprint.
    scene.update(scenario_role="eval", scenario_split="eval")
    validate_manifest({"scenarios": [scene]}, protocol["task"], evaluation=True)
    return scene


def verify_geometry_block(protocol, block):
    name, start, stop = block
    for index in range(start, stop):
        materialize(protocol, name, index)


def check_contract(runtime, protocol):
    core = runtime.env.unwrapped
    public = runtime.env.get_public_control_contract()
    actual = {key: public[key] for key in ("dt", "residual_scale_search", "residual_scale_executor", "use_residual_prior")}
    actual.update(agents=runtime.env.n_agents, role_order=list(runtime.env.role_order),
                  observation_dim=runtime.config["observation_dim"], action_dim=runtime.config["action_dim"],
                  critic_input_dim=runtime.config.get("critic_input_dim", 124), gamma=runtime.config["rl"]["gamma"],
                  repairs=runtime.config["repairs"])
    if any(actual[key] != protocol["control"][key] for key in actual):
        raise ValueError(f"native control contract changed: {actual}")
    scales = [actual["residual_scale_search"]] * 3 + [actual["residual_scale_executor"]]
    if not torch.equal(core._residual_scale, core._residual_scale.new_tensor(scales)):
        raise ValueError("per-vehicle physical residual scales differ from the public contract")
    actual["per_vehicle_residual_scales"] = scales
    task = protocol["task"]
    if (np.shape(runtime.observations) != (actual["agents"], actual["observation_dim"])
            or runtime.config["profile"] != task["profile"] or core.max_steps != task["max_steps"]
            or core.failure_penalty_steps != task["failure_penalty_steps"]
            or runtime.env.reward_accounting.identity["reward_objective"] != "team_mean_v1"):
        raise ValueError("native observations, task horizon, failure cost or reward contract changed")
    return actual


def completion_cost(outcome, steps, task):
    reason = outcome["termination_reason"]
    if (reason not in ("success", "obstacle_collision", "timeout") or not outcome["terminated"]
            or outcome["truncated"] or outcome["terminal_step"] != steps or not 1 <= steps <= task["max_steps"]
            or outcome["success"] != (reason == "success")
            or (reason == "timeout" and steps != task["max_steps"])):
        raise ValueError("incomplete or inconsistent native episode outcome")
    return steps if outcome["success"] else task["max_steps"] + task["failure_penalty_steps"]


@torch.no_grad()
def replay(scene, config, policy, protocol, *, tap_command=False):
    """Two independent measurement paths for the sent command, with native physics in both."""
    runtime = RepairedMissionRuntime(config, scene, seed=scene["scenario_seed"])
    try:
        contract = check_contract(runtime, protocol)
        core = runtime.env.unwrapped
        limits = torch.stack((core._a_xy_max, core._a_xy_max, core._a_z_max), dim=1)
        captured = []
        if tap_command:
            original = core._record_residual_prior_diagnostics

            def record(prior, residual, final):
                captured.append(final.detach().clone())
                return original(prior, residual, final)

            core._record_residual_prior_diagnostics = record
        trace = []
        nonzero_steps = 0
        while not runtime.terminal:
            assignments = sorted(runtime.guidance.agent_assignments, key=lambda item: item.agent_id)
            expected_gates = [item.reachable and not item.hold_state and runtime.observations[item.agent_id][20] < 0.5
                              for item in assignments]
            row = runtime.advance(policy, explore=False)
            if tap_command:
                if len(captured) != 1:
                    raise RuntimeError("expected one physical command per mission step")
                sent = captured.pop()
            else:
                sent = torch.clamp(core._last_prior_acc + core._last_residual_acc, min=-limits, max=limits)
            residual = row["actions"]
            gates = row["action_enabled"]
            if (row["t"] != len(trace) or residual.shape != (4, 3) or not np.array_equal(expected_gates, gates)
                    or np.any(residual[~gates])
                    or not torch.isfinite(sent).all() or not np.isfinite(row["team_reward"])):
                raise ValueError(f"invalid physical step or gated action at {row['t']}")
            if isinstance(policy, ZeroResidualPolicy) and (np.any(row["proposed_actions"]) or np.any(residual)):
                raise ValueError("native D2 proposed or applied a nonzero learned residual")
            nonzero_steps += int(np.any(residual))
            trace.append(dict(t=row["t"], residual=residual.tolist(), gates=gates.tolist(),
                              sent_acceleration=sent.tolist(), team_reward=row["team_reward"],
                              observation_sha256=hashlib.sha256(np.asarray(row["next_observations"], dtype="<f4").tobytes()).hexdigest()))
        outcome = runtime.env.get_episode_result()
        steps = len(trace)
        total = sum(contract["gamma"] ** row["t"] * row["team_reward"] for row in trace)
        if total != runtime.env.reward_accounting.team_discounted_return:
            raise ValueError("discounted return differs from native accounting")
        summary = dict(steps=steps, termination_reason=outcome["termination_reason"], success=outcome["success"],
                       penalized_completion_steps=completion_cost(outcome, steps, protocol["task"]),
                       team_discounted_return=total, nonzero_residual_steps=nonzero_steps,
                       nonzero_physical_command_steps=sum(bool(np.any(row["sent_acceleration"])) for row in trace))
        return trace, summary, contract
    finally:
        runtime.close()


def compare_replays(first, second):
    left, right = first[0], second[0]
    if not left or len(left) != len(right):
        raise ValueError("empty replay or different trajectory lengths")
    for a, b in zip(left, right):
        if a != b:
            fields = [key for key in a if a[key] != b[key]]
            raise ValueError(f"native replays differ at step {a['t']}: {fields}")
    if first[1:] != second[1:]:
        raise ValueError("replay outcome, return or control contract differs")
    if not first[1]["nonzero_physical_command_steps"]:
        raise ValueError("empty physical-command signal in nominal replay")
    left_sha, right_sha = digest(left), digest(right)
    if left_sha != right_sha:
        raise ValueError("reconstructed and observed command trace digests differ")
    return dict(**first[1], trace_sha256=left_sha, observed_trace_sha256=right_sha, repeats=2, per_step_exact=True,
                contract=first[2], evaluation_updates=0)


def check_b2_inputs(config, native, payload, groups):
    # Training seed, optimizer settings and output paths may differ. Execute the frozen native task.
    conditions = ("baseline", "method", "algorithm", "architecture_version", "base_candidate", "profile",
                  "max_steps", "observation_dim", "action_dim", "planner_protocol", "repairs", "reward",
                  "reward_objective", "source_reward_revision", "task_protocol", "collision_detection_revision",
                  "terminal_reward_revision", "collision_terminal_reward", "waypoint_reward_timing",
                  "execution_runtime_revision", "execution_runtime")
    if ("map_experiment" in config or any(config.get(key) != native[key] for key in conditions)
            or config["rl"]["gamma"] != native["rl"]["gamma"] or config.get("critic_input_dim", 124) != 124
            or payload["optimizer_updates"] <= 0):
        raise ValueError("checkpoint is not a trained, task-matched plain B2")
    ids, seeds, fingerprints = (set(payload["training_scenario_" + key]) for key in ("ids", "seeds", "fingerprints"))
    if not ids or not seeds or not fingerprints:
        raise ValueError("checkpoint has incomplete training scene identities")
    for name in ("calibration", "development", "confirmation_reserve"):
        group = groups[name]
        for index, fp in enumerate(group["fingerprints"]):
            seed = group["start_seed"] + index
            if seed in seeds or fp in fingerprints or f"ch4_dev_{seed}" in ids:
                raise ValueError("checkpoint training overlaps the registered physical scene bank")


def evaluate_b2(path, protocol, native):
    if path is None:
        return {"status": "not_supplied"}
    if not path.is_file():
        return {"status": "not_admitted", "reason": "checkpoint file missing"}
    info = {"checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    try:
        model, config, payload = load_for_evaluation(path, allow_mechanism=False)
        check_b2_inputs(config, native, payload, protocol["scene_groups"])
    except (ValueError, OSError, KeyError, TypeError, AttributeError, EOFError, pickle.UnpicklingError, RuntimeError) as exc:
        return dict(info, status="not_admitted", reason=str(exc))
    before = state_digest(model.training_state_dict())
    if before != payload["model_state_sha256"]:
        raise ValueError("loaded B2 policy state differs from verified checkpoint")
    criteria = protocol["b2_admission"]
    episodes = []
    for index in range(criteria["calibration_scenes"]):
        scene = materialize(protocol, "calibration", index)
        _, row, _ = replay(scene, native, model, protocol)
        episodes.append(dict(scenario_seed=scene["scenario_seed"], **row))
    if before != state_digest(model.training_state_dict()) or model.niter != payload["optimizer_updates"]:
        raise ValueError("frozen B2 changed during nominal admission")
    successes = sum(row["success"] for row in episodes)
    applied = sum(row["nonzero_residual_steps"] for row in episodes)
    admitted = successes >= criteria["minimum_successes"] and applied >= criteria["minimum_nonzero_applied_steps"]
    return dict(info, status="admitted" if admitted else "nominal_performance_insufficient",
                successes=successes, nonzero_residual_steps=applied, episodes=episodes, evaluation_updates=0)


@preserved_rng()
def execute(protocol, checkpoint=None, *, verify_fingerprints=False):
    catalog = audit_catalog(protocol)
    source = ch4_source_identity()
    native = validate(read_json(ROOT / protocol["nominal_references"]["native_ch3"]["config"]))
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        if verify_fingerprints:
            blocks = [(name, i, min(i + 8, entry["count"])) for name, entry in catalog.items()
                      for i in range(0, entry["count"], 8)]
            with ProcessPoolExecutor(max_workers=4) as pool:
                list(pool.map(verify_geometry_block, repeat(protocol), blocks))
        scene = materialize(protocol, "calibration", protocol["scene_generation"]["replay_index"])
        first = replay(scene, native, ZeroResidualPolicy(), protocol)
        second = replay(scene, native, ZeroResidualPolicy(), protocol, tap_command=True)
        d2 = compare_replays(first, second)
        d2.update(scenario_seed=scene["scenario_seed"], physical_fingerprint=scenario_fingerprint(scene))
        anchor = protocol["nominal_references"]["native_ch3"]["active_reference_replay"]
        d2["registered_active_reference"] = dict(
            project=anchor["development_project"], worktree_sha256=anchor["worktree_source_sha256"],
            matches=(anchor["calibration_seed"] == d2["scenario_seed"]
                     and anchor["physical_fingerprint"] == d2["physical_fingerprint"]
                     and anchor["trace_sha256"] == d2["trace_sha256"]))
        b2 = evaluate_b2(checkpoint, protocol, native)
    finally:
        torch.set_num_threads(previous_threads)
    files = {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
             for name in ("protocol.json", "run.py", "test_work1.py")}
    return dict(step1_passed=True, files_sha256=files, frozen_source=source, scene_groups=catalog,
                fingerprints_regenerated_in_this_run=verify_fingerprints, confirmation_rollouts=0,
                relative_water_rollouts=0, evaluation_updates=0, d2=d2, b2=b2)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--checkpoint", type=Path, help="optional existing frozen B2 candidate")
    parser.add_argument("--verify-fingerprints", action="store_true", help="also regenerate every registered geometry")
    parser.add_argument("--report-name", help="save results/NAME.json (create once)")
    args = parser.parse_args()
    protocol = read_json(HERE / "protocol.json")
    destination = None
    if args.report_name is not None:
        if not re.fullmatch("[A-Za-z0-9_-]+", args.report_name):
            parser.error("report name must be a simple file stem")
        destination = HERE / "results" / (args.report_name + ".json")
        if args.execute and destination.exists():
            raise FileExistsError(destination)
    if args.execute:
        result = execute(protocol, args.checkpoint, verify_fingerprints=args.verify_fingerprints)
    else:
        result = dict(executed=False, scene_groups=audit_catalog(protocol), scope=protocol["scope"])
    output = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.execute and destination is not None:
        destination.parent.mkdir(exist_ok=True)
        with destination.open("x", encoding="utf-8") as stream:
            stream.write(output + "\n")
    print(output)


if __name__ == "__main__":
    main()
