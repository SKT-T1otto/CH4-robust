"""Run the registered eight read-only runtime validation episodes to termination."""
import argparse
import copy
import gzip
import json
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone

import numpy as np
import torch

from chapter3_bser.learning.checkpoint import digest
from chapter4_interval_robust.provenance import write_new
from experiments.ch4_work1_recoverability import records
from experiments.ch4_work1_recoverability.run import materialize

from .config import DEFAULT_CONFIG, ROOT, file_sha, load, local_path, read_json
from .provenance import append_source, imported_local_files, verify_source
from .replay import NominalReplay
from .runtime import FrozenInitialPolicy, NominalRuntime, ZeroResidualPolicy, json_transition
from .scenarios import validate_registry


PROTOCOL = ROOT/"configs/ch4/nominal_maddpg_runtime.json"


def load_protocol(path=PROTOCOL):
    p = read_json(path)
    c = load()
    if (p["schema"] != "auv.ch4.nominal_maddpg.runtime_protocol.v1" or p["stage"] != 3
            or p["training_enabled"] is not False or p["optimizer_updates_allowed"] != 0
            or p["replay_capacity"] != 512 or len(p["cases"]) != 8):
        raise ValueError("expected the fixed eight-case, zero-update runtime protocol")
    for relative, sha in p["input_files"].items():
        if file_sha(local_path(relative)) != sha:
            raise ValueError("runtime validation reference changed: " + relative)
    registry = read_json(local_path(c["scene_manifest"]))
    validate_registry(c, registry, DEFAULT_CONFIG)
    names = set()
    expected = [("zero_timeout_a", "calibration", 0, None), ("zero_timeout_b", "calibration", 0, None),
                ("zero_collision", "calibration", 3, None), ("zero_success", "calibration", 5, None),
                ("frozen_calibration_a", "calibration", 5, 2729), ("frozen_calibration_b", "calibration", 5, 2729),
                ("frozen_validation_a", "validation", 0, 3733), ("frozen_validation_b", "validation", 0, 3733)]
    for case, spec in zip(p["cases"], expected):
        if (tuple(case[k] for k in ("name", "group", "index", "policy_seed")) != spec
                or case["name"] in names or type(case["stress_sampling"]) is not bool
                or case["stress_sampling"] != case["name"].endswith("_a")):
            raise ValueError("runtime fixtures differ from the registered calibration/validation cases")
        if case["policy_seed"] is None:
            expected_log = ("experiments/ch4_work1_recoverability/results/step3_d2_20261124/"
                            f"calibration_{case['index']:02d}.jsonl.gz")
            if case.get("reference_log") != expected_log or expected_log not in p["input_files"]:
                raise ValueError("archived zero residual fixture differs")
        names.add(case["name"])
    if p["repeat_pairs"] != [["zero_timeout_a", "zero_timeout_b"],
            ["frozen_calibration_a", "frozen_calibration_b"], ["frozen_validation_a", "frozen_validation_b"]]:
        raise ValueError("runtime repeat pairs changed")
    return p, c, registry


def scene_for(case, c, registry):
    if case["group"] == "calibration":
        old = read_json(ROOT/"experiments/ch4_work1_recoverability/protocol.json")
        return materialize(old, "calibration", case["index"])
    return copy.deepcopy(registry["groups"][case["group"]][case["index"]]["scene"])


def global_rng():
    return records.stamp((random.getstate(), np.random.get_state(), torch.get_rng_state()))


def replay_digest(replay):
    used = len(replay)
    data = [[b[:used] for b in buffers] for buffers in
            (replay.obs_buffs, replay.ac_buffs, replay.rew_buffs, replay.next_obs_buffs, replay.done_buffs)]
    return records.stamp([data, replay.enabled[:used], replay.next_enabled[:used],
                          replay.success_buffs[:used], replay.next_idx, used])


def run_case(job):
    case, p, c, registry, folder, source_sha = job
    torch.set_num_threads(1)
    source = verify_source()
    if source["sha256"] != source_sha:
        raise ValueError("runtime source differs from the execution registration")
    scene = scene_for(case, c, registry)
    policy = ZeroResidualPolicy() if case["policy_seed"] is None else FrozenInitialPolicy(c, case["policy_seed"])
    policy_id = "D2_zero_residual" if case["policy_seed"] is None else policy.policy_id
    replay = NominalReplay(p["replay_capacity"], scene["scenario_seed"] + 17000000)
    log = local_path(folder)/f"{case['name']}.jsonl.gz"
    if log.exists():
        raise FileExistsError(log)
    reference = None
    if case["policy_seed"] is None:
        reference = gzip.open(local_path(case["reference_log"]), "rt", encoding="utf-8")
    sequence = digest([scene["scenario_seed"], policy_id])
    samples = []
    try:
        with NominalRuntime(c, scene, policy, replay, episode_id=case["index"]) as runtime, gzip.open(
                log, "xt", encoding="utf-8", compresslevel=3) as output:
            initial_state = runtime.world.identity()
            if bool(runtime.world.plant.lag.any()):
                raise AssertionError("propulsion memory did not reset")
            while not runtime.terminal:
                rng = global_rng()
                transition = runtime.advance()
                if global_rng() != rng:
                    raise AssertionError("simulation consumed an external random stream")
                if reference is not None:
                    expected = reference.readline()
                    if not expected or json.loads(expected) != runtime.last_record["legacy"]:
                        raise AssertionError(f"zero residual differs from archived native trace at {transition['t']}")
                payload = dict(policy_id=policy_id, transition=json_transition(transition), record=runtime.last_record)
                output.write(json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n")
                sequence = digest([sequence, digest(payload)])
                if runtime.step in (1, 17, 128) or runtime.step % 32 == 0 or runtime.terminal:
                    before = global_rng()
                    samples.append(replay.verify_sample(min(128, len(replay))))
                    if global_rng() != before:
                        raise AssertionError("Replay sampling consumed an external random stream")
                if case["stress_sampling"] and runtime.step % 11 == 0:
                    before = global_rng()
                    replay.sample(1)
                    if global_rng() != before:
                        raise AssertionError("extra sampling changed an external random stream")
            if reference is not None and reference.readline():
                raise AssertionError("archived reference continued after the new terminal")
            summary = runtime.finalize()
            try:
                runtime.advance()
            except RuntimeError:
                pass
            else:
                raise AssertionError("terminated mission accepted another transition")
            if len(replay) != summary["steps"]:
                raise AssertionError("real transition count differs from Replay length")
            final_sha = None if case["policy_seed"] is None else policy.verify_frozen()
            result = dict(case=case, scene_fingerprint=runtime.world.fingerprint, policy_id=policy_id,
                initial_state_sha256=initial_state, summary=summary, transition_sequence_sha256=sequence,
                replay_data_sha256=replay_digest(replay), replay_length=len(replay),
                sample_calls=replay.sample_calls, sampled_rows=replay.sampled_rows, checked_samples=samples,
                initial_weights_sha256=getattr(policy, "initial_sha256", None), final_weights_sha256=final_sha,
                zero_reference_exact=reference is not None, global_rng_isolated=True,
                local_imports=imported_local_files(source), optimizer_updates=0)
    finally:
        if reference is not None:
            reference.close()
    result.update(log=log.relative_to(ROOT).as_posix(), log_sha256=file_sha(log))
    write_new(local_path(folder)/f"{case['name']}.json", result)
    return result


def execute(protocol_path, folder, workers):
    p, c, registry = load_protocol(protocol_path)
    source = verify_source()
    folder = local_path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if any(folder.glob("*.jsonl.gz")):
        raise FileExistsError("runtime validation output already has trajectory logs")
    spec = dict(protocol_sha256=file_sha(protocol_path), source_sha256=source["sha256"],
                cases=p["cases"], registered_utc=datetime.now(timezone.utc).isoformat(),
                maximum_complete_episodes=8, maximum_task_steps=3200, optimizer_updates_allowed=0)
    write_new(folder/"execution_registration.json", spec)
    results = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = {pool.submit(run_case, (case, p, c, registry, folder, source["sha256"])):case["name"] for case in p["cases"]}
        for future in as_completed(jobs):
            result = future.result()
            results[jobs[future]] = result
            print(f"runtime case completed: {jobs[future]} {result['summary']['steps']} {result['summary']['termination']}", flush=True)
    for left, right in p["repeat_pairs"]:
        a, b = results[left], results[right]
        if a["sample_calls"] <= b["sample_calls"]:
            raise AssertionError("registered extra Replay sampling was not exercised")
        for field in ("initial_state_sha256", "summary", "transition_sequence_sha256", "replay_data_sha256",
                      "initial_weights_sha256", "final_weights_sha256"):
            if a[field] != b[field]:
                raise AssertionError(f"independent frozen replays differ: {left}/{right}/{field}")
    zero = [r for r in results.values() if r["case"]["policy_seed"] is None]
    if {r["summary"]["termination"] for r in zero} != {"success", "obstacle_collision", "timeout"}:
        raise AssertionError("native success, collision and timeout were not all exercised")
    if sum(r["summary"]["suppressed_nonzero_boat_steps"] for r in results.values()) <= 0:
        raise AssertionError("no nonzero actor proposal was tested behind a closed gate")
    if verify_source()["sha256"] != source["sha256"]:
        raise ValueError("runtime source changed during validation")
    report = dict(schema="auv.ch4.nominal_maddpg.runtime_acceptance.v1", passed=True,
        created_utc=datetime.now(timezone.utc).isoformat(), completed_steps=[1, 2, 3],
        source_sha256=source["sha256"], protocol_sha256=file_sha(protocol_path),
        cases=[results[case["name"]] for case in p["cases"]], repeat_pairs=p["repeat_pairs"],
        task_episodes=8, real_transitions=sum(r["summary"]["steps"] for r in results.values()),
        optimizer_updates=0, training_enabled=False, runtime_validated=True, learner_update_validated=False,
        nominal_performance_verified=False, admission_rollouts=0, mismatch_rollouts=0)
    write_new(folder/"runtime_acceptance.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--register-source", action="store_true")
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--output", default="verification/nominal_maddpg_runtime_20261125")
    args = parser.parse_args()
    p, _, _ = load_protocol()
    if args.register_source:
        append_source(DEFAULT_CONFIG, stage=3,
                      reason="Connect read-only nominal MADDPG runtime, actual gated Replay transitions and runtime acceptance checks")
    if not args.execute:
        print(json.dumps(dict(executed=False, registered_cases=len(p["cases"]), optimizer_updates=0)))
        return
    report = execute(PROTOCOL, args.output, args.workers)
    print(json.dumps(dict(passed=report["passed"], episodes=report["task_episodes"],
                         transitions=report["real_transitions"], optimizer_updates=0)))


if __name__ == "__main__":
    main()
