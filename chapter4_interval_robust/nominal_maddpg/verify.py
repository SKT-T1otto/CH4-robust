"""Geometry, configuration and source verification. This command has no trainer."""
import argparse
import inspect
import platform
from datetime import datetime, timezone

import numpy as np
import torch

from chapter3_bser.learning.checkpoint import state_digest
from chapter3_bser.learning.provenance import verify_reviewed_changes
from chapter4_interval_robust.provenance import write_new
from core.config.active_environment import build_ch3_config
from core.env.mission_env import MissionCoreEnv, environment_kwargs_from_config
from core.env.task_protocol import PROTOCOL_FIELDS
from core.env.uav_env import _BaseUAVEnv

from .config import DEFAULT_CONFIG, ROOT, file_sha, load, local_path, read_json, task_config
from .learner import SingleCriticNetworks
from .provenance import append_source, imported_local_files, verify_source
from .scenarios import build_registry, validate_registry, verify_regeneration


def resolved_task_contract(config):
    native = task_config(config)
    settings = build_ch3_config(native["base_candidate"], native["profile"])
    settings.update({k: native[k] for k in (*PROTOCOL_FIELDS, "collision_terminal_reward") if k in native})
    settings["waypoint_reward_timing"] = native["waypoint_reward_timing"]
    kwargs = environment_kwargs_from_config(settings, device="cpu", max_steps=native["max_steps"], return_numpy=False)
    defaults = inspect.signature(_BaseUAVEnv.__init__).parameters
    actual = dict(agents=kwargs.get("n_agents", defaults["n_agents"].default),
                  dt=kwargs.get("dt", defaults["dt"].default), role_order=list(MissionCoreEnv.role_order))
    for key in ("residual_scale_search", "residual_scale_executor", "failure_penalty_steps", "max_steps",
                "task_protocol", "collision_detection_revision", "terminal_reward_revision",
                "collision_terminal_reward", "waypoint_reward_timing"):
        actual[key] = kwargs[key]
    for key, value in actual.items():
        if value != config["task"][key]:
            raise ValueError("resolved native task differs: " + key)
    return dict(static_contract=actual, reward_adapter=native["reward"], resolved_environment_kwargs=kwargs,
                runtime_executed=False)


def network_contract(config):
    rows = []
    for seed in config["training_seeds"]:
        model = SingleCriticNetworks(config, seed)
        before = state_digest(model.state_dict())
        with torch.no_grad():
            obs = torch.zeros(2, 4, 28)
            actions = model.actions(obs)
            values = model.values(obs, actions)
        if actions.shape != (2, 4, 3) or values.shape != (2, 4) or state_digest(model.state_dict()) != before:
            raise ValueError("network construction or inference mutated state")
        rows.append(dict(seed=seed, actors=len(model.actors), critics=len(model.critics),
                         target_actors=len(model.target_actors), target_critics=len(model.target_critics),
                         state_sha256=before, actor_parameters=sum(p.numel() for p in model.actors.parameters()),
                         critic_parameters=sum(p.numel() for p in model.critics.parameters())))
    return rows


def audit(config, config_path, regenerate=False, workers=4, regenerate_sample=False):
    registry_path = local_path(config["scene_manifest"])
    registry = read_json(registry_path)
    groups = validate_registry(config, registry, config_path)
    task = resolved_task_contract(config)
    networks = network_contract(config)
    regenerated = (verify_regeneration(config, registry, workers, sample=regenerate_sample)
                   if regenerate or regenerate_sample else dict(count=0, mode="not_requested", scene_ids=[]))
    record = verify_source()
    try:
        verify_reviewed_changes()
    except ValueError as exc:
        inherited = dict(passed=False, reason=str(exc))
    else:
        inherited = dict(passed=True)
    return dict(schema="auv.ch4.nominal_maddpg.steps12.verification.v1",
                created_utc=datetime.now(timezone.utc).isoformat(), passed=True, completed_steps=[1, 2],
                config_sha256=file_sha(config_path), scene_manifest_sha256=file_sha(registry_path),
                source_sha256=record["sha256"], registered_files=len(record["files"]),
                imported_files=imported_local_files(record), groups=groups,
                total_registered_scenes=sum(g["count"] for g in groups.values()),
                independently_regenerated_scenes=regenerated["count"], regeneration=regenerated,
                excluded_historical_fingerprints=len(registry["history_exclusion"]["fingerprints"]),
                excluded_historical_seeds=len(registry["history_exclusion"]["seeds"]),
                resolved_task=task, network_construction=networks, inherited_source_check=inherited,
                training_enabled=False, task_rollouts=0, optimizer_updates=0, checkpoints_written=0,
                runtime_validated=False, learner_update_validated=False, nominal_performance_verified=False,
                confirmation_sample_size_locked=False,
                environment=dict(python=platform.python_version(), torch=str(torch.__version__), numpy=np.__version__))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=4)
    parser.add_argument("--build-scenes", action="store_true")
    parser.add_argument("--register-source", action="store_true")
    regeneration = parser.add_mutually_exclusive_group()
    regeneration.add_argument("--regenerate", action="store_true")
    regeneration.add_argument("--regenerate-sample", action="store_true")
    parser.add_argument("--report", help="new verification JSON path within this project")
    args = parser.parse_args()
    config_path = local_path(args.config)
    config = load(config_path)
    if args.build_scenes:
        destination = local_path(config["scene_manifest"])
        if destination.exists():
            raise FileExistsError(destination)
        write_new(destination, build_registry(config, config_path, args.workers))
        print("Geometry registry created. Task rollouts and updates: 0", flush=True)
        return
    if args.register_source:
        append_source(config_path)
    report = audit(config, config_path, args.regenerate, args.workers, args.regenerate_sample)
    if args.report:
        write_new(local_path(args.report), report)
    import json
    print(json.dumps(dict(passed=report["passed"], groups=report["groups"],
                         regenerated=report["independently_regenerated_scenes"], training_enabled=False)))
