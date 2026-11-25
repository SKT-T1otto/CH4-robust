"""Materialize geometry and identities without running a mission or policy."""
from concurrent.futures import ProcessPoolExecutor

from chapter3_bser.learning.checkpoint import digest
from chapter3_bser.learning.config import scenario_fingerprint
from chapter3_bser.learning.scenarios import validate_manifest
from chapter4_interval_robust.scenarios import make_scene

from .config import GROUPS, ROOT, file_sha, local_path, read_json


def generate_scene(job):
    group, first, index, horizon = job
    # A worker constructs geometry only. One Torch thread avoids nested BLAS pools.
    import torch
    torch.set_num_threads(1)
    split = "train" if group == "train" else "dev"
    scene = make_scene(dict(splits={split: first}, max_steps=horizon), split, index)
    role = "train" if group == "train" else "eval"
    scene.update(scenario_id=f"ch4_nominal_maddpg_{group}_{first + index}",
                 scenario_role=role, scenario_split=role,
                 ch4_nominal_group=group)
    return dict(index=index, seed=scene["scenario_seed"], physical_fingerprint=scenario_fingerprint(scene), scene=scene)


def historical_geometry(root=ROOT):
    fingerprints, seeds, consumed = set(), set(), {}

    def visit(value):
        if isinstance(value, dict):
            if "initial_agent_positions" in value and ("target_position" in value or "target_initial_position" in value):
                fingerprints.add(scenario_fingerprint(value))
            if isinstance(value.get("physical_fingerprint"), str):
                fingerprints.add(value["physical_fingerprint"])
            if "fingerprints" in value and isinstance(value["fingerprints"], list):
                fingerprints.update(f for f in value["fingerprints"] if isinstance(f, str) and len(f) == 64)
            if type(value.get("scenario_seed")) is int:
                seeds.add(value["scenario_seed"])
            if type(value.get("start_seed")) is int and isinstance(value.get("fingerprints"), list):
                seeds.update(range(value["start_seed"], value["start_seed"] + len(value["fingerprints"])))
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    paths = sorted((root/"configs").rglob("*.json")) + sorted((root/"experiments").rglob("*.json"))
    for path in paths:
        if path.name.startswith("nominal_maddpg"):
            continue
        value = read_json(path)
        visit(value)
        consumed[path.relative_to(root).as_posix()] = file_sha(path)
    return dict(fingerprints=sorted(fingerprints), seeds=sorted(seeds), source_files=consumed)


def build_registry(config, config_path, workers=4):
    jobs = [(name, config["scene_groups"][name]["start_seed"], i, config["task"]["max_steps"])
            for name in GROUPS for i in range(config["scene_groups"][name]["count"])]
    rows = {name: [] for name in GROUPS}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for job, entry in zip(jobs, pool.map(generate_scene, jobs, chunksize=8)):
            rows[job[0]].append(entry)
            count = len(rows[job[0]])
            total = config["scene_groups"][job[0]]["count"]
            if count % 100 == 0 or count == total:
                print(f"geometry registered: {job[0]} {count}/{total}", flush=True)
    registry = dict(schema="auv.ch4.nominal_maddpg.scenes.v1", config_sha256=file_sha(config_path),
                    generator="one existing make_scene per scene seed", groups=rows,
                    history_exclusion=historical_geometry(), outcomes_generated=False)
    validate_registry(config, registry, config_path)
    return registry


def validate_registry(config, registry, config_path):
    if (registry["schema"] != "auv.ch4.nominal_maddpg.scenes.v1" or registry["outcomes_generated"] is not False
            or registry["config_sha256"] != file_sha(config_path) or set(registry["groups"]) != set(GROUPS)):
        raise ValueError("scene registry identity differs")
    history = registry["history_exclusion"]
    used_fp, used_seed, used_id = set(history["fingerprints"]), set(history["seeds"]), set()
    for relative, expected in history["source_files"].items():
        if file_sha(local_path(relative)) != expected:
            raise ValueError("historical scene catalog changed: " + relative)
    report = {}
    for name in GROUPS:
        spec, entries = config["scene_groups"][name], registry["groups"][name]
        if len(entries) != spec["count"]:
            raise ValueError("scene group count differs: " + name)
        for index, entry in enumerate(entries):
            scene = entry["scene"]
            fp, seed, identifier = scenario_fingerprint(scene), entry["seed"], scene["scenario_id"]
            expected_id = f"ch4_nominal_maddpg_{name}_{spec['start_seed'] + index}"
            if (entry["index"] != index or seed != spec["start_seed"] + index or scene["scenario_seed"] != seed
                    or entry["physical_fingerprint"] != fp or identifier != expected_id
                    or scene.get("ch4_nominal_group") != name):
                raise ValueError("scene identity or physical fingerprint differs")
            if fp in used_fp or seed in used_seed or identifier in used_id:
                raise ValueError("duplicate historical or cross-group physical scene/seed/id")
            if scene.get("flow_phase_x") != 0 or scene.get("flow_phase_y") != 0:
                raise ValueError("nominal scenes require zero flow phases")
            used_fp.add(fp)
            used_seed.add(seed)
            used_id.add(identifier)
        validate_manifest(dict(scenarios=[e["scene"] for e in entries]), config["task"], evaluation=name != "train")
        report[name] = dict(count=len(entries), start_seed=spec["start_seed"],
                            physical_fingerprints_sha256=digest([e["physical_fingerprint"] for e in entries]))
    return report


def verify_regeneration(config, registry, workers=4, sample=False):
    entries = [(name, e) for name in GROUPS for e in registry["groups"][name]]
    if sample:
        entries = [(name, registry["groups"][name][i]) for name in GROUPS
                   for i in sorted({0, len(registry["groups"][name])//2, len(registry["groups"][name])-1})]
    jobs = [(name, config["scene_groups"][name]["start_seed"], e["index"], config["task"]["max_steps"])
            for name, e in entries]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for (name, expected), actual in zip(entries, pool.map(generate_scene, jobs, chunksize=1 if sample else 8)):
            if digest(actual) != digest(expected):
                raise ValueError(f"independent geometry regeneration differs: {name}/{expected['index']}")
    return dict(count=len(entries), mode="first_middle_last_per_group" if sample else "all_scenes",
                scene_ids=[e["scene"]["scenario_id"] for _, e in entries])
