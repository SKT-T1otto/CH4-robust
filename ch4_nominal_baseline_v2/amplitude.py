"""Paired nominal validation of the frozen episode-16 policy at three residual amplitudes."""
import argparse
import json
import os
from pathlib import Path
import platform
from tempfile import NamedTemporaryFile

import numpy as np
import torch

from chapter3_bser.learning.checkpoint import digest
from chapter3_bser.learning.evaluate import preserved_rng
from chapter4_interval_robust.nominal_maddpg.checkpoint import read_boundary
from chapter4_interval_robust.nominal_maddpg.config import ROOT, file_sha, read_json
from chapter4_interval_robust.nominal_maddpg.execution import build_identity
from chapter4_interval_robust.nominal_maddpg.learner import SingleCriticLearner
from chapter4_interval_robust.nominal_maddpg.policy import FrozenEvaluationPolicy
from chapter4_interval_robust.nominal_maddpg.replay import NominalReplay
from chapter4_interval_robust.nominal_maddpg.runtime import NominalRuntime
from chapter4_interval_robust.nominal_maddpg.train_config import SceneCatalog
from .navigation import install_executor_navigation


SCALES = (0.0, 0.5, 1.0)
INDICES = tuple(range(32))
BOUNDARY_SHA = "e9261ae923399d5445371702d95ffc93e623c954a304cb2c667f91562dc50c59"
WEIGHTS_SHA = "0ffac82f732f2d977146f823c95225ed6f3fe2c26bc7a08bb19fd50c21c7c195"
DEFAULT_CHECKPOINT = (Path.home() / "ch4-nominal-dev/pilot_2729" /
                      "attempt_3117d9f11bbf4d93a1be926b36397e7d/boundaries/000016")
OUTPUT_ROOT = ROOT / "work/ch4"


class ScaledPolicy(FrozenEvaluationPolicy):
    def __init__(self, learner, scale):
        if scale not in SCALES:
            raise ValueError("expected residual multiplier 0, 0.5 or 1")
        super().__init__(learner)
        self.scale = scale
        # The runtime seeds its trajectory digest with the policy identity.
        # Keep both exact endpoints comparable with the original control traces.
        if scale == 0:
            self.policy_id = "D2/N0"
        elif scale != 1:
            self.policy_id = digest(dict(weights=self.initial_sha256, scale=scale))

    def step(self, observations, explore=False):
        actions = super().step(observations, explore)
        if self.scale == 0:
            return [torch.zeros_like(action) for action in actions]
        if self.scale == 1:
            return actions
        return [action * self.scale for action in actions]


def load_policies(checkpoint, catalog, identity):
    if not all((checkpoint / name).is_file() for name in ("committed.json", "state.pt")):
        raise ValueError(f"Missing episode-16 committed.json or state.pt under {checkpoint}. "
                         "Pass --checkpoint /path/to/boundaries/000016.")
    marker = read_json(checkpoint / "committed.json")
    training = marker["identity"]
    if (marker["completed_episodes"] != 16 or marker["file_sha256"] != BOUNDARY_SHA
            or training.get("run_seed") != 2729 or training.get("profile") != "pilot"
            or any(training.get(k) != v for k, v in identity.items() if k != "environment")
            or training["scenes_sha256"] != catalog.sha256):
        raise ValueError("The checkpoint is not the registered episode-16 development policy for this task.")
    state = read_boundary(checkpoint, training)
    learner = SingleCriticLearner(catalog.config, 0)
    learner.networks.load_state_dict(state["learner"]["networks"], strict=True)
    policies = {scale: ScaledPolicy(learner, scale) for scale in SCALES}
    if any(policy.verify_frozen() != WEIGHTS_SHA for policy in policies.values()):
        raise ValueError("Frozen inference weights differ from the selected episode-16 policy.")
    return policies


@preserved_rng()
def run_scene(catalog, scene, index, policy):
    horizon = catalog.config["task"]["max_steps"]
    replay = NominalReplay(horizon, 9100000 + index)
    with NominalRuntime(catalog.config, scene, policy, replay, episode_id=index,
                        mode="evaluation", explore=False, catalog=catalog) as runtime:
        install_executor_navigation(runtime.rt)
        magnitude, components = 0.0, 0
        for _ in range(horizon):
            step = runtime.advance()
            opened = np.asarray(step["action_enabled"], dtype=bool)
            applied = np.asarray(step["actions"])[opened]
            if np.any(np.abs(applied) > policy.scale):
                raise AssertionError("Applied normalized residual exceeded its experimental amplitude.")
            magnitude += float(np.abs(applied).sum(dtype=np.float64))
            components += applied.size
            if runtime.terminal:
                break
        result = runtime.finalize()  # rejects truncated episodes and verifies frozen networks
        task = runtime.rt.env.get_task_state()
        outcome = runtime.rt.env.get_episode_result()
        return dict(index=index, scale=policy.scale, source=runtime.source.as_dict(),
                    success=bool(result["success"]), termination=result["termination"], steps=result["steps"],
                    cost=result["penalized_completion_steps"], discovery=bool(task.target_found),
                    handoff=bool(task.executor_knows_target and task.handoff_step is not None),
                    handoff_step=task.handoff_step, max_consecutive_hold=int(runtime.world.core.capture_hold_counter_max),
                    collision_agent_ids=outcome["first_collision_agent_ids"], collision_phase=outcome["first_collision_phase"],
                    trajectory_sha256=result["trajectory_sha256"], nonzero_residual_steps=result["nonzero_residual_steps"],
                    residual_abs_sum=magnitude, open_components=components)


def summarize(rows):
    n = len(rows)
    success = sum(r["success"] for r in rows)
    handoffs = sum(r["handoff"] for r in rows)
    components = sum(r["open_components"] for r in rows)
    return dict(n=n, successes=success, success_rate=success/n, handoffs=handoffs,
                discoveries=sum(r["discovery"] for r in rows),
                success_given_handoff=success/handoffs if handoffs else None,
                collisions=sum(r["termination"] == "obstacle_collision" for r in rows),
                timeouts=sum(r["termination"] == "timeout" for r in rows),
                mean_penalized_steps=sum(r["cost"] for r in rows)/n,
                mean_abs_applied_residual_when_open=sum(r["residual_abs_sum"] for r in rows)/components if components else None)


def paired_report(rows):
    # Every list shares physical scenes in catalogue order.
    for index in INDICES:
        reference = rows[0][index]["source"]
        if any(rows[s][index]["source"] != reference or rows[s][index]["index"] != index
               or rows[s][index]["scale"] != s for s in SCALES):
            raise ValueError(f"Unpaired scene at validation index {index}")
    scores = {f"{s:g}": summarize(rows[s]) for s in SCALES}
    pairs = {}
    for old, new in ((0, 1), (0, 0.5), (1, 0.5)):
        before, after = rows[old], rows[new]
        pairs[f"{old:g}_to_{new:g}"] = dict(
            successes_gained=[i for i in INDICES if after[i]["success"] and not before[i]["success"]],
            successes_lost=[i for i in INDICES if before[i]["success"] and not after[i]["success"]],
            collisions_added=[i for i in INDICES if after[i]["termination"] == "obstacle_collision"
                              and before[i]["termination"] != "obstacle_collision"],
            collisions_removed=[i for i in INDICES if before[i]["termination"] == "obstacle_collision"
                                and after[i]["termination"] != "obstacle_collision"])
    extremes = pairs["0_to_1"]
    return dict(by_scale=scores, paired=pairs,
                full_scale_gains_kept_at_half=[i for i in extremes["successes_gained"] if rows[0.5][i]["success"]],
                full_scale_losses_recovered_at_half=[i for i in extremes["successes_lost"] if rows[0.5][i]["success"]])


def publish(path, data):
    """Publish whole JSON once, including when two processes finish the same scene."""
    with NamedTemporaryFile(dir=path.parent, prefix=".amplitude-", delete=False) as stream:
        temp = Path(stream.name)
    try:
        with temp.open("w", encoding="utf-8", newline="\n") as out:
            json.dump(data, out, indent=2, allow_nan=False)
            out.flush()
            os.fsync(out.fileno())
        try:
            if os.name == "nt":
                os.rename(temp, path)  # Windows rename fails if the destination exists.
            else:
                os.link(temp, path)
        except FileExistsError:
            if read_json(path) != data:
                raise ValueError(f"Different experiment or result already exists at {path}")
    finally:
        temp.unlink(missing_ok=True)


def run(args):
    torch.set_num_threads(1)
    catalog = SceneCatalog()
    identity = build_identity()
    with preserved_rng():
        policies = load_policies(args.checkpoint.expanduser(), catalog, identity)
    if args.check:
        print(f"Verified frozen episode 16, seed 2729, weights {WEIGHTS_SHA}")
        return
    folder = args.output.expanduser().resolve()
    if folder == OUTPUT_ROOT.resolve() or not folder.is_relative_to(OUTPUT_ROOT.resolve()):
        raise ValueError(f"Choose an output directory below {OUTPUT_ROOT}")
    folder.mkdir(parents=True, exist_ok=True)
    sources = {name: file_sha(Path(__file__).with_name(name)) for name in ("amplitude.py", "navigation.py")}
    plan = dict(schema="ch4.nominal_residual_amplitude.v1", indices=list(INDICES), group="validation",
                scales=list(SCALES), horizon=catalog.config["task"]["max_steps"], training_seed=2729,
                weights_sha256=WEIGHTS_SHA, checkpoint_sha256=BOUNDARY_SHA, host=platform.node(),
                catalog_sha256=catalog.sha256, base=identity, sources=sources, exploration=False, updates=0)
    if not (folder/"plan.json").exists() and any(p.name != "plan.json" and not p.name.startswith(".amplitude-")
                                                 for p in folder.iterdir()):
        raise ValueError(f"Output folder already contains unrelated files: {folder}")
    publish(folder/"plan.json", plan)
    plan_sha = digest(plan)
    rows = {s: [] for s in SCALES}
    completed = 0
    for i in INDICES:
        scene = catalog.scene("validation", i)
        origin = catalog.origin(scene, i, allowed_groups=("validation",)).as_dict()
        for s in SCALES:
            path = folder / f"scene_{i:03d}_scale_{s:g}.json"
            if not path.exists() and s in args.scales and i in args.indices:
                row = run_scene(catalog, scene, i, policies[s])
                publish(path, dict(plan_sha256=plan_sha, row_sha256=digest(row), row=row))
                print(f"scene {i:03d}, scale {s:g}: {row['termination']} at {row['steps']}", flush=True)
            if path.exists():
                entry = read_json(path)
                row = entry["row"]
                if (entry["plan_sha256"] != plan_sha or entry["row_sha256"] != digest(row)
                        or row["index"] != i or row["scale"] != s or row["source"] != origin):
                    raise ValueError(f"Incompatible or damaged scene receipt: {path}")
                rows[s].append(row)
                completed += 1
    total = len(INDICES) * len(SCALES)
    if completed == total:
        report = paired_report(rows)
        publish(folder/"summary.json", dict(plan_sha256=plan_sha, **report))
        print(json.dumps(report["by_scale"], indent=2))
        print(f"Paired results: {folder/'summary.json'}")
    else:
        print(f"{completed}/{total} complete. Run --execute without a subset to finish.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--check", action="store_true", help="verify source, scenes and checkpoint without simulation")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT, help="episode-16 boundary directory")
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT/"nominal_amplitude_ep16")
    parser.add_argument("--indices", type=int, nargs="+", choices=INDICES, default=INDICES)
    parser.add_argument("--scales", type=float, nargs="+", choices=SCALES, default=SCALES)
    args = parser.parse_args(argv)
    if not args.check and not args.execute:
        print(json.dumps(dict(executed=False, indices=args.indices, scales=args.scales,
                              checkpoint=str(args.checkpoint), output=str(args.output))))
        return
    try:
        run(args)
    except (FileNotFoundError, ValueError) as error:
        parser.exit(1, f"{error}\n")


if __name__ == "__main__":
    main()
