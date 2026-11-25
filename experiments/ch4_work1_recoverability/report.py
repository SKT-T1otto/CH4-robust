"""Preregister, run, and report the paired whole-mission mismatch experiment."""
import csv
import hashlib
import io
import json
import math
import os
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from chapter3_bser.learning.checkpoint import digest
from experiments.ch4_work1_recoverability import records, run
from experiments.ch4_work1_recoverability.plant_adapter import WaterParameters


DEV = "development"
CONF = "confirmation_reserve"


def publish(path, data):
    """Publish a complete file once. Concurrent identical results are harmless."""
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        tmp.write(data)
        tmp.flush()
        os.fsync(tmp.fileno())
    try:
        try:
            if os.name == "nt":
                # On Windows rename is atomic and refuses an existing destination.
                os.rename(tmp.name, path)
            else:
                os.link(tmp.name, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise ValueError(f"existing artifact differs: {path}")
    finally:
        Path(tmp.name).unlink(missing_ok=True)


def write_json(path, obj):
    publish(path, (json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode())


def seal(path, spec):
    if not path.exists():
        write_json(path, dict(utc=datetime.now(timezone.utc).isoformat(), sha256=digest(spec), spec=spec))
    saved = run.read_json(path)
    if saved["spec"] != spec or saved["sha256"] != digest(spec):
        raise ValueError(f"locked design changed: {path.name}")
    return saved


def cases(protocol):
    design = protocol["future_physical_design"]
    kl, kh = design["per_vehicle_drag_range"]
    sl, sh = design["per_vehicle_horizontal_thrust_range"]
    return [dict(beta=beta, drag=np.roll([kl, kl, kh, kh], i).tolist(),
                 thrust=np.roll([sl, sh, sl, sh], i).tolist(), law=design["primary_law"])
            for beta in (min(design["beta_m_per_s"]), max(design["beta_m_per_s"])) for i in range(4)]


def registration(protocol, checkpoint):
    groups = {}
    for name in (DEV, CONF):
        bank = protocol["scene_groups"][name]
        groups[name] = []
        for i, fp in enumerate(bank["fingerprints"]):
            seed = bank["start_seed"] + i
            rng = np.random.default_rng(np.random.SeedSequence([4636000, seed]))
            groups[name].append(dict(index=i, seed=seed, physical_fingerprint=fp, case=int(rng.integers(8))))
    files = ("protocol.json", "report.py", "run.py", "records.py", "plant_adapter.py")
    nominal = protocol["nominal_references"]["relative_water"]
    if (nominal["beta_m_per_s"], nominal["drag_multiplier"], nominal["horizontal_thrust_multiplier"],
            nominal["propulsion_lag"], nominal["noise"], nominal["law"], nominal["actuator_delay_steps"],
            nominal["estimator_delay_steps"], nominal["extra_vertical_bias_m_per_s2"]) != (.1, 1., 1., .15, "N0", "linear", 0, 0, 0.):
        raise ValueError("relative-water nominal reference changed")
    return dict(schema="ch4.paired_loss.v1", policy="frozen D2", nominal=nominal, shifts=cases(protocol),
                assignment="one iid uniform draw from eight cases per physical scene, SeedSequence([4636000,scene_seed])",
                groups=groups, sources={f: hashlib.sha256((run.HERE/f).read_bytes()).hexdigest() for f in files},
                frozen_source=run.ch4_source_identity(), checkpoint_sha256=(
                    hashlib.sha256(checkpoint.read_bytes()).hexdigest() if checkpoint and checkpoint.is_file() else None),
                independent_unit="physical scene", D="shift penalty steps minus nominal penalty steps, all scenes",
                sample_size="max(24, ceil((1.96 * development paired sample sd / 2.5)^2)), ordered prefix",
                overflow="Use the entire 64-scene registered reserve once as precision-limited independent evidence. "
                         "If the computed requirement exceeds the reserve, do not promote to recovery.",
                acceptance="confirmation mean D >= 5 and scene percentile 95% CI lower > 0; "
                           "size requirement met and baseline has resolution under the ceiling rule",
                bootstrap=dict(draws=10000, seed=4536000), ceiling="both arms at least 90% penalized at 500",
                window_metric="first registered approach per arm/phase; up to 24 consecutive controllable steps "
                              "on the same assignment and tracking waypoint. RMS horizontal velocity error vs "
                              "pre-step native approach reference, and mean positive excess closing speed (m/s). "
                              "Compare within scene only when both arms have a window; descriptive, post-treatment.",
                total_population="include every complete episode even if one or both phases or windows are absent")


def velocity_errors(before, after, window, vmax, slow_radius):
    hull = window["vehicle"]
    direction = np.asarray(window["tracking_waypoint"][:2]) - before["position_m"][hull][:2]
    distance = float(np.linalg.norm(direction))
    unit = direction / max(distance, 1e-8)
    desired = vmax * min(distance / slow_radius, 1.)
    actual = np.asarray(after["velocity_after_mps"][hull][:2])
    return float(np.linalg.norm(actual - desired * unit)), max(0., float(actual @ unit) - desired)


def arm(world, protocol):
    rt = world.rt
    counts = records.phase_counts()
    errors = {phase: [] for phase in records.PHASES}
    ended = set()
    target, streams, events = [], [], set()
    discounted = 0.
    radius = rt.env.get_public_control_contract()["prior_slow_radius_xy"]
    while not rt.terminal:
        before = world.decision()
        t = before["t"]
        # Compare target states at *decision* time. A collision can terminate before the next target update.
        target.append(records.stamp(world.core.target_state))
        streams.append(records.stamp(world.rng))
        for phase, hull in records.count_before(before, counts).items():
            h = before["hulls"][hull]
            counts[phase]["window"] = dict(t=t, vehicle=hull, assignment=h["assignment_id"],
                                            tracking_waypoint=h["tracking_waypoint"])
        after = world.advance(before)
        repeats = Counter()
        for physical_t, hull in after["planner_events"]:
            event = (physical_t, hull)
            # The native draw is keyed by seed/physical step/hull and within-call repeat.
            events.add((t, physical_t, hull, repeats[event]))
            repeats[event] += 1
        discounted += protocol["control"]["gamma"] ** t * after["team_reward"]
        for phase, c in counts.items():
            window = c["window"]
            if window is None or phase in ended:
                continue
            i = window["vehicle"]
            h = before["hulls"][i]
            if (t-window["t"] >= 24 or h["phase"] != phase or not h["controllable"]
                    or h["assignment_id"] != window["assignment"] or h["tracking_waypoint"] != window["tracking_waypoint"]):
                ended.add(phase)
                continue
            errors[phase].append(velocity_errors(before, after, window, float(world.core._v_xy_max[i]), radius))
    outcome = rt.env.get_episode_result()
    if discounted != rt.env.reward_accounting.team_discounted_return:
        raise AssertionError("team return differs from native accounting")
    cost = run.completion_cost(outcome, rt.step, protocol["task"])
    for phase, vals in errors.items():
        counts[phase]["velocity_errors_mps"] = vals
        counts[phase]["velocity_rmse_mps"] = float(np.sqrt(np.mean(np.square([x[0] for x in vals])))) if vals else None
        counts[phase]["excess_closing_mps"] = float(np.mean([x[1] for x in vals])) if vals else None
    row = dict(actual_steps=rt.step, completion_step=rt.step if outcome["success"] else None,
               P=cost, termination=outcome["termination_reason"], success=outcome["success"],
               collision=outcome["termination_reason"] == "obstacle_collision", timeout=outcome["termination_reason"] == "timeout",
               team_discounted_return=discounted, found_step=rt.found_step, handoff_step=rt.handoff_event_step,
               phase_counts=counts, trajectory_sha256=world.prefix)
    return row, target, streams, events


def paired_scene(job):
    group, entry, truth, protocol, config = job
    torch.set_num_threads(1)
    scene = run.materialize(protocol, group, entry["index"])
    with records.Branch(config, scene, entry["index"], protocol["task"]["approach_rule"]) as n, n.fork() as s:
        run.check_contract(n.rt, protocol)
        initial = n.identity()
        if n.seed != entry["seed"] or n.fingerprint != entry["physical_fingerprint"] or initial != s.identity():
            raise AssertionError("paired initial state mismatch")
        s.plant.set_parameters(WaterParameters(**truth))
        nominal, nt, nr, ne = arm(n, protocol)
        shift, st, sr, se = arm(s, protocol)
    overlap = min(len(nt), len(st))
    if not overlap or nt[:overlap] != st[:overlap] or nr[:overlap] != sr[:overlap]:
        raise AssertionError("paired target process or exogenous random streams diverged")
    return dict(group=group, **entry, truth=truth, initial_sha256=initial, nominal=nominal, shift=shift,
                D=shift["P"] - nominal["P"], shared_decision_steps=overlap,
                common_target_sha256=digest(nt[:overlap]), common_rng_sha256=digest(nr[:overlap]),
                common_planner_events=len(ne & se), nominal_only_events=len(ne-se), shift_only_events=len(se-ne))


def estimate(values):
    if not values:
        return dict(n=0, mean=None, ci95=None)
    a = np.asarray(values, dtype=float)
    rng = np.random.default_rng(4536000)
    means = rng.choice(a, (10000, len(a)), replace=True).mean(axis=1)
    return dict(n=len(values), mean=float(a.mean()), ci95=np.quantile(means, [.025, .975]).tolist())


def summarize(rows):
    ds = [r["D"] for r in rows]
    result = dict(D=estimate(ds), sample_sd=float(np.std(ds, ddof=1)), positive=sum(d>0 for d in ds),
                  negative=sum(d<0 for d in ds), zero=ds.count(0),
                  both_500=sum(r["nominal"]["P"] == r["shift"]["P"] == 500 for r in rows),
                  success_pairs=dict(Counter(f"{int(r['nominal']['success'])}->{int(r['shift']['success'])}" for r in rows)),
                  return_difference=estimate([r["shift"]["team_discounted_return"]-r["nominal"]["team_discounted_return"] for r in rows]),
                  arms={}, windows={})
    for name in ("nominal", "shift"):
        arms = [r[name] for r in rows]
        result["arms"][name] = dict(mean_P=float(np.mean([a["P"] for a in arms])),
            mean_team_discounted_return=float(np.mean([a["team_discounted_return"] for a in arms])),
            outcomes=dict(Counter(a["termination"] for a in arms)), failures_at_500=sum(a["P"] == 500 for a in arms),
            phases={p: records.aggregate(arms, p) for p in records.PHASES})
    result["ceiling_limited"] = all(x["failures_at_500"] / len(rows) >= .90 for x in result["arms"].values())
    for phase in records.PHASES:
        diffs = {}
        for metric in ("velocity_rmse_mps", "excess_closing_mps"):
            pairs = [(r["nominal"]["phase_counts"][phase][metric], r["shift"]["phase_counts"][phase][metric]) for r in rows]
            diffs[metric] = estimate([b-a for a,b in pairs if a is not None and b is not None])
        result["windows"][phase] = diffs
    return result


def size_lock(dev, reserve):
    sd = float(np.std([r["D"] for r in dev], ddof=1))
    required = max(24, math.ceil((1.96 * sd / 2.5) ** 2))
    return dict(development_sha256=digest(dev), development_n=len(dev), development_sd=sd,
                required_n=required, locked_n=min(required, reserve), precision_limited=required > reserve)


def collect(pool, group, entries, design, protocol, config, folder, lock_id):
    files = folder / group
    files.mkdir(exist_ok=True)
    rows, pending = [], {}
    for entry in entries:
        truth = design["shifts"][entry["case"]]
        path = files / f"scene_{entry['index']:03}.json"
        if path.exists():
            receipt = run.read_json(path)
            row = receipt["row"]
            if (receipt["lock_sha256"] != lock_id or receipt["row_sha256"] != digest(row) or row["group"] != group
                    or any(row[k] != v for k,v in entry.items()) or row["truth"] != truth):
                raise ValueError(f"invalid completed pair: {path}")
            rows.append(row)
        else:
            pending[pool.submit(paired_scene, (group, entry, truth, protocol, config))] = path
    for future in as_completed(pending):
        row = future.result()
        write_json(pending[future], dict(lock_sha256=lock_id, row_sha256=digest(row), row=row))
        rows.append(row)
        print(f"{group}: {len(rows)}/{len(entries)} complete pairs", flush=True)
    return sorted(rows, key=lambda r:r["index"])


def table(rows):
    output = io.StringIO(newline="")
    all_rows = []
    for row in rows:
        line = {k:row[k] for k in ("group", "index", "seed", "case", "D")}
        for name in ("nominal", "shift"):
            arm_row = row[name]
            for key in ("actual_steps", "completion_step", "P", "termination", "success", "collision", "timeout", "team_discounted_return"):
                line[f"{name}_{key}"] = arm_row[key]
            for phase, c in arm_row["phase_counts"].items():
                line[f"{name}_{phase}_entered"] = c["phase_boat_steps"] > 0
                line[f"{name}_{phase}_window"] = c["window"] is not None
        all_rows.append(line)
    writer = csv.DictWriter(output, fieldnames=all_rows[0])
    writer.writeheader()
    writer.writerows(all_rows)
    return output.getvalue().encode()


def run_stage(protocol, folder, workers, checkpoint):
    run.audit_catalog(protocol)
    folder.mkdir(parents=True, exist_ok=True)
    design = registration(protocol, checkpoint)
    registered = seal(folder/"preregistration.json", design)
    config = run.validate(run.read_json(run.ROOT/"configs/b2.json"))
    b2 = run.evaluate_b2(checkpoint, protocol, config)
    write_json(folder/"b2_admission.json", b2)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        dev = collect(pool, DEV, design["groups"][DEV], design, protocol, config, folder, registered["sha256"])
        development = summarize(dev)
        write_json(folder/"development_summary.json", development)
        size = size_lock(dev, len(design["groups"][CONF]))
        planned = design["groups"][CONF][:size["locked_n"]]
        lockfile = folder/"confirmation_lock.json"
        if not lockfile.exists() and any((folder/CONF).glob("scene_*.json")):
            raise ValueError("confirmation outcomes exist before sample-size lock")
        lock = seal(lockfile, dict(**size, preregistration_sha256=registered["sha256"], scenes=planned))
        print(f"Locked {size['locked_n']} confirmation scenes; required {size['required_n']}", flush=True)
        confirm = collect(pool, CONF, planned, design, protocol, config, folder, lock["sha256"])
    confirmation = summarize(confirm)
    effect = confirmation["D"]
    numerical_gate = effect["mean"] >= 5 and effect["ci95"][0] > 0
    report = dict(development=development, confirmation=confirmation, sample_size=size,
                  confirmation_mean_and_ci_gate=numerical_gate, proceed_to_recovery=(
                      numerical_gate and not size["precision_limited"] and not confirmation["ceiling_limited"]),
                  b2_admission=b2, training_updates=0, recovery_rollouts=0, development_lock=registered["sha256"],
                  confirmation_lock=lock["sha256"], confirmation_rows_sha256=digest(confirm))
    publish(folder/"paired_scenes.csv", table(dev+confirm))
    write_json(folder/"summary.json", report)
    return report
