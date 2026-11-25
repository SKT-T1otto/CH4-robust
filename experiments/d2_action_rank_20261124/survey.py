"""One-step physical action ranking on natural frozen-D2 states. No learner or critic."""
import argparse
import itertools
import math
import sys
import time
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True
import numpy as np
from plant import (CAL_SEED, DEV_SEED, DT, EPS_REWARD, GAMMA, LEVELS, RHO, ROOT, ZERO,
                   Pulse, World, advance, array, cpu_setup, current, digest_values, model_step,
                   observation, output_dir, parallel_jobs, read, restore_rng, rng_state,
                   state_digest, write_new)


def historical_support(world, agent, tolerance):
    # Positions, ground velocities and actual normalized sent commands of the
    # recipient are sufficient. Previous commands initialize the candidate lag.
    betas = np.array([0., .05, .10, .15, .20])
    ks = np.array([.85, .90, .95, 1., 1.05, 1.10, 1.15, 1.20])
    ss = np.array([.85, .90, .95, 1., 1.05, 1.10])
    rhos = np.array([0., .05, .10, .15, .20, .25])
    grid = np.array(list(itertools.product(betas, ks, ss, range(len(rhos)))))
    beta, k, s, ri = (grid[:, j:j+1] for j in range(4))
    ri = ri[:, 0].astype(int)
    fixed = np.flatnonzero((beta[:, 0] == .1) & (k[:, 0] == 1) & (s[:, 0] == 1) & (rhos[ri] == RHO)).item()
    errors = np.zeros((2, len(grid)))
    known_z = np.zeros((len(rhos), 2))
    used = []
    start = max(0, len(world.history)-8)
    specs = world.runtime.env.get_public_vehicle_specs()[agent]
    for step, (p, v, u, after, valid) in enumerate(world.history):
        known_z = rhos[:, None]*known_z + (1.-rhos[:, None])*u[agent, :2]
        if step < start or not valid[agent]:
            continue
        used.append(step)
        for idx, mix in enumerate((0., .5)):
            pred = model_step(p[agent].astype(float), v[agent, :2].astype(float), known_z[ri],
                              specs["drag_xy"], specs["a_xy_max"], DT, world.plant.w0, mix,
                              beta=beta, k=k, s=s)
            err = np.max(np.abs(pred-after[agent, :2]), axis=1)
            errors[idx] = np.maximum(errors[idx], err)
    enough = len(used) >= 2
    counts = (errors <= tolerance).sum(axis=1).tolist()
    loose = (errors <= .01).sum(axis=1).tolist()
    return dict(valid_history_steps=used, grid_size=len(grid), enough=enough,
                surviving_grid=counts if enough else None, min_error=errors.min(axis=1).tolist() if enough else None,
                tested_tuple_errors=errors[:, fixed].tolist() if enough else None,
                both_grid=bool(enough and min(counts) > 0),
                both_tested_tuples=bool(enough and np.all(errors[:, fixed] <= tolerance)),
                loose_grid_counts=loose if enough else None,
                both_grid_at_0p01=bool(enough and min(loose) > 0))


def run_branch(source, initial_rng, x, law, amount, horizon, initial_digest, tolerance):
    restore_rng(initial_rng)
    branch = source.fork()
    try:
        if state_digest(branch) != initial_digest:
            raise AssertionError("unequal starting state")
        branch.plant.set_law(law)
        agent = x["agent"]
        e, q = np.array(x["direction"]), np.array(x["target"])
        rewards, distances, overshoots, changed, trial = [], [], [], [], {}
        for j in range(horizon):
            if branch.runtime.terminal:
                break
            row = advance(branch, Pulse(agent, e, amount) if j == 0 else ZERO)
            if j == 0:
                trial = dict(proposed=row["proposed_actions"].tolist(), gated=row["actions"].tolist(),
                             sent=array(branch.plant.sent).tolist(), actuated=array(branch.plant.z).tolist())
            pos = array(branch.core._agent_pos)[agent]
            rewards.append(float(row["team_reward"]))
            distances.append(float(np.linalg.norm(pos-q)))
            overshoots.append(max(0.,float(np.dot(pos[:2]-q[:2], e))))
            next_x = observation(branch, agent)
            changed.append(bool(next_x["assignment"] != x["assignment"] or
                                not np.allclose(next_x["target"], q, rtol=0., atol=1e-7)))
        if max(branch.stats["free_errors"], default=0.) > tolerance:
            raise AssertionError("relative-water integrator crosscheck failed")
        result = branch.runtime.env.get_episode_result()
        return dict(law=law, amount=amount, first=trial, rewards=rewards,
                    distances_to_initial_point=distances, projected_overshoot=overshoots,
                    target_changed=changed, terminal=branch.runtime.terminal, outcome=result,
                    return12=discount(rewards[:12]), return_total=discount(rewards))
    finally:
        branch.close()


def discount(rewards):
    return sum(GAMMA**i * r for i, r in enumerate(rewards))


def metrics(runs, agent, horizon):
    table = np.array([[discount(runs[m][j]["rewards"][:horizon]) for j in range(3)] for m in ("L", "Q")])
    commands = [np.array(r["first"]["sent"])[agent] for r in runs["L"]]
    # Preserve zero as the canonical choice if clipping made a pulse identical.
    keep = []
    for j in (1, 0, 2):
        if not any(np.allclose(commands[j], commands[c], rtol=0, atol=1e-6) for c in keep):
            keep.append(j)
    for j in range(3):
        for key in ("proposed", "gated", "sent", "actuated"):
            if not np.array_equal(runs["L"][j]["first"][key], runs["Q"][j]["first"][key]):
                raise AssertionError("cross-model command or filter mismatch")
    u = table[:, keep]
    regret = u.max(axis=1)[:, None]-u
    worst = regret.max(axis=0)
    ordinary = int(u.mean(axis=0).argmax())
    robust = int((u.mean(axis=0)-worst).argmax())
    C = float(worst.min())
    pair_strength = 0.
    for j, k in itertools.combinations(range(len(keep)), 2):
        d1, d2 = u[:, j]-u[:, k]
        if d1*d2 < 0.:
            pair_strength = max(pair_strength, float(min(abs(d1),abs(d2))))
    all_regrets = (table.max(axis=1)[:, None]-table).max(axis=0)
    return dict(values_L=table[0].tolist(), values_Q=table[1].tolist(),
                distinct=len(keep), canonical_actions=[LEVELS[j] for j in keep],
                best_L=LEVELS[keep[u[0].argmax()]], best_Q=LEVELS[keep[u[1].argmax()]],
                reversal_margin=pair_strength, conflict_cost=C, headroom=float(worst[ordinary]-C),
                worst_regret_D2=float(all_regrets[1]), worst_regret_brake=float(all_regrets[0]),
                worst_regret_mean=float(worst[ordinary]), mean_action=LEVELS[keep[ordinary]],
                regret_action=LEVELS[keep[robust]], worst_regret_lambda1=float(worst[robust]),
                mean_tradeoff=float(u[:,ordinary].mean()-u[:,robust].mean()))


def measure_window(snapshot, initial_rng, x, kind, index, cfg):
    h = historical_support(snapshot, x["agent"], cfg["history_eps"])
    if h["enough"] and h["tested_tuple_errors"][int(snapshot.plant.label=="Q")] > cfg["history_eps"]:
        raise AssertionError("true branch excluded by public history predictor")
    anchor = state_digest(snapshot)
    def rollouts(horizon):
        return {law:[run_branch(snapshot, initial_rng, x, law, amount, horizon, anchor, cfg["history_eps"])
                     for amount in LEVELS] for law in ("L","Q")}
    # Every sixth scene was assigned to horizon sensitivity before results. Any
    # threshold-crossing sign reversal also gets an extended, identical rerun.
    horizon = 24 if index % 6 == 0 else 12
    runs = rollouts(horizon)
    stats = metrics(runs, x["agent"], 12)
    calls = sum(len(r["rewards"]) for block in runs.values() for r in block)
    extra = False
    if horizon == 12 and stats["reversal_margin"] > EPS_REWARD:
        horizon = 24
        runs = rollouts(horizon)
        calls += sum(len(r["rewards"]) for block in runs.values() for r in block)
        if metrics(runs, x["agent"], 12) != stats:
            raise AssertionError("horizon extension changed earlier rollout")
        extra = True
    if state_digest(snapshot) != anchor:
        raise AssertionError("mutated source snapshot")
    return dict(**x, stratum=kind, prefix_model=snapshot.plant.label, state_sha256=anchor,
                rng_sha256=digest_values(initial_rng), history=h, metrics12=stats, requested_horizon=horizon,
                metrics24=metrics(runs, x["agent"], 24) if horizon==24 else None,
                branch_count=12 if extra else 6, branch_steps=calls, traces=runs)


def scene_job(job):
    cpu_setup()
    run, index = job
    folder = output_dir(run)
    cfg = read(folder/"calibration.json")
    if not read(folder/"controls.json")["passed"]:
        raise ValueError("missing branch controls")
    label = "L" if index%2 == 0 else "Q"
    seed = DEV_SEED+index
    world = World(seed, index, cfg["w0"], label)
    world.pair_planner_events()
    rng = np.random.default_rng(np.random.SeedSequence([seed, 33417]))
    counts, populations, samples = Counter(), Counter(), {}
    elapsed = time.monotonic()
    try:
        while not world.runtime.terminal:
            replacement = {}
            for agent, enabled in enumerate(world.runtime._gates()):
                counts["all"] += 1
                if not enabled:
                    counts["closed"] += 1
                    continue
                x = observation(world, agent)
                kind = "near" if x["near"] else "other"
                counts[kind] += 1
                populations[f"{kind}|{agent}|{x['kind']}"] += 1
                if rng.integers(counts[kind]) == 0:
                    replacement[kind] = x
            for kind, x in replacement.items():
                if kind in samples:
                    samples[kind][0].close()
                samples[kind] = (world.fork(), rng_state(), x)
            advance(world)
            if world.runtime.step % 100 == 0:
                print("prefix", index, world.runtime.step, flush=True)
        rows = []
        for kind in ("near", "other"):
            if kind in samples:
                snap, initial_rng, x = samples[kind]
                rows.append(measure_window(snap, initial_rng, x, kind, index, cfg))
        result = dict(index=index, generator_seed=seed, scenario_seed=world.runtime.scenario["scenario_seed"],
                      prefix_model=label, prefix_steps=world.runtime.step, counts=dict(counts),
                      role_exposure=dict(populations), prefix_outcome=world.runtime.env.get_episode_result(),
                      max_free_prediction_error=max(world.stats["free_errors"], default=0.),
                      wall_seconds=time.monotonic()-elapsed, windows=rows)
        write_new(folder/f"scene_{index:02}.json", result)
        return dict(index=index, steps=world.runtime.step, windows=len(rows),
                    conflicts=sum(r["metrics12"]["conflict_cost"]>EPS_REWARD for r in rows))
    finally:
        for snap, _, _ in samples.values():
            snap.close()
        world.close()


def summarize(rows):
    windows = [(s, x) for s in rows for x in s["windows"]]
    eligible = [(s,x) for s,x in windows if x["metrics12"]["distinct"]>=2]
    def describe(pair, threshold):
        d = [x["metrics12"] for _, x in pair]
        return dict(sampled=len(pair), distinct_two_or_more=sum(m["distinct"]>=2 for m in d),
                    raw_reversals=sum(m["reversal_margin"]>1e-9 for m in d),
                    reversals=sum(m["reversal_margin"]>threshold for m in d),
                    conflicts=sum(m["conflict_cost"]>threshold for m in d),
                    supported_grid_conflicts=sum(x["history"]["both_grid"] and x["metrics12"]["conflict_cost"]>threshold for _,x in pair),
                    tested_tuple_conflicts=sum(x["history"]["both_tested_tuples"] and x["metrics12"]["conflict_cost"]>threshold for _,x in pair),
                    headroom_above_threshold=sum(m["headroom"]>threshold for m in d))
    def weighted(f):
        weights = [s["counts"][x["stratum"]] for s,x in windows]
        return float(np.average([f(x) for _,x in windows], weights=weights))
    stats = {}
    for key in ("reversal_margin","conflict_cost","headroom","worst_regret_mean","worst_regret_D2","worst_regret_brake","worst_regret_lambda1"):
        a = np.array([x["metrics12"][key] for _,x in eligible])
        stats[key] = dict(mean=float(a.mean()), median=float(np.median(a)), max=float(a.max()),
                          weighted_per_open_step=weighted(lambda x:x["metrics12"][key]))
    stats_by_tau = {}
    for eps in (.0125, EPS_REWARD, .05):
        stats_by_tau[str(eps)] = {kind:describe([(s,x) for s,x in windows if kind=="all" or x["stratum"]==kind],eps)
                                  for kind in ("all","near","other")}
    cluster = {}
    for kind in ("near", "other"):
        a = [x for _,x in windows if x["stratum"] == kind]
        if a and not any(x["metrics12"]["conflict_cost"]>EPS_REWARD for x in a):
            cluster[kind] = 1.-.05**(1./len(a))
    horizons = [x for _,x in windows if x["metrics24"] is not None]
    by_agent = {str(i):describe([(s,x) for s,x in windows if x["agent"]==i],EPS_REWARD) for i in range(4)}
    count = Counter()
    for s in rows:
        count.update(s["counts"])
    wcounts = Counter()
    for _,x in windows:
        wcounts[str(x["metrics12"]["distinct"])] += 1
    return dict(scenes=len(rows), prefix_steps=sum(s["prefix_steps"] for s in rows),
                exposures=dict(count), windows=len(windows), valid_windows=len(eligible), distinct_action_counts=dict(wcounts),
                by_threshold=stats_by_tau, costs_in_discounted_team_reward=stats,
                weighted_raw_rate=weighted(lambda x:x["metrics12"]["reversal_margin"]>1e-9),
                weighted_material_conflict_rate=weighted(lambda x:x["metrics12"]["conflict_cost"]>EPS_REWARD),
                weighted_material_pair_reversal_rate=weighted(lambda x:x["metrics12"]["reversal_margin"]>EPS_REWARD),
                weighted_dual_supported_conflict_rate=weighted(lambda x:x["history"]["both_tested_tuples"] and x["metrics12"]["conflict_cost"]>EPS_REWARD),
                independent_scene_zero_event_upper_95=cluster, by_agent=by_agent,
                history=dict(enough=sum(x["history"]["enough"] for _,x in windows),
                             both_grid=sum(x["history"]["both_grid"] for _,x in windows),
                             both_fixed_tuples=sum(x["history"]["both_tested_tuples"] for _,x in windows),
                             both_grid_at_0p01=sum(x["history"]["both_grid_at_0p01"] for _,x in windows)),
                extra_horizon=dict(windows=len(horizons),
                                   material_pair_reversals=sum(x["metrics24"]["reversal_margin"]>EPS_REWARD for x in horizons),
                                   material_conflicts=sum(x["metrics24"]["conflict_cost"]>EPS_REWARD for x in horizons)),
                branches=sum(x["branch_count"] for _,x in windows), branch_steps=sum(x["branch_steps"] for _,x in windows),
                prefix_models=dict(Counter(s["prefix_model"] for s in rows)),
                prefix_outcomes=dict(Counter(s["prefix_outcome"].get("termination_reason","unknown") for s in rows)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--execute", action="store_true", required=True)
    args = ap.parse_args()
    folder = output_dir(args.run)
    if not read(folder/"controls.json")["passed"]:
        raise ValueError("controls required before survey")
    settings = dict(calibration_seed=CAL_SEED, development_seed=DEV_SEED, held_out_test_used=False,
                    calibration_scenes=8, development_scenes=24, numerical_action_levels=list(LEVELS),
                    threshold=EPS_REWARD, thresholds=[.0125, EPS_REWARD, .05], horizon=12, extended_horizon=24,
                    gamma=GAMMA, rho=RHO, k=1., s=1., b=0., command_delay=0, noise="N0",
                    mission_config="configs/b2.json", action_gate="unchanged D2", training_updates=0,
                    rng="natural reset; scenario and physical-step keyed Python, NumPy, CPU Torch. Per-agent waypoint event stream.",
                    sample="one uniform reservoir per scene and pre-action stratum over all open agent-steps; population-weighted estimates",
                    direction="world-horizontal unit vector to installed tracking waypoint; zero if undefined",
                    near="0.75 < horizontal distance <= 2.4; horizontal/3D >= 0.8; closing speed >= .25 * role vmax",
                    uncertainty="two or more free measured horizontal transitions within last eight; actual sent history for lag; no latent labels as policy inputs",
                    calibration=read(folder/"calibration.json"))
    protocol = folder/"survey_protocol.json"
    if not protocol.exists():
        write_new(protocol,settings)
    elif read(protocol)!=settings:
        raise ValueError("resume protocol mismatch")
    jobs = [(args.run,i) for i in range(24) if not (folder/f"scene_{i:02}.json").exists()]
    for status in parallel_jobs(scene_job,jobs,args.workers):
        print("scene complete",status,flush=True)
    rows = [read(folder/f"scene_{i:02}.json") for i in range(24)]
    if len({s["scenario_seed"] for s in rows}) != 24:
        raise AssertionError("duplicate development scenario")
    write_new(folder/"summary.json",summarize(rows))
    print("completed full survey",flush=True)


if __name__=="__main__":
    main()
