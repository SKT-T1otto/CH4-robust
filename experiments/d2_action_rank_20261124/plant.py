"""Frozen D2, two relative-water drag laws and decision-boundary branch checks."""
import argparse
import copy
import dataclasses
import hashlib
import json
import os
import random
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np
import torch
from chapter3_bser.learning.reference import ZeroResidualPolicy
from chapter3_bser.learning.runtime import RepairedMissionRuntime
from chapter4_interval_robust.scenarios import make_scene

CAL_SEED = 2239500
DEV_SEED = 2739500
BETA = 0.10    # m/s, amplitude of each public spatial basis component
RHO = 0.15
DT = 0.2
GAMMA = 0.95
EPS_REWARD = 0.025
ZERO = ZeroResidualPolicy()
LEVELS = (-0.5, 0.0, 0.5)


def cpu_setup():
    torch.set_num_threads(1)
    torch.set_grad_enabled(False)


def output_dir(name):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise ValueError("invalid experiment result directory")
    return HERE / "results" / name


def write_new(path, value):
    encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(encoded + "\n")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def array(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().copy()
    return np.asarray(value).copy()


def current(p, beta=BETA):
    return beta * np.stack((np.sin(0.25*p[..., 1]), np.cos(0.22*p[..., 0])), axis=-1)


def model_step(p, v, z, drag, accel, dt, w0, mix, beta=BETA, k=1., s=1.):
    """Independent NumPy horizontal prediction. Batch shapes may include candidates."""
    vplus = v + dt * accel * s * z
    w = vplus - current(p, beta)
    return vplus - dt * drag * k * (1.0 - mix + mix*np.linalg.norm(w, axis=-1, keepdims=True)/w0) * w


def rng_state():
    return (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())


def restore_rng(state):
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])


def new_scene(seed):
    return make_scene({"splits": {"dev": seed}, "max_steps": 400}, "dev", 0)


class Pulse:
    def __init__(self, agent, direction, amount):
        self.values = torch.zeros((4, 3))
        self.values[agent, :2] = torch.tensor(np.asarray(direction) * amount)

    def step(self, observations, explore=False):
        return list(self.values.split(1))


class Plant:
    """Two instance hooks retain the parent's reference, clipping and entire mission step."""
    def __init__(self, core, w0, label):
        self.core = core
        self.w0 = w0
        self.set_law(label)
        self.nominal_drag = core._drag_xy.clone()
        self.limits = torch.stack((core._a_xy_max, core._a_xy_max, core._a_z_max), dim=1)
        self.z = torch.zeros_like(self.limits)
        self.sent = torch.zeros_like(self.z)
        self.old_diag = core._record_residual_prior_diagnostics
        core._record_residual_prior_diagnostics = self.on_composed_command
        # The replacement returns the entire horizontal damping term. Vertical stays native.
        core._drag_xy = torch.zeros_like(self.nominal_drag)
        core._flow_at = self.horizontal_acceleration

    def set_law(self, label, identical=False):
        if label not in ("L", "Q"):
            raise ValueError(label)
        self.label = label
        self.mix = 0.0 if identical else 0.5

    @property
    def coefficient(self):
        return 0. if self.label == "L" else self.mix

    def on_composed_command(self, prior, residual, final):
        self.old_diag(prior, residual, final)
        self.sent = (final / self.limits).clone()
        self.z = RHO*self.z + (1.-RHO)*self.sent
        final.copy_(self.z*self.limits)

    def horizontal_acceleration(self, position):
        core = self.core
        c = BETA * torch.stack((torch.sin(.25*position[:, 1]), torch.cos(.22*position[:, 0])), dim=1)
        w = core._agent_vel[:, :2] + core.dt*core._agent_acc[:, :2] - c
        factor = 1. if self.label == "L" else (1.-self.mix) + self.mix*torch.linalg.vector_norm(w, dim=1, keepdim=True)/self.w0
        out = torch.zeros_like(position)
        out[:, :2] = -self.nominal_drag[:, None]*factor*w
        return out


class World:
    def __init__(self, seed, index, w0, law):
        scene = new_scene(seed)
        self.seed = seed
        self.runtime = RepairedMissionRuntime(read(ROOT/"configs/b2.json"), scene,
                                              seed=int(scene["scenario_seed"]), episode_id=index)
        self.core = self.runtime.env.unwrapped
        self.plant = Plant(self.core, w0, law)
        contract = self.runtime.env.get_public_control_contract()
        if (contract["dt"], contract["prior_slow_radius_xy"], contract["residual_scale_search"],
                contract["residual_scale_executor"], contract["use_residual_prior"]) != (.2, 2.4, .2, .15, True):
            raise ValueError("unexpected parent control contract")
        self.history = []
        self.stats = {"free_errors": [], "visited_speeds": []}

    def pair_planner_events(self):
        # In the mission step this is the only site that enters a stochastic
        # search-waypoint selection. One source event per searcher per physical step.
        self._original_next_waypoint = self.core._choose_next_search_waypoint
        self.core._choose_next_search_waypoint = self.paired_next_waypoint

    def paired_next_waypoint(self, agent_id, reserved_positions=None):
        state = torch.get_rng_state()
        key = [self.seed, self.core.step_count, int(agent_id), 49032]
        local_seed = int(np.random.SeedSequence(key).generate_state(1)[0])
        torch.random.default_generator.manual_seed(local_seed)
        try:
            return self._original_next_waypoint(agent_id, reserved_positions)
        finally:
            torch.set_rng_state(state)

    def step(self, policy=ZERO):
        core = self.core
        p, v = array(core._agent_pos), array(core._agent_vel)
        active = ~array(core.agent_finished)
        row = self.runtime.advance(policy, explore=False)
        p1, v1 = array(core._agent_pos), array(core._agent_vel)
        size = array(core.space_size)
        speed = np.linalg.norm(v1[:, :2], axis=1)
        valid = (active & ~array(core.agent_finished) & ~array(core._collision_flags) &
                 ~(np.minimum(p1, size-p1) < 1e-5).any(axis=1) &
                 (speed < array(core._v_xy_max)-1e-5))
        pred = model_step(p.astype(float), v[:, :2].astype(float), array(self.plant.z)[:, :2],
                          array(self.plant.nominal_drag)[:, None], array(core._a_xy_max)[:, None],
                          core.dt, self.plant.w0, self.plant.coefficient)
        errors = np.max(np.abs(pred-v1[:, :2]), axis=1)
        self.stats["free_errors"].extend(errors[valid].tolist())
        moving = valid & (speed > 0.2)
        self.stats["visited_speeds"].extend(np.linalg.norm(v1[:, :2]-current(p1), axis=1)[moving].tolist())
        self.history.append((p, v, array(self.plant.sent), v1, valid))
        return row

    def fork(self):
        # The guided-navigation hook is a closure over its owning wrapper. Rebind it
        # on the new graph. Explicit shells preserve cyclic forwarding-wrapper links.
        rt = self.runtime
        wrappers = [rt.env, rt.env.env, rt.env.env.env, rt.controller]
        memo = {id(src): object.__new__(type(src)) for src in wrappers}
        for src in wrappers:
            memo[id(src)].__dict__.update(copy.deepcopy(vars(src), memo))
        other = copy.deepcopy(self, memo)
        guided = other.runtime.env.env.env
        other.core._update_nav_targets = guided._original_update_nav_targets
        guided._install_navigation_hook()
        if not (other.runtime.env.unwrapped is other.core is other.plant.core and
                other.core._flow_at.__self__ is other.plant and
                other.core._record_residual_prior_diagnostics.__self__ is other.plant and
                guided._original_update_nav_targets.__self__ is other.core and
                other.runtime.bridge.owner is other.runtime):
            raise AssertionError("cross-branch owner reference")
        return other

    def close(self):
        self.runtime.close()


def digest_values(*values):
    h = hashlib.sha256()
    def feed(x):
        if isinstance(x, (torch.Tensor, np.ndarray)):
            a = np.ascontiguousarray(array(x))
            h.update(str((a.dtype.str, a.shape)).encode() + b"@" + a.tobytes())
        elif dataclasses.is_dataclass(x):
            feed(dataclasses.asdict(x))
        elif isinstance(x, dict):
            for key in sorted(x, key=str):
                feed(key)
                feed(x[key])
        elif isinstance(x, (list, tuple)):
            h.update(str(len(x)).encode()+b"[")
            for part in x:
                feed(part)
            h.update(b"]")
        else:
            h.update(repr(x).encode()+b"|")
    for value in values:
        feed(value)
    return h.hexdigest()


def state_digest(world):
    r, env = world.runtime, world.core
    maps = {k: v for k, v in vars(env.map_module).items() if isinstance(v, (torch.Tensor, np.ndarray))}
    tensors = {k: v for k, v in vars(env).items() if isinstance(v, torch.Tensor)}
    return digest_values(tensors, maps, r.observations, r.step, r.guidance, r.bridge.path_tracker._state,
                         env.target_state, env.executor_delivered_target_state, r.env.reward_accounting.last,
                         r.env.reward_accounting.team_discounted_return, r.env.env.reward_adapter.__dict__,
                         world.plant.z, world.plant.sent, world.history, r.safe_pending,
                         r.env.get_episode_result())


def observation(world, agent):
    item = next(a for a in world.runtime.guidance.agent_assignments if a.agent_id == agent)
    q = np.array(item.tracking_waypoint)
    p = array(world.core._agent_pos)[agent]
    delta = q-p
    dist = np.linalg.norm(delta[:2])
    direction = delta[:2]/dist if dist > 1e-8 else np.zeros(2)
    v = array(world.core._agent_vel)[agent, :2]
    near = (.75 < dist <= 2.4 and dist >= .8*np.linalg.norm(delta) and
            np.dot(v, direction) >= .25*float(world.core._v_xy_max[agent]))
    return dict(agent=agent, step=world.runtime.step, direction=direction.tolist(), target=q.tolist(),
                distance_xy=float(dist), near=bool(near), kind=item.assignment_kind,
                final_target=list(item.final_waypoint), assignment=str(item.assignment_id),
                post_found=bool(world.core.task_found))


def step_rng(seed, step):
    # Per-step substreams prevent earlier branch-dependent draws leaking into later
    # steps. CPU Torch is the parent's stochastic planning source. N0 has no plant noise.
    seeds = np.random.SeedSequence([seed, step, 49031]).generate_state(4)
    random.seed(int(seeds[0]))
    np.random.seed(int(seeds[1]))
    torch.random.default_generator.manual_seed(int(seeds[2]))


def advance(world, policy=ZERO):
    step_rng(world.seed, world.runtime.step)
    return world.step(policy)


def calibration_job(index):
    cpu_setup()
    world = World(CAL_SEED+index, index, 1., "L")
    try:
        for _ in range(32):
            if world.runtime.terminal:
                break
            advance(world)
        speeds = world.stats["visited_speeds"]
        return dict(index=index, generator_seed=world.seed, scenario_seed=world.runtime.scenario["scenario_seed"],
                    steps=world.runtime.step, moving=len(speeds), free=len(world.stats["free_errors"]),
                    speed_median=float(np.median(speeds)) if speeds else None,
                    max_free_prediction_error=max(world.stats["free_errors"], default=0.0))
    finally:
        world.close()


def parallel_jobs(fn, jobs, workers):
    if workers == 1:
        for job in jobs:
            yield fn(job)
        return
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = [pool.submit(fn, job) for job in jobs]
        for result in as_completed(pending):
            yield result.result()


def calibrate(folder, workers):
    existing = [i for i in range(8) if not (folder/f"cal_{i:02}.json").exists()]
    for row in parallel_jobs(calibration_job, existing, workers):
        write_new(folder/f"cal_{row['index']:02}.json", row)
        print("calibration", row["index"], row["steps"], flush=True)
    rows = [read(folder/f"cal_{i:02}.json") for i in range(8)]
    w0 = float(np.median([r["speed_median"] for r in rows if r["moving"]]))
    eps = max(2e-6, 1.25*max(r["max_free_prediction_error"] for r in rows))
    write_new(folder/"calibration.json", dict(w0=w0, beta=BETA, rho=RHO, dt=DT, history_eps=eps,
              rule="median of per-scene medians of free moving relative speeds at end of steps",
              speed_unit="m/s", beta_unit="m/s per component of declared spatial basis",
              counts=[r["moving"] for r in rows], free_steps=sum(r["free"] for r in rows),
              law_calibration="independent linear-law N0 prefixes up to 32 steps"))


def checked_trace(world, x, amount, steps=12):
    trace = []
    for j in range(steps):
        if world.runtime.terminal:
            break
        row = advance(world, Pulse(x["agent"], x["direction"], amount) if j == 0 else ZERO)
        trace.append(digest_values(state_digest(world), rng_state(), row["team_reward"], row["dones"],
                                   row["proposed_actions"], row["actions"]))
    return trace


def controls(folder):
    cpu_setup()
    cfg = read(folder/"calibration.json")
    seed = CAL_SEED
    world = World(seed, 0, cfg["w0"], "L")
    world.pair_planner_events()
    replay = None
    try:
        # A live pre-decision anchor with nonzero memory and a nonzero legal pulse.
        for _ in range(6):
            advance(world)
        agent = next(i for i, g in enumerate(world.runtime._gates()) if g)
        x = observation(world, agent)
        if np.linalg.norm(x["direction"]) < .99 or np.max(np.abs(array(world.plant.z))) <= 0.:
            raise AssertionError("degenerate control anchor")
        anchor = state_digest(world)
        replay = World(seed, 0, cfg["w0"], "L")
        replay.pair_planner_events()
        for _ in range(world.runtime.step):
            advance(replay)
        if state_digest(replay) != anchor:
            raise AssertionError("independent prefix replay mismatch")
        passed = {}
        def run(label, amount, identical=False):
            branch = world.fork()
            try:
                if state_digest(branch) != anchor:
                    raise AssertionError("fork mismatch at decision")
                branch.plant.set_law(label, identical)
                t = checked_trace(branch, x, amount)
                if len(t) != 12:
                    raise AssertionError("a blank control terminated before 12 steps")
                if state_digest(world) != anchor:
                    raise AssertionError("source mutated by branch")
                if max(branch.stats["free_errors"], default=0.) > cfg["history_eps"]:
                    raise AssertionError("independent horizontal integration disagrees")
                return t
            finally:
                branch.close()
        pure = run("L", .5)
        actual = checked_trace(replay, x, .5)
        if pure != actual or run("L", .5) != pure:
            raise AssertionError("same law/action replay differs")
        passed["identical_law_action"] = dict(steps=12, exact=True, trace=pure, independent_replay=True)
        blank = {}
        for a in LEVELS:
            one = run("L", a)
            other = run("Q", a, identical=True)
            if one != other:
                raise AssertionError("same drag curve with different model label differs")
            blank[str(a)] = one
        passed["identical_drag_curves"] = dict(steps=12, actions=len(blank), false_flips=0, exact=True, traces=blank)
        different = run("Q", .5)
        if different == pure or len({tuple(t) for t in blank.values()}) < 2:
            raise AssertionError("model or action positive control has no effect")
        equilibrium = []
        for ratio in (0., 1., 2.):
            speed = ratio*cfg["w0"]
            equilibrium.append(dict(relative_speed=speed, linear=speed, mixed=(.5+.5*ratio)*speed))
        write_new(folder/"controls.json", dict(passed=True, anchor=x, anchor_digest=anchor, controls=passed,
                  true_mixed_law_diverged=True, analytic_drag_per_unit_damping=equilibrium,
                  fork="entire in-memory D2 graph; re-bound guidance closure; independent reset-and-prefix replay"))
    finally:
        if replay is not None:
            replay.close()
        world.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("calibrate", "controls"))
    ap.add_argument("--run", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--execute", action="store_true", required=True)
    args = ap.parse_args()
    out = output_dir(args.run)
    if args.stage == "calibrate":
        calibrate(out, args.workers)
    else:
        controls(out)
        print("same-state and both blank controls passed", flush=True)
