"""Decision-time records and isolated forks of the Chapter 4 D2 simulator (CPU, N0)."""
import copy
import dataclasses
import gzip
import hashlib
import json
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch
from chapter3_bser.learning.checkpoint import digest
from chapter3_bser.learning.config import scenario_fingerprint
from chapter3_bser.learning.evaluate import preserved_rng
from chapter3_bser.learning.reference import ZeroResidualPolicy
from chapter3_bser.learning.runtime import RepairedMissionRuntime
from experiments.ch4_work1_recoverability.plant_adapter import RelativeWaterPlant, WaterParameters


SEARCH = "search_before_discovery"
EXECUTOR = "executor_after_handoff"
PHASES = (SEARCH, EXECUTOR)


def stamp(value):
    """Canonical fingerprint of the dynamic components used in the decision audit."""
    def encode(x):
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()
        if isinstance(x, np.ndarray):
            return [str(x.dtype), list(x.shape), hashlib.sha256(x.tobytes()).hexdigest()]
        if isinstance(x, np.generic):
            return encode(x.item())
        if dataclasses.is_dataclass(x):
            return {f.name: encode(getattr(x, f.name)) for f in dataclasses.fields(x)}
        if isinstance(x, dict):
            return {str(k): encode(v) for k, v in x.items()}
        if isinstance(x, (set, frozenset)):
            return {"unordered": sorted((encode(v) for v in x), key=repr)}
        if isinstance(x, (list, tuple)):
            return [encode(v) for v in x]
        if isinstance(x, float):
            return x.hex()
        return x
    return digest(encode(value))


def step_stream(seed, t):
    keys = np.random.SeedSequence([seed, t, 49031]).generate_state(4)
    return (random.Random(int(keys[0])).getstate(), np.random.RandomState(int(keys[1])).get_state(),
            torch.Generator().manual_seed(int(keys[2])).get_state())


def approaching(position, velocity, waypoint, vmax, rule):
    offset = np.asarray(waypoint) - position
    distance = np.linalg.norm(offset[:2])
    return bool(.75 < distance <= 2.4 and distance >= rule["minimum_horizontal_fraction_of_3d_distance"] * np.linalg.norm(offset)
                and np.dot(velocity[:2], offset[:2]) / distance >= rule["minimum_closing_speed_fraction_of_vehicle_xy_max"] * vmax)


def phase_for(hull, found, delivered):
    if hull < 3 and not found:
        return SEARCH
    if hull == 3 and found and delivered:
        return EXECUTOR
    return None


def delivered_view(receipt, t, seed, finder, receiver):
    """Only the native target handoff has a receiver receipt in this task."""
    if receiver != 3 or receipt is None:
        return None
    sample, arrival = receipt["payload"]["sample_step"], receipt["received_step"]
    if not 0 <= sample <= arrival <= t or receipt["physical_step"] != t:
        raise ValueError("noncausal target receipt")
    return dict(kind="target_handoff", id=f"{seed}/target/{sample}/{arrival}/{finder}/3",
                source_vehicle=finder, receiver_vehicle=3, sample_step=sample, arrival_step=arrival,
                payload=copy.deepcopy(receipt["payload"]))


class Branch:
    def __init__(self, config, scene, index, rule):
        self.seed = scene["scenario_seed"]
        self.fingerprint = scenario_fingerprint(scene)
        self.rule = rule
        if rule["horizontal_distance_m"] != "(0.75, 2.4]":
            raise ValueError("unrecognized registered distance interval")
        with preserved_rng():
            self.rt = RepairedMissionRuntime(config, scene, seed=self.seed, episode_id=index)
        self.core = self.rt.env.unwrapped
        self.plant = RelativeWaterPlant(self.core, WaterParameters()).__enter__()
        self.original_waypoint = self.core._choose_next_search_waypoint
        self.core._choose_next_search_waypoint = self._waypoint
        self.rng = step_stream(self.seed, 0)
        self.events = []
        self.prefix = digest([self.fingerprint, self.seed, "D2/N0"])
        self.policy = ZeroResidualPolicy()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        del self.core._choose_next_search_waypoint
        self.plant.__exit__(None, None, None)
        self.rt.close()

    def _waypoint(self, agent_id, reserved_positions=None):
        t, hull = self.core.step_count, int(agent_id)
        key = [self.seed, t, hull, 49032]
        repeated = self.events.count([t, hull])
        self.events.append([t, hull])
        if repeated:
            key.append(repeated)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(np.random.SeedSequence(key).generate_state(1)[0]))
            return self.original_waypoint(agent_id, reserved_positions)

    def fork(self):
        rt = self.rt
        wrappers = [rt.env, rt.env.env, rt.env.env.env, rt.controller]
        memo = {id(obj): object.__new__(type(obj)) for obj in wrappers}
        for obj in wrappers:
            memo[id(obj)].__dict__.update(copy.deepcopy(vars(obj), memo))
        other = copy.deepcopy(self, memo)
        guided = other.rt.env.env.env
        other.core._update_nav_targets = guided._original_update_nav_targets
        guided._install_navigation_hook()  # deepcopy cannot rebind the closure's captured GuidedEnv.
        if not (other.core is other.rt.env.unwrapped is other.plant.core
                and other.plant._episode is other.core.episode_outcome
                and other.core._flow_at.__self__ is other.plant
                and other.core._record_residual_prior_diagnostics.__self__ is other.plant
                and other.core._choose_next_search_waypoint.__self__ is other
                and other.original_waypoint.__self__ is other.core
                and guided._original_update_nav_targets.__self__ is other.core
                and other.rt.bridge.owner is other.rt):
            raise RuntimeError("fork contains cross-branch hook ownership")
        return other

    def identity(self):
        c, rt = self.core, self.rt
        def arrays(obj):
            return {k: v for k, v in vars(obj).items() if torch.is_tensor(v) or isinstance(v, np.ndarray)}
        return stamp([self.seed, self.fingerprint, rt.step, self.prefix, self.rng, self.plant.parameters,
                      arrays(self.plant), arrays(c), arrays(c.map_module), rt.observations, rt.guidance,
                      vars(rt.bridge.path_tracker), c.target_state, c.executor_delivered_target_state,
                      vars(c.fixed_reliable_handoff), c.get_episode_result(), rt.safe_pending,
                      rt.env.reward_accounting.last, rt.env.reward_accounting.team_discounted_return])

    def decision(self):
        rt = self.rt
        t = rt.step
        task = rt.env.get_task_state()
        target = delivered_view(rt.env.get_public_delivered_target(), t, self.seed, task.finder_id, 3)
        received = task.executor_knows_target and task.handoff_step is not None and task.handoff_step <= t and target is not None
        positions = self.core._agent_pos.tolist()
        velocities = self.core._agent_vel.tolist()
        gates = rt._gates().tolist()
        hulls = []
        for a in sorted(rt.guidance.agent_assignments, key=lambda x: x.agent_id):
            i = a.agent_id
            phase = phase_for(i, task.target_found, received)
            kind_ok = (phase == SEARCH and a.assignment_kind == "search") or (
                phase == EXECUTOR and a.assignment_kind == "executor_execution")
            controllable = bool(kind_ok and gates[i])
            near = controllable and approaching(positions[i], velocities[i], a.tracking_waypoint,
                                                 float(self.core._v_xy_max[i]), self.rule)
            hulls.append(dict(vehicle=i, role=rt.env.role_order[i], assignment=a.assignment_kind,
                              assignment_id=a.assignment_id, tracking_waypoint=list(a.tracking_waypoint),
                              final_waypoint=list(a.final_waypoint), gate=gates[i], phase=phase,
                              controllable=controllable, qualifies=near))
        return dict(t=t, found=task.target_found, handoff_step=task.handoff_step, executor_received=bool(received),
                    executor_target_receipt=target, hulls=hulls, position_m=positions, velocity_mps=velocities,
                    lag_before=self.plant.lag.tolist())

    def snapshot_data(self):
        py, np_state, cpu = self.rng
        return dict(decision=self.decision(), actor_observations=np.asarray(self.rt.observations).tolist(),
                    previous_command_mps2=self.plant.command.tolist() if hasattr(self.plant, "command") else None,
                    plant_parameters=dataclasses.asdict(self.plant.parameters),
                    navigation=[dataclasses.asdict(self.rt.bridge.path_tracker.snapshot(i)) for i in range(4)],
                    stream=dict(python=py, numpy=[np_state[0], np_state[1].tolist(), *np_state[2:]],
                                torch_cpu_hex=cpu.numpy().tobytes().hex()))

    @torch.no_grad()
    def advance(self, before):
        t = self.rt.step
        if t != before["t"] or self.rt.terminal:
            raise ValueError("stale decision or terminated branch")
        self.events = []
        with preserved_rng():
            random.setstate(self.rng[0])
            np.random.set_state(self.rng[1])
            torch.set_rng_state(self.rng[2])
            row = self.rt.advance(self.policy, explore=False)
        self.rng = step_stream(self.seed, t + 1)
        if row["action_enabled"].tolist() != [h["gate"] for h in before["hulls"]] or np.any(row["proposed_actions"]):
            raise AssertionError("decision-time gates or frozen D2 policy changed")
        outcome = self.rt.env.get_episode_result()
        # This command/response record is written only after the transition. No peer command channel exists.
        evidence = [dict(id=f"{self.seed}/motion/{t}/{i}", kind="command_response", source_vehicle=i,
                         command_step=t, sample_step=t+1, arrival_step=None, receiver_vehicle=None,
                         teammate_visibility="offline_only") for i in range(4)]
        record = dict(before=before, residual=row["actions"].tolist(), command_mps2=self.plant.command.tolist(),
                      lag_after=self.plant.lag.tolist(), thrust_mps2=self.plant.thrust_acc.tolist(),
                      position_after_m=self.core._agent_pos.tolist(), velocity_after_mps=self.core._agent_vel.tolist(),
                      team_reward=row["team_reward"], rewards=row["rewards"].tolist(), done=row["dones"],
                      termination=outcome["termination_reason"], motion_evidence=evidence,
                      planner_events=self.events.copy(), next_observations_sha256=stamp(row["next_observations"]))
        self.prefix = digest([self.prefix, digest(record)])
        return record


def phase_counts():
    return {p: dict(phase_boat_steps=0, controllable_boat_steps=0, controllable_task_steps=0,
                    qualifying_boat_steps=0, window=None) for p in PHASES}


def count_before(before, counts):
    selected = {}
    for phase in PHASES:
        items = [h for h in before["hulls"] if h["phase"] == phase]
        eligible = [h["vehicle"] for h in items if h["qualifies"]]
        c = counts[phase]
        c["phase_boat_steps"] += len(items)
        c["controllable_boat_steps"] += sum(h["controllable"] for h in items)
        c["controllable_task_steps"] += int(any(h["controllable"] for h in items))
        c["qualifying_boat_steps"] += len(eligible)
        if eligible and c["window"] is None:
            selected[phase] = min(eligible)
    return selected


def scan(job):
    group, index, protocol, config, folder = job
    from experiments.ch4_work1_recoverability.run import materialize, check_contract, completion_cost
    torch.set_num_threads(1)
    scene = materialize(protocol, group, index)
    name = f"{group}_{index:02}.jsonl.gz"
    counts = phase_counts()
    with Branch(config, scene, index, protocol["task"]["approach_rule"]) as world, gzip.open(
            Path(folder) / name, "xt", encoding="utf-8", compresslevel=3) as log:
        check_contract(world.rt, protocol)
        while not world.rt.terminal:
            before = world.decision()
            for phase, hull in count_before(before, counts).items():
                with world.fork() as snap:
                    anchor = world.identity()
                    if snap.identity() != anchor:
                        raise AssertionError("snapshot differs from the live predecision state")
                    data = snap.snapshot_data()
                    snapshot_name = f"{group}_{index:02}_{phase}_snapshot.json.gz"
                    with gzip.open(Path(folder) / snapshot_name, "xt", encoding="utf-8") as output:
                        json.dump(data, output, separators=(",", ":"), allow_nan=False)
                    counts[phase]["window"] = dict(group=group, index=index, seed=world.seed, t=before["t"],
                                                  vehicle=hull, physical_fingerprint=world.fingerprint,
                                                  phase=phase, state_sha256=anchor, prefix_sha256=world.prefix,
                                                  snapshot=snapshot_name, snapshot_data_sha256=digest(data))
            log.write(json.dumps(world.advance(before), separators=(",", ":"), allow_nan=False) + "\n")
        outcome = world.rt.env.get_episode_result()
        return dict(group=group, index=index, seed=world.seed, log=name, steps=world.rt.step,
                    termination=outcome["termination_reason"], completion_cost=completion_cost(outcome, world.rt.step, protocol["task"]),
                    team_discounted_return=world.rt.env.reward_accounting.team_discounted_return, phase_counts=counts)


def restore(config, scene, recipe, rule):
    """Rebuild a persisted decision snapshot without deserializing executable objects."""
    world = Branch(config, scene, recipe["index"], rule)
    try:
        if world.fingerprint != recipe["physical_fingerprint"] or world.seed != recipe["seed"]:
            raise ValueError("snapshot scene mismatch")
        for _ in range(recipe["t"]):
            world.advance(world.decision())
        if (world.identity() != recipe["state_sha256"] or world.prefix != recipe["prefix_sha256"]
                or digest(world.snapshot_data()) != recipe["snapshot_data_sha256"]):
            raise ValueError("independent prefix replay differs from the captured snapshot")
        return world
    except BaseException:
        world.__exit__(None, None, None)
        raise


def bind_control(world, model_label, controller_label):
    # Both aliases resolve through actual plant/actor constructors. D2 has no learned weights.
    law = {"linear_A": "linear", "linear_B": "linear"}[model_label]
    factory = {"D2_A": ZeroResidualPolicy, "D2_B": ZeroResidualPolicy}[controller_label]
    world.plant.set_parameters(WaterParameters(law=law))
    world.policy = factory()


def check_window(job):
    recipe, entry, protocol, config, folder = job
    from experiments.ch4_work1_recoverability.run import materialize
    torch.set_num_threads(1)
    scene = materialize(protocol, recipe["group"], recipe["index"])
    anchor = recipe["state_sha256"]
    with ExitStack() as stack:
        def own(branch):
            return stack.enter_context(branch)
        rule = protocol["task"]["approach_rule"]
        source = own(restore(config, scene, recipe, rule))
        direct = own(source.fork())
        reset = own(restore(config, scene, recipe, rule))
        aliases = own(source.fork())
        positive = own(source.fork())
        for branch in (direct, reset, aliases, positive):
            if branch.identity() != anchor:
                raise AssertionError("branch did not start at captured decision")
        # Mutate only a sacrificial clone, including RNG storage and navigation memory.
        isolated = own(source.fork())
        isolated.plant.lag.add_(1)
        isolated.rt.bridge.path_tracker._last_targets.clear()
        isolated.rng[1][1][0] ^= np.uint32(1)
        isolated.rng[2][0] ^= 1
        if isolated.identity() == anchor or any(w.identity() != anchor for w in (source, direct, reset, aliases, positive)):
            raise AssertionError("mutation escaped the sacrificial clone")
        bind_control(direct, "linear_A", "D2_A")
        bind_control(reset, "linear_A", "D2_A")
        bind_control(aliases, "linear_B", "D2_B")
        positive.plant.set_parameters(WaterParameters(beta=.15))
        delta = None
        steps = 0
        rewards = 0.
        with gzip.open(Path(folder) / entry["log"], "rt", encoding="utf-8") as log:
            for line in log:
                expected = json.loads(line)
                t = expected["before"]["t"]
                if t < recipe["t"]:
                    continue
                for w in (direct, reset, aliases):
                    actual = w.advance(w.decision())
                    if actual != expected:
                        keys = [k for k in expected if actual[k] != expected[k]]
                        raise AssertionError(f"paired replay diverged at {t}: {keys}")
                rewards += expected["team_reward"]
                steps += 1
                if delta is None:
                    changed = positive.advance(positive.decision())
                    if changed["command_mps2"] != expected["command_mps2"]:
                        raise AssertionError("positive control changed commands before a physical response")
                    dv = np.asarray(changed["velocity_after_mps"]) - expected["velocity_after_mps"]
                    peak = float(np.linalg.norm(dv[:, :2], axis=1).max())
                    if peak > 2e-6:
                        delta = dict(step=t, max_xy_velocity_difference_mps=peak)
        if (not steps or delta is None or source.identity() != anchor or
                any(not w.rt.terminal or w.rt.step != entry["steps"] or
                    w.rt.env.get_episode_result()["termination_reason"] != entry["termination"]
                    for w in (direct, reset, aliases))):
            raise AssertionError("incomplete terminal controls or ineffective physics intervention")
        if len({w.identity() for w in (direct, reset, aliases)}) != 1:
            raise AssertionError("terminal dynamic states differ")
        return dict(window=recipe, tail_steps=steps, termination=entry["termination"],
                    tail_team_reward_sum=rewards, direct_reset_and_archived_trace_exact=True,
                    identical_physics_and_controller_aliases_exact=True, fork_storage_isolated=True,
                    positive_beta_from=.10, positive_beta_to=.15, first_effect=delta)


def aggregate(entries, phase):
    counts = [r["phase_counts"][phase] for r in entries]
    return dict(total_scenes=len(entries), entered_scenes=sum(c["phase_boat_steps"] > 0 for c in counts),
                controllable_boat_steps=sum(c["controllable_boat_steps"] for c in counts),
                controllable_task_steps=sum(c["controllable_task_steps"] for c in counts),
                qualifying_boat_steps=sum(c["qualifying_boat_steps"] for c in counts),
                window_scenes=sum(c["window"] is not None for c in counts),
                no_window_scenes=sum(c["window"] is None for c in counts))


def run_stage(protocol, folder, workers):
    from experiments.ch4_work1_recoverability import run
    catalog = run.audit_catalog(protocol)
    source = run.ch4_source_identity()
    config = run.validate(run.read_json(run.ROOT / "configs/b2.json"))
    folder.mkdir(parents=True, exist_ok=False)
    def job(group, i):
        return group, i, protocol, config, folder
    with ProcessPoolExecutor(max_workers=workers) as pool:
        scenes = list(pool.map(scan, [job("calibration", i) for i in range(catalog["calibration"]["count"])]))
        fixture = []
        selected = []
        for phase in PHASES:
            row = next((r for r in scenes if r["phase_counts"][phase]["window"]), None)
            if row is None:
                if not fixture:
                    fixture.append(pool.submit(scan, job("legacy_regression", 23)).result())
                row = fixture[0]
            window = row["phase_counts"][phase]["window"]
            if window is None:
                raise ValueError(f"no qualifying {phase} anchor for the terminal controls")
            selected.append((window, row, protocol, config, folder))
        controls = list(pool.map(check_window, selected))
    names = [r["log"] for r in scenes + fixture]
    for scene in scenes + fixture:
        for c in scene["phase_counts"].values():
            if c["window"] is not None:
                window = c["window"]
                with gzip.open(folder / window["snapshot"], "rt", encoding="utf-8") as file:
                    if digest(json.load(file)) != window["snapshot_data_sha256"]:
                        raise ValueError("persisted snapshot content changed")
                names.append(window["snapshot"])
    files = {name: hashlib.sha256((folder / name).read_bytes()).hexdigest() for name in names}
    sources = {name: hashlib.sha256((run.HERE / name).read_bytes()).hexdigest()
               for name in ("protocol.json", "run.py", "records.py", "plant_adapter.py", "test_work1.py")}
    report = dict(passed=True, source=source, files_sha256=files, sources_sha256=sources, training_updates=0,
                  confirmation_rollouts=0, population="all eight registered calibration scenes, nominal relative water",
                  phase_summary={p: aggregate(scenes, p) for p in PHASES}, total_steps=sum(r["steps"] for r in scenes),
                  outcomes=dict(Counter(r["termination"] for r in scenes)), scenes=scenes,
                  regression_only_fixtures=fixture, terminal_controls=controls,
                  selection="first decision per scene/phase, lowest eligible vessel ID on ties; controls use lowest scene index",
                  evidence_contract="Peer command/response is offline_only; the executor target receipt is a separate delivered payload.")
    with (folder / "summary.json").open("x", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, allow_nan=False, indent=2)
    return report
