"""Independent score reconstruction and balanced longer-horizon/speed spot checks."""
import argparse
import hashlib
import itertools
import sys

sys.dont_write_bytecode = True
import numpy as np
from plant import (DEV_SEED, EPS_REWARD, GAMMA, HERE, LEVELS, ROOT, World, advance, array, cpu_setup,
                   current, digest_values, output_dir, parallel_jobs, read, restore_rng, rng_state,
                   state_digest, write_new)
from survey import metrics, run_branch

REPLAY_INDICES = (0, 1, 6, 7, 12, 13, 18, 19)


def replay_job(job):
    cpu_setup()
    folder, idx = job
    result = read(folder/f"scene_{idx:02}.json")
    config = read(folder/"calibration.json")
    world = World(DEV_SEED+idx, idx, config["w0"], result["prefix_model"])
    world.pair_planner_events()
    remaining = result["windows"]
    entries, ratios, replayed = [], [], 0
    try:
        while not world.runtime.terminal:
            before = rng_state() if any(x["step"]==world.runtime.step for x in remaining) else None
            for x in remaining:
                if x["step"] != world.runtime.step:
                    continue
                if state_digest(world) != x["state_sha256"]:
                    raise AssertionError(f"independent prefix disagrees in scene {idx}, step {x['step']}")
                if digest_values(before) != x["rng_sha256"]:
                    raise AssertionError("decision-time RNG state disagrees on independent replay")
                replayed += 1
                value = x["metrics24"]
                extra_steps = 0
                if value is None:
                    runs = {m:[run_branch(world, before, x, m, a, 24, x["state_sha256"], config["history_eps"])
                               for a in LEVELS] for m in ("L","Q")}
                    if metrics(runs, x["agent"], 12) != x["metrics12"]:
                        raise AssertionError("recomputed 12-step ranking changed")
                    value = metrics(runs,x["agent"],24)
                    extra_steps = sum(len(r["rewards"]) for block in runs.values() for r in block)
                entries.append(dict(stratum=x["stratum"], step=x["step"], agent=x["agent"], metrics24=value,
                                    additional_steps=extra_steps))
            if before is not None:
                restore_rng(before)
            enabled=world.runtime._gates()
            positions=array(world.core._agent_pos)
            velocities=array(world.core._agent_vel)
            w=velocities[:,:2]-current(positions)
            ratios.extend((np.linalg.norm(w,axis=1)[enabled]/config["w0"]).tolist())
            advance(world)
        if (replayed != len(remaining) or world.runtime.step != result["prefix_steps"]
                or world.runtime.env.get_episode_result() != result["prefix_outcome"]):
            raise AssertionError("natural prefix outcome did not reproduce")
        data = dict(index=idx, sample_count=len(ratios), open_relative_speed_over_w0=ratios,
                    steps=world.runtime.step, windows=entries)
        write_new(folder/f"replay_{idx:02}.json", data)
        return dict(index=idx, reproduced=replayed)
    finally:
        world.close()


def score_check(window):
    traces=window["traces"]
    x=window["metrics12"]
    q=np.array([[sum(GAMMA**k*v for k,v in enumerate(branch["rewards"][:12]))
                 for branch in traces[law]] for law in ("L","Q")])
    np.testing.assert_allclose(q,[x["values_L"],x["values_Q"]],rtol=0,atol=1e-12)
    canonical = [LEVELS.index(a) for a in x["canonical_actions"]]
    if len(canonical) != x["distinct"]:
        raise AssertionError("distinct-action count")
    q=q[:,canonical]
    gaps = [max(max(q[m])-q[m,a] for m in (0,1)) for a in range(q.shape[1])]
    margin=0.
    for a,b in itertools.combinations(range(q.shape[1]),2):
        diff=q[:,a]-q[:,b]
        if np.sign(diff[0])*np.sign(diff[1]) < 0:
            margin=max(margin,min(abs(diff)))
    avg_action=int(np.mean(q,axis=0).argmax())
    np.testing.assert_allclose([min(gaps),gaps[avg_action]-min(gaps),margin],
                               [x["conflict_cost"],x["headroom"],x["reversal_margin"]],rtol=0,atol=1e-12)
    for m in ("L","Q"):
        for b in traces[m]:
            if len(b["rewards"]) < window["requested_horizon"] and not b["terminal"]:
                raise AssertionError("silently truncated branch")
            proposed=np.array(b["first"]["proposed"])
            proposed[window["agent"]]=0.
            if np.any(proposed):
                raise AssertionError("a teammate received a nonzero residual")


def distributions(data):
    a=np.asarray(data, dtype=float)
    if not len(a):
        return None
    return dict(n=len(a), mean=float(a.mean()), median=float(np.median(a)), p95=float(np.quantile(a,.95)),
                min=float(a.min()), max=float(a.max()))


def wrong_model_cost(m):
    l=np.asarray(m["values_L"])
    q=np.asarray(m["values_Q"])
    # Evaluate on the actual model after choosing the other model's best action.
    return [float(l.max()-l[LEVELS.index(m["best_Q"])]),
            float(q.max()-q[LEVELS.index(m["best_L"])])]


def revisit_raw_flips(folder):
    # Exploratory confirmation of all raw best-action changes. Keep this selected
    # group separate from the preassigned frequency sample and balanced horizon check.
    cpu_setup()
    cfg=read(folder/"calibration.json")
    checks=[]
    for path in sorted(folder.glob("scene_*.json")):
        s=read(path)
        xs=[x for x in s["windows"] if x["metrics12"]["best_L"]!=x["metrics12"]["best_Q"]
            and x["metrics12"]["reversal_margin"]>1e-9]
        if not xs:
            continue
        world=World(s["generator_seed"],s["index"],cfg["w0"],s["prefix_model"])
        world.pair_planner_events()
        try:
            for x in sorted(xs,key=lambda r:r["step"]):
                while world.runtime.step < x["step"]:
                    advance(world)
                state=rng_state()
                if state_digest(world)!=x["state_sha256"] or digest_values(state)!=x["rng_sha256"]:
                    raise AssertionError("raw-flip replay mismatch")
                runs={m:[run_branch(world,state,x,m,a,24,x["state_sha256"],cfg["history_eps"])
                         for a in LEVELS] for m in ("L","Q")}
                restore_rng(state)
                if metrics(runs,x["agent"],12)!=x["metrics12"]:
                    raise AssertionError("raw-flip score mismatch")
                checks.append(dict(index=s["index"],step=x["step"],agent=x["agent"],
                                   metrics24=metrics(runs,x["agent"],24),traces=runs))
        finally:
            world.close()
    write_new(folder/"raw_flip_followup.json",dict(selection="exploratory: all raw 12-step winner changes; separate from prevalence",windows=checks))


def verify(folder):
    scenes=[read(folder/f"scene_{i:02}.json") for i in range(24)]
    summary=read(folder/"summary.json")
    all_items=[(s,x) for s in scenes for x in s["windows"]]
    valid=[(s,x) for s,x in all_items if x["metrics12"]["distinct"]>=2]
    for s,x in all_items:
        score_check(x)
    if len(valid)!=summary["valid_windows"]:
        raise AssertionError("valid window discrepancy")
    scene_matrix=[]
    for s in scenes:
        r=np.zeros(6)
        for x in s["windows"]:
            cnt=s["counts"][x["stratum"]]
            m=x["metrics12"]
            r+=cnt*np.array([1.,m["reversal_margin"]>1e-9,m["reversal_margin"]>EPS_REWARD,
                             m["conflict_cost"]>EPS_REWARD,m["headroom"],m["worst_regret_D2"]])
        if int(r[0])!=s["counts"].get("near",0)+s["counts"].get("other",0):
            raise AssertionError("stratum weights did not reproduce census denominator")
        scene_matrix.append(r)
    mat=np.stack(scene_matrix)
    rng=np.random.default_rng(174331)
    a=rng.choice(np.arange(0,24,2),size=(10000,12))
    b=rng.choice(np.arange(1,24,2),size=(10000,12))
    boot=mat[a].sum(axis=1)+mat[b].sum(axis=1)
    boot=boot[:,1:]/boot[:,:1]
    names=["raw_flip","material_pair_flip","material_conflict","mean_headroom","worst_regret_D2"]
    point=mat.sum(0)
    rates={k:dict(point=float(point[i+1]/point[0]),
                  bootstrap_95=np.quantile(boot[:,i],[.025,.975]).tolist()) for i,k in enumerate(names)}
    raw=[(s,x) for s,x in valid if x["metrics12"]["reversal_margin"]>1e-9]
    different=[(s,x) for s,x in valid if x["metrics12"]["best_L"]!=x["metrics12"]["best_Q"]]
    wrong=[wrong_model_cost(x["metrics12"]) for _,x in valid]
    wrong_diff=[wrong_model_cost(x["metrics12"]) for _,x in different]
    weighted_wrong=float(np.average([max(wrong_model_cost(x["metrics12"])) for s,x in all_items],
                                    weights=[s["counts"][x["stratum"]] for s,x in all_items]))
    biggest={k:max(valid,key=lambda sx:sx[1]["metrics12"][k]) for k in ("conflict_cost","headroom","worst_regret_D2")}
    details={k:dict(scene=s["index"],step=x["step"],agent=x["agent"],stratum=x["stratum"],
                    measured=x["metrics12"],support=x["history"]) for k,(s,x) in biggest.items()}
    shifts=[abs(x["metrics12"]["values_L"][1]-x["metrics12"]["values_Q"][1]) for _,x in valid]
    sh_scene,sh_window=valid[int(np.argmax(shifts))]
    rep=[read(folder/f"replay_{i:02}.json") for i in REPLAY_INDICES]
    longer=[x for r in rep for x in r["windows"]]
    ratios=np.concatenate([r["open_relative_speed_over_w0"] for r in rep])
    data=dict(passed=True, independently_rescored=len(all_items),valid=len(valid), weighted=rates,
              bootstrap="10,000 scene resamples stratified by prefix law; all-zero bootstrap has no upper-tail information",
              raw_flip_costs={k:distributions([x["metrics12"][k] for s,x in raw])
                              for k in ("conflict_cost","headroom","reversal_margin")},
              any_two_best_differ=sum(x["metrics12"]["best_L"]!=x["metrics12"]["best_Q"] for s,x in valid),
              selecting_other_models_winner=dict(worst_at_state=distributions([max(e) for e in wrong]),
                                                 mean_across_models_at_state=distributions([np.mean(e) for e in wrong]),
                                                 worst_when_winners_differ=distributions([max(e) for e in wrong_diff]),
                                                 per_open_step_weighted_worst=weighted_wrong),
              fixed_D2_return_shift=dict(distribution=distributions(shifts),above_threshold=sum(v>EPS_REWARD for v in shifts),
                                         largest=dict(scene=sh_scene["index"],step=sh_window["step"],agent=sh_window["agent"],
                                                      best_L=sh_window["metrics12"]["best_L"],best_Q=sh_window["metrics12"]["best_Q"])),
              above_numerical_margins={str(t):sum(x["metrics12"]["reversal_margin"]>t for s,x in valid)
                                       for t in (1e-9,1e-6,1e-4,.001,.0125,.025,.05)},
              worst_examples=details,free_prediction_error_max=max(s["max_free_prediction_error"] for s in scenes),
              terminal_before_12_windows=sum(any(len(b["rewards"])<12 for rows in x["traces"].values() for b in rows)
                                             for s,x in all_items),
              balanced_replay=dict(scenes=list(REPLAY_INDICES),rows=len(longer),replayed_steps=sum(r["steps"] for r in rep),
                                   material_pair_reversals=sum(x["metrics24"]["reversal_margin"]>EPS_REWARD for x in longer),
                                   material_conflicts=sum(x["metrics24"]["conflict_cost"]>EPS_REWARD for x in longer),
                                   max_headroom=max(x["metrics24"]["headroom"] for x in longer),
                                   extra_branch_steps=sum(x["additional_steps"] for x in longer),
                                   speed_ratio=distributions(ratios),fraction_above_2w0=float((ratios>=2).mean())))
    write_new(folder/"independent_check.json",data)


def integrity(folder):
    records={}
    before=read(folder/"integrity_before.json")["before"]
    for name in before:
        root=ROOT if name=="CH4_independent_original" else ROOT.parent/name
        digest=hashlib.sha256()
        n=size=0
        for path in sorted(root.rglob("*")):
            local=path.relative_to(root)
            if root==ROOT and (local.parts[0]==".git" or path.is_relative_to(HERE)):
                continue
            if not path.is_file():
                continue
            content=hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda:stream.read(1048576),b""):
                    content.update(chunk)
            digest.update(local.as_posix().encode()+bytes([0])+content.digest())
            n+=1
            size+=path.stat().st_size
        records[name]=dict(files=n, bytes=size, sha256=digest.hexdigest())
    if before!=records:
        raise AssertionError("protected content differs from its pre-experiment inventory")
    inputs=("configs/b2.json","configs/environment.json","core/env/uav_env.py",
            "chapter3_bser/learning/reference.py","chapter3_bser/learning/runtime.py",
            "chapter3_bser/learning/base_runtime.py","chapter3_bser/learning/gated_runtime.py",
            "chapter3_bser/rewards/team_objective.py","chapter3_bser/d2/assembly.py",
            "chapter3_bser/safety/safe_guidance.py")
    parent=read(ROOT/"PACKAGE_SHA256.json")
    matched={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in inputs}
    if any(parent.get(p)!=h for p,h in matched.items()):
        raise AssertionError("frozen D2 input differs from package manifest")
    files={p.relative_to(HERE).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
           for p in sorted(HERE.rglob("*")) if p.is_file()}
    dest=folder/"integrity.json"
    write_new(dest,dict(passed=True,after=records,unchanged_files=sum(x["files"] for x in records.values()),
                        package_matched=matched,created_file_sha256=files,
                        self_inventory=dest.relative_to(HERE).as_posix(),created_count_including_self=len(files)+1))


if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("mode",choices=("replay","verify","integrity","raw"))
    ap.add_argument("--run",required=True)
    ap.add_argument("--workers",type=int,default=4)
    ap.add_argument("--indices",nargs="*",type=int)
    ap.add_argument("--execute",action="store_true",required=True)
    args=ap.parse_args()
    folder=output_dir(args.run)
    if args.mode=="replay":
        wanted=REPLAY_INDICES if args.indices is None else args.indices
        if not set(wanted)<=set(REPLAY_INDICES):
            raise ValueError("unregistered replay scene")
        jobs=[(folder,i) for i in wanted if not (folder/f"replay_{i:02}.json").exists()]
        for result in parallel_jobs(replay_job,jobs,args.workers):
            print("independent replay",result,flush=True)
    elif args.mode=="raw":
        revisit_raw_flips(folder)
        print("raw-flip horizon follow-up completed",flush=True)
    elif args.mode=="verify":
        verify(folder)
        print("independent scores, weights, branch outcomes and replay check passed",flush=True)
    else:
        integrity(folder)
        print("immutable reference hashes and created-file inventory verified",flush=True)
