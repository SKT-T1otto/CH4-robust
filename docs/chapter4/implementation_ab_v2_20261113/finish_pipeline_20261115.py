"""Bounded, fresh-output CLI exercise for the current v2 source. CPU only.

This script does not search checkpoints or run a research matrix. It restores
only the two temporary checkpoints it creates during this invocation.
"""
import hashlib,json,os,subprocess,sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
PREFIX='work/ch4/ab_v2_final_v9_20261115'
REPORT=HERE/'pipeline_20261115_002.json'

def equal(a,b,where='root'):
    if isinstance(a,torch.Tensor):
        if not isinstance(b,torch.Tensor) or not torch.equal(a,b):raise AssertionError(where)
    elif isinstance(a,np.ndarray):
        if not isinstance(b,np.ndarray) or not np.array_equal(a,b):raise AssertionError(where)
    elif isinstance(a,dict):
        if a.keys()!=b.keys():raise AssertionError(where+' keys')
        for key in a:equal(a[key],b[key],where+'.'+str(key))
    elif isinstance(a,(tuple,list)):
        if len(a)!=len(b):raise AssertionError(where+' length')
        for i,(x,y) in enumerate(zip(a,b)):equal(x,y,where+'.'+str(i))
    elif a!=b:raise AssertionError(where)

def main():
    if REPORT.exists():raise FileExistsError(REPORT)
    os.environ['CUDA_VISIBLE_DEVICES']='-1';os.environ['PYTHONDONTWRITEBYTECODE']='1'
    steps=[]
    def command(name,args):
        full=[sys.executable,'-B','ch4_cli.py',*args]
        result=subprocess.run(full,cwd=ROOT,env=os.environ.copy(),capture_output=True,text=True,encoding='utf-8')
        (HERE/(REPORT.stem+'_'+name+'_stdout.txt')).open('x',encoding='utf-8').write(result.stdout)
        (HERE/(REPORT.stem+'_'+name+'_stderr.txt')).open('x',encoding='utf-8').write(result.stderr)
        steps.append(dict(name=name,args=full,exit_code=result.returncode))
        print(name,'exit',result.returncode,flush=True)
        if result.returncode:raise RuntimeError(name+' failed; logs retained')
        return json.loads(result.stdout)
    common=['--config','configs/ch4/experiment_ab_v2.json','--execute','--mechanism']
    train=PREFIX+'_train_001';evaluation=PREFIX+'_eval_001';resume=PREFIX+'_resume_001';nominal=PREFIX+'_nominal_001';identify=PREFIX+'_identify_001'
    try:
        a=command('final_train',['train',*common,'--protocol','piecewise','--output',train])
        assert a['total_updates']<=4 and a['total_physical_steps']<=64
        b=command('final_eval',['evaluate',*common,'--protocol','piecewise','--split','dev','--weights',train+'/checkpoint_00002.pt','--output',evaluation])
        assert b['total_updates']==0 and b['physical_steps_executed_this_run']<=32
        r=command('final_resume',['train',*common,'--protocol','piecewise','--resume',train+'/checkpoint_00001.pt','--output',resume])
        assert r['updates_executed_this_run']<=1 and r['physical_steps_executed_this_run']<=16
        sys.path.insert(0,str(ROOT))
        from chapter4_interval_robust.ab_v2.checkpoint import load
        continuous=load(ROOT/train/'checkpoint_00002.pt');resumed=load(ROOT/resume/'checkpoint_00002.pt')
        for key in ('learner','numpy_global_rng','python_global_rng','next_episode','config'):
            equal(continuous[key],resumed[key],key)
        original=[json.loads(line) for line in (ROOT/train/'transitions.jsonl').read_text(encoding='utf-8').splitlines() if json.loads(line)['episode']==1]
        repeated=[json.loads(line) for line in (ROOT/resume/'transitions.jsonl').read_text(encoding='utf-8').splitlines()]
        for row in original+repeated:row.pop('estimation_seconds',None)
        equal(original,repeated,'physical_trajectory')
        n=command('final_nominal_eval',['evaluate',*common,'--protocol','nominal','--split','dev','--weights',train+'/checkpoint_00002.pt','--output',nominal])
        summary=command('final_summary',['summarize','--config','configs/ch4/experiment_ab_v2.json','--execute','--inputs',evaluation,nominal,'--output',PREFIX+'_summary_001'])
        disturbed=next(v for k,v in summary['groups'].items() if ':piecewise:' in k)
        assert disturbed['paired_nominal_scene_count']==2
        i=command('final_identify',['identify',*common,'--protocol','piecewise','--split','calib','--output',identify])
        calls=[v['prediction_calls'] for v in i['groups'].values()]
        assert len(calls)==4 and len(set(calls))==1
        with (ROOT/identify/'identification.jsonl').open(encoding='utf-8') as f:
            records=[json.loads(line) for line in f]
        assert len(records)==2*16*4
        report=dict(passed=True,source=continuous['source'],steps=steps,train=a,evaluation=b,resume=r,nominal=n,
            identification=i,physical_and_optimizer_resume_equal=True,paired_nominal_summary=True,
            matching_factorial_prediction_budget=True,model_budget=dict(train_steps=a['total_physical_steps'],
                eval_steps=b['physical_steps_executed_this_run'],nominal_eval_steps=n['physical_steps_executed_this_run'],
                total_continuous_policy_episodes=6,total_continuous_policy_steps=96,total_updates=a['total_updates']),
            checkpoint_scope='only checkpoints created by this script',historical_models_loaded=False,
            formal_matrix_started=False,performance_passed=False)
    except Exception as exc:
        report=dict(passed=False,steps=steps,error=repr(exc),performance_passed=False)
        raise
    finally:
        with REPORT.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print('PIPELINE PASSED',flush=True)

if __name__=='__main__':main()
