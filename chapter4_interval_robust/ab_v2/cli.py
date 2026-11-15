"""v2 dispatcher behind the explicit --config contract."""
import json
from .configuration import load,validate
from ..provenance import identity,record

def dispatch(a):
    c=load(a.config,a.method,a.mechanism)
    if a.seed is not None:c['seed']=a.seed
    if a.domain:c['domain']=a.domain
    if a.protocol:c['disturbance']['protocol']=a.protocol
    c=validate(c);split=a.split or ('train' if a.command=='train' else 'calib' if a.command=='identify' else 'dev')
    if a.command=='matrix':
        result=dict(executed=False,schema=c['schema'],methods=list(c['methods']),factorial=['s0t0','s1t0','s0t1','s1t1'],
            schemas=c['schemas'],noise=c['noise_protocol'],splits=c['splits'],candidate_budget=c['estimation'],performance_passed=False)
    elif not a.execute:
        result=dict(executed=False,schema=c['schema'],command=a.command,method=c['method'],split=split,noise=c['noise_protocol'],
            episodes=c['episodes'] if a.command=='train' else c['evaluation_episodes'],max_steps=c['max_steps'],output=a.output,requires_explicit_execute=True)
    elif a.command=='source-record':result=record(a.reason)
    elif a.command=='source-verify':result=identity()
    elif a.command=='verify':
        from .validation import execute
        result=execute(c,a.output)
        print(json.dumps(result,ensure_ascii=False,indent=2))
        if not result['passed']:raise SystemExit(1)
        return
    else:
        if not a.output:raise ValueError('fresh output path required')
        if split=='test' and (not a.ack_final_test or not a.run_label):raise ValueError('final test acknowledgement/frozen label required')
        from .runner import run,identify,summarize
        if a.command=='identify':result=identify(c,a.output,split)
        elif a.command=='summarize':result=summarize(a.inputs,a.output)
        elif a.command in ('train','evaluate'):result=run(c,a.output,mode=a.command,split=split,validation=a.validation,weights=a.weights,resume=a.resume,run_label=a.run_label)
        else:raise ValueError('command is not a v2 research entry')
    print(json.dumps(result,ensure_ascii=False,indent=2))
