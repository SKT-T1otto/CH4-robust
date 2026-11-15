"""Plan-first commands for isolated fourth-chapter research."""
import argparse
import copy
import json
from pathlib import Path
from .configuration import ROOT,load,validate


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    for name in ('train','evaluate','verify','source-record','source-verify','summarize','upstream-reference','matrix','identify'):
        command=sub.add_parser(name)
        command.add_argument('--execute',action='store_true')
        command.add_argument('--config',default=str(ROOT/'configs/ch4/experiment.json'))
        command.add_argument('--method')
        command.add_argument('--seed',type=int)
        command.add_argument('--domain',choices=('in_domain','heldout_flow','heldout_response'))
        command.add_argument('--output')
        command.add_argument('--mechanism',action='store_true')
        command.add_argument('--protocol',choices=('nominal','episode','piecewise','drift'))
        command.add_argument('--split',choices=('train','calib','dev','test'))
        command.add_argument('--validation')
        command.add_argument('--resume')
        command.add_argument('--weights')
        command.add_argument('--reason')
        command.add_argument('--inputs',nargs='+')
        command.add_argument('--run-label')
        command.add_argument('--ack-final-test',action='store_true')
    a=p.parse_args(argv)
    if json.loads(Path(a.config).read_text(encoding='utf-8')).get('schema')=='auv.ch4.sirsa_ab.v2':
        from .ab_v2.cli import dispatch
        return dispatch(a)
    if a.command=='identify' or a.split=='calib':p.error('identify/calib require an explicit v2 config')
    c=load(a.config,a.method,a.mechanism)
    if a.seed is not None:c['seed']=a.seed;c=validate(c)
    if a.domain:c['domain']=a.domain;c=validate(c)
    if a.protocol:c['disturbance']['protocol']=a.protocol;c=validate(c)
    split=a.split or ('train' if a.command=='train' else 'dev')
    if a.command=='matrix':
        result=dict(executed=False,methods=list(c['methods']),protocols=['nominal','episode','piecewise','drift'],
                    splits=c['splits'],domains=list(c['domains']),budgets=dict(episodes=c['episodes'],max_steps=c['max_steps']),
                    acceptance=c['acceptance'],note='No commands launched; each learned ablation requires independent training.')
    elif not a.execute:
        result=dict(executed=False,command=a.command,project=str(ROOT),method=c['method'],split=split,
                    protocol=c['disturbance']['protocol'],output=a.output,mechanism=a.mechanism,
                    domain=c['domain'],
                    episodes=c['episodes'] if a.command=='train' else c['evaluation_episodes'],
                    max_steps=c['max_steps'],git_writes=False,requires_explicit_execute=True)
    elif a.command=='source-record':
        from .provenance import record
        result=record(a.reason)
    elif a.command=='source-verify':
        from .provenance import identity
        result=identity()
    elif a.command=='verify':
        if not a.output:p.error('--output is required')
        from .validation import execute
        result=execute(a.output)
        print(json.dumps(result,ensure_ascii=False,indent=2))
        if not result['passed']:raise SystemExit(1)
        return
    elif a.command=='upstream-reference':
        if not a.output:p.error('--output is required')
        from .upstream_reference import prepare
        result=prepare(a.output)
    elif a.command=='summarize':
        if not a.output or not a.inputs:p.error('--output and --inputs are required')
        from .runner import summarize
        result=summarize(a.inputs,a.output,c['risk']['alpha'])
    else:
        if not a.output:p.error('--output is required')
        if split=='test' and (not a.ack_final_test or not a.run_label):
            p.error('final test requires --ack-final-test and a frozen --run-label; do not tune on test')
        from .runner import run
        result=run(c,a.output,mode=a.command,split=split,validation=a.validation,resume=a.resume,weights=a.weights,run_label=a.run_label)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
