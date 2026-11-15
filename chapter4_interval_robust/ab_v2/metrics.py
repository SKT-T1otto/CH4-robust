"""Pre-action empirical coverage and explicit competing recovery endpoints."""
import numpy as np
from ..metrics import tail_mean,wilson

class EpisodeMetrics:
    def __init__(self,c):self.config=c;self.rows=[]
    def add(self,row):self.rows.append(row)
    def summary(self,runtime,elapsed):
        rows=self.rows;n=len(rows);c=self.config
        if not n:raise ValueError('empty episode')
        outcome=rows[-1]['outcome'];box=[r['interval_coverage']['joint_box'] for r in rows]
        support=[r['interval_coverage']['conditional_support'] for r in rows if r['interval_coverage']['conditional_support'] is not None]
        recovery=[];changes=[e for e in runtime.process.events if e['kind']!='initial'];window=c['recovery']['window']
        for j,e in enumerate(changes):
            start=e['step'];end=min(n,changes[j+1]['step'] if j+1<len(changes) else n);found=None
            for k in range(start,end-window+1):
                segment=rows[k:k+window]
                same=all(np.array_equal(r['phase'],segment[0]['phase']) for r in segment)
                good=same and all(max(r['model_velocity_error'])<=c['recovery']['speed_error'] and not r['collision'] for r in segment)
                for i,phase in enumerate(segment[0]['phase']):
                    if not segment[0]['active'][i]:continue
                    last=segment[-1]['stage_diagnostics']
                    if phase in (0,2):good&=(sum(r['stage_diagnostics']['tracking_before'][i]-r['stage_diagnostics']['tracking_after'][i] for r in segment)>0 or last['tracking_after'][i]<=.8)
                    elif phase==1:good&=last['tracking_after'][i]<=1 and last['speed'][i]<=.25
                    else:good&=last['hold_progress'][i]>0
                if good:found=k+window-start;break
            terminal=rows[end-1]['outcome'] if end else outcome
            if found is not None:cause='recovered'
            elif end<n:cause='next_change'
            elif any(r['collision'] for r in rows[start:end]):cause='competing_collision'
            elif terminal['success']:cause='task_success_unobserved_recovery'
            elif terminal.get('termination_reason')=='timeout':cause='administrative_end'
            elif terminal['terminated']:cause='competing_failure'
            else:cause='administrative_end'
            recovery.append(dict(change_step=start,recovered=found is not None,recovery_steps=found,endpoint=cause,observed_steps=end-start,
                                 right_censored=cause in ('next_change','administrative_end')))
        value=runtime.summary()
        value.update(schema='auv.ch4.ab_episode.v2',method=c['method'],steps=n,success=bool(outcome['success']),collision=any(r['collision'] for r in rows),
          team_return=float(sum(r['reward'] for r in rows)),joint_box_coverage=float(np.mean(box)),conditional_joint_support=None if not support else float(np.mean(support)),
          valid_rate=float(np.mean([r['interval_coverage']['valid'] for r in rows])),
          per_dimension_coverage=np.mean([r['interval_coverage']['per_dimension'] for r in rows],0).tolist(),
          mean_width=np.mean([r['interval_coverage']['width'] for r in rows],0).tolist(),
          set_reasons={reason:sum(r['interval_coverage']['reason']==reason for r in rows) for reason in {r['interval_coverage']['reason'] for r in rows}},
          prediction_error=float(np.mean([r['model_velocity_error'] for r in rows])),recovery=recovery,
          intervention_rate=0.,wall_seconds=elapsed,seconds_per_step=elapsed/n,
          estimation_seconds=sum(r['estimation_seconds'] for r in rows),performance_passed=False,
          disturbance_events=runtime.process.events)
        return value

def aggregate(rows,alpha=.25):
    if not rows:raise ValueError('no episodes')
    returns=[r['team_return'] for r in rows];events=[e for r in rows for e in r.get('recovery',[])]
    recovered=[e['recovery_steps'] for e in events if e['recovered']];seeds=sorted({r['training_seed'] for r in rows})
    conditional=[r['conditional_joint_support'] for r in rows if r.get('conditional_joint_support') is not None]
    return dict(episodes=len(rows),success_rate=float(np.mean([r['success'] for r in rows])),success_wilson_95=wilson(sum(r['success'] for r in rows),len(rows)),
        collision_rate=float(np.mean([r['collision'] for r in rows])),mean_return=float(np.mean(returns)),lower_return_tail_mean=tail_mean(returns,alpha),
        alpha_report=alpha,joint_box_coverage=float(np.mean([r['joint_box_coverage'] for r in rows])),valid_rate=float(np.mean([r['valid_rate'] for r in rows])),
        conditional_joint_support=None if not conditional else float(np.mean(conditional)),conditional_support_episodes=len(conditional),
        per_dimension_coverage=np.mean([r['per_dimension_coverage'] for r in rows],0).tolist(),
        mean_width=np.mean([r['mean_width'] for r in rows],0).tolist(),
        intervention_rate=float(np.mean([r['intervention_rate'] for r in rows])),
        wall_seconds=float(sum(r['wall_seconds'] for r in rows)),estimation_seconds=float(sum(r['estimation_seconds'] for r in rows)),
        prediction_error=float(np.mean([r['prediction_error'] for r in rows])),recovery_events=len(events),
        recovery_unrecovered_rate=None if not events else 1-len(recovered)/len(events),recovery_conditional_mean=None if not recovered else float(np.mean(recovered)),
        recovery_endpoints={name:sum(e['endpoint']==name for e in events) for name in {e['endpoint'] for e in events}},
        per_training_seed_mean={str(s):float(np.mean([r['team_return'] for r in rows if r['training_seed']==s])) for s in seeds},
        independent_training_seeds=len(seeds),training_seed_uncertainty_established=False,performance_passed=False)
