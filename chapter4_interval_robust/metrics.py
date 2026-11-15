"""Physical outcomes, empirical set diagnostics and explicitly censored recovery."""
import math
import numpy as np


def tail_mean(values,alpha=.25):
    x=np.asarray(values,float)
    if x.ndim!=1 or not len(x) or not 0<alpha<=1:raise ValueError('nonempty returns and valid alpha required')
    return float(np.sort(x)[:max(1,int(math.floor(alpha*len(x))))].mean())


def wilson(successes,count,z=1.96):
    if count==0:return None
    p=successes/count;d=1+z*z/count
    center=(p+z*z/(2*count))/d
    half=z*math.sqrt(p*(1-p)/count+z*z/(4*count*count))/d
    return [max(0.,center-half),min(1.,center+half)]


class EpisodeMetrics:
    def __init__(self,config):
        self.config=config;self.rows=[]

    def add(self,row):
        self.rows.append(row)

    def summary(self,runtime,elapsed):
        rows=self.rows;n=len(rows)
        if not n:raise ValueError('empty episode cannot be reported as successful')
        result=runtime.summary();outcome=rows[-1]['outcome']
        coverage=np.asarray([r['interval_coverage']['per_dimension'] for r in rows],bool)
        widths=np.asarray([r['interval_coverage']['width'] for r in rows],float)
        errors=np.asarray([r['model_velocity_error'] for r in rows],float)
        phases=np.asarray([r['phase'] for r in rows])
        by_phase={}
        for i,name in enumerate(('search','standby','approach','hold')):
            mask=phases==i
            by_phase[name]=dict(agent_steps=int(mask.sum()),coverage=float(coverage[mask].mean()) if mask.any() else None,
                                mean_width=float(widths[mask].mean()) if mask.any() else None,
                                mean_prediction_error=float(errors[mask].mean()) if mask.any() else None)
        recovery=[];window=self.config['recovery']['window'];threshold=self.config['recovery']['speed_error']
        changes=sorted((event for event in runtime.process.events if event['kind']!='initial'),
                       key=lambda event:event['step'])
        for event_index,event in enumerate(changes):
            start=event['step'];first=None
            next_change=changes[event_index+1]['step'] if event_index+1<len(changes) else n
            end=min(n,next_change)
            # Physical transition indices in [start, end) belong to this regime.
            # A later parameter change cannot establish recovery from this one.
            censor_reason='next_change' if next_change<n else 'episode_end'
            for index in range(start,end-window+1):
                segment=rows[index:index+window]
                # A phase transition invalidates the old phase's recovery window.
                same_phase=all(np.array_equal(r['phase'],segment[0]['phase']) for r in segment)
                stable=same_phase and all(max(r['model_velocity_error'])<=threshold and not r['collision'] for r in segment)
                # Do not call a stopped searcher recovered merely because its
                # dynamics prediction is accurate. Require phase-specific task motion.
                if stable:
                    last_state=segment[-1]['stage_diagnostics']
                    for agent,phase in enumerate(segment[0]['phase']):
                        if not segment[0]['active'][agent]:continue
                        if phase in (0,2):
                            gains=[r['stage_diagnostics']['tracking_before'][agent]-r['stage_diagnostics']['tracking_after'][agent] for r in segment]
                            stable &= sum(gains)>0 or last_state['tracking_after'][agent]<=.8
                        elif phase==1:
                            stable &= last_state['tracking_after'][agent]<=1. and last_state['speed'][agent]<=.25
                        else:
                            stable &= last_state['hold_progress'][agent]>0
                if stable:first=index+window-1;break
            recovery.append(dict(change_step=start,recovered=first is not None,
                                 recovery_steps=None if first is None else first-start+1,
                                 censored_steps=end-start if first is None else None,
                                 censor_reason=censor_reason if first is None else None,end_step=end,
                                 criterion='phase-consistent dynamics window plus progress/dwell/hold; not a safety proof'))
        result.update(method=self.config['method'],schema='auv.ch4.episode.v1',steps=n,
                      success=bool(outcome['success']),collision=any(r['collision'] for r in rows),
                      team_return=float(sum(r['reward'] for r in rows)),
                      discounted_return=float(sum(self.config['sac']['gamma']**i*r['reward'] for i,r in enumerate(rows))),
                      terminated=rows[-1]['terminated'],truncated=rows[-1]['truncated'],
                      interval_per_dimension_coverage=coverage.mean(0).tolist(),
                      interval_joint_coverage=float(coverage.all(axis=(1,2)).mean()),
                      interval_mean_width=widths.mean(0).tolist(),interval_valid_rate=float(np.mean([r['set_after']['valid'] for r in rows])),
                      prediction_error=float(errors.mean()),by_phase=by_phase,
                      recovery=recovery,intervention_rate=float(np.mean([g['intervened'] for r in rows for g in r['guard']])),
                      wall_seconds=float(elapsed),seconds_per_physical_step=float(elapsed/n),
                      disturbance_events=runtime.process.events,safety_guarantee=False,performance_acceptance=False)
        return result


def aggregate(episodes,alpha=.25):
    if not episodes:raise ValueError('no episode outcomes to aggregate')
    successes=sum(bool(r['success']) for r in episodes)
    returns=[r['team_return'] for r in episodes]
    recovery=[event for episode in episodes for event in episode.get('recovery',[])]
    recovered_steps=[event['recovery_steps'] for event in recovery if event['recovered']]
    return dict(episodes=len(episodes),success_rate=successes/len(episodes),
                success_wilson_95=wilson(successes,len(episodes)),
                collision_rate=float(np.mean([r['collision'] for r in episodes])),
                mean_return=float(np.mean(returns)),lower_return_tail_mean=tail_mean(returns,alpha),
                alpha=alpha,tail_samples=max(1,int(alpha*len(episodes))),
                joint_coverage=float(np.mean([r['interval_joint_coverage'] for r in episodes])),
                interval_valid_rate=float(np.mean([r['interval_valid_rate'] for r in episodes])),
                intervention_rate=float(np.mean([r['intervention_rate'] for r in episodes])),
                prediction_error=float(np.mean([r['prediction_error'] for r in episodes])),
                recovery_event_count=len(recovery),
                recovery_unrecovered_rate=None if not recovery else 1-len(recovered_steps)/len(recovery),
                # Conditional on observed recovery; censored events are never
                # converted into artificial recovery times.
                recovery_mean_steps=None if not recovered_steps else float(np.mean(recovered_steps)),
                wall_seconds=float(sum(r['wall_seconds'] for r in episodes)),
                independently_trained_seeds=len(set(r.get('training_seed') for r in episodes)),
                return_std_across_episodes=float(np.std(returns)),
                training_seed_uncertainty_established=False,performance_acceptance=False)
