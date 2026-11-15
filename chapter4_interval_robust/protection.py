"""Optional matched public finite-scenario probe; no safety certificate."""
from collections import deque
import numpy as np
from core.mapping.planning_state import extract_planning_state
from chapter3_bser.safe_rl.evidence import ScanEvidence,EvidenceGeometry
from .dynamics import predict_velocity


class PublicProtection:
    def __init__(self,env,options,specs,contract,seed):
        self.env=env;self.options=options;self.specs=specs;self.contract=contract
        self.rng=np.random.default_rng(np.random.SeedSequence([seed,4301]))
        self.evidence=ScanEvidence();self.commands=deque(maxlen=17)

    def _simulate(self,p,v,command,scenarios,geometry,active):
        dt=self.contract['dt'];q=np.broadcast_to(p,(len(scenarios),4,3)).copy()
        vel=np.broadcast_to(v,q.shape).copy()
        # Bound unknown actuator memory by both extreme outputs, not a privileged
        # read of the simulator's queue/lag state. Still only finitely sampled.
        for sign in (-1.,1.):
            position=q.copy();velocity=vel.copy();filtered=np.full_like(vel,sign)
            commands=[x.copy() for x in self.commands]
            for k in range(self.options['horizon']):
                scale=np.array([[s['a_xy_max']]*2+[s['a_z_max']] for s in self.specs])
                control=command if k==0 else np.clip(-velocity.mean(0)/(dt*scale),-1,1)
                commands.append(control)
                delivered=np.zeros_like(vel)
                for n in range(len(scenarios)):
                    for i in range(4):
                        index=len(commands)-1-int(scenarios[n,i,7])
                        delivered[n,i]=commands[index][i] if index>=0 else 0
                lag=scenarios[:,:,6,None]
                filtered=lag*filtered+(1-lag)*delivered
                noise=self.rng.normal(size=filtered.shape)*scenarios[:,:,8,None]
                nxt=predict_velocity(velocity,np.clip(filtered+noise,-1,1),scenarios,self.specs,dt,position)
                nxt[:,~active]=0. # Match the inherited inactive-agent freeze.
                new_position=position+nxt*dt
                for n in range(len(scenarios)):
                    for i in range(4):
                        point=new_position[n,i]
                        if np.any(point<0) or np.any(point>self.contract['bounds']):return False
                        if not geometry.segment(position[n,i],point):return False
                position,velocity=new_position,nxt
            if np.max(np.linalg.norm(velocity,axis=-1))>.15:return False
        return True

    def choose(self,proposal,estimate,previous,active):
        command=np.clip(np.asarray(proposal,float),-1,1)
        command[~active]=0
        if self.options['mode']=='off':
            self.commands.append(command.copy())
            return command,[dict(status='protection_off',intervened=False,safety_guarantee=False) for _ in range(4)]
        self.evidence.consume(self.env.read_public_events(self.evidence.cursor))
        state=extract_planning_state(self.env)
        geometry=EvidenceGeometry(state,self.evidence,self.options)
        p=np.asarray([a.position for a in state.agents]);v=np.asarray([a.velocity for a in state.agents])
        scenarios=estimate.joint_samples(self.rng,self.options['samples'])
        supported=estimate.valid and self._simulate(p,v,command,scenarios,geometry,active)
        if supported:status='finite_scenarios_checked'
        else:
            scale=np.array([[s['a_xy_max']]*2+[s['a_z_max']] for s in self.specs])
            command=np.clip(-v/(self.contract['dt']*scale),-1,1);command[~active]=0
            status='no_verified_candidate' if not estimate.valid else 'limited_unverified_brake'
        rows=[dict(status=status,intervened=bool(np.max(np.abs(command[i]-proposal[i]))>1e-6),
                   safety_guarantee=False,unknown_space_certified=False) for i in range(4)]
        self.commands.append(command.copy())
        return command,rows

    def last_prediction_error(self,p,v,command,next_v,estimate):
        # One-step nominal-response diagnostic; delayed candidates are assessed by
        # the estimator separately. Never used as a hidden-state policy feature.
        center=(estimate.lower+estimate.upper)/2
        prediction=predict_velocity(v,command,center,self.specs,self.contract['dt'],p)
        return np.linalg.norm(prediction-next_v,axis=1).tolist()
