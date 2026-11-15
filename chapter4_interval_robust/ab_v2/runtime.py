"""Unchanged parent mission with public165 input and pre-action context33 labels."""
import copy,time
import numpy as np
import torch
from chapter3_bser.learning.base_runtime import BaselineMissionRuntime,_make_base_env
from ..runtime import public_nominal,public_phase_codes
from ..configuration import reference_config
from ..dynamics import DisturbanceProcess,PhysicalIntervention
from ..scenarios import make_scene as parent_scene
from .configuration import physical_options
from .estimation import History,Estimate,ConditionalEstimator,diagnostics
from .parameters import PolicyInput,from_legacy,context

def make_scene(c,split,index):
    if split not in c['splits'] or index<0:raise ValueError('isolated v2 scene split required')
    # Parent generator accepts train/dev/test; calib is its own seed block.
    translated=copy.deepcopy(c);translated['splits']={'train':c['splits'][split],'dev':c['splits'][split],'test':c['splits'][split]}
    scene=parent_scene(translated,'train' if split=='train' else 'dev',index)
    rng=np.random.default_rng(np.random.SeedSequence([scene['disturbance_seed'],6301]))
    scene.update(scenario_id=f"ch4_ab_v2_{split}_{c['splits'][split]+index}",scenario_role=split,scenario_split=split,
                 change_steps=sorted(int(rng.integers(lo,hi+1)) for lo,hi in c['disturbance']['change_windows']),
                 ch4_split_schema='ch4.scenes.calib4.v2')
    return scene

class Intervention(PhysicalIntervention):
    def apply(self,actions):
        pre=self.process.actuator.filtered.copy()
        result=super().apply(actions);self.last['pre_filter']=pre
        return result

class ReferenceRuntime(BaselineMissionRuntime):
    def __init__(self,c,scene,*,seed,episode_id,identifier=None):
        self.ch4=c;self.method=c['method_options'];self.identifier=identifier;self.history=History()
        self.previous_command=np.zeros((4,3),np.float32);self.estimate=Estimate(c['prior'])
        if self.method['estimator']=='fixed':self.estimate.valid[:]=True;self.estimate.reason='declared_fixed_prior'
        self.query_rng=np.random.default_rng(np.random.SeedSequence([seed,6401]));self.estimator=None
        super().__init__(reference_config(c),scene,seed=seed,episode_id=episode_id)
        self.specs=self.env.get_public_vehicle_specs();self.contract=self.env.get_public_control_contract()
        self.process=DisturbanceProcess(physical_options(c,scene),scene['disturbance_seed'])
        self.intervention=Intervention(self.env.unwrapped,self.process)
        if self.method['estimator']=='conditional':
            self.estimator=ConditionalEstimator(c['prior'],self.specs,self.contract['dt'],c['estimation'],seed,
                                                shared=self.method['shared'],aligned=self.method['aligned'])
        self.last_guard=[]
    @staticmethod
    def _make_environment(config):
        env=_make_base_env(config);env.unwrapped.use_residual_prior=False;env.unwrapped._residual_scale.fill_(1.)
        return env
    def policy_input(self):
        o=np.asarray(self.observations,np.float32)
        phases=np.eye(4)[public_phase_codes(self.env,o)]
        base=np.concatenate((o,np.eye(4),phases,self.previous_command,public_nominal(self.env),
                             np.full((4,1),self.step/self.ch4['max_steps'])),1)
        values=np.concatenate((base,self.estimate.features(self.method['input']),self.history.arrays().transpose(1,0,2).reshape(4,104)),1)
        return PolicyInput(values.astype(np.float32),o[:,20]<.5)
    def _actions_for_step(self,actions):
        self.proposal=actions.detach().cpu().numpy().copy();command=np.clip(self.proposal,-1,1)
        active=np.asarray(self.observations)[:,20]<.5;command[~active]=0
        self.last_guard=[dict(status='protection_off',intervened=False,safety_guarantee=False) for _ in range(4)]
        return torch.tensor(command,dtype=torch.float32)
    def transition(self,proposal):
        before=self.policy_input();old=self.estimate
        bank=old.bank(self.query_rng,self.ch4['risk']['samples'],before.active)
        public=self.env.get_agent_state();p=np.asarray(public.positions);v=np.asarray(public.velocities)
        phases=public_phase_codes(self.env,np.asarray(self.observations));nominal=public_nominal(self.env)
        class Fixed:
            def step(self,observations,explore=True):return [torch.tensor(proposal[i:i+1],dtype=torch.float32) for i in range(4)]
        row=super().advance(Fixed(),explore=False)
        state=self.env.get_agent_state();nv=np.asarray(state.velocities);np_=np.asarray(state.positions)
        diag=self.intervention.last;u=diag['sent_command'];collisions=np.asarray(state.collision_flags,bool)
        # Eligibility uses only public geometry, speed and phase state.
        limits=np.array([s['v_xy_max'] for s in self.specs]);zlimits=np.array([s['v_z_max'] for s in self.specs])
        eligible=(before.active&(np.asarray(self.observations)[:,20]<.5)&~collisions&
                  (np.linalg.norm(nv[:,:2],axis=1)<limits-1e-5)&(np.abs(nv[:,2])<zlimits-1e-5)&
                  ~np.any((np_<=1e-5)|(np_>=np.asarray(self.contract['bounds'])-1e-5),axis=1))
        started=time.perf_counter();self.history.append(p,v,u,nv,eligible)
        if self.estimator:self.estimate=self.estimator.update(p,v,u,nv,eligible)
        elif self.method['estimator']=='ensemble':
            if self.identifier is None:raise ValueError('trained identifier interface required')
            self.estimate=self.identifier.infer(self.history.arrays())
        else:
            self.estimate=Estimate(self.ch4['prior'],self.history.arrays())
            self.estimate.valid[:]=True;self.estimate.reason='declared_fixed_prior'
        estimation_seconds=time.perf_counter()-started
        self.previous_command=u.astype(np.float32);after=self.policy_input();truth=from_legacy(diag['truth'].local)
        # Scores use the pre-action estimate; post-action self-fit is separate.
        pre_score=diagnostics(old,truth,self.ch4['estimation'])
        new_score=diagnostics(self.estimate,truth,self.ch4['estimation'])
        candidate_error=np.full(4,np.nan)
        from ..dynamics import predict_velocity
        from .estimation import reconstruct
        from .parameters import to_legacy
        center=(old.lower+old.upper)/2;center[:,0]=center[:,0].mean();center[:,5]=np.round(center[:,5])
        # Reconstruct memory then apply this command with the hypothesized delay.
        h=old.history;z=reconstruct(h,center);commands=[f[:,6:9] for f in h if f[0,12]>=0]+[u]
        delivered=np.stack([commands[-1-int(center[i,5])][i] if len(commands)>int(center[i,5]) else np.zeros(3) for i in range(4)])
        z=center[:,4,None]*z+(1-center[:,4,None])*delivered
        candidate_error=np.linalg.norm(predict_velocity(v,z,to_legacy(center),self.specs,self.contract['dt'],p)-nv,axis=1)
        result=dict(observations=before.observations,active=before.active,proposal=self.proposal.copy(),
            next_observations=after.observations,next_active=after.active,reward=float(row['team_reward']),
            rewards=np.asarray(row['rewards']),terminated=bool(self.env.get_episode_result()['terminated']),
            truncated=bool(self.env.get_episode_result()['truncated']),context=context(truth,diag['pre_filter'],self.ch4['prior']),
            privileged_target=truth,next_history=self.history.arrays(),scene_bank=bank,
            set_before=old.as_dict(),set_after=self.estimate.as_dict(),eligible=eligible,
            sent_command=u,delivered_command=diag['delivered_command'],actuated_normalized=diag['actuated_normalized'],
            actual_acceleration=diag['actual_acceleration'],hidden_actuator=diag['hidden_actuator'],
            privileged_pre_filter=diag['pre_filter'],privileged_theta=truth,
            interval_coverage=pre_score,postfit_coverage=new_score,model_velocity_error=candidate_error.tolist(),
            estimation_seconds=estimation_seconds,physical_step=int(row['t']),phase=phases,collision=bool(collisions.any()),
            outcome=self.env.get_episode_result(),guard=self.last_guard,nominal_command=nominal,
            stage_diagnostics=dict(tracking_before=np.linalg.norm(p-np.asarray(public.navigation_targets),axis=1),
                tracking_after=np.linalg.norm(np_-np.asarray(public.navigation_targets),axis=1),speed=np.linalg.norm(nv,axis=1),
                hold_progress=np.asarray(self.observations)[:,21].copy()))
        return result
    def close(self):
        if hasattr(self,'intervention'):self.intervention.restore()
        super().close()
