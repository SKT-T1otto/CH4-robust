"""OLD/D2 task reference plus independently versioned absolute control input."""
from dataclasses import dataclass
import copy
import numpy as np
import torch
from chapter3_bser.learning.base_runtime import BaselineMissionRuntime, _make_base_env
from .configuration import reference_config
from .dynamics import DisturbanceProcess, PhysicalIntervention
from .estimation import History, HistorySetEstimator
from .parameters import PublicSet, coverage, joint, normalize

BASE_DIM=43
OBS_DIM=61
ACTION_SCHEMA='ch4.reference_shifted_absolute_gaussian.v1'


@dataclass(frozen=True)
class PolicyInput:
    """Only deployable fields. No environment, info, truth or diagnostic objects."""
    observations: np.ndarray
    active: np.ndarray


def public_phase_codes(env,original):
    task=env.get_task_state();codes=np.empty(4,np.int64)
    for i in range(4):
        value=0 if i<3 and not task.target_found else 1
        if i==3 and task.executor_knows_target:value=3 if original[i,21]>0 else 2
        codes[i]=value
    return codes


def public_nominal(env):
    """Same nominal prior arithmetic as core, computed from public views only."""
    state=env.get_agent_state();contract=env.get_public_control_contract();specs=env.get_public_vehicle_specs()
    p=torch.tensor(state.positions,dtype=torch.float32);v=torch.tensor(state.velocities,dtype=torch.float32)
    target=torch.tensor(state.navigation_targets,dtype=torch.float32);delta=target-p
    axy=torch.tensor([s['a_xy_max'] for s in specs]);az=torch.tensor([s['a_z_max'] for s in specs])
    vmax=torch.tensor([s['v_xy_max'] for s in specs]);vz=torch.tensor([s['v_z_max'] for s in specs])
    distance=torch.norm(delta[:,:2],dim=1,keepdim=True).clamp_min(contract['eps'])
    desired=delta[:,:2]/distance*vmax[:,None]*torch.clamp(distance/contract['prior_slow_radius_xy'],0,1)
    acc=torch.zeros((4,3));acc[:,:2]=contract['prior_kv_xy']*(desired-v[:,:2])
    acc[:,2]=contract['prior_kv_z']*(torch.clamp(delta[:,2]/contract['prior_slow_radius_z'],-1,1)*vz-v[:,2])
    acc[:,:2]=torch.clamp(acc[:,:2],-axy[:,None],axy[:,None]);acc[:,2]=torch.clamp(acc[:,2],-az,az)
    strengths=torch.tensor([contract['prior_strength_search']]*3+[contract['prior_strength_executor']])
    scales=torch.stack((axy,axy,az),dim=1)
    return torch.clamp(acc*strengths[:,None]/scales,-1,1).numpy()


class ReferenceRuntime(BaselineMissionRuntime):
    def __init__(self,c,scene,*,seed,episode_id,identifier=None):
        self.ch4=c;self.identifier=identifier;self.previous_command=np.zeros((4,3),np.float32)
        self.method=c['method_options'];self.estimate=PublicSet.prior(c['disturbance']['bounds'])
        self.history=History(c['estimation']['history']);self.last_guard=[]
        super().__init__(reference_config(c),scene,seed=seed,episode_id=episode_id)
        self.specs=self.env.get_public_vehicle_specs();self.contract=self.env.get_public_control_contract()
        self.process=DisturbanceProcess(c['disturbance'],int(scene['disturbance_seed']))
        self.intervention=PhysicalIntervention(self.env.unwrapped,self.process)
        self.estimator=None
        if self.method['estimator']=='history_set':
            self.estimator=HistorySetEstimator(c['disturbance']['bounds'],self.specs,self.contract,
                                               c['estimation'],seed,shared=self.method['shared'])
        from .protection import PublicProtection
        self.protection=PublicProtection(self.env,c['protection'],self.specs,self.contract,int(scene['scenario_seed']))
        self.env.enable_public_event_capture('ch4:'+str(episode_id),capacity=2048)

    @staticmethod
    def _make_environment(config):
        env=_make_base_env(config)
        env.unwrapped.use_residual_prior=False
        env.unwrapped._residual_scale.fill_(1.)
        return env

    def policy_input(self):
        original=np.asarray(self.observations,dtype=np.float32)
        if original.shape!=(4,28):raise ValueError('public observation contract changed')
        phase=np.eye(4,dtype=np.float32)[public_phase_codes(self.env,original)]
        if not self.method['phase']:phase.fill(0)
        ref=public_nominal(self.env)
        clock=np.full((4,1),self.step/self.ch4['max_steps'],np.float32)
        base=np.concatenate((original,np.eye(4,dtype=np.float32),phase,self.previous_command,ref,clock),axis=-1)
        bounds=self.ch4['disturbance']['bounds']
        features=self.estimate.features(bounds,mode=self.method['input'])
        values=np.concatenate((base,features),axis=-1).astype(np.float32)
        if values.shape!=(4,OBS_DIM) or not np.isfinite(values).all():raise ValueError('invalid CH4 observation')
        return PolicyInput(values,original[:,20]<.5)

    def _actions_for_step(self,actions):
        self.proposal=actions.detach().cpu().numpy().copy()
        active=np.asarray(self.observations)[:,20]<.5
        command,self.last_guard=self.protection.choose(self.proposal,self.estimate,self.previous_command,active)
        return torch.as_tensor(command,dtype=torch.float32)

    def transition(self,proposal):
        before=self.policy_input();history,history_mask=self.history.arrays()
        phase_codes=public_phase_codes(self.env,np.asarray(self.observations))
        public=self.env.get_agent_state();p=np.asarray(public.positions);v=np.asarray(public.velocities)
        nominal=public_nominal(self.env)
        old_set=copy.deepcopy(self.estimate)
        class FixedProposal:
            def step(self,observations,explore=True):
                return [torch.as_tensor(proposal[i:i+1],dtype=torch.float32) for i in range(4)]
        row=super().advance(FixedProposal(),explore=False)
        state=self.env.get_agent_state();nv=np.asarray(state.velocities);np_=np.asarray(state.positions)
        diagnostic=self.intervention.last
        command=diagnostic['sent_command']
        # Conservative *observable* eligibility. No hidden applied action/parameter
        # or true clip flag is consumed by either estimator.
        b=np.asarray(self.ch4['disturbance']['bounds'])
        min_speed=np.array([s['v_xy_max'] for s in self.specs])*b[5,0]
        min_z=np.array([s['v_z_max'] for s in self.specs])*b[5,0]
        speed_mask=(np.linalg.norm(nv[:,:2],axis=1)<min_speed-1e-5)&(np.abs(nv[:,2])<min_z-1e-5)
        walls=np.any((np_<=1e-5)|(np_>=np.asarray(self.contract['bounds'])-1e-5),axis=1)
        collisions=np.asarray(state.collision_flags,bool)
        eligible=before.active & (np.asarray(self.observations)[:,20]<.5) & speed_mask & ~walls & ~collisions
        self.history.append(p,v,command,nv,eligible)
        if self.estimator is not None:
            self.estimate=self.estimator.update(p,v,command,nv,eligible)
        elif self.method['estimator']=='ensemble':
            if self.identifier is None:raise ValueError('ensemble identifier required')
            # Fixed broad prior matches identifier training and allows a moving
            # history window to expand after change; no unsupported recursive squeeze.
            self.estimate=self.identifier.infer(self.history,PublicSet.prior(self.ch4['disturbance']['bounds']))
        self.previous_command=command.astype(np.float32)
        after=self.policy_input();next_history,next_mask=self.history.arrays()
        outcome=self.env.get_episode_result()
        truth=diagnostic['truth']
        context=joint(normalize(truth.local,b)).astype(np.float32)
        result=dict(observations=before.observations,active=before.active,proposal=self.proposal.copy(),
                    rewards=np.asarray(row['rewards'],np.float32),reward=float(row['team_reward']),
                    next_observations=after.observations,next_active=after.active,
                    terminated=bool(outcome['terminated']),truncated=bool(outcome['truncated']),
                    context=context,history=history,history_mask=history_mask,
                    next_history=next_history,next_history_mask=next_mask,
                    privileged_target=truth.local.copy(),set_before=old_set.as_dict(),set_after=self.estimate.as_dict(),
                    sent_command=command,delivered_command=diagnostic['delivered_command'],
                    actuated_normalized=diagnostic['actuated_normalized'],actual_acceleration=diagnostic['actual_acceleration'],
                    hidden_actuator=diagnostic['hidden_actuator'],eligible=eligible,
                    physical_step=int(row['t']),phase=phase_codes,
                    collision=bool(collisions.any()),outcome=outcome,guard=self.last_guard,
                    interval_coverage=coverage(self.estimate,truth),
                    nominal_command=nominal,velocity=nv,reference_velocity=v,
                    stage_diagnostics=dict(tracking_before=np.linalg.norm(p-np.asarray(public.navigation_targets),axis=1),
                                           tracking_after=np.linalg.norm(np_-np.asarray(public.navigation_targets),axis=1),
                                           speed=np.linalg.norm(nv,axis=1),
                                           hold_progress=np.asarray(self.observations)[:,21].copy()),
                    model_velocity_error=self.protection.last_prediction_error(p,v,command,nv,old_set))
        return result

    def close(self):
        if hasattr(self,'intervention'):self.intervention.restore()
        super().close()
