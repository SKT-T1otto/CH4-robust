"""Instance-local physical intervention; original core integration stays intact."""
from collections import deque
import types
import numpy as np
import torch
from .parameters import NOMINAL, PrivilegedContext, sample_local


def predict_velocity(velocity, applied_normalized, local, specs, dt, position, phase=(0.,0.)):
    """Vectorized public model; local shape (...,4,9), order matches the core."""
    theta = np.asarray(local,dtype=np.float64)
    v = np.broadcast_to(np.asarray(velocity,dtype=np.float64),(*theta.shape[:-1],3)).copy()
    u = np.broadcast_to(np.asarray(applied_normalized),v.shape)
    amax = np.array([[s['a_xy_max']]*2+[s['a_z_max']] for s in specs])
    drag = np.array([[s['drag_xy']]*2+[s['drag_z']] for s in specs])
    base_buoy = np.array([s['buoyancy_bias'] for s in specs])
    v += np.clip(u,-1,1)*amax*theta[...,4,None]*dt
    p = np.asarray(position)
    flow = np.stack((np.sin(.25*p[...,1]+phase[0]),np.cos(.22*p[...,0]+phase[1])),axis=-1)
    v[...,:2] += (-drag[...,:2]*theta[...,2,None]*v[...,:2] + theta[...,0,None]*flow)*dt
    v[...,2] += (-drag[...,2]*theta[...,2]*v[...,2]+base_buoy+theta[...,3]+theta[...,1])*dt
    cap_xy = np.array([s['v_xy_max'] for s in specs])*theta[...,5]
    scale = np.minimum(1.,cap_xy/np.maximum(np.linalg.norm(v[...,:2],axis=-1),1e-12))
    v[...,:2] *= scale[...,None]
    cap_z = np.array([s['v_z_max'] for s in specs])*theta[...,5]
    v[...,2] = np.clip(v[...,2],-cap_z,cap_z)
    return v


class Actuator:
    def __init__(self,max_delay):
        self.max_delay=int(max_delay)
        self.reset()

    def reset(self):
        self.queue=deque([np.zeros((4,3)) for _ in range(self.max_delay+1)],maxlen=self.max_delay+1)
        self.filtered=np.zeros((4,3))
        self.last_delivered=np.zeros((4,3))

    def apply(self,command,theta,noise):
        command=np.asarray(command,dtype=np.float64)
        self.queue.append(command.copy())
        delayed=np.stack([self.queue[-1-int(theta[i,7])][i] for i in range(4)])
        lag=theta[:,6,None]
        self.filtered=lag*self.filtered+(1-lag)*delayed
        self.last_delivered=delayed.copy()
        return np.clip(self.filtered+noise*theta[:,8,None],-1.,1.)

    def state_dict(self):
        return dict(queue=[x.tolist() for x in self.queue],filtered=self.filtered.tolist())


class DisturbanceProcess:
    def __init__(self,options,seed):
        self.options=options
        self.rng=np.random.default_rng(np.random.SeedSequence([int(seed),4101]))
        self.noise_rng=np.random.default_rng(np.random.SeedSequence([int(seed),4102]))
        self.bounds=np.asarray(options.get('sampling_bounds',options['bounds']),float)
        self.protocol=options['protocol']
        self.actuator=Actuator(int(self.bounds[7,1]))
        self.events=[]
        self.current=None

    def reset(self):
        self.actuator.reset();self.events=[];self.current=None

    def at(self,step):
        change=self.current is None or (self.protocol=='piecewise' and step in self.options['change_steps'])
        if change:
            x=np.tile(NOMINAL,(4,1)) if self.protocol=='nominal' else sample_local(self.rng,self.bounds,1)[0]
            self.current=PrivilegedContext(x)
            self.events.append(dict(step=int(step),kind='initial' if step==0 else 'parameter_change'))
        elif self.protocol=='drift':
            x=self.current.local.copy()
            delta=np.asarray(self.options['drift_rate'],float)
            inc=self.rng.uniform(-delta,delta,size=(4,9));inc[:,:2]=inc[0,:2]
            inc[:,7]=0. # Integer delay changes belong to the piecewise protocol.
            x=np.clip(x+inc,self.bounds[:,0],self.bounds[:,1])
            self.current=PrivilegedContext(x)
        return self.current


class PhysicalIntervention:
    """Hooks only a new CH4 environment instance; restore() removes the hook."""
    def __init__(self,core,process):
        self.core=core;self.process=process;self.last={}
        self.original=core._apply_agent_dynamics
        self.base={k:getattr(core,k).clone() for k in
                   ('_a_xy_max','_a_z_max','_v_xy_max','_v_z_max','_drag_xy','_drag_z','_buoyancy_bias')}
        self.flow=(core.flow_gain,core.flow_z_gain)
        owner=self
        def hooked(instance,actions):
            return owner.apply(actions)
        core._apply_agent_dynamics=types.MethodType(hooked,core)

    def apply(self,actions):
        env=self.core
        truth=self.process.at(env.step_count)
        x=truth.local
        command=torch.as_tensor(actions).detach().cpu().numpy().copy()
        if self.process.protocol=='nominal':
            # Exact fast path: no new RNG draws, tensor operations or queue mapping.
            delivered=command.copy();actuated=command.copy()
            result=self.original(actions)
        else:
            noise=self.process.noise_rng.normal(size=(4,3))
            actuated=self.process.actuator.apply(command,x,noise)
            delivered=self.process.actuator.last_delivered.copy()
            factors={'_a_xy_max':4,'_a_z_max':4,'_v_xy_max':5,'_v_z_max':5,'_drag_xy':2,'_drag_z':2}
            try:
                for key,index in factors.items():
                    setattr(env,key,self.base[key]*torch.as_tensor(x[:,index].copy(),dtype=env.dtype,device=env.device))
                env._buoyancy_bias=self.base['_buoyancy_bias']+torch.as_tensor((x[:,3]+x[:,1]).copy(),dtype=env.dtype,device=env.device)
                env.flow_gain=float(x[0,0]);env.flow_z_gain=float(x[0,1])
                result=self.original(torch.as_tensor(actuated,dtype=env.dtype,device=env.device))
            finally:
                for key,value in self.base.items():setattr(env,key,value)
                env.flow_gain,env.flow_z_gain=self.flow
        self.last=dict(sent_command=command,delivered_command=delivered,actuated_normalized=actuated,
                       actual_acceleration=env._agent_acc.detach().cpu().numpy().copy(),truth=truth,
                       hidden_actuator=self.process.actuator.state_dict())
        return result

    def restore(self):
        self.core._apply_agent_dynamics=self.original
