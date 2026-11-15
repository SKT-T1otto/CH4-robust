"""A: conditional field/response pools. B: candidate command-time propagation.

All sets are empirical; candidate support and observed-data support are separate.
"""
from collections import deque
import copy,hashlib
import numpy as np
from ..dynamics import predict_velocity
from .parameters import NOMINAL,normalize,to_legacy,context

class History:
    def __init__(self):self.rows=deque(maxlen=8);self.total=0
    def append(self,p,v,u,nv,eligible):
        row=np.concatenate((np.asarray(p)/20,np.asarray(v)/3,u,np.asarray(nv)/3,np.asarray(eligible)[:,None]),-1)
        if row.shape!=(4,13) or not np.isfinite(row).all():raise ValueError('invalid public history')
        self.rows.append(row.astype(np.float32));self.total+=1
    def arrays(self):
        a=np.zeros((8,4,13),np.float32);a[:,:,-1]=-1
        if self.rows:a[-len(self.rows):]=self.rows
        return a

def reconstruct(frames,local,initial=None):
    """Candidate pre-decision filter from completed public transitions only."""
    frames=np.asarray(frames);x=np.asarray(local)
    z=np.zeros((*x.shape[:-1],3)) if initial is None else np.array(initial,copy=True)
    observed=frames[:,0,12]>=0
    f=frames[observed]
    for k in range(len(f)):
        delivered=np.zeros_like(z)
        for i in range(4):
            delay=x[...,i,5].astype(int);idx=k-delay
            for d in np.unique(delay):
                use=delay==d
                value=f[k-int(d),i,6:9] if k>=int(d) else np.zeros(3)
                delivered[...,i,:]=np.where(use[...,None],value,delivered[...,i,:])
        lag=x[...,4,None];z=lag*z+(1-lag)*delivered
    return z

class Estimate:
    def __init__(self,b,history=None):
        self.bounds=np.asarray(b,float);self.lower=np.tile(self.bounds[:,0],(4,1));self.upper=np.tile(self.bounds[:,1],(4,1))
        self.valid=np.zeros(4,bool);self.support=np.zeros(4,int);self.hypotheses=np.zeros(4,int);self.age=np.zeros(4,int)
        self.delay_mask=np.ones((4,3),bool);self.reason='initial_prior';self.history=np.zeros((8,4,13),np.float32)
        self.history[:,:,12]=-1
        if history is not None:self.history=np.asarray(history).copy()
        self.pool=None;self.accepted=None;self.z=None;self.z_radius=None;self.shared=True
    def features(self,mode='interval'):
        center=normalize((self.lower+self.upper)/2,self.bounds);width=(self.upper-self.lower)/np.maximum(np.ptp(self.bounds,axis=1),1e-12)
        if mode=='none':center=np.zeros_like(center);width=np.ones_like(width)
        if mode=='point':width=np.zeros_like(width)
        meta=np.stack((self.valid.astype(float),np.minimum(self.support/8,1),np.minimum(self.age/8,1)),1)
        if mode=='none':meta[:]=0
        return np.concatenate((center,width,meta,self.delay_mask),1).astype(np.float32)
    def as_dict(self):
        return dict(schema='ch4.public_summary6.v2',lower=self.lower.tolist(),upper=self.upper.tolist(),
                    valid=self.valid.tolist(),data_support=self.support.tolist(),hypothesis_count=self.hypotheses.tolist(),
                    age=self.age.tolist(),delay_mask=self.delay_mask.tolist(),reason=self.reason)
    def samples(self,rng,count):
        fallback=0
        if self.pool is None:
            x=rng.uniform(self.lower,self.upper,size=(count,4,6))
            g=rng.uniform(self.lower[:,0].min(),self.upper[:,0].max(),(count,1));x[:,:,0]=g
            for i in range(4):
                ds=np.flatnonzero(self.delay_mask[i]);x[:,i,5]=rng.choice(ds,count)
            z=reconstruct(self.history,x)
        else:
            G=self.pool.shape[0];x=np.empty((count,4,6));z=np.empty((count,4,3))
            eligible=self.accepted.any(-1) # G,4
            if self.shared:
                informed=self.support>0
                mass=np.all(eligible[:,informed],axis=1).astype(float) if informed.any() else np.ones(G)
            else:
                mass=(eligible/np.maximum(eligible.sum(0),1)).mean(1)
            if mass.sum()==0:mass=np.ones(G)
            choices=rng.choice(G,count,p=mass/mass.sum())
            for n,g in enumerate(choices):
                for i in range(4):
                    ids=np.flatnonzero(self.accepted[g,i])
                    if not len(ids):ids=np.arange(self.pool.shape[2]);fallback+=1
                    k=int(rng.choice(ids));x[n,i]=self.pool[g,i,k]
                    radius=self.z_radius[g,i,k]
                    z[n,i]=np.clip(self.z[g,i,k]+rng.uniform(-radius,radius,3),-1,1)
        # Exactly one public-field instance in every world, including S0.
        if not np.all(x[:,:,0]==x[:,:1,0]):raise AssertionError('inconsistent joint field')
        return x,z,dict(local_prior_fallbacks=fallback,measure='uniform_shared_branch_then_uniform_conditional_candidates' if self.shared else 'equal_vehicle_union_mixture')
    def bank(self,rng,count,active):
        x,z,diag=self.samples(rng,count)
        values=np.stack([context(a,s,self.bounds) for a,s in zip(x,z)])
        valid=bool(self.valid[np.asarray(active,bool)].all())
        fingerprint=hashlib.sha256(values.tobytes()+self.history.tobytes()).hexdigest()
        return dict(schema='ch4.conditional_scene_bank33.v2',contexts=values.tolist(),valid=valid,
                    summary=self.as_dict(),source_bank_hash=fingerprint,**diag)

class ConditionalEstimator:
    def __init__(self,b,specs,dt,opts,seed,*,shared=True,aligned=True):
        self.b=np.asarray(b,float);self.specs=specs;self.dt=dt;self.opts=opts
        self.rng=np.random.default_rng(seed);self.shared=shared;self.aligned=aligned
        self.history=History();self.commands=[];self.last_update=np.zeros(4,int);self.reasons={};self.cooldown=0
        self.reset();self.estimate=Estimate(b)
        self.prediction_calls=0
        self.padding_prediction_calls=0
    def reset(self):
        G=self.opts['shared_candidates'];K=self.opts['per_delay']*3
        # Same field strata for all S/T groups; only evidence aggregation changes.
        g=np.linspace(*self.b[0],G)
        self.pool=self.rng.uniform(self.b[:,0],self.b[:,1],(G,4,K,6));self.pool[:,:,:,0]=g[:,None,None]
        self.pool[:,:,:,5]=np.tile(np.arange(3),self.opts['per_delay'])
    def _evaluate(self):
        frames=list(self.history.rows);pool=self.pool;z=np.zeros((*pool.shape[:-1],3));radius=np.zeros(pool.shape[:-1])
        # Resetting evidence must not silently reduce a factorial group's
        # physical-kernel budget. Missing history slots get real masked kernel
        # calls, excluded from residuals, support and filter evolution. Report
        # this wasted/padding work separately; it is not additional evidence.
        for _ in range(8-len(frames)):
            predict_velocity(np.zeros((4,3)),np.zeros((pool.shape[0],pool.shape[2],4,3)),
                to_legacy(pool.transpose(0,2,1,3)),self.specs,self.dt,np.zeros((4,3)))
            count=int(np.prod(pool.shape[:-1]));self.prediction_calls+=count;self.padding_prediction_calls+=count
        shifted=self.history.total>len(frames)
        if shifted:radius.fill(1.)
        worst=np.zeros(pool.shape[:-1]);counts=np.zeros(4,int)
        start=self.history.total-len(frames)
        for k,f in enumerate(frames):
            delayed=np.zeros_like(z)
            for i in range(4):
                delays=pool[:,i,:,5].astype(int)
                for d in (0,1,2):
                    idx=start+k-d
                    delivered=self.commands[idx][i] if idx>=0 else np.zeros(3)
                    delayed[:,i]=np.where((delays==d)[...,None],delivered,delayed[:,i])
            lag=pool[...,4];z=lag[...,None]*z+(1-lag[...,None])*delayed;radius*=lag
            control=z if self.aligned else np.broadcast_to(f[None,:,None,6:9],z.shape)
            # Public physics kernel expects (..., vehicle, xyz), whereas these
            # conditional pools are (field, vehicle, candidate, xyz).
            predicted=predict_velocity(f[:,3:6]*3,control.transpose(0,2,1,3),
                to_legacy(pool.transpose(0,2,1,3)),self.specs,self.dt,f[:,:3]*20).transpose(0,2,1,3)
            self.prediction_calls+=int(np.prod(pool.shape[:-1]))
            residual=np.max(np.abs(predicted-f[None,:,None,9:12]*3),-1)
            # Unknown window-initial memory: bounded contribution, no hidden read.
            amax=np.array([max(s['a_xy_max'],s['a_z_max']) for s in self.specs])
            allowance=radius*amax[None,:,None]*pool[...,2]*self.dt*2
            valid=f[:,12]==1;counts+=valid
            worst=np.maximum(worst,np.where(valid[None,:,None],residual-allowance,0))
        return worst,z,radius,counts
    def update(self,p,v,u,nv,eligible):
        self.history.append(p,v,u,nv,eligible);self.commands.append(np.asarray(u).copy())
        # Commands are kept per episode; histories and replay remain bounded.
        score,z,radius,counts=self._evaluate();accepted=score<=self.opts['velocity_tolerance']
        sparse=(accepted.sum(-1)<self.opts['minimum_candidates'])&(counts[None,:]>0)
        if self.opts['replenish']:
            # One bounded local re-proposal. Evaluate the second pass even if
            # no pool is sparse, so every S/T group has the same kernel budget.
            for g,i in zip(*np.where(sparse)):
                values=self.rng.uniform(self.b[:,0],self.b[:,1],self.pool[g,i].shape)
                values[:,0]=self.pool[g,i,0,0];values[:,5]=self.pool[g,i,:,5];self.pool[g,i]=values
            score,z,radius,counts=self._evaluate();accepted=score<=self.opts['velocity_tolerance']
        result=Estimate(self.b,self.history.arrays());result.shared=self.shared;result.support=counts
        informed=counts>=self.opts['minimum_data'];enough=accepted.sum(-1)>=self.opts['minimum_candidates']
        if self.shared:
            common=np.all(enough[:,informed],1) if informed.any() else np.ones(len(enough),bool)
            accepted&=common[:,None,None]
        if not accepted.any() and informed.any():
            self.reasons['inconsistent_reset']=self.reasons.get('inconsistent_reset',0)+1
            self.reset();self.cooldown=2;result.reason='inconsistent_reset'
            # Discard inconsistent residual evidence, retain the actual command
            # history and global timestamps needed for nonzero actuator memory.
            self.history.rows.clear()
        else:
            result.pool=self.pool.copy();result.accepted=accepted.copy();result.z=z.copy();result.z_radius=radius.copy()
            result.reason='supported' if informed.any() else 'insufficient_data'
            for i in range(4):
                values=self.pool[:,i][accepted[:,i]]
                result.hypotheses[i]=len(values)
                if informed[i] and len(values)>=self.opts['minimum_candidates']:
                    pad=self.opts['padding']*np.ptp(self.b,axis=1)
                    result.lower[i]=np.maximum(self.b[:,0],values.min(0)-pad)
                    result.upper[i]=np.minimum(self.b[:,1],values.max(0)+pad)
                    result.delay_mask[i]=[np.any(values[:,5]==d) for d in range(3)]
                    result.lower[i,5]=np.flatnonzero(result.delay_mask[i])[0]
                    result.upper[i,5]=np.flatnonzero(result.delay_mask[i])[-1]
                    result.valid[i]=True
                    if eligible[i]:self.last_update[i]=self.history.total
            if self.cooldown:
                self.cooldown-=1;result.valid[:]=False;result.reason='memory_warmup'
        result.age=self.history.total-self.last_update
        self.estimate=result
        return result

def diagnostics(estimate,truth,opts):
    x=np.asarray(truth);covered=(x>=estimate.lower)&(x<=estimate.upper)
    conditional=None
    if estimate.pool is not None:
        # Continuous coordinates use predeclared resolution; delay is exact.
        delta=np.abs(normalize(estimate.pool,estimate.bounds)-normalize(x,estimate.bounds)[None,:,None])
        near=(delta[...,:5]<=opts['match_radius']).all(-1)&(estimate.pool[...,5]==x[None,:,None,5])&estimate.accepted
        conditional=bool(near.any(-1).all(1).any())
    return dict(per_dimension=covered.tolist(),joint_box=bool(covered.all()),conditional_support=conditional,
                valid=estimate.valid.tolist(),width=(estimate.upper-estimate.lower).tolist(),
                reason=estimate.reason,data_support=estimate.support.tolist(),hypotheses=estimate.hypotheses.tolist())
