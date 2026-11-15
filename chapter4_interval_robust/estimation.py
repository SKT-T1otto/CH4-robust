"""Public-history estimators. No method receives simulator truth at inference."""
from collections import deque
import numpy as np
import torch
from torch import nn
from .parameters import PublicSet, sample_local, normalize
from .dynamics import predict_velocity


class History:
    def __init__(self,length=5):
        self.length=int(length);self.frames=deque(maxlen=self.length);self.masks=deque(maxlen=self.length)

    def append(self,position,velocity,command,next_velocity,valid):
        frame=np.concatenate((np.asarray(position)/20,np.asarray(velocity)/3,
                              np.asarray(command),np.asarray(next_velocity)/3),axis=-1)
        if frame.shape!=(4,12) or not np.isfinite(frame).all():raise ValueError('invalid public history')
        self.frames.append(frame.astype(np.float32));self.masks.append(np.asarray(valid,bool))

    def arrays(self):
        frames=np.zeros((self.length,4,12),np.float32);mask=np.zeros((self.length,4),bool)
        if self.frames:
            frames[-len(self.frames):]=np.asarray(self.frames);mask[-len(self.masks):]=np.asarray(self.masks)
        return frames,mask


class EnsembleIdentifier(nn.Module):
    """Independent PyTorch SIRSA-style supervised ensemble; not an exact reproduction.

    Per-transition residual context estimates are bounded by the prior and averaged
    across valid history. Ensemble spread is a heuristic, never a coverage claim.
    """
    def __init__(self,bounds,ensemble=4,hidden=64,shared_encoder=False):
        super().__init__()
        self.bounds=np.asarray(bounds,float)
        self.ensemble=int(ensemble);self.shared_encoder=bool(shared_encoder)
        # Shared role-conditioned model or four independent vehicle models.
        count=self.ensemble if self.shared_encoder else self.ensemble*4
        self.models=nn.ModuleList([nn.Sequential(nn.Linear(34,hidden),nn.ReLU(),
                                                nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,9))
                                   for _ in range(count)])
        self.optimizer=torch.optim.Adam(self.parameters(),lr=1e-3)
        self.updates=0

    def predictions(self,frames,masks,prior_features):
        # frames B,H,4,12; masks B,H,4; prior B,4,18.
        batch,length=frames.shape[:2]
        role=torch.eye(4,device=frames.device)[None,None].expand(batch,length,4,4)
        prior=prior_features[:,None].expand(batch,length,4,18)
        center=prior_features[...,:9]
        radius=prior_features[...,9:]/2
        inputs=torch.cat((frames,role,prior),dim=-1)
        outputs=[]
        for member in range(self.ensemble):
            agents=[]
            for i in range(4):
                model=self.models[member if self.shared_encoder else member*4+i]
                per_transition=center[:,None,i]+radius[:,None,i]*torch.tanh(model(inputs[:,:,i]))
                weight=masks[:,:,i,None].to(frames.dtype)
                mean=(per_transition*weight).sum(1)/weight.sum(1).clamp_min(1)
                has_data=weight.sum(1)>0
                agents.append(torch.where(has_data,mean,center[:,i]))
            outputs.append(torch.stack(agents,1))
        return torch.stack(outputs,0)

    @torch.no_grad()
    def infer(self,history,prior):
        frames,mask=history.arrays()
        if self.updates==0:
            result=PublicSet(prior.lower,prior.upper,False,0,'identifier_untrained_prior')
            return result
        if not mask.any():return PublicSet.prior(self.bounds)
        features=prior.features(self.bounds)
        values=self.predictions(torch.from_numpy(frames[None]),torch.from_numpy(mask[None]),
                                torch.from_numpy(features[None]))[:,0].cpu().numpy()
        center=values.mean(0);radius=values.std(0)
        b=self.bounds;span=b[:,1]-b[:,0]
        lo=np.clip(center-radius,0,1)*span+b[:,0]
        hi=np.clip(center+radius,0,1)*span+b[:,0]
        # Unsupported vehicles retain the prior instead of collapsing to a point.
        for i in range(4):
            if not mask[:,i].any():lo[i]=prior.lower[i];hi[i]=prior.upper[i]
        lo[:,7]=np.floor(lo[:,7]);hi[:,7]=np.ceil(hi[:,7])
        return PublicSet(lo,hi,True,int(mask.sum()),'ensemble_mean_std_heuristic')

    def update_supervised(self,frames,masks,targets):
        f=torch.as_tensor(frames,dtype=torch.float32)
        mask=torch.as_tensor(masks,dtype=torch.bool)
        target=torch.as_tensor(normalize(targets,self.bounds),dtype=torch.float32)
        prior=PublicSet.prior(self.bounds).features(self.bounds)
        prior=torch.from_numpy(prior)[None].expand(len(f),4,18)
        pred=self.predictions(f,mask,prior)
        losses=[]
        # Disjoint member minibatches when possible; bootstrap otherwise.
        for j in range(self.ensemble):
            index=torch.arange(j,len(f),self.ensemble)
            if not len(index):index=torch.randint(len(f),(1,))
            valid=mask[index].any(1).float()
            error=(pred[j,index]-target[index]).square().mean(-1)
            losses.append((error*valid).sum()/valid.sum().clamp_min(1))
        loss=torch.stack(losses).mean()
        self.optimizer.zero_grad();loss.backward();self.optimizer.step();self.updates+=1
        return float(loss.detach())


class HistorySetEstimator:
    """Finite-hypothesis, physics-aligned empirical set estimator (A+B).

    Feasible particle hulls are approximations, NOT certified set membership.
    A shared field is tested jointly against eligible vehicles, while vehicle
    response parameters stay separate. A-off uses independent masks and a union
    envelope when a later joint-world query requires one common field.
    """
    def __init__(self,bounds,specs,contract,options,seed,shared=True):
        self.bounds=np.asarray(bounds,float);self.specs=specs;self.contract=contract;self.options=options
        self.rng=np.random.default_rng(np.random.SeedSequence([int(seed),4201]))
        self.shared=bool(shared);self.count=int(options['particles'])
        self.window=int(options['history']);self.max_delay=int(self.bounds[7,1])
        self.reasons={};self.step=0;self.cooldown=0
        self.reset_particles(initial=True)
        self.estimate=PublicSet.prior(self.bounds)

    def reset_particles(self,initial=False):
        self.particles=sample_local(self.rng,self.bounds,self.count,shared=self.shared)
        self.filtered=np.zeros((self.count,4,3)) if initial else self.rng.uniform(-1,1,(self.count,4,3))
        self.commands=deque([np.zeros((4,3)) for _ in range(self.max_delay+1)],maxlen=self.max_delay+1)
        self.errors=deque(maxlen=self.window)
        self.cooldown=0 if initial else self.max_delay+6

    def update(self,position,velocity,command,next_velocity,valid):
        valid=np.asarray(valid,bool)
        self.commands.append(np.asarray(command,float).copy())
        history=np.asarray(self.commands)
        delay=self.particles[:,:,7].astype(int)
        delivered=np.stack([history[-1-delay[:,i],i,:] for i in range(4)],axis=1)
        lag=self.particles[:,:,6,None]
        self.filtered=lag*self.filtered+(1-lag)*delivered
        predicted=predict_velocity(velocity,self.filtered,self.particles,self.specs,self.contract['dt'],position)
        # Command noise/model mismatch tolerance is a declared empirical setting.
        residual=np.max(np.abs(predicted-np.asarray(next_velocity)),axis=-1)
        tolerance=float(self.options['velocity_tolerance'])
        eligible=valid & np.isfinite(next_velocity).all(axis=1)
        error=np.where(eligible[None],residual,0.)
        self.errors.append(error)
        self.step+=1
        if self.cooldown:
            self.cooldown-=1
            self.estimate=PublicSet.prior(self.bounds)
            self.estimate.valid=False;self.estimate.reason='reset_memory_warmup'
            return self.estimate
        if not eligible.any():
            self.estimate.reason='no_eligible_transition'
            return self.estimate
        worst=np.max(np.asarray(self.errors),axis=0)
        accepted=worst<=tolerance
        if self.shared:
            common=accepted.all(axis=1)
            accepted=np.repeat(common[:,None],4,axis=1)
        minimum=int(self.options['minimum_particles'])
        if np.any(accepted.sum(0)<minimum):
            reason='empty_or_undersupported_hypotheses_reset'
            self.reasons[reason]=self.reasons.get(reason,0)+1
            # Reinitialize both parameters and unknown actuator state. A full
            # prior is returned immediately; new evidence must support shrinkage.
            old_commands=list(self.commands)
            self.reset_particles(initial=False)
            self.commands=deque(old_commands,maxlen=self.max_delay+1)
            self.estimate=PublicSet.prior(self.bounds)
            self.estimate.valid=False;self.estimate.reason=reason
            return self.estimate
        lo=[];hi=[];padding=float(self.options['hull_padding'])*(self.bounds[:,1]-self.bounds[:,0])
        for i in range(4):
            values=self.particles[accepted[:,i],i]
            lo.append(np.maximum(self.bounds[:,0],values.min(0)-padding))
            hi.append(np.minimum(self.bounds[:,1],values.max(0)+padding))
        lo=np.asarray(lo);hi=np.asarray(hi)
        lo[:,7]=np.floor(lo[:,7]);hi[:,7]=np.ceil(hi[:,7])
        self.estimate=PublicSet(lo,hi,True,int(accepted.sum(0).min()),
                                'joint_shared_empirical_hull' if self.shared else 'independent_empirical_hull')
        return self.estimate
