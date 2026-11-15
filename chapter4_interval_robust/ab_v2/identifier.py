"""Six-coordinate supervised SIRSA-style reference, not original reproduction."""
import numpy as np
import torch
from torch import nn
from .parameters import normalize
from .estimation import Estimate

class EnsembleIdentifier(nn.Module):
    def __init__(self,b,options):
        super().__init__();self.bounds=np.asarray(b);self.members=options['ensemble'];h=options['ensemble_hidden']
        self.models=nn.ModuleList([nn.Sequential(nn.Linear(28,h),nn.ReLU(),nn.Linear(h,h),nn.ReLU(),nn.Linear(h,6)) for _ in range(4*self.members)])
        self.optimizer=torch.optim.Adam(self.parameters(),lr=.001);self.updates=0
    def predict(self,frames):
        B,H=frames.shape[:2];features=frames[...,:12]
        prior=np.tile(np.r_[np.full(6,.5),np.ones(6)],(4,1))
        pri=torch.tensor(prior,dtype=torch.float32,device=frames.device)[None,None].expand(B,H,4,12)
        role=torch.eye(4,device=frames.device)[None,None].expand(B,H,4,4)
        x=torch.cat((features,role,pri),-1);valid=(frames[...,12]==1).float()
        outputs=[]
        for member in range(self.members):
            agents=[]
            for i in range(4):
                y=torch.sigmoid(self.models[member*4+i](x[:,:,i]));weights=valid[:,:,i,None]
                mean=(y*weights).sum(1)/weights.sum(1).clamp_min(1)
                agents.append(torch.where(weights.sum(1)>0,mean,torch.full_like(mean,.5)))
            outputs.append(torch.stack(agents,1))
        return torch.stack(outputs)
    @torch.no_grad()
    def infer(self,frames):
        result=Estimate(self.bounds,frames);counts=(np.asarray(frames)[:,:,12]==1).sum(0);result.support=counts
        if not self.updates:result.reason='identifier_untrained';return result
        pred=self.predict(torch.tensor(frames[None],dtype=torch.float32)).numpy()[:,0]
        center=pred.mean(0);radius=pred.std(0);span=np.ptp(self.bounds,axis=1)
        result.lower=np.clip(center-radius,0,1)*span+self.bounds[:,0]
        result.upper=np.clip(center+radius,0,1)*span+self.bounds[:,0]
        for i in range(4):
            if not counts[i]:result.lower[i]=self.bounds[:,0];result.upper[i]=self.bounds[:,1]
        result.lower[:,5]=np.floor(result.lower[:,5]);result.upper[:,5]=np.ceil(result.upper[:,5])
        result.delay_mask=np.array([[lo<=d<=hi for d in range(3)] for lo,hi in zip(result.lower[:,5],result.upper[:,5])])
        result.valid=counts>0
        result.age=np.array([7-np.flatnonzero(frames[:,i,12]==1)[-1] if counts[i] else 8 for i in range(4)])
        result.reason='ensemble_uncalibrated_spread';return result
    def update_supervised(self,frames,targets):
        f=torch.tensor(frames,dtype=torch.float32);target=torch.tensor(normalize(targets,self.bounds),dtype=torch.float32)
        prediction=self.predict(f);valid=(f[...,12]==1).any(1).float();losses=[]
        for member in range(self.members):
            ids=torch.arange(member,len(f),self.members)
            if not len(ids):ids=torch.tensor([member%len(f)])
            error=(prediction[member,ids]-target[ids]).square().mean(-1)
            losses.append((error*valid[ids]).sum()/valid[ids].sum().clamp_min(1))
        loss=torch.stack(losses).mean();self.optimizer.zero_grad();loss.backward();self.optimizer.step();self.updates+=1
        return float(loss.detach())
