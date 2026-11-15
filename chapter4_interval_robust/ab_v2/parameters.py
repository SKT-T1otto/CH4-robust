"""Reduced identifiable coordinates. Legacy physics is an explicit adapter."""
from dataclasses import dataclass
import numpy as np
from ..parameters import NOMINAL as LEGACY_NOMINAL

KEYS=('flow_gain','drag_scale','a_max_scale','net_vertical_bias','actuator_lag','delay_steps')
NOMINAL=np.array([.18,1.,1.,0.,0.,0.])
BOUNDS=np.array([[.15,.28],[.85,1.2],[.85,1.1],[-.02,.025],[0.,.25],[0.,2.]])
LEGACY_INDICES=(0,2,4,3,6,7)
THETA_DIM=21
CONTEXT_DIM=33
ACTOR_DIM=165
CRITIC_DIM=705
SCHEMAS=dict(parameters='ch4.theta21.v2',context='ch4.theta21_filter12.v2',
             observation='ch4.public165.status3.v2',action='ch4.reference_shifted_absolute_gaussian.v2',
             estimate='ch4.conditional_set.v2',history='ch4.history8x13.v2')

def bounds(value):
    b=np.asarray(value,float)
    if b.shape!=(6,2) or not np.isfinite(b).all() or np.any(b[:,0]>b[:,1]):raise ValueError('six ordered finite bounds required')
    if np.any(b[[1,2],0]<=0) or not 0<=b[4,0]<=b[4,1]<1:raise ValueError('invalid response scales')
    if np.any(b[5]!=np.floor(b[5])) or b[5,0]<0 or b[5,1]>2:raise ValueError('v2 policy supports integer delays 0..2 only')
    return b.copy()

def normalize(x,b):
    b=np.asarray(b);return (np.asarray(x)-b[:,0])/np.maximum(b[:,1]-b[:,0],1e-12)

def to_legacy(x,noise=0.):
    x=np.asarray(x,float)
    if x.shape[-1]!=6 or not np.isfinite(x).all() or np.any(x[...,5]!=np.floor(x[...,5])):raise ValueError('invalid reduced physical coordinates')
    out=np.broadcast_to(LEGACY_NOMINAL,(*x.shape[:-1],9)).copy()
    out[...,list(LEGACY_INDICES)]=x;out[...,1]=0.;out[...,5]=1.;out[...,8]=noise
    return out

def from_legacy(x):
    x=np.asarray(x,float)
    if np.any(x[...,1]!=0) or np.any(x[...,5]!=1):raise ValueError('legacy world outside reduced main domain')
    return x[...,list(LEGACY_INDICES)].copy()

def joint(local):
    x=np.asarray(local,float)
    if x.shape[-2:]!=(4,6) or not np.isfinite(x).all() or not np.all(x[...,0,0,None]==x[..., :,0]):raise ValueError('one shared physical flow required')
    return np.concatenate((x[...,0,:1],x[..., :,1:].reshape(*x.shape[:-2],20)),axis=-1)

def expand(x):
    x=np.asarray(x,float)
    if x.shape[-1]!=21:raise ValueError('v2 theta must have 21 coordinates')
    g=np.broadcast_to(x[...,:1][...,None,:],(*x.shape[:-1],4,1))
    return np.concatenate((g,x[...,1:].reshape(*x.shape[:-1],4,5)),axis=-1)

def context(local,pre_filter,b):
    z=np.asarray(pre_filter,float)
    if z.shape!=(4,3) or not np.isfinite(z).all() or np.any(np.abs(z)>1.000001):raise ValueError('invalid pre-action filter state')
    return np.concatenate((joint(normalize(local,b)),((z+1)/2).ravel())).astype(np.float32)

def legacy_bounds(b,noise):
    out=np.stack((LEGACY_NOMINAL,LEGACY_NOMINAL),-1)
    out[list(LEGACY_INDICES)]=b;out[8]=noise
    return out

@dataclass(frozen=True)
class PolicyInput:
    observations:np.ndarray
    active:np.ndarray
    def __post_init__(self):
        if np.asarray(self.observations).shape!=(4,165) or not np.isfinite(self.observations).all() or np.asarray(self.active).shape!=(4,):raise ValueError('public165 contract violated')
