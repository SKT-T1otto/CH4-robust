"""Physical parameter registry. Truth and public uncertainty are separate types."""
from dataclasses import dataclass
import numpy as np

KEYS = ('flow_gain', 'flow_z_gain', 'drag_scale', 'buoyancy_bias_delta',
        'a_max_scale', 'v_max_scale', 'actuator_lag', 'action_delay_steps', 'action_noise_std')
NOMINAL = np.array([.18, 0., 1., 0., 1., 1., 0., 0., 0.], dtype=np.float64)
# Development domain, deliberately separate from the old CH4 stress domain.
DEFAULT_BOUNDS = np.array([[.15,.28],[-.025,.025],[.85,1.2],[-.02,.025],
                           [.85,1.1],[.9,1.05],[0.,.25],[0.,2.],[0.,.02]])
SHARED = 2
JOINT_DIM = 2 + 4 * 7


def validate_bounds(bounds,require_nominal=True):
    b = np.asarray(bounds, dtype=np.float64)
    if b.shape != (9,2) or not np.isfinite(b).all() or np.any(b[:,0] > b[:,1]):
        raise ValueError('bounds require nine finite, ordered pairs')
    if np.any(b[[2,4,5],0] <= 0) or not (0 <= b[6,0] <= b[6,1] < 1):
        raise ValueError('positive physical scales and 0 <= lag < 1 required')
    if b[7,0] < 0 or np.any(b[7] != np.floor(b[7])) or b[7,1] > 16 or b[8,0] < 0:
        raise ValueError('delay must be integer 0..16; noise must be nonnegative')
    if require_nominal and (np.any(NOMINAL < b[:,0]) or np.any(NOMINAL > b[:,1])):
        raise ValueError('development prior must contain the nominal point')
    return b.copy()


def sample_local(rng, bounds, count, shared=True):
    b = np.asarray(bounds)
    x = rng.uniform(b[:,0], b[:,1], size=(count,4,9))
    x[:,:,7] = rng.integers(int(b[7,0]), int(b[7,1])+1, size=(count,4))
    if shared:
        x[:,:,0:2] = x[:,0:1,0:2]
    return x


def joint(local):
    """One public-field realization, four independent vehicle realizations."""
    x = np.asarray(local)
    if x.shape[-2:] != (4,9) or not np.allclose(x[..., :, :2], x[..., :1, :2], atol=0, rtol=0):
        raise ValueError('inconsistent shared field in a joint physical world')
    return np.concatenate((x[...,0,:2], x[...,2:].reshape(*x.shape[:-2],28)), axis=-1)


def expand(value):
    x = np.asarray(value)
    if x.shape[-1] != JOINT_DIM:
        raise ValueError('joint context has 30 parameters')
    shared = np.broadcast_to(x[...,:2][...,None,:], (*x.shape[:-1],4,2))
    return np.concatenate((shared,x[...,2:].reshape(*x.shape[:-1],4,7)), axis=-1)


def normalize(local, bounds):
    b = np.asarray(bounds)
    return (np.asarray(local)-b[:,0]) / np.maximum(b[:,1]-b[:,0],1e-12)


@dataclass(frozen=True)
class PrivilegedContext:
    """Simulator-only. Never accepted by Actor or online estimator APIs."""
    local: np.ndarray

    def __post_init__(self):
        x = np.asarray(self.local,dtype=np.float64).copy()
        joint(x)
        if not np.isfinite(x).all() or np.any(x[:,7] != np.floor(x[:,7])):
            raise ValueError('nonfinite context or noninteger delay')
        x.setflags(write=False)
        object.__setattr__(self,'local',x)


@dataclass
class PublicSet:
    lower: np.ndarray
    upper: np.ndarray
    valid: bool = True
    support: int = 0
    reason: str = 'initial_prior'

    def __post_init__(self):
        self.lower = np.asarray(self.lower,dtype=np.float64).copy()
        self.upper = np.asarray(self.upper,dtype=np.float64).copy()
        if self.lower.shape != (4,9) or self.upper.shape != (4,9):
            raise ValueError('public set requires 4 x 9 local bounds')
        if not np.isfinite([self.lower,self.upper]).all() or np.any(self.lower > self.upper):
            raise ValueError('invalid/empty public set')
        # Delay is a set of integers; never execute a fractional delay.
        self.lower[:,7] = np.ceil(self.lower[:,7])
        self.upper[:,7] = np.floor(self.upper[:,7])
        if np.any(self.lower > self.upper):
            raise ValueError('empty integer delay support')

    @classmethod
    def prior(cls,bounds):
        b = np.asarray(bounds)
        return cls(np.tile(b[:,0],(4,1)),np.tile(b[:,1],(4,1)))

    def features(self,bounds,mode='interval'):
        b = np.asarray(bounds)
        center = normalize((self.lower+self.upper)/2,b)
        width = (self.upper-self.lower)/np.maximum(b[:,1]-b[:,0],1e-12)
        if mode == 'point':
            width = np.zeros_like(width)
        elif mode == 'fixed_width':
            width = np.ones_like(width)
        elif mode == 'none':
            center = np.zeros_like(center); width = np.zeros_like(width)
        return np.concatenate((center,width),axis=-1).astype(np.float32)

    def joint_samples(self,rng,count):
        if count < 1:
            raise ValueError('positive scenario count required')
        x = rng.uniform(self.lower,self.upper,size=(count,4,9))
        for i in range(4):
            x[:,i,7] = rng.integers(int(self.lower[i,7]),int(self.upper[i,7])+1,size=count)
        # Independent-estimator ablation uses a conservative UNION envelope for
        # common parameters; never an unjustified intersection of vehicle estimates.
        lo = self.lower[:,:2].min(0); hi = self.upper[:,:2].max(0)
        field = rng.uniform(lo,hi,size=(count,1,2))
        x[:,:,:2] = field
        return x

    def as_dict(self):
        return dict(lower=self.lower.tolist(),upper=self.upper.tolist(),valid=bool(self.valid),
                    support=int(self.support),reason=self.reason)


def coverage(estimate, truth):
    x = truth.local
    covered = (x >= estimate.lower) & (x <= estimate.upper)
    return dict(per_dimension=covered.tolist(), joint=bool(covered.all()),
                width=(estimate.upper-estimate.lower).tolist(),valid=estimate.valid)
