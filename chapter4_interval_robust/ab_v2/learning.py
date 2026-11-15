"""Factual SAC and SIRSA-style tail queries on frozen conditional scene banks."""
import copy,math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from ..learning import mlp,lower_tail
from .parameters import PolicyInput,ACTOR_DIM,CRITIC_DIM
from .identifier import EnsembleIdentifier

class Actor(nn.Module):
    def __init__(self,hidden,options):
        super().__init__();self.options=options;self.agents=nn.ModuleList([mlp(165,6,hidden) for _ in range(4)])
        for a in self.agents:
            nn.init.zeros_(a[-1].weight);nn.init.zeros_(a[-1].bias)
            with torch.no_grad():a[-1].bias[3:]=-2
    def sample(self,obs,active,deterministic=False):
        # Actor-only ablations do not rewrite Critic observations or query banks.
        x=obs.clone()
        if self.options.get('width_off'):x[:,:,49:55]=1
        if self.options.get('memory_off'):x[:,:,61:]=0
        actions=[];densities=[]
        for i,a in enumerate(self.agents):
            mean,logstd=a(x[:,i]).chunk(2,-1);mean=mean+torch.atanh(x[:,i,39:42].clamp(-.98,.98))
            distribution=torch.distributions.Normal(mean,logstd.clamp(-5,1).exp())
            z=mean if deterministic else distribution.rsample();u=torch.tanh(z)
            logp=(distribution.log_prob(z)-2*(math.log(2.)-z-F.softplus(-2*z))).sum(-1)
            enabled=active[:,i].float();actions.append(u*enabled[:,None]);densities.append(logp*enabled)
        return torch.stack(actions,1),torch.stack(densities,1).sum(-1)

class Critic(nn.Module):
    def __init__(self,h):
        super().__init__();self.q1=mlp(CRITIC_DIM,1,h);self.q2=mlp(CRITIC_DIM,1,h)
    def forward(self,o,a,xi):
        x=torch.cat((o.flatten(1),a.flatten(1),xi),-1)
        return self.q1(x).squeeze(-1),self.q2(x).squeeze(-1)

FIELDS=('observations','active','proposal','reward','next_observations','next_active','context','next_context',
        'terminated','truncated','scene_bank','next_history','privileged_target')
class Replay:
    def __init__(self,n,seed):self.capacity=n;self.rows=[];self.cursor=0;self.rng=np.random.default_rng(seed)
    def append(self,row):
        clean={k:copy.deepcopy(row[k]) for k in FIELDS}
        if np.asarray(clean['context']).shape!=(33,) or np.asarray(clean['next_context']).shape!=(33,):raise ValueError('factual context33 required')
        if clean['scene_bank']['schema']!='ch4.conditional_scene_bank33.v2':raise ValueError('v1 boxes cannot replace conditional scene banks')
        if len(self.rows)<self.capacity:self.rows.append(clean)
        else:self.rows[self.cursor]=clean
        self.cursor=(self.cursor+1)%self.capacity
    def sample(self,n):return [self.rows[i] for i in self.rng.choice(len(self.rows),n,replace=False)]
    def state_dict(self):return dict(capacity=self.capacity,rows=self.rows,cursor=self.cursor,rng=self.rng.bit_generator.state)
    def load_state_dict(self,s):
        if s['capacity']!=self.capacity:raise ValueError('replay capacity changed')
        self.rows=copy.deepcopy(s['rows']);self.cursor=s['cursor'];self.rng.bit_generator.state=s['rng']

def history_summary(o):
    return np.asarray(o)[...,61:].reshape(*np.asarray(o).shape[:-1],8,13)[...,:12].mean(-2)

class Support:
    """Empirical joint-history/action gate, not an extrapolation certificate."""
    def __init__(self,opts):self.opts=opts;self.rows=[]
    def add(self,row):
        self.rows.append((np.asarray(row['context']).copy(),np.asarray(row['observations']).copy(),np.asarray(row['proposal']).copy()))
        self.rows=self.rows[-self.opts['maximum_facts']:]
    def supported(self,obs,action,queries):
        out=np.zeros(queries.shape[:2],bool)
        if len(self.rows)<self.opts['minimum_facts']:return out
        xi=np.asarray([r[0] for r in self.rows]);o=np.asarray([r[1] for r in self.rows]);a=np.asarray([r[2] for r in self.rows])
        h=history_summary(o);r=self.opts['radii']
        for i,(state,act,q) in enumerate(zip(obs,action,queries)):
            same=np.all(o[:,:,32:36]==state[None,:,32:36],axis=(1,2))
            same&=np.sqrt(np.mean((o[:,:,:28]-state[:,:28])**2,axis=(1,2)))<=r['state']
            same&=np.sqrt(np.mean((h-history_summary(state))**2,axis=(1,2)))<=r['history']
            same&=np.sqrt(np.mean((o[:,:,43:61]-state[:,43:61])**2,axis=(1,2)))<=r['set']
            same&=np.all((o[:,:,20]<.5)==(state[None,:,20]<.5),axis=1)
            same&=np.sqrt(np.mean((a-act)**2,axis=(1,2)))<=r['action']
            bank=xi[same]
            if not len(bank):continue
            for j,query in enumerate(q):
                nearby=(np.sqrt(np.mean((bank[:,:21]-query[:21])**2,1))<=r['theta'])&(
                         np.sqrt(np.mean((bank[:,21:]-query[21:])**2,1))<=r['memory'])
                out[i,j]=len(np.unique(np.round(bank[nearby,:21],5),axis=0))>=self.opts['minimum_facts']
        return out

class MultiAgentSAC:
    def __init__(self,c):
        self.config=copy.deepcopy(c);self.options=c['method_options'];torch.set_num_threads(1);torch.manual_seed(c['seed'])
        h=c['sac']['hidden'];self.actor=Actor(h,self.options);self.critic=Critic(h);self.target=copy.deepcopy(self.critic).requires_grad_(False)
        self.log_temperature=nn.Parameter(torch.tensor(math.log(c['sac']['initial_temperature'])))
        self.actor_optimizer=torch.optim.Adam(self.actor.parameters(),lr=c['sac']['lr'])
        self.critic_optimizer=torch.optim.Adam(self.critic.parameters(),lr=c['sac']['lr'])
        self.temperature_optimizer=torch.optim.Adam([self.log_temperature],lr=c['sac']['lr'])
        self.identifier=EnsembleIdentifier(c['prior'],c['estimation'])
        self.replay=Replay(c['sac']['replay_size'],c['seed']+7401);self.support=Support(c['risk'])
        self.rng=np.random.default_rng(c['seed']+7402);self.updates=0;self.environment_steps=0
    @torch.no_grad()
    def act(self,public,deterministic=False):
        if not isinstance(public,PolicyInput):raise TypeError('v2 Actor accepts public165 PolicyInput only')
        return self.actor.sample(torch.tensor(public.observations[None]),torch.tensor(public.active[None]),deterministic)[0][0].numpy()
    def remember(self,row):self.replay.append(row);self.support.add(row)
    def update(self):
        if self.config.get('mechanism') and self.updates>=self.config['maximum_updates']:raise RuntimeError('four-update mechanism ceiling')
        c=self.config;rows=self.replay.sample(c['sac']['batch_size'])
        def tensor(k,dtype=torch.float32):return torch.tensor(np.asarray([r[k] for r in rows]),dtype=dtype)
        o=tensor('observations');a=tensor('proposal');nxt=tensor('next_observations');xi=tensor('context');nxi=tensor('next_context')
        active=tensor('active',torch.bool);na=tensor('next_active',torch.bool);term=tensor('terminated');temp=self.log_temperature.exp().detach()
        with torch.no_grad():
            action,logp=self.actor.sample(nxt,na);qa,qb=self.target(nxt,action,nxi)
            y=tensor('reward')+c['sac']['gamma']*(1-term)*(torch.minimum(qa,qb)-temp*logp)
        qa,qb=self.critic(o,a,xi);qloss=F.mse_loss(qa,y)+F.mse_loss(qb,y)
        self.critic_optimizer.zero_grad();qloss.backward();self.critic_optimizer.step()
        self.critic.requires_grad_(False);action,logp=self.actor.sample(o,active)
        qa,qb=self.critic(o,action,xi);utility=torch.minimum(qa,qb);supported=np.zeros(len(rows),bool);disagreement=0.
        if self.options['risk']:
            queries=np.array([r['scene_bank']['contexts'] for r in rows],np.float32);B,N=queries.shape[:2]
            passed=self.support.supported(o.detach().numpy(),action.detach().numpy(),queries)
            ro=o[:,None].expand(B,N,4,165).reshape(B*N,4,165);ra=action[:,None].expand(B,N,4,3).reshape(B*N,4,3)
            q1,q2=self.critic(ro,ra,torch.tensor(queries.reshape(-1,33)))
            q1=q1.reshape(B,N);q2=q2.reshape(B,N)
            diff=(q1-q2).abs()/(1+((q1+q2)/2).abs());disagreement=float(diff.detach().mean())
            supported=passed.all(1)&np.array([r['scene_bank']['valid'] for r in rows])&(
                diff.detach().numpy()<=c['risk']['maximum_q_disagreement']).all(1)
            tail=lower_tail(torch.minimum(q1,q2),self.options.get('risk_alpha',c['risk']['alpha']),c['risk']['minimum_tail_samples'])
            utility=torch.where(torch.tensor(supported),tail,utility)
        utility_mean=float(utility.detach().mean())
        aloss=(temp*logp-utility).mean();self.actor_optimizer.zero_grad();aloss.backward();self.actor_optimizer.step();self.critic.requires_grad_(True)
        tloss=-(self.log_temperature*(logp.detach()-3*active.float().sum(-1))).mean()
        self.temperature_optimizer.zero_grad();tloss.backward();self.temperature_optimizer.step()
        iloss=0.
        if self.options['estimator']=='ensemble':iloss=self.identifier.update_supervised(np.array([r['next_history'] for r in rows]),np.array([r['privileged_target'] for r in rows]))
        with torch.no_grad():
            for target,value in zip(self.target.parameters(),self.critic.parameters()):target.lerp_(value,c['sac']['tau'])
        self.updates+=1
        values=dict(update=self.updates,critic_loss=float(qloss.detach()),actor_loss=float(aloss.detach()),temperature=float(self.log_temperature.exp().detach()),
                    identifier_loss=iloss,risk_supported_rows=int(supported.sum()),risk_fallback_rows=int((~supported).sum()),q_disagreement=disagreement,
                    target_mean=float(y.mean()),actor_utility_mean=utility_mean,bootstrap_rows=int((1-term).sum()))
        if not all(np.isfinite(v) for v in values.values()):raise FloatingPointError('nonfinite SAC update')
        return values
    def state_dict(self):
        return dict(actor=self.actor.state_dict(),critic=self.critic.state_dict(),target=self.target.state_dict(),identifier=self.identifier.state_dict(),
            identifier_optimizer=self.identifier.optimizer.state_dict(),identifier_updates=self.identifier.updates,
            actor_optimizer=self.actor_optimizer.state_dict(),critic_optimizer=self.critic_optimizer.state_dict(),
            temperature_optimizer=self.temperature_optimizer.state_dict(),log_temperature=self.log_temperature.detach(),
            replay=self.replay.state_dict(),support=self.support.rows,rng=self.rng.bit_generator.state,torch_rng=torch.get_rng_state(),
            updates=self.updates,environment_steps=self.environment_steps,boundary='episode_complete_reset_before_next')
    def load_state_dict(self,s):
        if s['boundary']!='episode_complete_reset_before_next':raise ValueError('only episode-boundary restore supported')
        for name in ('actor','critic','target','identifier'):getattr(self,name).load_state_dict(s[name])
        for name in ('actor_optimizer','critic_optimizer','temperature_optimizer'):getattr(self,name).load_state_dict(s[name])
        self.identifier.optimizer.load_state_dict(s['identifier_optimizer']);self.identifier.updates=s['identifier_updates']
        with torch.no_grad():self.log_temperature.copy_(s['log_temperature'])
        self.replay.load_state_dict(s['replay']);self.support.rows=copy.deepcopy(s['support']);self.rng.bit_generator.state=s['rng'];torch.set_rng_state(s['torch_rng'])
        self.updates=s['updates'];self.environment_steps=s['environment_steps']
