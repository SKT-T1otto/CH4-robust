"""Independent multi-agent SAC with factual context critic and supported tail queries."""
import copy
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .parameters import PublicSet, normalize, joint
from .estimation import EnsembleIdentifier
from .runtime import OBS_DIM, PolicyInput


def mlp(inputs,outputs,hidden):
    return nn.Sequential(nn.Linear(inputs,hidden),nn.ReLU(),nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,outputs))


def lower_tail(values,alpha,minimum=1):
    count=max(1,int(math.floor(values.shape[-1]*alpha)))
    if not 0<alpha<=1 or count<minimum:raise ValueError('insufficient lower-tail support')
    return torch.sort(values,dim=-1).values[...,:count].mean(-1)


class JointActor(nn.Module):
    def __init__(self,hidden):
        super().__init__();self.agents=nn.ModuleList([mlp(OBS_DIM,6,hidden) for _ in range(4)])
        for actor in self.agents:
            nn.init.zeros_(actor[-1].weight);nn.init.zeros_(actor[-1].bias)
            with torch.no_grad():actor[-1].bias[3:]=-2.

    def sample(self,observations,active,deterministic=False):
        values=[];densities=[]
        for i,actor in enumerate(self.agents):
            mean,logstd=actor(observations[:,i]).chunk(2,-1)
            # State-dependent shift is part of the Gaussian mean; the resulting
            # physical command is sent only once through the absolute interface.
            reference=observations[:,i,39:42].clamp(-.98,.98)
            mean=mean+torch.atanh(reference)
            logstd=logstd.clamp(-5.,1.)
            distribution=torch.distributions.Normal(mean,logstd.exp())
            z=mean if deterministic else distribution.rsample()
            u=torch.tanh(z)
            # Stable exact tanh change of variables (no post-filter action substitution).
            correction=2*(math.log(2.)-z-F.softplus(-2*z))
            logp=(distribution.log_prob(z)-correction).sum(-1)
            enabled=active[:,i].to(u.dtype)
            values.append(u*enabled[:,None]);densities.append(logp*enabled)
        return torch.stack(values,1),torch.stack(densities,1).sum(-1)


class TwinCritic(nn.Module):
    def __init__(self,hidden):
        super().__init__();dimension=4*OBS_DIM+12+30
        self.q1=mlp(dimension,1,hidden);self.q2=mlp(dimension,1,hidden)

    def forward(self,observations,action,context):
        x=torch.cat((observations.flatten(1),action.flatten(1),context),-1)
        return self.q1(x).squeeze(-1),self.q2(x).squeeze(-1)


REPLAY_FIELDS=('observations','active','proposal','reward','next_observations','next_active',
               'terminated','truncated','context','next_context','next_history','next_history_mask',
               'privileged_target','set_before')


class Replay:
    def __init__(self,capacity,seed):
        self.capacity=int(capacity);self.rows=[];self.cursor=0;self.rng=np.random.default_rng(seed)

    def append(self,row):
        if 'next_context' not in row:raise ValueError('factual next context required')
        clean={key:copy.deepcopy(row[key]) for key in REPLAY_FIELDS}
        if len(self.rows)<self.capacity:self.rows.append(clean)
        else:self.rows[self.cursor]=clean
        self.cursor=(self.cursor+1)%self.capacity

    def sample(self,count):
        if len(self.rows)<count:raise ValueError('insufficient factual replay')
        index=self.rng.choice(len(self.rows),size=count,replace=False)
        return [self.rows[int(i)] for i in index]

    def state_dict(self):
        return dict(capacity=self.capacity,rows=self.rows,cursor=self.cursor,rng=self.rng.bit_generator.state)

    def load_state_dict(self,state):
        if state['capacity']!=self.capacity:raise ValueError('replay capacity changed')
        self.rows=copy.deepcopy(state['rows']);self.cursor=state['cursor'];self.rng.bit_generator.state=state['rng']


class ContextSupport:
    """Conservative empirical neighborhood gate, not a Q extrapolation certificate."""
    def __init__(self,minimum,radius):
        self.minimum=minimum;self.radius=radius;self.contexts={}

    def add(self,context):
        x=np.asarray(context,np.float32)
        # Distinct physical contexts, not repeated samples from one episode.
        self.contexts[tuple(np.round(x,5))]=x.copy()

    def supported(self,queries):
        shape=queries.shape[:-1]
        if len(self.contexts)<self.minimum:return np.zeros(shape,bool)
        bank=np.stack(list(self.contexts.values()))
        q=queries.reshape(-1,30)
        result=[]
        for start in range(0,len(q),128):
            distances=np.sqrt(np.mean((q[start:start+128,None]-bank[None])**2,axis=-1))
            result.extend((distances<=self.radius).sum(-1)>=self.minimum)
        return np.asarray(result,bool).reshape(shape)


class MultiAgentSAC:
    def __init__(self,c,seed=None):
        self.config=copy.deepcopy(c);self.method=c['method_options'];self.bounds=np.asarray(c['disturbance']['bounds'])
        torch.set_num_threads(1);torch.manual_seed(c['seed'] if seed is None else seed)
        options=c['sac'];self.actor=JointActor(options['hidden']);self.critic=TwinCritic(options['hidden'])
        self.target=copy.deepcopy(self.critic).requires_grad_(False)
        self.log_temperature=nn.Parameter(torch.tensor(math.log(options['initial_temperature'])))
        self.actor_optimizer=torch.optim.Adam(self.actor.parameters(),lr=options['lr'])
        self.critic_optimizer=torch.optim.Adam(self.critic.parameters(),lr=options['lr'])
        self.temperature_optimizer=torch.optim.Adam([self.log_temperature],lr=options['lr'])
        e=c['estimation']
        self.identifier=EnsembleIdentifier(self.bounds,e['ensemble'],e['hidden'],self.method['shared'])
        self.replay=Replay(options['replay_size'],c['seed']+4401)
        self.rng=np.random.default_rng(c['seed']+4402)
        self.support=ContextSupport(c['risk']['minimum_contexts'],c['risk']['support_radius'])
        self.updates=0;self.environment_steps=0

    @torch.no_grad()
    def act(self,public,deterministic=False):
        if not isinstance(public,PolicyInput):raise TypeError('Actor accepts only PolicyInput')
        o=torch.as_tensor(public.observations[None],dtype=torch.float32)
        mask=torch.as_tensor(public.active[None],dtype=torch.bool)
        return self.actor.sample(o,mask,deterministic)[0][0].cpu().numpy()

    def remember(self,row):
        self.replay.append(row);self.support.add(row['context'])

    def risk_queries(self,rows):
        count=self.config['risk']['samples'];contexts=[];valid=[]
        for row in rows:
            saved=row['set_before']
            estimate=PublicSet(saved['lower'],saved['upper'],saved['valid'],saved['support'],saved['reason'])
            if self.method['input']=='fixed_width':estimate=PublicSet.prior(self.bounds)
            x=estimate.joint_samples(self.rng,count)
            contexts.append(joint(normalize(x,self.bounds)))
            valid.append(estimate.valid)
        queries=np.asarray(contexts,np.float32)
        supported=self.support.supported(queries)&np.asarray(valid)[:,None]
        return torch.from_numpy(queries),supported.all(1)

    def update(self):
        if self.config.get('mechanism') and self.updates>=self.config['maximum_updates']:
            raise RuntimeError('bounded multi-agent update budget exhausted')
        opts=self.config['sac'];rows=self.replay.sample(opts['batch_size'])
        def tensor(name,dtype=torch.float32):return torch.as_tensor(np.asarray([r[name] for r in rows]),dtype=dtype)
        obs=tensor('observations');active=tensor('active',torch.bool);proposal=tensor('proposal')
        nxt=tensor('next_observations');next_active=tensor('next_active',torch.bool)
        context=tensor('context');next_context=tensor('next_context')
        reward=tensor('reward');terminated=tensor('terminated');truncated=tensor('truncated')
        temperature=self.log_temperature.exp().detach()
        with torch.no_grad():
            next_action,next_logp=self.actor.sample(nxt,next_active)
            qa,qb=self.target(nxt,next_action,next_context)
            # The inherited task timeout is terminal. Collector truncation alone
            # is not terminal and must retain the bootstrap term.
            target=reward+opts['gamma']*(1-terminated)*(torch.minimum(qa,qb)-temperature*next_logp)
        qa,qb=self.critic(obs,proposal,context)
        critic_loss=F.mse_loss(qa,target)+F.mse_loss(qb,target)
        self.critic_optimizer.zero_grad();critic_loss.backward();self.critic_optimizer.step()
        self.critic.requires_grad_(False)
        action,logp=self.actor.sample(obs,active)
        qa,qb=self.critic(obs,action,context);utility=torch.minimum(qa,qb)
        supported=np.zeros(len(rows),bool)
        if self.method['risk']:
            queries,supported=self.risk_queries(rows)
            batch,count=queries.shape[:2]
            repeat_obs=obs[:,None].expand(batch,count,4,OBS_DIM).reshape(batch*count,4,OBS_DIM)
            repeat_action=action[:,None].expand(batch,count,4,3).reshape(batch*count,4,3)
            ca,cb=self.critic(repeat_obs,repeat_action,queries.reshape(batch*count,30))
            tail=lower_tail(torch.minimum(ca,cb).reshape(batch,count),self.method.get('risk_alpha',self.config['risk']['alpha']),
                            self.config['risk']['minimum_tail_samples'])
            utility=torch.where(torch.from_numpy(supported),tail,utility)
        actor_loss=(temperature*logp-utility).mean()
        self.actor_optimizer.zero_grad();actor_loss.backward();self.actor_optimizer.step()
        self.critic.requires_grad_(True)
        target_entropy=-3*active.float().sum(-1)
        temperature_loss=-(self.log_temperature*(logp.detach()+target_entropy)).mean()
        self.temperature_optimizer.zero_grad();temperature_loss.backward();self.temperature_optimizer.step()
        identifier_loss=0.
        if self.method['estimator']=='ensemble':
            identifier_loss=self.identifier.update_supervised(
                np.asarray([r['next_history'] for r in rows]),np.asarray([r['next_history_mask'] for r in rows]),
                np.asarray([r['privileged_target'] for r in rows]))
        with torch.no_grad():
            for target_parameter,parameter in zip(self.target.parameters(),self.critic.parameters()):
                target_parameter.lerp_(parameter,opts['tau'])
        self.updates+=1
        metrics=dict(update=self.updates,critic_loss=float(critic_loss.detach()),actor_loss=float(actor_loss.detach()),
                     temperature=float(self.log_temperature.exp().detach()),temperature_loss=float(temperature_loss.detach()),
                     identifier_loss=identifier_loss,risk_supported_rows=int(supported.sum()),
                     target_mean=float(target.mean()),
                     identifier_updated=int(self.method['estimator']=='ensemble'),
                     risk_fallback_rows=int((~supported).sum()),distinct_contexts=len(self.support.contexts),
                     bootstrap_rows=int(((1-terminated)>0).sum()),collector_truncation_rows=int(truncated.sum()))
        if not all(np.isfinite(v) for v in metrics.values()):raise FloatingPointError('nonfinite SAC update')
        return metrics

    def state_dict(self):
        return dict(actor=self.actor.state_dict(),critic=self.critic.state_dict(),target=self.target.state_dict(),
                    log_temperature=self.log_temperature.detach(),actor_optimizer=self.actor_optimizer.state_dict(),
                    critic_optimizer=self.critic_optimizer.state_dict(),temperature_optimizer=self.temperature_optimizer.state_dict(),
                    identifier=self.identifier.state_dict(),identifier_optimizer=self.identifier.optimizer.state_dict(),
                    identifier_updates=self.identifier.updates,replay=self.replay.state_dict(),rng=self.rng.bit_generator.state,
                    contexts=self.support.contexts,updates=self.updates,environment_steps=self.environment_steps,
                    torch_rng=torch.get_rng_state(),training_boundary='episode_complete_reset_before_next',
                    estimator_state=None,actuator_state=None)

    def load_state_dict(self,state):
        if state['training_boundary']!='episode_complete_reset_before_next':raise ValueError('mid-episode resume unsupported')
        for name in ('actor','critic','target','identifier'):getattr(self,name).load_state_dict(state[name])
        with torch.no_grad():self.log_temperature.copy_(state['log_temperature'])
        for name in ('actor_optimizer','critic_optimizer','temperature_optimizer'):
            getattr(self,name).load_state_dict(state[name])
        self.identifier.optimizer.load_state_dict(state['identifier_optimizer']);self.identifier.updates=state['identifier_updates']
        self.replay.load_state_dict(state['replay']);self.rng.bit_generator.state=state['rng']
        self.support.contexts=copy.deepcopy(state['contexts']);self.updates=state['updates'];self.environment_steps=state['environment_steps']
        torch.set_rng_state(state['torch_rng'])
