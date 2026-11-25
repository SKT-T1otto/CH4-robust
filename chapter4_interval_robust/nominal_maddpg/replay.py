"""Real executed transitions with a sampling stream separate from the simulator."""
import numpy as np
import torch
import copy
from core.replay.ch3_buffer import CH3ReplayBuffer

from chapter3_bser.learning.replay import ExecutionReplay


class NominalReplay(ExecutionReplay):
    def __init__(self, capacity, sampling_seed):
        super().__init__(capacity)
        if type(sampling_seed) is not int or sampling_seed < 0:
            raise ValueError("sampling seed must be a nonnegative integer")
        self.sampling_rng = torch.Generator().manual_seed(sampling_seed).get_state()
        self.sample_calls = 0
        self.sampled_rows = 0

    def state_dict(self):
        state = CH3ReplayBuffer.state_dict(self)
        state.update(schema='auv.ch4.executed_replay.v1',
            enabled=self.enabled[:len(self)].clone(), next_enabled=self.next_enabled[:len(self)].clone(),
            gate_initialized=self.gate_initialized[:len(self)].clone(),
            sampling_rng=self.sampling_rng.clone(), sample_calls=self.sample_calls, sampled_rows=self.sampled_rows)
        return state

    def load_state_dict(self, state):
        """Validate extensions before the parent performs its own atomic validation."""
        if state.get('schema') != 'auv.ch4.executed_replay.v1':
            raise ValueError('independent executed Replay schema differs')
        used = state.get('filled_i')
        if type(used) is not int or not 0 <= used <= self.max_steps:
            raise ValueError('invalid Replay valid length')
        for key, shape in (('enabled', (used,4)), ('next_enabled', (used,4)), ('gate_initialized', (used,))):
            value = state.get(key)
            if not torch.is_tensor(value) or value.dtype != torch.bool or value.device.type != 'cpu' or value.shape != shape:
                raise ValueError('invalid Replay '+key)
        if not bool(state['gate_initialized'].all()):
            raise ValueError('unwritten Replay gates')
        for key in ('sample_calls', 'sampled_rows'):
            if type(state.get(key)) is not int or state[key] < 0:
                raise ValueError('invalid Replay '+key)
        if (state['sampled_rows'] < state['sample_calls']
                or state.get('frame') != 1.0+state['sample_calls']/4):
            raise ValueError('Replay sampling counters disagree')
        rng = torch.Generator()
        rng.set_state(state['sampling_rng'])
        for key, shape in (('ac_buffs',(used,3)), ('rew_buffs',(used,)), ('done_buffs',(used,))):
            values = state.get(key)
            if not isinstance(values,list) or len(values)!=4 or any(not torch.is_tensor(v) or v.shape!=shape for v in values):
                raise ValueError('invalid Replay '+key)
        actions = torch.stack(state['ac_buffs'],1)
        if bool((actions.abs()>1).any()) or bool((actions[~state['enabled']]!=0).any()):
            raise ValueError('Replay actions disagree with gates')
        dones = state['done_buffs'][0].bool()
        if (any(not torch.equal(state['done_buffs'][0],v) for v in state['done_buffs'])
                or any(not torch.equal(state['rew_buffs'][0],v) for v in state['rew_buffs'])
                or bool(state['next_enabled'][dones].any())
                or not torch.is_tensor(state.get('success_buffs'))
                or state['success_buffs'].shape != (used,)
                or bool((state['success_buffs'].bool() & ~dones).any())):
            raise ValueError('Replay team reward/terminal/success contract differs')
        CH3ReplayBuffer.load_state_dict(self, state)
        for key in ('enabled','next_enabled','gate_initialized'):
            getattr(self,key).zero_()
            getattr(self,key)[:used].copy_(state[key])
        self.sampling_rng = rng.get_state().clone()
        self.sample_calls, self.sampled_rows = state['sample_calls'], state['sampled_rows']

    def push_transition(self, transition):
        packed = dict(transition)
        for name in ("observations", "next_observations"):
            packed[name] = np.asarray(transition[name], dtype=np.float32)
        super().push_transition(packed)

    def sample(self, n, *, norm_rews=False, device="cpu"):
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(self.sampling_rng)
            result = super().sample(n, norm_rews=norm_rews, device=device)
            self.sampling_rng = torch.get_rng_state().clone()
        self.sample_calls += 1
        self.sampled_rows += result[6].numel()
        return result

    def verify_slot(self, index, transition):
        fields = (("observations", self.obs_buffs), ("actions", self.ac_buffs),
                  ("rewards", self.rew_buffs), ("next_observations", self.next_obs_buffs),
                  ("dones", self.done_buffs))
        for name, buffers in fields:
            for hull in range(4):
                expected = torch.as_tensor(transition[name][hull], dtype=buffers[hull].dtype)
                if not torch.equal(buffers[hull][index], expected):
                    raise AssertionError("real transition differs from Replay: " + name)
        for name, buffer in (("action_enabled", self.enabled), ("next_action_enabled", self.next_enabled)):
            if not torch.equal(buffer[index], torch.as_tensor(transition[name], dtype=torch.bool)):
                raise AssertionError("Replay execution gate differs")
        if bool(self.success_buffs[index]) != bool(transition["success_flags"][0]):
            raise AssertionError("Replay mission success differs")

    def verify_sample(self, n):
        sample = self.sample(n)
        obs, actions, rewards, next_obs, dones, weights, indices, successes = sample
        buffers = (self.obs_buffs, self.ac_buffs, self.rew_buffs, self.next_obs_buffs, self.done_buffs)
        for values, stored in zip((obs, actions, rewards, next_obs, dones), buffers):
            for hull in range(4):
                if not torch.equal(values[hull], stored[hull][indices]):
                    raise AssertionError("sampled transition was transformed or relabeled")
        if (not torch.equal(sample.enabled, self.enabled[indices])
                or not torch.equal(sample.next_enabled, self.next_enabled[indices])
                or not torch.equal(successes, self.success_buffs[indices])
                or not torch.equal(weights, torch.ones_like(weights))):
            raise AssertionError("sample gates, success flags or uniform weights differ")
        critic_input = torch.cat((*obs, *actions), dim=1)
        if critic_input.shape != (len(indices), 124) or not bool(torch.isfinite(critic_input).all()):
            raise AssertionError("invalid 124D real critic batch")
        return dict(rows=len(indices), critic_shape=list(critic_input.shape),
                    terminal_rows=int(dones[0].sum()), indices=indices.tolist())


class TrainingReplay(NominalReplay):
    """Training-only real transitions with registry-checked ring-slot provenance."""
    def __init__(self, capacity, sampling_seed, catalog):
        from .train_config import SceneCatalog
        if not isinstance(catalog, SceneCatalog):
            raise TypeError('training Replay requires a verified scene catalog')
        super().__init__(capacity, sampling_seed)
        self.catalog = catalog
        self.slot_sources = [None]*capacity
        self.insertions = 0

    def state_dict(self):
        state = super().state_dict()
        state.update(catalog_sha256=self.catalog.sha256, insertions=self.insertions,
                     slot_sources=copy.deepcopy(self.slot_sources[:len(self)]))
        return state

    def load_state_dict(self, state):
        from .train_config import SceneOrigin
        used, count = state.get('filled_i'), state.get('insertions')
        if (state.get('catalog_sha256') != self.catalog.sha256 or type(count) is not int or count < 0
                or used != min(count,self.max_steps) or state.get('next_idx') != count%self.max_steps):
            raise ValueError('training Replay catalog/count/pointer differs')
        sources = state.get('slot_sources')
        if not isinstance(sources,list) or len(sources)!=used:
            raise ValueError('missing training Replay slot sources')
        for index, record in enumerate(sources):
            origin = SceneOrigin(**{key:record[key] for key in SceneOrigin.__dataclass_fields__})
            self.catalog.validate_origin(origin, allowed_groups=('train',))
            insertion = record.get('insertion')
            if (type(insertion) is not int or not count-used <= insertion < count
                    or insertion%self.max_steps!=index or type(record.get('transition_step')) is not int
                    or not 0 <= record['transition_step'] < self.catalog.config['task']['max_steps']):
                raise ValueError('invalid ring-slot insertion or task step')
        super().load_state_dict(state)
        self.slot_sources = copy.deepcopy(sources)+[None]*(self.max_steps-used)
        self.insertions = count

    def bind_scene(self, scene, episode_id):
        return self.catalog.origin(scene, episode_id, allowed_groups=('train',))

    def push_transition(self, transition, *, source=None):
        self.catalog.validate_origin(source, allowed_groups=('train',))
        step = transition.get('t')
        if type(step) is not int or step < 0:
            raise ValueError('training Replay requires the actual transition step')
        index = self.next_idx
        record = dict(**source.as_dict(), transition_step=step, insertion=self.insertions)
        super().push_transition(transition)
        self.slot_sources[index] = record
        self.insertions += 1

    def sample(self, n, *, norm_rews=False, device='cpu'):
        result = super().sample(n, norm_rews=norm_rews, device=device)
        records = []
        for index in result[6].cpu().tolist():
            record = self.slot_sources[index]
            if record is None or record['group'] != 'train' or record['catalog_sha256'] != self.catalog.sha256:
                raise ValueError('training Replay has an invalid scene source')
            records.append(copy.deepcopy(record))
        result.sources = records
        return result

    def verify_slot(self, index, transition, *, source=None):
        super().verify_slot(index, transition)
        if source is not None:
            expected = dict(**source.as_dict(), transition_step=transition['t'], insertion=self.insertions-1)
            if self.slot_sources[index] != expected:
                raise AssertionError('training Replay slot source differs from the real transition')
