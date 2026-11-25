"""Live learner policy and isolated evaluation snapshots for the existing runtime."""
import copy

import numpy as np
import torch

from chapter3_bser.learning.checkpoint import digest, state_digest
from experiments.ch4_work1_recoverability.records import stamp
from .learner import SingleCriticLearner
from .noise import OUNoise
from .train_config import load_training_protocol


def observation_tensor(observations):
    array = np.asarray(observations, dtype=np.float32)
    if array.shape != (4, 28) or not np.isfinite(array).all():
        raise ValueError('policy requires four finite local 28D observations')
    return array, torch.from_numpy(array.copy()).unsqueeze(0)


class TrainingPolicy:
    def __init__(self, learner, exploration_seed):
        if not isinstance(learner, SingleCriticLearner):
            raise TypeError('training policy requires the independent single-critic learner')
        p = load_training_protocol()['exploration']
        self.learner = learner
        self.noise = OUNoise(exploration_seed, **{k:p[k] for k in (
            'initial_sigma','minimum_sigma','decay_per_environment_step','theta','dt','mean')})
        self.last_decision = None
        self.last_input_sha256 = None
        self.policy_id = digest(dict(kind='live_single_critic_maddpg',
            initial_weights=state_digest(learner.networks.state_dict()), exploration_seed=exploration_seed))
        self._active_episode = False

    def begin_episode(self):
        if self._active_episode:
            raise RuntimeError('a training policy cannot own two live episodes')
        self.noise.reset_episode()
        self._active_episode = True

    def end_episode(self):
        self._active_episode = False

    @torch.no_grad()
    def step(self, observations, explore=False):
        if not self._active_episode:
            raise RuntimeError('training policy must be bound to an explicit training episode')
        if type(explore) is not bool:
            raise ValueError('exploration switch must be boolean')
        array, tensor = observation_tensor(observations)
        actor = self.learner.networks.actions(tensor)[0]
        if not bool(torch.isfinite(actor).all()):
            raise ValueError('nonfinite live actor output')
        tick = self.noise.steps
        noise, sigma = self.noise.sample(enabled=explore)
        proposal = actor + noise
        clipped = proposal.clamp(-1, 1)
        self.last_input_sha256 = stamp(array)
        self.last_decision = dict(actor=actor.tolist(), noise=noise.tolist(), proposal=proposal.tolist(),
            clipped=clipped.tolist(), sigma=sigma, exploration_enabled=explore, noise_step=tick,
            learner_counters=self.learner.counters(),
            actor_sha256=state_digest(self.learner.networks.actors.state_dict()))
        return [clipped[i:i+1].clone() for i in range(4)]

    def evaluation_snapshot(self):
        return FrozenEvaluationPolicy(self.learner)


class FrozenEvaluationPolicy:
    def __init__(self, learner):
        self.networks = copy.deepcopy(learner.networks).requires_grad_(False).eval()
        self.networks.zero_grad(set_to_none=True)
        self.initial_sha256 = state_digest(self.networks.state_dict())
        self.policy_id = digest(dict(kind='frozen_evaluation_single_critic_maddpg', weights=self.initial_sha256))
        self.last_input_sha256 = None

    @torch.no_grad()
    def step(self, observations, explore=False):
        if explore:
            raise ValueError('evaluation snapshot forbids exploration')
        array, tensor = observation_tensor(observations)
        self.last_input_sha256 = stamp(array)
        actions = self.networks.actions(tensor)[0]
        return [actions[i:i+1].clone() for i in range(4)]

    def verify_frozen(self):
        if (state_digest(self.networks.state_dict()) != self.initial_sha256
                or any(p.grad is not None or p.requires_grad for p in self.networks.parameters())):
            raise AssertionError('evaluation snapshot changed weights or gradient permissions')
        return self.initial_sha256
