"""Independent equation and gradient tests; only bounded temporary-model updates."""
import gzip
import json
import unittest

import numpy as np
import torch
from torch import nn

from chapter3_bser.learning.checkpoint import state_digest
from chapter3_bser.learning.replay_base import ExecutionSample
from chapter4_interval_robust.nominal_maddpg.config import ROOT, load, read_json, file_sha
from chapter4_interval_robust.nominal_maddpg.learner import SingleCriticLearner, prepare_batch
from chapter4_interval_robust.nominal_maddpg.replay import NominalReplay


def synthetic_sample(size=9, seed=1837):
    rng = torch.Generator().manual_seed(seed)
    obs = torch.randn(size, 4, 28, generator=rng)
    next_obs = torch.randn(size, 4, 28, generator=rng)
    enabled = torch.arange(size * 4).reshape(size, 4) % 3 != 0
    next_enabled = torch.arange(size * 4).reshape(size, 4) % 5 != 0
    done = torch.zeros(size)
    done[-1] = 1
    next_enabled[-1] = False
    actions = (torch.rand(size, 4, 3, generator=rng)*1.8 - .9)*enabled[:, :, None]
    reward = torch.linspace(-.7, .4, size)
    values = ([obs[:, i].clone() for i in range(4)], [actions[:, i].clone() for i in range(4)],
              [reward.clone() for _ in range(4)], [next_obs[:, i].clone() for i in range(4)],
              [done.clone() for _ in range(4)], torch.ones(size), torch.arange(size),
              torch.zeros(size, dtype=torch.bool))
    return ExecutionSample(values, enabled, next_enabled)


def numpy_mlp(module, inputs, *, actor=False):
    """Independent float64 NumPy forward; never calls Torch forward or learner code."""
    state = {k: v.detach().numpy().astype(np.float64) for k, v in module.state_dict().items()}
    x = np.asarray(inputs, dtype=np.float64)
    x = (x - x.mean(axis=-1, keepdims=True))/np.sqrt(x.var(axis=-1, keepdims=True) + module.in_fn.eps)
    x = x*state['in_fn.weight'] + state['in_fn.bias']
    for layer in ('fc1', 'fc2', 'fc3'):
        x = x @ state[layer+'.weight'].T + state[layer+'.bias']
        if layer != 'fc3':
            x = np.maximum(x, 0)
    return np.tanh(x) if actor else x


def numpy_targets(learner, sample):
    observations = np.stack([v.numpy() for v in sample[3]], axis=1)
    actions = np.zeros((len(observations), 4, 3))
    targets = np.stack([v.numpy() for v in sample[2]], axis=1).astype(np.float64)
    active = sample[4][0].numpy() == 0
    for i in range(4):
        mask = active & sample.next_enabled[:, i].numpy()
        if mask.any():
            actions[mask, i] = numpy_mlp(learner.networks.target_actors[i], observations[mask, i], actor=True)
    if active.any():
        joint = np.concatenate([observations[active].reshape(-1, 112), actions[active].reshape(-1, 12)], axis=1)
        for i in range(4):
            targets[active, i] += learner.gamma*numpy_mlp(learner.networks.target_critics[i], joint)[:, 0]
    return targets, actions


def load_real_replay():
    protocol = read_json(ROOT/'configs/ch4/nominal_maddpg_learner.json')
    for relative, sha in protocol['input_files'].items():
        if file_sha(ROOT/relative) != sha:
            raise ValueError('sealed real-transition fixture changed: '+relative)
    replay = NominalReplay(protocol['replay_capacity'], protocol['mechanism_seed'])
    terminal_rows, lengths = [], {}
    for case in protocol['cases']:
        folder = ROOT/'verification/nominal_maddpg_runtime_20261125'
        receipt = read_json(folder/(case+'.json'))
        rows = 0
        with gzip.open(folder/(case+'.jsonl.gz'), 'rt', encoding='utf-8') as stream:
            for line in stream:
                payload = json.loads(line)
                transition = payload['transition']
                if transition['t'] != rows:
                    raise AssertionError('nonconsecutive real transition clock')
                index = replay.next_idx
                replay.push_transition(transition)
                replay.verify_slot(index, transition)
                if transition['dones'][0]:
                    terminal_rows.append(dict(case=case, index=index, t=transition['t'],
                        reason=payload['record']['outcome']['termination_reason']))
                rows += 1
        if rows != receipt['replay_length']:
            raise AssertionError('real fixture length differs from its sealed receipt')
        lengths[case] = rows
    return replay, dict(lengths=lengths, terminal_rows=terminal_rows)


class Bomb(nn.Module):
    def forward(self, inputs):
        raise AssertionError('terminal or closed-gate network must not be evaluated')


class AffineQ(nn.Module):
    def __init__(self, weights=None, bias=0.):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(124) if weights is None else weights.clone())
        self.bias = nn.Parameter(torch.tensor(float(bias)))

    def forward(self, inputs):
        return (inputs @ self.weight + self.bias).unsqueeze(1)


class BiasActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor([.1, -.2, .3]))

    def forward(self, inputs):
        return self.bias.tanh().expand(len(inputs), 3)


class NominalLearnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.config = load()
        cls.protocol = read_json(ROOT/'configs/ch4/nominal_maddpg_learner.json')
        cls.replay, cls.real_metadata = load_real_replay()

    def learner(self, seed=2729):
        return SingleCriticLearner(self.config, seed)

    def close(self, a, b, *, soft=False):
        tolerance = self.protocol['soft_update_tolerance' if soft else 'tolerance']
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), **tolerance)

    def test_td_targets_and_actions_match_independent_numpy(self):
        learner, sample = self.learner(), synthetic_sample()
        actual, actions = learner.td_targets(sample)
        expected, reference_actions = numpy_targets(learner, sample)
        self.close(actual.numpy(), expected)
        self.close(actions.numpy(), reference_actions)
        self.assertFalse(actual.requires_grad)
        self.assertTrue(torch.equal(actions[~sample.next_enabled], torch.zeros_like(actions[~sample.next_enabled])))

    def test_critic_loss_is_unweighted_mse_on_replay_actions(self):
        learner, sample = self.learner(), synthetic_sample()
        targets, _ = numpy_targets(learner, sample)
        joint = np.concatenate([np.stack([x.numpy() for x in sample[0]], axis=1).reshape(-1,112),
                                np.stack([x.numpy() for x in sample[1]], axis=1).reshape(-1,12)], axis=1)
        for i in range(4):
            q = numpy_mlp(learner.networks.critics[i], joint)[:, 0]
            self.close(learner.critic_loss(sample, i).detach().numpy(), np.mean((q-targets[:, i])**2))

    def test_real_success_collision_and_400_step_timeout_have_reward_only_targets(self):
        learner = self.learner()
        sample = self.replay.sample(len(self.replay))
        targets, _ = learner.td_targets(sample)
        reasons = {r['reason'] for r in self.real_metadata['terminal_rows']}
        self.assertEqual(reasons, {'success', 'obstacle_collision', 'timeout'})
        self.assertTrue(any(r['reason'] == 'timeout' and r['t'] == 399 for r in self.real_metadata['terminal_rows']))
        terminal = sample[4][0].bool()
        self.assertEqual(int(terminal.sum()), len(self.real_metadata['terminal_rows']))
        self.assertTrue(torch.equal(targets[terminal], torch.stack(sample[2], 1)[terminal]))

    def test_terminal_rows_never_call_target_networks(self):
        learner, sample = self.learner(), synthetic_sample()
        for done in sample[4]:
            done.fill_(1)
        sample.next_enabled.fill_(False)
        learner.networks.target_actors = nn.ModuleList([Bomb() for _ in range(4)])
        learner.networks.target_critics = nn.ModuleList([Bomb() for _ in range(4)])
        targets, actions = learner.td_targets(sample)
        self.assertTrue(torch.equal(targets, torch.stack(sample[2], 1)))
        self.assertFalse(bool(actions.any()))

    def test_all_next_gates_closed_still_bootstraps_nonterminal_value_without_clipping(self):
        learner, sample = self.learner(), synthetic_sample()
        sample.next_enabled.fill_(False)
        learner.networks.target_actors = nn.ModuleList([Bomb() for _ in range(4)])
        learner.networks.target_critics = nn.ModuleList([AffineQ(bias=1000).requires_grad_(False) for _ in range(4)])
        targets, actions = learner.td_targets(sample)
        expected = torch.stack(sample[2], 1) + learner.gamma*1000*(1-sample[4][0][:, None])
        self.close(targets.numpy(), expected.numpy())
        self.assertFalse(bool(actions.any()))
        self.assertGreater(float(targets[0, 0]), 900)

    def test_actor_loss_uses_only_open_rows_and_other_agents_replay_actions(self):
        learner, sample = self.learner(), synthetic_sample()
        for i in range(4):
            mask = sample.enabled[:, i].numpy()
            obs = np.stack([x.numpy() for x in sample[0]], axis=1)[mask]
            actions = np.stack([x.numpy() for x in sample[1]], axis=1)[mask].astype(np.float64)
            actions[:, i] = numpy_mlp(learner.networks.actors[i], obs[:, i], actor=True)
            joint = np.concatenate([obs.reshape(-1, 112), actions.reshape(-1, 12)], axis=1)
            expected = -numpy_mlp(learner.networks.critics[i], joint).mean()
            self.close(learner.actor_loss(sample, i).detach().numpy(), expected)

    def test_actor_update_isolates_other_actors_critics_and_targets(self):
        learner, sample = self.learner(), synthetic_sample()
        before = {k:v.clone() for k,v in learner.networks.state_dict().items()}
        opt_before = [state_digest(o.state_dict()) for o in learner.critic_optimizers]
        result = learner.update_actor(sample, 2)
        after = learner.networks.state_dict()
        self.assertFalse(result['skipped'])
        self.assertGreater(result['gradient_norm'], 0)
        self.assertTrue(any(not torch.equal(v, after[k]) for k,v in before.items() if k.startswith('actors.2.')))
        for k,v in before.items():
            if not k.startswith('actors.2.'):
                self.assertTrue(torch.equal(v, after[k]), k)
        self.assertEqual(opt_before, [state_digest(o.state_dict()) for o in learner.critic_optimizers])
        self.assertTrue(all(p.grad is None for p in learner.networks.parameters()))

    def test_closed_batch_preserves_existing_adam_momentum_and_parameter_state(self):
        learner, sample = self.learner(), synthetic_sample()
        learner.update_actor(sample, 0)
        self.assertTrue(learner.actor_optimizers[0].state)
        weights = state_digest(learner.networks.state_dict())
        state = state_digest(learner.actor_optimizers[0].state_dict())
        closed = synthetic_sample()
        closed.enabled.fill_(False)
        for actions in closed[1]:
            actions.zero_()
        result = learner.update_actor(closed, 0)
        self.assertTrue(result['skipped'])
        self.assertEqual(learner.actor_steps[0], 1)
        self.assertEqual(learner.actor_skips[0], 1)
        self.assertEqual(state, state_digest(learner.actor_optimizers[0].state_dict()))
        self.assertEqual(weights, state_digest(learner.networks.state_dict()))

    def test_closed_gate_samples_still_update_critics(self):
        learner, sample = self.learner(), synthetic_sample()
        sample.enabled.fill_(False)
        for actions in sample[1]:
            actions.zero_()
        before = state_digest(learner.networks.critics[1].state_dict())
        result = learner.update_critic(sample, 1)
        self.assertGreater(result['gradient_norm'], 0)
        self.assertNotEqual(before, state_digest(learner.networks.critics[1].state_dict()))
        self.assertEqual(learner.actor_steps, [0]*4)

    def test_analytic_actor_gradient_and_first_adam_step_direction(self):
        learner, sample = self.learner(), synthetic_sample()
        actor = BiasActor()
        weights = torch.linspace(-.2, .3, 124)
        weights[112:115] = torch.tensor([2., -3., .5])
        learner.networks.actors[0] = actor
        learner.networks.critics[0] = AffineQ(weights)
        learner.actor_optimizers[0] = torch.optim.Adam(actor.parameters(), lr=.001)
        gradient, = torch.autograd.grad(learner.actor_loss(sample, 0), actor.bias)
        expected = -weights[112:115]*(1-actor.bias.detach().tanh().square())
        self.close(gradient.detach().numpy(), expected.numpy())
        before = actor.bias.detach().clone()
        learner.update_actor(sample, 0)
        expected_after = before - .001*expected/(expected.abs()+1e-8)
        self.close(actor.bias.detach().numpy(), expected_after.numpy())
        self.assertTrue(torch.equal(torch.sign(actor.bias.detach()-before), torch.sign(weights[112:115])))

    def test_soft_update_matches_independent_numpy_and_has_no_target_gradients(self):
        learner = self.learner()
        with torch.no_grad():
            for group in (learner.networks.actors, learner.networks.critics):
                for p in group.parameters():
                    p.add_(.123)
        before = {k:v.clone() for k,v in learner.networks.state_dict().items()}
        learner.soft_update_targets()
        for name, value in learner.networks.state_dict().items():
            if name.startswith('target_'):
                expected = (1-.005)*before[name].numpy().astype(np.float64) + .005*before[name[7:]].numpy()
                self.close(value.numpy(), expected, soft=True)
            else:
                self.assertTrue(torch.equal(value, before[name]))
        self.assertEqual(learner.target_updates, 1)
        for group in (learner.networks.target_actors, learner.networks.target_critics):
            self.assertTrue(all(not p.requires_grad and p.grad is None for p in group.parameters()))

    def test_real_replay_four_rounds_have_finite_metrics_and_exact_counters(self):
        learner = self.learner()
        expected = [0]*4
        before = state_digest(learner.networks.state_dict())
        for _ in range(4):
            sample = self.replay.sample(128)
            original = state_digest(dict(values=tuple(sample), gates=sample.enabled, next_gates=sample.next_enabled))
            for i in range(4):
                expected[i] += int(bool(sample.enabled[:, i].any()))
            result = learner.update(sample)
            json.dumps(result, allow_nan=False)
            self.assertEqual(original, state_digest(dict(values=tuple(sample), gates=sample.enabled, next_gates=sample.next_enabled)))
            self.assertTrue(result['closed_target_actions_zero'])
        self.assertEqual(learner.counters(), dict(rounds=4, target_updates=4, actor_steps=expected,
                         critic_steps=[4]*4, actor_skips=[4-v for v in expected]))
        self.assertNotEqual(before, state_digest(learner.networks.state_dict()))
        self.assertTrue(all(p.grad is None for p in learner.networks.parameters()))

    def test_identical_initialization_and_batch_reproduce_update_without_rng_consumption(self):
        a, b, sample = self.learner(), self.learner(), synthetic_sample()
        before = torch.get_rng_state().clone()
        self.assertEqual(a.update(sample), b.update(sample))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertEqual(state_digest(a.networks.state_dict()), state_digest(b.networks.state_dict()))
        for x,y in zip(a.actor_optimizers+a.critic_optimizers, b.actor_optimizers+b.critic_optimizers):
            self.assertEqual(state_digest(x.state_dict()), state_digest(y.state_dict()))

    def test_invalid_samples_rejected_before_any_parameter_or_optimizer_update(self):
        for kind in ('nan', 'local_done', 'closed_action', 'terminal_gate', 'reward', 'weights', 'shape', 'success'):
            sample = synthetic_sample()
            if kind == 'nan': sample[0][0][0, 0] = float('nan')
            if kind == 'local_done': sample[4][0][0] = 1
            if kind == 'closed_action': sample[1][0][0, 0] = .5
            if kind == 'terminal_gate': sample.next_enabled[-1, 0] = True
            if kind == 'reward': sample[2][0][0] += 1
            if kind == 'weights': sample[5][0] = .5
            if kind == 'shape': sample[0][0] = sample[0][0][:, :27]
            if kind == 'success': sample[7][0] = True
            learner = self.learner()
            before = state_digest(learner.networks.state_dict())
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                learner.update(sample)
            self.assertEqual(before, state_digest(learner.networks.state_dict()))
            self.assertFalse(any(o.state for o in learner.actor_optimizers+learner.critic_optimizers))
            self.assertEqual(learner.rounds, 0)

    def test_update_rejects_no_grad_context(self):
        learner = self.learner()
        with torch.no_grad(), self.assertRaisesRegex(RuntimeError, 'outside the no_grad'):
            learner.update(synthetic_sample())

    def test_critic_update_isolates_actors_other_critics_and_targets(self):
        learner, sample = self.learner(), synthetic_sample()
        before = {k:v.clone() for k,v in learner.networks.state_dict().items()}
        learner.update_critic(sample, 1)
        after = learner.networks.state_dict()
        self.assertTrue(any(not torch.equal(v, after[k]) for k,v in before.items() if k.startswith('critics.1.')))
        for k,v in before.items():
            if not k.startswith('critics.1.'):
                self.assertTrue(torch.equal(v, after[k]), k)
        self.assertFalse(any(o.state for o in learner.actor_optimizers))
        self.assertEqual(learner.critic_steps, [0, 1, 0, 0])

    def test_nonfinite_target_rejected_before_any_online_update(self):
        learner, sample = self.learner(), synthetic_sample()
        with torch.no_grad():
            learner.networks.target_critics[0].fc3.bias.fill_(float('nan'))
        before = state_digest(dict(actors=learner.networks.actors.state_dict(),
                                   critics=learner.networks.critics.state_dict()))
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            learner.update(sample)
        self.assertEqual(before, state_digest(dict(actors=learner.networks.actors.state_dict(),
                                                   critics=learner.networks.critics.state_dict())))
        self.assertEqual(learner.rounds, 0)
        self.assertFalse(any(o.state for o in learner.actor_optimizers+learner.critic_optimizers))

    def test_complete_closed_round_updates_critics_but_skips_all_primed_actors(self):
        learner, sample = self.learner(), synthetic_sample()
        learner.update(sample)
        weights = state_digest(learner.networks.actors.state_dict())
        states = [state_digest(o.state_dict()) for o in learner.actor_optimizers]
        sample.enabled.fill_(False)
        for action in sample[1]:
            action.zero_()
        result = learner.update(sample)
        self.assertTrue(all(v['skipped'] for v in result['actors']))
        self.assertEqual(weights, state_digest(learner.networks.actors.state_dict()))
        self.assertEqual(states, [state_digest(o.state_dict()) for o in learner.actor_optimizers])
        self.assertEqual(learner.counters(), dict(rounds=2, target_updates=2,
            actor_steps=[1]*4, critic_steps=[2]*4, actor_skips=[1]*4))

    def test_batch_conversion_is_detached_and_does_not_alias_replay_input(self):
        sample = synthetic_sample()
        batch = prepare_batch(sample)
        saved = batch.observations.clone()
        sample[0][0].add_(100)
        self.assertTrue(torch.equal(batch.observations, saved))
        self.assertFalse(batch.observations.requires_grad)


if __name__ == '__main__':
    unittest.main()
