"""Bounded step-4 audit using sealed real transitions, never task training."""
import argparse
import copy
import json
from datetime import datetime, timezone

import numpy as np
import torch

from chapter3_bser.learning.checkpoint import state_digest
from chapter4_interval_robust.provenance import write_new
from checks.chapter4.test_nominal_learner import load_real_replay, numpy_mlp, numpy_targets

from .config import DEFAULT_CONFIG, ROOT, load, read_json, file_sha, local_path
from .learner import SingleCriticLearner
from .provenance import append_source, verify_source, imported_local_files


PROTOCOL = ROOT/'configs/ch4/nominal_maddpg_learner.json'


def load_protocol():
    p = read_json(PROTOCOL)
    if (p['schema'] != 'auv.ch4.nominal_maddpg.learner_protocol.v1' or p['stage'] != 4
            or p['task_training_enabled'] is not False or p['new_task_episodes'] != 0
            or p['maximum_rounds_per_temporary_model'] != 4 or p['batch_size'] != 128
            or p['replay_capacity'] != 2048 or p['temporary_models_discarded'] is not True
            or p['training_checkpoints_written'] != 0
            or p['nominal_admission_opened'] is not False or p['mismatch_sets_opened'] is not False
            or p['tolerance'] != dict(atol=2e-6, rtol=1e-5)
            or p['soft_update_tolerance'] != dict(atol=2e-7, rtol=2e-6)):
        raise ValueError('expected the bounded step-4 learner protocol')
    for name, sha in p['input_files'].items():
        if file_sha(local_path(name)) != sha:
            raise ValueError('sealed learner fixture changed: '+name)
    return p


def reference_round(networks, actor_optimizers, critic_optimizers, sample, gamma, tau):
    """Direct equations, separate optimizers; does not call any learner method."""
    obs, actions, rewards, next_obs, dones = sample[:5]
    active = dones[0] == 0
    with torch.no_grad():
        next_actions = [torch.zeros_like(a) for a in actions]
        for i in range(4):
            mask = active & sample.next_enabled[:, i]
            if bool(mask.any()):
                next_actions[i][mask] = networks.target_actors[i](next_obs[i][mask])
        targets = [r.clone() for r in rewards]
        if bool(active.any()):
            joint_next = torch.cat([o[active] for o in next_obs]+[a[active] for a in next_actions], 1)
            for i in range(4):
                targets[i][active] += gamma*networks.target_critics[i](joint_next)[:, 0]
    losses_q, losses_actor = [], []
    joint = torch.cat(list(obs)+list(actions), 1)
    for i in range(4):
        optimizer = critic_optimizers[i]
        optimizer.zero_grad(set_to_none=True)
        loss = ((networks.critics[i](joint)[:, 0]-targets[i])**2).mean()
        losses_q.append(float(loss.detach()))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    for i in range(4):
        mask = sample.enabled[:, i]
        if not bool(mask.any()):
            losses_actor.append(None)
            continue
        optimizer = actor_optimizers[i]
        optimizer.zero_grad(set_to_none=True)
        selected_actions = [a[mask] for a in actions]
        selected_actions[i] = networks.actors[i](obs[i][mask])
        networks.critics[i].requires_grad_(False)
        loss = -networks.critics[i](torch.cat([o[mask] for o in obs]+selected_actions, 1)).mean()
        losses_actor.append(float(loss.detach()))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        networks.critics[i].requires_grad_(True)
    with torch.no_grad():
        for online, target in ((networks.actors, networks.target_actors),
                               (networks.critics, networks.target_critics)):
            for value, other in zip(online.parameters(), target.parameters()):
                other.copy_((1-tau)*other + tau*value)
    return dict(critics=losses_q, actors=losses_actor)


def compare_state(actual, expected, tolerance):
    """Compare nested parameter/Adam states with exact metadata and numeric bounds."""
    if torch.is_tensor(actual):
        if not torch.is_tensor(expected) or actual.shape != expected.shape or actual.dtype != expected.dtype:
            raise AssertionError('reference state shape or dtype differs')
        np.testing.assert_allclose(actual.detach().numpy(), expected.detach().numpy(), **tolerance)
        return float((actual.detach().double()-expected.detach().double()).abs().max()) if actual.numel() else 0.
    if isinstance(actual, dict):
        if actual.keys() != expected.keys():
            raise AssertionError('reference state keys differ')
        return max([compare_state(actual[k], expected[k], tolerance) for k in actual]+[0.])
    if isinstance(actual, (list, tuple)):
        if len(actual) != len(expected):
            raise AssertionError('reference state lengths differ')
        return max([compare_state(a,b,tolerance) for a,b in zip(actual,expected)]+[0.])
    if actual != expected:
        raise AssertionError('reference state metadata differs')
    return 0.


def execute(folder):
    p, config, source = load_protocol(), load(), verify_source()
    folder = local_path(folder)
    if folder.exists():
        raise FileExistsError('use a new learner audit output directory')
    folder.mkdir(parents=True)
    write_new(folder/'execution_registration.json', dict(utc=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=file_sha(PROTOCOL), source_sha256=source['sha256'],
        task_training_enabled=False, new_task_episodes=0, maximum_rounds_per_model=4,
        input_files=p['input_files']))
    torch.set_num_threads(1)
    rng = torch.get_rng_state().clone()
    replay, metadata = load_real_replay()
    learner = SingleCriticLearner(config, 2729)
    all_rows = replay.sample(len(replay))
    expected_targets, expected_actions = numpy_targets(learner, all_rows)
    targets, target_actions = learner.td_targets(all_rows)
    np.testing.assert_allclose(targets.numpy(), expected_targets, **p['tolerance'])
    np.testing.assert_allclose(target_actions.numpy(), expected_actions, **p['tolerance'])
    joint = np.concatenate([np.stack([v.numpy() for v in all_rows[0]], 1).reshape(-1,112),
                            np.stack([v.numpy() for v in all_rows[1]], 1).reshape(-1,12)], 1)
    critic_errors, actor_errors = [], []
    for i in range(4):
        reference_q = numpy_mlp(learner.networks.critics[i], joint)[:, 0]
        expected_loss = ((reference_q-expected_targets[:, i])**2).mean()
        actual_loss = float(learner.critic_loss(all_rows, i).detach())
        np.testing.assert_allclose(actual_loss, expected_loss, **p['tolerance'])
        critic_errors.append(abs(actual_loss-expected_loss))
        mask = all_rows.enabled[:, i].numpy()
        obs = np.stack([v.numpy() for v in all_rows[0]], 1)[mask]
        actions = np.stack([v.numpy() for v in all_rows[1]], 1)[mask].astype(np.float64)
        actions[:, i] = numpy_mlp(learner.networks.actors[i], obs[:, i], actor=True)
        actor_joint = np.concatenate([obs.reshape(-1,112), actions.reshape(-1,12)], 1)
        expected_actor_loss = -numpy_mlp(learner.networks.critics[i], actor_joint).mean()
        actual_actor_loss = float(learner.actor_loss(all_rows, i).detach())
        np.testing.assert_allclose(actual_actor_loss, expected_actor_loss, **p['tolerance'])
        actor_errors.append(abs(actual_actor_loss-expected_actor_loss))
    terminal = all_rows[4][0].bool()
    if not torch.equal(targets[terminal], torch.stack(all_rows[2], 1)[terminal]):
        raise AssertionError('terminal target must exactly equal original team reward')
    reference = copy.deepcopy(learner.networks)
    ref_actors = [torch.optim.Adam(a.parameters(), lr=config['learning']['actor_lr']) for a in reference.actors]
    ref_critics = [torch.optim.Adam(q.parameters(), lr=config['learning']['critic_lr']) for q in reference.critics]
    initial_hash = state_digest(learner.networks.state_dict())
    rounds, expected_actor_steps = [], [0]*4
    for iteration in range(4):
        sample = replay.sample(p['batch_size'])
        before = {k:v.clone() for k,v in learner.networks.state_dict().items() if k.startswith('target_')}
        result = learner.update(sample)
        expected = reference_round(reference, ref_actors, ref_critics, sample, learner.gamma, learner.tau)
        soft_error = 0.
        for name, value in learner.networks.state_dict().items():
            if name.startswith('target_'):
                online = learner.networks.state_dict()[name[7:]].numpy().astype(np.float64)
                reference_soft = (1-learner.tau)*before[name].numpy().astype(np.float64)+learner.tau*online
                np.testing.assert_allclose(value.numpy(), reference_soft, **p['soft_update_tolerance'])
                soft_error = max(soft_error, float(np.abs(value.numpy()-reference_soft).max()))
        parameter_error = compare_state(learner.networks.state_dict(), reference.state_dict(), p['tolerance'])
        optimizer_error = max(compare_state(a.state_dict(), b.state_dict(), p['tolerance']) for a,b in
            zip(learner.actor_optimizers+learner.critic_optimizers, ref_actors+ref_critics))
        for i in range(4):
            np.testing.assert_allclose(result['critics'][i]['loss'], expected['critics'][i], **p['tolerance'])
            expected_actor_steps[i] += int(bool(sample.enabled[:, i].any()))
            if expected['actors'][i] is not None:
                np.testing.assert_allclose(result['actors'][i]['loss'], expected['actors'][i], **p['tolerance'])
        rounds.append(dict(iteration=iteration+1, sample_indices=sample[6].tolist(), actual=result,
            reference_losses=expected, max_parameter_error=parameter_error,
            max_optimizer_error=optimizer_error, max_soft_update_error=soft_error))
    expected_counts = dict(rounds=4, target_updates=4, actor_steps=expected_actor_steps,
                           critic_steps=[4]*4, actor_skips=[4-v for v in expected_actor_steps])
    if learner.counters() != expected_counts or not torch.equal(rng, torch.get_rng_state()):
        raise AssertionError('update counts or external Torch RNG changed unexpectedly')
    if any(p.grad is not None or p.requires_grad for group in
           (learner.networks.target_actors,learner.networks.target_critics) for p in group.parameters()):
        raise AssertionError('target network gradient isolation failed')
    result = dict(schema='auv.ch4.nominal_maddpg.learner_acceptance.v1', passed=True,
        source_sha256=source['sha256'], protocol_sha256=file_sha(PROTOCOL),
        real_transition_rows=len(replay), real_transition_metadata=metadata,
        max_td_target_error=float(np.abs(targets.numpy()-expected_targets).max()),
        max_target_action_error=float(np.abs(target_actions.numpy()-expected_actions).max()),
        max_critic_loss_error=max(critic_errors), max_actor_loss_error=max(actor_errors),
        terminal_target_exact=True, all_real_rows_checked=True, rounds=rounds,
        initial_weights_sha256=initial_hash, final_weights_sha256=state_digest(learner.networks.state_dict()),
        counters=learner.counters(), independent_reference_rounds=4,
        global_torch_rng_unchanged=True, new_task_episodes=0, training_checkpoints_written=0,
        task_training_enabled=False, learner_update_validated=True, nominal_performance_verified=False,
        nominal_admission_opened=False, mismatch_sets_opened=False,
        local_imports=imported_local_files(source))
    write_new(folder/'learner_acceptance.json', result)
    print(json.dumps({k:result[k] for k in ('passed','real_transition_rows','max_td_target_error',
        'max_target_action_error','max_critic_loss_error','max_actor_loss_error','counters')}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--register-source', action='store_true')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--output', default='verification/nominal_maddpg_learner_20261125/equations')
    args = parser.parse_args()
    load_protocol()
    if args.register_source:
        append_source(DEFAULT_CONFIG, stage=4, reason='Add independent single-critic MADDPG learning equations and bounded learner validation; preserve all task and historical identities')
    if args.execute:
        execute(args.output)
    else:
        print(json.dumps(dict(stage=4,execute=False,new_task_episodes=0,maximum_rounds_per_model=4)))


if __name__ == '__main__':
    main()
