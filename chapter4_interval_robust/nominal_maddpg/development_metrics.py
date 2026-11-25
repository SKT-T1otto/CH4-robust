"""Observational diagnostics for a nominal development episode. Nothing here drives the policy."""
from collections import Counter
import numpy as np
import torch

from experiments.ch4_work1_recoverability.plant_adapter import RHO


class EpisodeMetrics:
    def __init__(self, construction_seconds):
        self.steps = 0
        self.rounds = 0
        self.found = self.handoff = self.success = False
        self.gates = np.zeros(4, dtype=int)
        self.saturated = np.zeros(4, dtype=int)
        self.amplitude = np.zeros(4)
        self.command_delta = np.zeros(4)
        self.thrust_delta = np.zeros(4)
        self.nonzero_delta = np.zeros(4, dtype=int)
        self.actor_rows = np.zeros(4, dtype=int)
        self.post_handoff_open_steps = self.post_handoff_sample_rows = 0
        self.seconds = dict(scene_construction=construction_seconds, environment_steps=0., updates=0.)
        self.ranges = {}

    def observe(self, runtime, row, elapsed):
        self.steps += 1
        self.seconds['environment_steps'] += elapsed
        actions = np.asarray(row['actions'])
        gates = np.asarray(row['action_enabled'], dtype=bool)
        self.gates += gates
        self.saturated += (np.abs(actions) >= 1.).sum(axis=1)
        self.amplitude += np.abs(actions).sum(axis=1)
        task = runtime.rt.env.get_task_state()
        self.found |= bool(task.target_found)
        self.handoff |= bool(task.executor_knows_target and task.handoff_step is not None)
        self.success |= bool(task.mission_complete)
        before = runtime.last_record['legacy']['before']
        post = bool(before['executor_received'])
        self.post_handoff_open_steps += int(post and gates[3])

        # Compare at the SAME pre-action state, prior, limits and actuator memory.
        # This measures a direct command/thrust change, not a trajectory-level effect.
        plant = runtime.world.plant
        prior = runtime.world.core._last_prior_acc.clamp(-plant.limits, plant.limits)
        delta = (plant.command - prior).detach()
        self.command_delta += delta.norm(dim=1).numpy()
        self.nonzero_delta += (delta != 0).any(dim=1).numpy()
        memory = torch.as_tensor(before['lag_before'], dtype=plant.lag.dtype)
        baseline = (RHO * memory + (1 - RHO) * (prior / plant.limits)) * plant.limits
        baseline[:, :2] *= plant.s
        self.thrust_delta += (plant.thrust_acc - baseline).norm(dim=1).numpy()
        return post

    def update(self, sample, update, elapsed):
        self.seconds['updates'] += elapsed
        self.rounds += 1
        self.actor_rows += [actor['rows'] for actor in update['actors']]
        self.post_handoff_sample_rows += sum(source['executor_post_handoff'] and bool(enabled)
            for source, enabled in zip(sample.sources, sample.enabled[:, 3]))
        for key, low, high in (
                ('q', min(c['q_min'] for c in update['critics']), max(c['q_max'] for c in update['critics'])),
                ('td_target', update['target_min'], update['target_max']),
                ('critic_gradient_norm', min(c['gradient_norm'] for c in update['critics']),
                                         max(c['gradient_norm'] for c in update['critics'])),
                ('actor_gradient_norm', None, None)):
            if key == 'actor_gradient_norm':
                applied = [a['gradient_norm'] for a in update['actors'] if not a['skipped']]
                if not applied:
                    continue
                low, high = min(applied), max(applied)
            if not np.isfinite([low, high]).all():
                raise FloatingPointError('nonfinite ' + key)
            prev = self.ranges.get(key, [low, high])
            self.ranges[key] = [min(prev[0], low), max(prev[1], high)]

    def report(self):
        return dict(steps=self.steps, rounds=self.rounds, discovery=self.found, handoff=self.handoff,
            execution_success=self.success, gate_open=self.gates.tolist(),
            absolute_residual_sum=self.amplitude.tolist(), saturated_components=self.saturated.tolist(),
            command_delta_l2_sum_mps2=self.command_delta.tolist(), thrust_delta_l2_sum_mps2=self.thrust_delta.tolist(),
            nonzero_command_steps=self.nonzero_delta.tolist(), actor_effective_rows=self.actor_rows.tolist(),
            executor_post_handoff_open_steps=self.post_handoff_open_steps,
            executor_post_handoff_effective_rows=self.post_handoff_sample_rows,
            seconds=self.seconds.copy(), ranges=self.ranges.copy())


def summarize(rows):
    """Only native complete outcomes enter the denominator, one per scene/episode."""
    if not rows or any(not r['completed'] for r in rows):
        raise ValueError('development statistics require complete episodes')
    diagnostics = [r['diagnostics'] for r in rows]
    steps = sum(d['steps'] for d in diagnostics)
    def total(name):
        return np.sum([d[name] for d in diagnostics], axis=0)
    gates = total('gate_open')
    def per_open(values, divisor):
        return [float(n / k) if k else None for n, k in zip(values, divisor)]
    ranges = {}
    for d in diagnostics:
        for k, (lo, hi) in d['ranges'].items():
            old = ranges.get(k, [lo, hi])
            ranges[k] = [min(old[0], lo), max(old[1], hi)]
    return dict(episodes=len(rows), team_steps=steps, update_rounds=int(total('rounds')),
        discoveries=int(total('discovery')), handoffs=int(total('handoff')), execution_successes=int(total('execution_success')),
        success_rate=float(np.mean([r['success'] for r in rows])),
        mean_penalized_completion_steps=float(np.mean([r['penalized_completion_steps'] for r in rows])),
        mean_team_return=float(np.mean([r['team_undiscounted_return'] for r in rows])),
        mean_discounted_return=float(np.mean([r['team_discounted_return'] for r in rows])),
        terminations=dict(Counter(r['termination'] for r in rows)),
        gate_open_rate=(gates/steps).tolist(), actor_effective_rows=total('actor_effective_rows').tolist(),
        mean_abs_applied_residual_per_open_component=per_open(total('absolute_residual_sum'),3*gates),
        saturated_fraction_per_open_component=per_open(total('saturated_components'),3*gates),
        mean_command_delta_l2_per_team_step_mps2=(total('command_delta_l2_sum_mps2')/steps).tolist(),
        mean_thrust_delta_l2_per_team_step_mps2=(total('thrust_delta_l2_sum_mps2')/steps).tolist(),
        nonzero_command_steps=total('nonzero_command_steps').tolist(),
        executor_post_handoff_open_steps=int(total('executor_post_handoff_open_steps')),
        executor_post_handoff_effective_rows=int(total('executor_post_handoff_effective_rows')),
        ranges=ranges, seconds={k:sum(d['seconds'][k] for d in diagnostics) for k in diagnostics[0]['seconds']})
