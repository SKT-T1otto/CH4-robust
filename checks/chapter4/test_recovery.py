"""Recovery censoring regressions using synthetic public trajectories only."""
from types import SimpleNamespace
import unittest
import numpy as np
from chapter4_interval_robust.metrics import EpisodeMetrics, aggregate


def summarize_trajectory(errors, changes=(), window=3):
    config=dict(method='full',sac={'gamma':.95},
                recovery={'window':window,'speed_error':.25})
    metric=EpisodeMetrics(config)
    events=[dict(step=0,kind='initial')]
    events.extend(dict(step=step,kind='parameter_change') for step in changes)
    runtime=SimpleNamespace(process=SimpleNamespace(events=events),summary=lambda:{})
    for step,error in enumerate(errors):
        # The active searcher advances toward its public navigation target, so a
        # low prediction error is sufficient for this fixture's task criterion.
        stage=dict(tracking_before=np.full(4,2.),tracking_after=np.full(4,1.5),
                   speed=np.full(4,.2),hold_progress=np.zeros(4))
        metric.add(dict(phase=np.zeros(4,int),active=np.array([True,False,False,False]),
                        interval_coverage=dict(per_dimension=np.ones((4,9),bool),
                                               width=np.ones((4,9))),
                        model_velocity_error=np.full(4,error),collision=False,
                        stage_diagnostics=stage,reward=1.,outcome={'success':False},
                        terminated=step==len(errors)-1,truncated=False,
                        set_after={'valid':True},guard=[{'intervened':False}]*4))
    return metric.summary(runtime,1.)


class RecoveryCensoring(unittest.TestCase):
    def test_later_change_cannot_recover_previous_regime(self):
        summary=summarize_trajectory([1.,1.,1.,1.,1.,0.,0.,0.],changes=(1,5))
        first,second=summary['recovery']
        self.assertFalse(first['recovered'])
        self.assertEqual(first['end_step'],5)
        self.assertEqual(first['censored_steps'],4)
        self.assertEqual(first['censor_reason'],'next_change')
        self.assertTrue(second['recovered'])
        self.assertEqual(second['recovery_steps'],3)
        self.assertIsNone(second['censor_reason'])
        combined=aggregate([summary])
        self.assertEqual(combined['recovery_event_count'],2)
        self.assertEqual(combined['recovery_unrecovered_rate'],.5)
        self.assertEqual(combined['recovery_mean_steps'],3.)

    def test_stable_window_cannot_cross_next_change(self):
        summary=summarize_trajectory([1.,1.,0.,0.,0.,0.,0.],changes=(2,4))
        first,second=summary['recovery']
        # Without censoring, transitions 2, 3, 4 form a false recovery window.
        self.assertFalse(first['recovered'])
        self.assertEqual(first['censored_steps'],2)
        self.assertEqual(first['censor_reason'],'next_change')
        self.assertTrue(second['recovered'])
        self.assertEqual(second['recovery_steps'],3)

    def test_incomplete_window_is_censored_at_episode_end(self):
        summary=summarize_trajectory([1.,1.,0.,0.],changes=(2,))
        event,=summary['recovery']
        self.assertFalse(event['recovered'])
        self.assertEqual(event['end_step'],4)
        self.assertEqual(event['censored_steps'],2)
        self.assertEqual(event['censor_reason'],'episode_end')
        self.assertIsNone(event['recovery_steps'])
        combined=aggregate([summary])
        self.assertEqual(combined['recovery_unrecovered_rate'],1.)
        self.assertIsNone(combined['recovery_mean_steps'])

    def test_complete_window_ending_before_change_counts(self):
        summary=summarize_trajectory([1.,0.,0.,0.,1.],changes=(1,4))
        first,second=summary['recovery']
        self.assertTrue(first['recovered'])
        self.assertEqual(first['recovery_steps'],3)
        self.assertEqual(first['end_step'],4)
        self.assertFalse(second['recovered'])

    def test_no_changes_has_no_recovery_rate_or_time(self):
        summary=summarize_trajectory([0.,0.,0.])
        combined=aggregate([summary])
        self.assertEqual(combined['recovery_event_count'],0)
        self.assertIsNone(combined['recovery_unrecovered_rate'])
        self.assertIsNone(combined['recovery_mean_steps'])


if __name__=='__main__':unittest.main(verbosity=2)
