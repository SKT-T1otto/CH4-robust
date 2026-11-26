import argparse
from contextlib import redirect_stdout
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from chapter3_bser.learning.reference import ZeroResidualPolicy
from chapter3_bser.learning.evaluate import preserved_rng
from chapter4_interval_robust.nominal_maddpg.config import load, read_json
from chapter4_interval_robust.nominal_maddpg.policy import FrozenEvaluationPolicy
from ch4_nominal_baseline_v2 import amplitude as a


class AmplitudeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.learner = a.SingleCriticLearner(load(), 79)

    def test_frozen_proposals_on_all_four_hulls(self):
        obs = np.arange(112, dtype=np.float32).reshape(4, 28)/50
        base = FrozenEvaluationPolicy(self.learner)
        actions = torch.cat(base.step(obs))
        self.assertTrue((actions.abs().sum(dim=1) > 0).all())
        for scale in a.SCALES:
            policy = a.ScaledPolicy(self.learner, scale)
            self.assertTrue(torch.equal(torch.cat(policy.step(obs)), actions*scale))
            self.assertEqual(policy.last_input_sha256, base.last_input_sha256)
            self.assertEqual(policy.verify_frozen(), base.verify_frozen())
            with self.assertRaises(ValueError):
                policy.step(obs, explore=True)

    def test_native_controls_and_half_scale_gate(self):
        catalog = a.SceneCatalog()
        scene = catalog.scene("validation", 3)

        @preserved_rng()
        def prefix(policy):
            mode = "evaluation" if isinstance(policy, FrozenEvaluationPolicy) else "regression"
            with a.NominalRuntime(catalog.config, scene, policy, a.NominalReplay(400, 9100003),
                                  episode_id=3, mode=mode, catalog=catalog) as runtime:
                a.install_executor_navigation(runtime.rt)
                transitions = [runtime.advance() for _ in range(3)]
                return runtime.world.prefix, transitions

        zero, _ = prefix(ZeroResidualPolicy())
        one, _ = prefix(FrozenEvaluationPolicy(self.learner))
        self.assertEqual(prefix(a.ScaledPolicy(self.learner, 0))[0], zero)
        self.assertEqual(prefix(a.ScaledPolicy(self.learner, 1))[0], one)
        _, steps = prefix(a.ScaledPolicy(self.learner, 0.5))
        reference = FrozenEvaluationPolicy(self.learner)
        used = 0
        for row in steps:
            actions = torch.cat(reference.step(row["observations"])).numpy()*0.5
            expected = actions*np.asarray(row["action_enabled"])[:, None]
            np.testing.assert_array_equal(row["actions"], expected)
            used += np.count_nonzero(expected)
        self.assertGreater(used, 0)

    def test_partial_resume_and_paired_report(self):
        catalog = a.SceneCatalog()
        def result(catalog, scene, index, scale):
            success = index == 1 or (index == 0 and scale in (0, 0.5)) or (index == 2 and scale > 0)
            cause = "success" if success else ("obstacle_collision" if scale == 1 else "timeout")
            return dict(index=index, scale=scale, source=catalog.origin(scene, index, allowed_groups=("validation",)).as_dict(),
                        success=success, discovery=success, handoff=success, termination=cause,
                        cost=100 if success else 500, steps=100 if cause != "timeout" else 400,
                        open_components=3, residual_abs_sum=3*scale)

        with TemporaryDirectory() as temp, patch.object(a, "OUTPUT_ROOT", Path(temp)), \
                patch.object(a, "SceneCatalog", return_value=catalog), patch.object(a, "build_identity", return_value={}), \
                patch.object(a, "load_policies", return_value={s:s for s in a.SCALES}), \
                patch.object(a, "run_scene", side_effect=result) as run_scene, redirect_stdout(io.StringIO()):
            args = argparse.Namespace(check=False, checkpoint=Path(temp), output=Path(temp)/"trial", indices=[0, 1], scales=[0.5])
            a.run(args)
            self.assertEqual(run_scene.call_count, 2)
            self.assertFalse((args.output/"summary.json").exists())
            args.indices, args.scales = a.INDICES, a.SCALES
            a.run(args)
            a.run(args)
            self.assertEqual(run_scene.call_count, 96)
            report = read_json(args.output/"summary.json")
            self.assertEqual(report["by_scale"]["0.5"]["successes"], 3)
            self.assertEqual(report["by_scale"]["0.5"]["collisions"], 0)
            self.assertEqual(report["full_scale_gains_kept_at_half"], [2])
            self.assertEqual(report["full_scale_losses_recovered_at_half"], [0])
            corrupt = args.output/"scene_000_scale_0.5.json"
            corrupt.write_text(corrupt.read_text().replace('"cost": 100', '"cost": 99'), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "damaged scene receipt"):
                a.run(args)

    def test_preview_and_checkpoint_rejection(self):
        with redirect_stdout(io.StringIO()), patch.object(a, "run", side_effect=AssertionError):
            a.main([])
        with TemporaryDirectory() as temp:
            path = Path(temp)
            with self.assertRaisesRegex(ValueError, "Pass --checkpoint"):
                a.load_policies(path, None, {})
            a.publish(path/"committed.json", {"identity": {}, "completed_episodes": 4})
            (path/"state.pt").touch()
            with self.assertRaisesRegex(ValueError, "episode-16"):
                a.load_policies(path, None, {})


if __name__ == "__main__":
    unittest.main()
