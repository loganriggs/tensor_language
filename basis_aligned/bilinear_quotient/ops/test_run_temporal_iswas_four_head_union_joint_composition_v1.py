#!/usr/bin/env python3
import importlib
import os
from pathlib import Path
import subprocess
import sys
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
runner = importlib.import_module("run_temporal_iswas_four_head_union_joint_composition_v1")


class FourHeadJointRunnerTests(unittest.TestCase):
    def test_hashes(self):
        observed = {"prior": runner.sha(runner.PRIOR), "augmentation": runner.sha(runner.AUGMENTATION),
                    "augmentation_runner": runner.sha(runner.AUGMENTATION_RUNNER),
                    "base_runner": runner.sha(runner.BASE_RUNNER), "producer": runner.sha(runner.PRODUCER)}
        self.assertEqual(observed, runner.EXPECTED)

    def test_union_and_price(self):
        self.assertEqual(runner.UNION, {9: (1, 4), 11: (3,), 15: (5,)})
        self.assertEqual(runner.PRICE["model_forwards"], 4)
        self.assertEqual(runner.PRICE["sequence_evaluations"], 512)

    def test_dryrun(self):
        env = dict(os.environ, BQLIB_DRYRUN="1", BQLIB_NO_MODEL="1")
        completed = subprocess.run([sys.executable, str(Path(runner.__file__))], env=env,
                                   text=True, capture_output=True, check=True)
        self.assertIn('"model_loaded": false', completed.stdout)
        self.assertIn('"9": [1, 4]', completed.stdout)


if __name__ == "__main__": unittest.main()
