#!/usr/bin/env python3
import importlib
import os
from pathlib import Path
import subprocess
import sys
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
runner = importlib.import_module("run_temporal_iswas_three_head_union_joint_composition_v1")


class ThreeHeadUnionCompositionRunnerTests(unittest.TestCase):
    def test_authority_hashes(self):
        observed = {"prior": runner.sha(runner.PRIOR),
                    "parent_result": runner.sha(runner.PARENT_RESULT),
                    "parent_runner": runner.sha(runner.PARENT_RUNNER),
                    "joint": runner.sha(runner.JOINT), "accounting": runner.sha(runner.ACCOUNTING),
                    "producer": runner.sha(runner.PRODUCER)}
        self.assertEqual(observed, runner.EXPECTED)

    def test_masks_partition_exact_population(self):
        endpoints, _lookup = runner.parent.endpoint_bank(runner.parent.candidate.build_rows())
        for phase in ("FIT", "HOLDOUT"):
            self.assertEqual(int(runner.masks(endpoints, phase, "ALL").sum()), 64)
            for template in runner.parent.candidate.TEMPLATES:
                self.assertEqual(int(runner.masks(endpoints, phase, template).sum()), 32)

    def test_price_and_union_are_exact(self):
        self.assertEqual(runner.UNION, {9: (1,), 11: (3,), 15: (5,)})
        self.assertEqual(runner.PRICE, {"checkpoint_loads": 1, "model_forwards": 4,
            "sequence_evaluations": 512, "scored_token_positions": 1024,
            "transformer_backwards": 0, "model_updates": 0, "fit_parameters": 0})

    def test_dryrun_loads_no_model(self):
        env = dict(os.environ, BQLIB_DRYRUN="1", BQLIB_NO_MODEL="1")
        completed = subprocess.run([sys.executable, str(Path(runner.__file__))], env=env,
                                   text=True, capture_output=True, check=True)
        self.assertIn('"model_loaded": false', completed.stdout)
        self.assertIn('"model_forwards": 4', completed.stdout)


if __name__ == "__main__": unittest.main()
