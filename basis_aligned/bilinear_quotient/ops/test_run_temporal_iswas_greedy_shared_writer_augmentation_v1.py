#!/usr/bin/env python3
import importlib
import os
from pathlib import Path
import subprocess
import sys
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
runner = importlib.import_module("run_temporal_iswas_greedy_shared_writer_augmentation_v1")


class GreedySharedWriterRunnerTests(unittest.TestCase):
    def test_authority_hashes(self):
        observed = {"prior": runner.sha(runner.PRIOR),
                    "joint_result": runner.sha(runner.JOINT_RESULT), "atlas": runner.sha(runner.ATLAS),
                    "parent_runner": runner.sha(runner.PARENT_RUNNER),
                    "accounting": runner.sha(runner.ACCOUNTING), "producer": runner.sha(runner.PRODUCER)}
        self.assertEqual(observed, runner.EXPECTED)

    def test_arm_inventory(self):
        self.assertEqual(set(runner.ARMS), {"base_H3", "H3_plus_L8H1", "H3_plus_L9H4", "H3_plus_both"})
        self.assertEqual(runner.ARMS["H3_plus_both"], {8: (1,), 9: (1, 4), 11: (3,), 15: (5,)})

    def test_selection_is_smallest_then_best(self):
        reports = []
        for arm in runner.ARMS:
            for role, value in (("temporal", .85), ("iswas", .70)):
                reports.append({"role": role, "phase": "FIT", "template_id": "ALL", "arm": arm,
                    "signed_recovery": value + (.01 if arm == "H3_plus_L9H4" else 0),
                    "cosine": 1., "direction_agreement": 1., "non_target_to_target_gold_norm": 0.})
        self.assertEqual(runner.select_fit(reports), "H3_plus_L9H4")

    def test_dryrun(self):
        env = dict(os.environ, BQLIB_DRYRUN="1", BQLIB_NO_MODEL="1")
        completed = subprocess.run([sys.executable, str(Path(runner.__file__))], env=env,
                                   text=True, capture_output=True, check=True)
        self.assertIn('"model_loaded": false', completed.stdout)
        self.assertIn('"model_forwards": 9', completed.stdout)


if __name__ == "__main__": unittest.main()
