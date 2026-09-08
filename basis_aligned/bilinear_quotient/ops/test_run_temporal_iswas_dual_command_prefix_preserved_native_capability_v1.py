#!/usr/bin/env python3
import importlib
import os
from pathlib import Path
import subprocess
import sys
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
runner = importlib.import_module("run_temporal_iswas_dual_command_prefix_preserved_native_capability_v1")


class PrefixPreservedCapabilityRunnerTests(unittest.TestCase):
    def test_wrapper_hashes(self):
        observed = {"prior": runner.sha(runner.PRIOR), "builder": runner.sha(runner.BUILDER),
                    "predecessor": runner.sha(runner.PREDECESSOR),
                    "base_runner": runner.sha(runner.BASE_RUNNER)}
        self.assertEqual(observed, runner.EXPECTED_WRAPPER)

    def test_population_prefix_identity(self):
        rows = runner.candidate.build_rows()
        self.assertEqual(runner.candidate.validate_rows(rows),
                         runner.candidate.EXPECTED_AUTHORITY_SHA256)
        self.assertTrue(all(endpoint["temporal_prefix_ids"]
                            == endpoint["ids"][:endpoint["temporal_position"]]
                            for row in rows for endpoint in row["endpoints"].values()))

    def test_dryrun_loads_no_model(self):
        env = dict(os.environ, BQLIB_DRYRUN="1", BQLIB_NO_MODEL="1")
        completed = subprocess.run([sys.executable, str(Path(runner.__file__))], env=env,
                                   text=True, capture_output=True, check=True)
        self.assertIn('"model_loaded": false', completed.stdout)
        self.assertIn('"sequences": 128', completed.stdout)


if __name__ == "__main__": unittest.main()
