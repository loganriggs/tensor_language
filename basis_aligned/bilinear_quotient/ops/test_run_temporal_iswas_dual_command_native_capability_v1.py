#!/usr/bin/env python3
import importlib
import os
from pathlib import Path
import subprocess
import sys
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
runner = importlib.import_module("run_temporal_iswas_dual_command_native_capability_v1")


class DualCommandCapabilityRunnerTests(unittest.TestCase):
    def _records(self, failures=()):
        records = []
        failed = set(failures)
        for phase in ("FIT", "HOLDOUT"):
            for template in runner.authority.TEMPLATES:
                for cell in runner.authority.CELLS:
                    for role in ("temporal", "iswas"):
                        for group in range(8):
                            key = (phase, template, cell, role, group)
                            records.append({"phase": phase, "template_id": template,
                                            "cell": cell, "role": role,
                                            "correct": key not in failed})
        return records

    def test_exact_price(self):
        self.assertEqual(runner.PRICE, {
            "model_forwards": 1, "sequence_evaluations": 128,
            "scored_token_positions": 256, "transformer_backwards": 0,
            "model_updates": 0, "fit_parameters": 0})

    def test_aggregation_requires_eight_and_six_correct(self):
        records = self._records()
        self.assertEqual(len(runner.capability_cells(records)), 32)
        self.assertTrue(all(cell["passed"] for cell in runner.capability_cells(records)))
        failures = [("HOLDOUT", "forecast_while", "11", "iswas", group)
                    for group in range(3)]
        cells = runner.capability_cells(self._records(failures))
        target = next(cell for cell in cells if cell["phase"] == "HOLDOUT"
                      and cell["template_id"] == "forecast_while"
                      and cell["cell"] == "11" and cell["role"] == "iswas")
        self.assertEqual(target["correct_count"], 5)
        self.assertFalse(target["passed"])

    def test_authority_hashes(self):
        observed = {"prior": runner.sha(runner.PRIOR), "builder": runner.sha(runner.BUILDER),
                    "audit": runner.sha(runner.AUDIT), "producer": runner.sha(runner.PRODUCER)}
        self.assertEqual(observed, runner.EXPECTED)

    def test_dryrun_loads_no_model(self):
        env = dict(os.environ, BQLIB_DRYRUN="1", BQLIB_NO_MODEL="1")
        completed = subprocess.run([sys.executable, str(Path(runner.__file__))], env=env,
                                   text=True, capture_output=True, check=True)
        self.assertIn('"model_loaded": false', completed.stdout)
        self.assertIn('"sequences": 128', completed.stdout)


if __name__ == "__main__":
    unittest.main()
