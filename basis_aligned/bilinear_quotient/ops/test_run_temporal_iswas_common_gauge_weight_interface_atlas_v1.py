#!/usr/bin/env python3
import hashlib
import importlib
import os
from pathlib import Path
import subprocess
import sys
import unittest

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
runner = importlib.import_module("run_temporal_iswas_common_gauge_weight_interface_atlas_v1")


class CommonGaugeWeightAtlasRunnerTests(unittest.TestCase):
    def test_authority_hashes(self):
        observed = {"prior": runner.sha(runner.PRIOR), "basis": runner.sha(runner.BASIS),
                    "helper": runner.sha(runner.HELPER), "producer": runner.sha(runner.PRODUCER)}
        self.assertEqual(observed, runner.EXPECTED)

    def test_restore_column_major_basis(self):
        basis = torch.linalg.qr(torch.arange(1, 25, dtype=torch.float32).reshape(6, 4)).Q[:, :2]
        stored_values = basis.T.contiguous().reshape(-1)
        stored = {"shape": [6, 2], "layout": "columns_then_rows_float32",
                  "values": stored_values.tolist(),
                  "sha256": hashlib.sha256(stored_values.numpy().tobytes()).hexdigest()}
        with self.assertRaises(ValueError):
            runner.restore_basis(torch, stored, "cpu")
        stored["shape"] = [1152, 2]
        expanded = torch.linalg.qr(torch.arange(1, 2305, dtype=torch.float32).reshape(1152, 2)).Q
        values = expanded.T.contiguous().reshape(-1)
        stored["values"] = values.tolist()
        stored["sha256"] = hashlib.sha256(values.numpy().tobytes()).hexdigest()
        self.assertTrue(torch.equal(runner.restore_basis(torch, stored, "cpu"), expanded))

    def test_exact_price(self):
        self.assertEqual(runner.PRICE["weight_interfaces_scored"], 1026)
        self.assertEqual(runner.PRICE["basis_weight_contractions"], 3078)
        self.assertEqual(runner.PRICE["model_forwards"], 0)

    def test_dryrun_loads_no_model(self):
        env = dict(os.environ, BQLIB_DRYRUN="1", BQLIB_NO_MODEL="1")
        completed = subprocess.run([sys.executable, str(Path(runner.__file__))], env=env,
                                   text=True, capture_output=True, check=True)
        self.assertIn('"model_loaded": false', completed.stdout)
        self.assertIn('"weight_interfaces_scored": 1026', completed.stdout)


if __name__ == "__main__": unittest.main()
