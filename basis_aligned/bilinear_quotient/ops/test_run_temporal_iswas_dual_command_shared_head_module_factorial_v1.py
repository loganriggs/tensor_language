#!/usr/bin/env python3
import importlib
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
runner = importlib.import_module("run_temporal_iswas_dual_command_shared_head_module_factorial_v1")


class DualCommandSharedHeadRunnerTests(unittest.TestCase):
    def test_pair_indices_are_role_specific_involutions(self):
        endpoints, lookup = runner.endpoint_bank(runner.candidate.build_rows())
        self.assertEqual(len(endpoints), 128)
        for role in ("temporal", "iswas"):
            pairs = runner.pair_indices(endpoints, lookup, role)
            self.assertEqual(sorted(pairs), list(range(128)))
            self.assertTrue(all(pairs[pairs[index]] == index for index in range(128)))
            self.assertTrue(all(endpoints[index][2][f"{role}_position"]
                                == endpoints[pair][2][f"{role}_position"]
                                for index, pair in enumerate(pairs)))

    def test_price_is_literal(self):
        self.assertEqual(runner.PRICE, {"checkpoint_loads": 1, "model_forwards": 15,
            "sequence_evaluations": 1920, "scored_token_positions": 3840,
            "transformer_backwards": 0, "model_updates": 0, "fit_parameters": 0})

    def test_static_authorities_except_pending_capability(self):
        for name, path in (("prior", runner.PRIOR), ("atlas", runner.ATLAS),
                           ("builder", runner.BUILDER), ("accounting", runner.ACCOUNTING),
                           ("producer", runner.PRODUCER)):
            self.assertEqual(runner.sha(path), runner.EXPECTED[name])
        self.assertEqual(runner.EXPECTED["capability"], "PENDING_CAPABILITY_RESULT_SHA256")


if __name__ == "__main__": unittest.main()
