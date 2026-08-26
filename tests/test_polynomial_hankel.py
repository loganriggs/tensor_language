import importlib.util
from pathlib import Path
import unittest

import torch


ROOT = Path(__file__).parents[1]
PATH = ROOT / "basis_aligned" / "polynomial_causal" / "hankel.py"
SPEC = importlib.util.spec_from_file_location("polynomial_causal_hankel", PATH)
HANKEL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HANKEL)


class HankelTests(unittest.TestCase):
    def test_low_rank_completion_beats_additive_baseline(self):
        gen = torch.Generator().manual_seed(4)
        row = torch.randn(24, 3, generator=gen)
        col = torch.randn(20, 3, generator=gen)
        matrix = 2 + row @ col.T
        observed = torch.rand(24, 20, generator=gen) > 0.2
        baseline = HANKEL.row_column_baseline(matrix, observed)
        completed = HANKEL.complete_low_rank(matrix, observed, rank=3)
        self.assertLess(HANKEL.heldout_rmse(completed, matrix, observed),
                        0.3 * HANKEL.heldout_rmse(baseline, matrix, observed))

    def test_spectrum_recovers_small_rank(self):
        gen = torch.Generator().manual_seed(8)
        matrix = torch.randn(30, 2, generator=gen) @ torch.randn(
            2, 25, generator=gen)
        report = HANKEL.spectrum(matrix)
        self.assertLessEqual(report["rank95"], 2)


if __name__ == "__main__":
    unittest.main()
