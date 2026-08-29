from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest
import torch

import collect_mlp2_cmr_v1_fit_mean as fit


def test_runner_imports_from_outside_repository() -> None:
    script = Path(fit.__file__).resolve()
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import runpy; "
                f"runpy.run_path({str(script)!r}, run_name='fit_import_smoke')"
            ),
        ],
        cwd="/tmp",
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_moment_finalization_matches_direct_population_statistics() -> None:
    values = torch.tensor([[1.0, 2.0], [3.0, -2.0], [5.0, 4.0]], dtype=torch.float64)
    mean, variance, second = fit.finalize_moments(
        values.sum(0), values.square().sum(0), len(values),
    )
    assert torch.allclose(mean, values.mean(0))
    assert torch.allclose(second, values.square().mean(0))
    assert torch.allclose(variance, values.var(0, unbiased=False))


def test_top_selection_is_descending_stable_and_jaccard_exact() -> None:
    score = torch.tensor([1.0, 3.0, 3.0, 2.0])
    selected = fit.select_top(score, 3)
    assert selected.tolist() == [1, 2, 3]
    assert fit.support_jaccard(selected, selected.flip(0)) == 1.0
    assert fit.support_jaccard(selected, torch.tensor([0, 1, 2])) == 0.5


def test_malformed_moments_and_scores_fail_closed() -> None:
    with pytest.raises(ValueError, match="moments"):
        fit.finalize_moments(torch.ones(2), torch.ones(3), 2)
    with pytest.raises(ValueError, match="score"):
        fit.select_top(torch.tensor([1.0, float("nan")]), 1)
