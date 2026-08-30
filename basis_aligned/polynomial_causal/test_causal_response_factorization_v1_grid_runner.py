from pathlib import Path

import torch

from causal_response_factorization_v1_fit_adapter import (
    FitArtifactBinding,
    FitTrainingInput,
)
from causal_response_factorization_v1_grid_runner import run_grid


def _sha(character: str) -> str:
    return character * 64


def _tiny_input() -> FitTrainingInput:
    generator = torch.Generator().manual_seed(7)
    response = torch.randn((2, 4, 3, 7), generator=generator, dtype=torch.float64)
    valid = torch.ones_like(response, dtype=torch.bool)
    return FitTrainingInput(
        response=response.contiguous(), valid=valid.contiguous(),
        document_ids=torch.arange(7, dtype=torch.int64),
        original_document_indices=torch.arange(7, dtype=torch.int64),
        source_groups=torch.tensor([0, 0, 1, 1], dtype=torch.int64),
        phases=("off", "on"), source_tags=("a", "b", "c", "d"),
        target_tags=("a", "b", "c", "d")[:3],
        source_components=("x", "x", "y", "y"), owner_components=("x", "y"),
        artifacts=FitArtifactBinding(
            parent_binding_sha256=_sha("0"), receipt_sha256=_sha("1"),
            terminal_sha256=_sha("1"), authority_artifact_sha256=_sha("2"),
            authority_logical_sha256=_sha("3"), bundle_sha256=_sha("4"),
            manifest_artifact_sha256=_sha("5"), manifest_logical_sha256=_sha("6"),
            source_closure_sha256=_sha("7"),
        ),
    )


def test_tiny_grid_publishes_and_resumes_exactly(tmp_path: Path) -> None:
    training = _tiny_input()
    kwargs = dict(
        rank_pairs=((1, 0), (0, 1), (1, 1)), seeds=(11, 12), steps=60,
        learning_rate=0.03, optimizer_device="cpu", require_published_source=False,
    )
    first = run_grid(training, tmp_path / "grid", **kwargs)
    assert first["expected_cells"] == 6
    assert first["result_cells"] == 6
    assert first["failure_cells"] == 0
    assert len(list((tmp_path / "grid").glob("*.pt"))) == 6
    before = (tmp_path / "grid" / "terminal.json").read_bytes()
    second = run_grid(training, tmp_path / "grid", **kwargs)
    assert second == first
    assert (tmp_path / "grid" / "terminal.json").read_bytes() == before
    assert first["validation_values_read"] is False
    assert first["eval_values_read"] is False


def test_failure_is_a_preserved_terminal_cell(tmp_path: Path) -> None:
    def failing(*args, **kwargs):
        raise ArithmeticError("planted failure")

    terminal = run_grid(
        _tiny_input(), tmp_path / "failed", rank_pairs=((1, 0),), seeds=(13,),
        steps=1, learning_rate=0.03, optimizer_device="cpu",
        require_published_source=False, fitter=failing,
    )
    assert terminal["result_cells"] == 0
    assert terminal["failure_cells"] == 1
    assert terminal["cells"][0]["error_type"] == "ArithmeticError"
    assert "planted failure" in terminal["cells"][0]["error_message"]
