from pathlib import Path
import json

import pytest
import torch

from causal_response_factorization_v1_fit_adapter import (
    FitArtifactBinding,
    FitTrainingInput,
)
from causal_response_factorization_v1_grid_runner import (
    _validate_failure_cell,
    run_grid,
)


def _sha(character: str) -> str:
    return character * 64


def _tiny_input() -> FitTrainingInput:
    generator = torch.Generator().manual_seed(7)
    response = torch.randn((2, 4, 4, 7), generator=generator, dtype=torch.float64)
    valid = torch.ones_like(response, dtype=torch.bool)
    return FitTrainingInput(
        response=response.contiguous(), valid=valid.contiguous(),
        document_ids=torch.arange(7, dtype=torch.int64),
        original_document_indices=torch.arange(7, dtype=torch.int64),
        source_groups=torch.tensor([0, 0, 1, 1], dtype=torch.int64),
        phases=("off", "on"), source_tags=("a", "b", "c", "d"),
        target_tags=("a", "b", "c", "d"),
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


def test_resume_semantically_replays_factor_tensors(tmp_path: Path) -> None:
    kwargs = dict(
        rank_pairs=((1, 0),), seeds=(21,), steps=20, learning_rate=0.03,
        optimizer_device="cpu", require_published_source=False,
    )
    output = tmp_path / "tampered"
    run_grid(_tiny_input(), output, **kwargs)
    cell = next(output.glob("*.pt"))
    payload = torch.load(cell, map_location="cpu", weights_only=True)
    payload["document_codes"][0, 0] += 1.0
    torch.save(payload, cell)
    with pytest.raises(RuntimeError, match="semantic replay changed"):
        run_grid(_tiny_input(), output, **kwargs)


def test_published_surface_cannot_accept_caller_protocol(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="only through main"):
        run_grid(
            _tiny_input(), tmp_path / "not-production", rank_pairs=((1, 0),),
            seeds=(1,), steps=1, learning_rate=0.03, optimizer_device="cpu",
            require_published_source=True,
        )


def test_registered_failure_validator_rejects_forged_cause(tmp_path: Path) -> None:
    path = tmp_path / "forged.failure.json"
    path.write_text(json.dumps({
        "schema": "causal_response_factorization_v1_grid_failure",
        "status": "failed_training_only", "source_closure_sha256": _sha("a"),
        "input_binding_sha256": _sha("b"), "global_rank": 1,
        "private_rank_each_owner": 0, "seed": 9, "steps": 20,
        "learning_rate": 0.03, "optimizer_device": "cuda", "elapsed_seconds": 1.0,
        "error_type": "IntegrityError", "error_message": "not numerical",
        "validation_values_read": False, "eval_values_read": False,
    }))
    with pytest.raises(RuntimeError, match="not a registered numerical outcome"):
        _validate_failure_cell(
            path, source_sha256=_sha("a"), input_sha256=_sha("b"), global_rank=1,
            private_rank=0, seed=9, steps=20, learning_rate=0.03,
            optimizer_device="cuda", registered_only=True,
        )


def test_resume_recomputes_seeded_initial_health(tmp_path: Path) -> None:
    kwargs = dict(
        rank_pairs=((1, 0),), seeds=(31,), steps=20, learning_rate=0.03,
        optimizer_device="cpu", require_published_source=False,
    )
    output = tmp_path / "initial"
    run_grid(_tiny_input(), output, **kwargs)
    (output / "terminal.json").unlink()
    cell = next(output.glob("*.pt"))
    payload = torch.load(cell, map_location="cpu", weights_only=True)
    forged_initial = payload["receipt"]["initial_mse"] * 2
    forged_improvement = (
        (forged_initial - payload["receipt"]["final_mse"]) / forged_initial
    )
    payload["receipt"]["initial_mse"] = forged_initial
    payload["receipt"]["improvement_fraction"] = forged_improvement
    payload["receipt"]["healthy"] = True
    payload["metrics"]["initial_mse"] = forged_initial
    payload["metrics"]["improvement_fraction"] = forged_improvement
    torch.save(payload, cell)
    with pytest.raises(RuntimeError, match="initial_mse"):
        run_grid(_tiny_input(), output, **kwargs)


def test_resume_rejects_prediction_preserving_noncanonical_gauge(tmp_path: Path) -> None:
    kwargs = dict(
        rank_pairs=((1, 0),), seeds=(41,), steps=20, learning_rate=0.03,
        optimizer_device="cpu", require_published_source=False,
    )
    output = tmp_path / "gauge"
    run_grid(_tiny_input(), output, **kwargs)
    (output / "terminal.json").unlink()
    cell = next(output.glob("*.pt"))
    payload = torch.load(cell, map_location="cpu", weights_only=True)
    payload["program"]["global_phase"][:, 0] *= 2
    payload["document_codes"][:, 0] /= 2
    torch.save(payload, cell)
    with pytest.raises(RuntimeError, match="canonical_gauge"):
        run_grid(_tiny_input(), output, **kwargs)
