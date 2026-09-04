"""CPU-only contract tests for Task 14 projector Program A."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch


OPS = Path(__file__).parent
for name in ("task14_causal_spectral_rank_one", "task14_head11_3_projector_adapter"):
    spec = importlib.util.spec_from_file_location(name, OPS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
spec = importlib.util.spec_from_file_location(
    "run_task14_head11_3_projector_discovery",
    OPS / "run_task14_head11_3_projector_discovery.py",
)
PROGRAM = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = PROGRAM
spec.loader.exec_module(PROGRAM)


def _health() -> PROGRAM.FitHealth:
    return PROGRAM.FitHealth(
        True, True, True, True, True, True, 0.0, 0.1, 1.0, 0.8, 100
    )


class FakeBackend:
    def __init__(
        self, *, leak: bool = False, wrong_ordinals: bool = False,
        missing_cell: bool = False, unhealthy_permutations: bool = False,
        permutation_passes: bool = False,
    ):
        self.leak = leak
        self.wrong_ordinals = wrong_ordinals
        self.missing_cell = missing_cell
        self.unhealthy_permutations = unhealthy_permutations
        self.permutation_passes = permutation_passes

    def collect_spectral_inputs(self, relations):
        n = len(relations)
        deltas = torch.zeros(n, 128, dtype=torch.float64)
        deltas[:, 0] = 1
        return PROGRAM.SpectralInputs(
            tuple(x.ordinal for x in relations), tuple(x.cell_key for x in relations),
            deltas, deltas.clone(), torch.ones(n, dtype=torch.float64),
            validation_records_seen=int(self.leak), model_counts={
                "forward_calls": 0, "backward_calls": 0,
                "example_evaluations": 0,
            },
        )

    def _result(self, relations, frame, rank, start, *, passing, healthy=True):
        ordinals = tuple(x.ordinal for x in relations)
        if self.wrong_ordinals:
            ordinals = ordinals[:-1]
        target_cells = {
            row.cell_key: PROGRAM.TargetCellScore(
                0.9 if passing else 0.1, 0.8, 0.5,
                row.cell_key.startswith("C_to_ordinary_singular|C|"),
            ) for row in relations if row.role == "target"
        }
        control_cells = {
            row.cell_key: PROGRAM.ControlCellScore(0.05, 0.05, 0.05)
            for row in relations if row.role == "control"
        }
        if self.missing_cell:
            target_cells.pop(next(iter(target_cells)))
        health = _health() if healthy else PROGRAM.FitHealth(
            False, True, True, True, True, True, 0.0, 0.1, 1.0, 0.8, 100
        )
        return PROGRAM.FitResult(
            rank, start, frame, health, target_cells, control_cells,
            tuple(0.5 for row in relations if row.role == "target"),
            {"forward_calls": 0, "backward_calls": 0, "example_evaluations": 0},
            validation_token_sequences_seen=int(self.leak), scored_ordinals=ordinals,
            normalized_row_effect_ordinals=tuple(
                row.ordinal for row in relations if row.role == "target"
            ),
        )

    def fit_and_score(self, *, select_relations, rank, start, initial_frame,
                      permutation_id, **unused):
        return self._result(
            select_relations, initial_frame, rank, start,
            passing=permutation_id is None or self.permutation_passes,
            healthy=not (self.unhealthy_permutations and permutation_id is not None),
        )

    def score_fixed_frame(self, *, select_relations, frame, control_id):
        return self._result(select_relations, frame, frame.shape[1], 0, passing=False)


def test_exact_discovery_partition_and_dryrun_contract() -> None:
    plan = PROGRAM.compile_discovery_plan()
    assert len(plan.fit) == 153 and len(plan.select) == 145
    assert max(x.ordinal for x in plan.fit + plan.select) < 544
    fit_endpoints = {v for x in plan.fit for v in (x.target_endpoint_id, x.donor_endpoint_id)}
    select_endpoints = {v for x in plan.select for v in (x.target_endpoint_id, x.donor_endpoint_id)}
    assert fit_endpoints.isdisjoint(select_endpoints)
    dryrun = PROGRAM.compile_dryrun()
    assert dryrun["authority_parsed"] is False
    assert dryrun["validation_rows_loaded"] == 0
    assert dryrun["fit_objective_constants"] == PROGRAM.asdict(PROGRAM.FIT_OBJECTIVE)
    assert dryrun["fit_objective_constants_blocking"] is False
    assert dryrun["primary_price"] == {
        "forward_calls": 1206,
        "backward_calls": 902,
        "example_evaluations": 37700,
        "stored_frame_bytes": 141824,
    }
    assert dryrun["conditional_price"] == {
        "forward_calls": 420,
        "backward_calls": 400,
        "example_evaluations": 13380,
    }


def test_equal_cell_operator_and_top_algebraic_frames() -> None:
    # The 3:1 row imbalance favors axis 0 under a flat mean, while equal-cell
    # weighting gives the stronger singleton cell on axis 1 the leading value.
    d = torch.zeros(4, 128, dtype=torch.float64)
    d[:3, 0], d[3, 1] = 1.0, 2.0
    inputs = PROGRAM.SpectralInputs(
        (0, 1, 2, 3), ("many", "many", "many", "one"), d, d.clone(),
        torch.ones(4, dtype=torch.float64),
    )
    operator = PROGRAM._equal_cell_spectral_operator(inputs)
    assert operator[1, 1] > operator[0, 0]
    for rank in (1, 2, 4):
        frame = PROGRAM.top_algebraic_frame(operator, rank)
        assert frame.shape == (128, rank)
        assert torch.allclose(frame.T @ frame, torch.eye(rank, dtype=torch.float64))
    assert abs(PROGRAM.top_algebraic_frame(operator, 1)[1, 0]) == pytest.approx(1.0)


def test_fake_lifecycle_is_create_only_and_records_selection(tmp_path) -> None:
    receipt_path, bundle_path = tmp_path / "receipt.json", tmp_path / "bundle.pt"
    receipt = PROGRAM.execute_program_a(
        FakeBackend(), receipt_path=receipt_path, bundle_path=bundle_path
    )
    assert receipt["selected_rank"] == 1
    assert receipt["program_b_opened"] is False
    assert receipt["validation_rows_loaded"] == 0
    assert receipt["analytic_operator_sha256"]
    assert receipt["terminal"] == "program_a_selected"
    assert len(receipt["projector_overlap_pairs"]) == 10
    with pytest.raises(FileExistsError):
        PROGRAM.execute_program_a(
            FakeBackend(), receipt_path=receipt_path, bundle_path=bundle_path
        )


@pytest.mark.parametrize("backend", [FakeBackend(leak=True), FakeBackend(wrong_ordinals=True)])
def test_backend_access_or_select_mismatch_fails_closed(tmp_path, backend) -> None:
    with pytest.raises(PROGRAM.ProgramAError):
        PROGRAM.execute_program_a(
            backend, receipt_path=tmp_path / "receipt.json", bundle_path=tmp_path / "bundle.pt"
        )


def test_missing_select_cell_fails_closed(tmp_path) -> None:
    with pytest.raises(PROGRAM.ProgramAError, match="omits or invents"):
        PROGRAM.execute_program_a(
            FakeBackend(missing_cell=True), receipt_path=tmp_path / "receipt.json",
            bundle_path=tmp_path / "bundle.pt",
        )


def test_unhealthy_permutation_is_instrument_invalid_not_scientific_null(tmp_path) -> None:
    receipt = PROGRAM.execute_program_a(
        FakeBackend(unhealthy_permutations=True),
        receipt_path=tmp_path / "receipt.json", bundle_path=tmp_path / "bundle.pt",
    )
    assert receipt["terminal"] == "instrument_invalid"
    assert "permutation_fit_health_failed" in receipt["instrument_invalid_reasons"]


def test_passing_permutation_is_nonidentification_not_small_subspace_null(tmp_path) -> None:
    receipt = PROGRAM.execute_program_a(
        FakeBackend(permutation_passes=True),
        receipt_path=tmp_path / "receipt.json", bundle_path=tmp_path / "bundle.pt",
    )
    assert receipt["terminal"] == "program_a_not_identified"
    assert receipt["pred_c_small_subspace_null"] is False
    assert receipt["nonidentification_reasons"] == ["permutation_control_not_rejected"]


def test_cli_managed_environment_is_model_free_and_rejects_unknown_args(
    monkeypatch, capsys,
) -> None:
    monkeypatch.setenv("BQLIB_DRYRUN", "1")
    assert PROGRAM.main([]) == 0
    assert json.loads(capsys.readouterr().out)["model_loaded"] is False
    with pytest.raises(SystemExit):
        PROGRAM.main(["--unknown"])
