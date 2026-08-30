import json
import math

import pytest
import torch

import causal_response_factorization_v1_validation_scorer as scorer
from causal_response_factorization_v1 import ResponseProgram
from test_causal_response_factorization_v1_validation_input import _input


ARMS = (1, 2)


def _program(source_groups, *, global_rank=1, private_rank=0, seed=0, shape=(2, 4, 4)):
    generator = torch.Generator().manual_seed(seed)
    p, s, t = shape
    groups = int(source_groups.max()) + 1

    def factor(rows, rank):
        value = torch.randn((rows, rank), dtype=torch.float64, generator=generator)
        return value.contiguous()

    private = []
    for group in range(groups):
        members = int((source_groups == group).sum())
        private.append((
            factor(p, private_rank), factor(members, private_rank), factor(t, private_rank),
        ))
    return ResponseProgram(
        factor(p, global_rank), factor(s, global_rank), factor(t, global_rank),
        private_phase=tuple(block[0] for block in private),
        private_source=tuple(block[1] for block in private),
        private_target=tuple(block[2] for block in private),
        source_groups=source_groups.clone(),
    )


def _candidate(program, *, global_rank, private_rank, seed, index, rms=0.5, documents=3):
    codes = torch.randn(
        (documents, program.code_dimension), dtype=torch.float64,
        generator=torch.Generator().manual_seed(100 + index),
    ).contiguous()
    return scorer.FrozenCandidate(
        global_rank=global_rank, private_rank_each_owner=private_rank, seed=seed,
        artifact=f"cell_{index}.pt", artifact_sha256=f"{index:x}" * 64,
        bytes=10 + index, persistent_values=program.persistent_values,
        per_document_values=program.code_dimension, training_response_rms=rms,
        program=program, training_codes=codes,
    )


def _freeze_record(candidate):
    return {
        "global_rank": candidate.global_rank,
        "private_rank_each_owner": candidate.private_rank_each_owner,
        "seed": candidate.seed, "artifact": candidate.artifact,
        "artifact_sha256": candidate.artifact_sha256, "bytes": candidate.bytes,
        "persistent_values": candidate.persistent_values,
        "per_document_values": candidate.per_document_values,
    }


def _library():
    _, validation = _input()
    groups = validation.source_groups
    specs = [(1, 0, 1), (1, 0, 2), (0, 1, 1)]
    candidates = [
        _candidate(
            _program(groups, global_rank=g, private_rank=q, seed=seed),
            global_rank=g, private_rank=q, seed=seed, index=index,
        )
        for index, (g, q, seed) in enumerate(specs)
    ]
    freeze = {"candidate_programs": [_freeze_record(c) for c in candidates]}
    return validation, candidates, freeze


def test_table_scores_every_candidate_and_panel_without_selecting():
    validation, candidates, freeze = _library()
    table = scorer.score_library(
        candidates, validation, freeze, arm_budgets=ARMS, require_production=False,
    )
    assert table["schema"] == scorer.TABLE_SCHEMA
    assert table["candidate_count"] == 3
    assert [row["seed"] for row in table["candidates"]] == [1, 2, 1]
    assert table["candidate_selected"] is False
    assert table["pareto_frontier_formed"] is False
    assert table["candidates_dropped_after_scoring"] == 0
    assert table["eval_values_read"] is False
    assert not any("winner" in key or "best" in key for key in table)
    for row in table["candidates"]:
        assert set(row["calibrated"]) == set(scorer.DESIGNS)
        for design in scorer.DESIGNS:
            assert set(row["calibrated"][design]) == {"1", "2"}
            for panel in row["calibrated"][design].values():
                assert panel["status"] in ("scored", "failed")
                assert panel["costs"]["physical_source_arms"] in ARMS
        assert row["unconditional"]["uses_validation_responses"] is False
        assert "pooled" in row["unconditional"] and "owner_pairs" in row["unconditional"]
        assert len(row["unconditional"]["slices"]["phase"]) == 2
    json.dumps(table, allow_nan=False)


def test_scored_panels_report_support_conditioning_and_owner_pairs():
    validation, candidates, freeze = _library()
    table = scorer.score_library(
        candidates, validation, freeze, arm_budgets=ARMS, require_production=False,
    )
    scored = [
        panel
        for row in table["candidates"]
        for design in row["calibrated"].values()
        for panel in design.values()
        if panel["status"] == "scored"
    ]
    assert scored, "synthetic library produced no scored panel"
    for panel in scored:
        assert 0.0 <= panel["supported_document_fraction"] <= 1.0
        assert panel["support_gate"] == scorer.SUPPORT_GATE
        assert panel["eligible_for_frontier"] == panel["support_gate_passes"]
        assert panel["anchor_cells"] == panel["costs"]["calibration_cells"]
        assert "worst_owner_pair_nrmse" in panel["calibrated"]
        assert len(panel["calibrated"]["owner_pairs"]) == len(validation.owner_components) ** 2
        assert set(panel["conditioning"]) == {
            "supported_documents", "smallest_singular_value_min",
            "smallest_singular_value_median", "valid_selected_cells_min",
        }
        assert panel["claim_boundary"]["calibrated_is_zero_shot_ood"] is False


def test_unconditional_arm_is_identical_across_designs_and_budgets():
    validation, candidates, freeze = _library()
    first = scorer.score_candidate(
        candidates[0], validation, training_rms=0.5, arm_budgets=(1,),
        designs=("sha256_outcome_blind_blocks",),
    )
    second = scorer.score_candidate(
        candidates[0], validation, training_rms=0.5, arm_budgets=(2,),
        designs=("training_only_block_d_optimal",),
    )
    assert first["unconditional"] == second["unconditional"]


def test_numerical_panel_failure_is_recorded_not_dropped():
    validation, candidates, freeze = _library()
    groups = validation.source_groups
    base = _program(groups, global_rank=2, private_rank=0, seed=7)
    duplicated = ResponseProgram(
        base.global_phase[:, [0, 0]].contiguous(), base.global_source[:, [0, 0]].contiguous(),
        base.global_target[:, [0, 0]].contiguous(),
        private_phase=base.private_phase, private_source=base.private_source,
        private_target=base.private_target, source_groups=base.source_groups,
    )
    candidate = _candidate(duplicated, global_rank=2, private_rank=0, seed=7, index=9)
    row = scorer.score_candidate(
        candidate, validation, training_rms=0.5, arm_budgets=ARMS,
    )
    d_optimal = row["calibrated"]["training_only_block_d_optimal"]
    assert all(panel["status"] == "failed" for panel in d_optimal.values())
    assert all("full column rank" in panel["error_message"] for panel in d_optimal.values())
    assert all(panel["eligible_for_frontier"] is False for panel in d_optimal.values())
    blind = row["calibrated"]["sha256_outcome_blind_blocks"]
    assert set(blind) == {"1", "2"}
    assert row["basis_rank"] == 1


def test_library_validation_rejects_freeze_mismatch_and_rms_disagreement():
    validation, candidates, freeze = _library()
    groups = validation.source_groups
    forged = dict(freeze)
    forged["candidate_programs"] = [dict(record) for record in freeze["candidate_programs"]]
    forged["candidate_programs"][1]["artifact_sha256"] = "e" * 64
    with pytest.raises(RuntimeError, match="differs from the freeze"):
        scorer.validate_candidate_library(
            candidates, forged, source_groups=groups, require_production=False,
        )
    mixed = list(candidates)
    mixed[2] = _candidate(
        candidates[2].program, global_rank=0, private_rank=1, seed=1, index=2, rms=0.25,
    )
    with pytest.raises(RuntimeError, match="training RMS currency"):
        scorer.validate_candidate_library(
            mixed, freeze, source_groups=groups, require_production=False,
        )
    with pytest.raises(RuntimeError, match="census differs"):
        scorer.validate_candidate_library(
            candidates[:2], freeze, source_groups=groups, require_production=False,
        )


def test_frozen_candidate_rejects_price_rank_and_code_mismatch():
    _, validation = _input()
    program = _program(validation.source_groups)
    good = _candidate(program, global_rank=1, private_rank=0, seed=1, index=0)
    with pytest.raises(ValueError, match="literal price"):
        scorer.FrozenCandidate(**{**good.__dict__, "persistent_values": 1})
    with pytest.raises(ValueError, match="registered ranks"):
        scorer.FrozenCandidate(**{**good.__dict__, "global_rank": 2})
    with pytest.raises(ValueError, match="training codes"):
        scorer.FrozenCandidate(**{
            **good.__dict__, "training_codes": good.training_codes[:, :0].contiguous(),
        })
    with pytest.raises(ValueError, match="training RMS"):
        scorer.FrozenCandidate(**{**good.__dict__, "training_response_rms": 0.0})


def test_production_guard_requires_registered_panels_and_role():
    validation, candidates, freeze = _library()
    with pytest.raises(RuntimeError, match="production validation panels or role"):
        scorer.score_library(candidates, validation, freeze, arm_budgets=ARMS)


def test_jsonable_replaces_non_finite_and_refuses_tensors():
    assert scorer._jsonable({"a": math.nan, "b": [1.0, math.inf], "c": True}) == {
        "a": None, "b": [1.0, None], "c": True,
    }
    with pytest.raises(TypeError):
        scorer._jsonable({"t": torch.zeros(1)})


def test_scorer_has_no_filesystem_model_or_eval_surface():
    assert not hasattr(scorer, "Path")
    assert not hasattr(scorer, "open")
    assert not hasattr(scorer, "load")
    assert not any("eval" in name.lower() for name in dir(scorer) if not name.startswith("_"))
