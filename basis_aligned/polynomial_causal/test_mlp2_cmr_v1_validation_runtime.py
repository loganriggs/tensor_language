from __future__ import annotations

import copy

import pytest
import torch

import mlp2_cmr_v1_validation_runtime as runtime
import mlp2_cmr_v1_validation_statistics as statistics


def _cell(document: int, cell: int, arm: int) -> statistics.CellSums:
    count = 2
    native = 4.0 + document * 0.01
    candidate = native + arm * 1e-5
    return statistics.CellSums(
        count=count,
        native_nll_sum=native,
        candidate_nll_sum=candidate,
        teacher_kl_sum=(arm + 1) * count * 1e-5,
        centered_logit_sse=(arm + 1) * 1e-5,
        native_centered_logit_energy=10.0,
        raw_logit_sse=(arm + 1) * 1e-5,
        native_correct_count=1,
        candidate_correct_count=1,
        native_top1_agreement_count=2,
        support_sha256=f"{document:04x}{cell:04x}".ljust(64, "0"),
    )


def ledger_fixture():
    return {
        arm: {
            document: {
                cell: _cell(document, cell_index, arm_index)
                for cell_index, cell in enumerate(statistics.CELL_NAMES)
            } for document in range(runtime.DOCUMENTS)
        } for arm_index, arm in enumerate(runtime.ALL_ARMS)
    }


def test_pack_unpack_ledgers_is_lossless_and_support_shared() -> None:
    ledgers = ledger_fixture()
    packed = runtime.pack_ledgers(ledgers)
    replay = runtime.unpack_ledgers(packed)
    assert replay == ledgers
    changed = copy.deepcopy(ledgers)
    value = changed["ZERO"][0]["all_scored"]
    changed["ZERO"][0]["all_scored"] = statistics.CellSums(
        **{**value.__dict__, "support_sha256": "f" * 64}
    )
    with pytest.raises(ValueError, match="support hashes"):
        runtime.pack_ledgers(changed)


def test_call_ledger_requires_exact_arm_site_and_physical_counts() -> None:
    ledger = runtime.new_call_ledger()
    for arm in runtime.ALL_ARMS:
        ledger[arm]["forward_calls"] = runtime.CALLS
        ledger[arm]["forward_returns"] = runtime.CALLS
        ledger[arm]["attention_calls_by_site"] = [runtime.CALLS] * 18
        ledger[arm]["native_mlp_calls_by_site"] = [runtime.CALLS] * 18
        if arm == "ZERO" or arm in runtime.PHYSICAL_ARMS:
            ledger[arm]["native_mlp_calls_by_site"][runtime.SITE] = 0
        ledger[arm]["physical_mlp2_calls"] = (
            runtime.CALLS if arm in runtime.PHYSICAL_ARMS or arm in runtime.SIGNED_T else 0
        )
        ledger[arm]["zero_mlp2_calls"] = runtime.CALLS if arm == "ZERO" else 0
    assert runtime.call_ledger_passes(ledger)
    ledger["SUFFIX"]["native_mlp_calls_by_site"][2] = 1
    assert not runtime.call_ledger_passes(ledger)


def test_geometry_pack_unpack_is_lossless() -> None:
    pair = statistics.PairSums(1.0, 2.0, 3.0)
    batches = [{
        cell: {name: pair for name in statistics.GEOMETRY_PAIRS}
        for cell in statistics.CELL_NAMES
    } for _ in range(runtime.CALLS)]
    replay = runtime.unpack_geometry(runtime.pack_geometry(batches))
    assert replay == batches


def test_validate_supports_rejects_duplicates_and_hashes_exact() -> None:
    supports = {
        arm: torch.arange(512, dtype=torch.long) + index * 512
        for index, arm in enumerate(runtime.PHYSICAL_ARMS)
    }
    hashes = runtime.validate_supports(supports)
    assert set(hashes) == set(runtime.PHYSICAL_ARMS)
    supports["SUFFIX"][1] = supports["SUFFIX"][0]
    with pytest.raises(ValueError, match="malformed"):
        runtime.validate_supports(supports)


def test_source_closed_scorer_replays_every_gate_from_sufficient_statistics() -> None:
    pair = statistics.PairSums(1.0, 1.0, 1.0)
    geometry = [{
        cell: {name: pair for name in statistics.GEOMETRY_PAIRS}
        for cell in statistics.CELL_NAMES
    } for _ in range(runtime.CALLS)]
    bundle = {
        "schema": "mlp2_cmr_v1_validation_ledger",
        "ledgers": runtime.pack_ledgers(ledger_fixture()),
        "margin_counts": torch.zeros(runtime.DOCUMENTS, 2, dtype=torch.long),
        "margin_support_counts": torch.full(
            (runtime.DOCUMENTS,), 2, dtype=torch.long,
        ),
        "epsilon_grid": torch.tensor([0.1, 1.0], dtype=torch.float64),
        "geometry": runtime.pack_geometry(geometry),
        "additivity": torch.tensor(
            [[2.0, 1.0, 1.0]] * runtime.DOCUMENTS, dtype=torch.float64,
        ),
    }
    audits = {
        "exact_price_and_support_replay": True,
        "gauge_and_permutation_replay": True,
        "physical_materialization_replay": True,
        "physical_call_ledger_replay": True,
        "float32_cpu_float64_precision_audit": True,
    }
    result = runtime.score_validation_bundle(bundle, protocol_audits=audits)
    assert result["validation_passed"]
    assert result["replication_authorized"]
    assert result["shared_document_bootstrap"]["simultaneous_lower_bound"] >= 0.05
    failed = runtime.score_validation_bundle(
        bundle, protocol_audits={**audits, "physical_call_ledger_replay": False},
    )
    assert not failed["validation_passed"] and not failed["replication_authorized"]
