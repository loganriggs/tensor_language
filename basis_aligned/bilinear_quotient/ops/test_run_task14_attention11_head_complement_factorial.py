"""CPU-only tests for the Task-14 attention-11 head/complement screen."""

from __future__ import annotations

import json

import pytest

import circuit_fast_screen_candidate_task14_agreement as candidate
import circuit_fast_screen_producer as producer
import run_task14_attention11_head_complement_factorial as run


class FakeBackend:
    def __init__(self, complement_recovery: float = 0.0) -> None:
        self.complement_recovery = complement_recovery
        self.native_calls = 0
        self.patch_calls = 0
        old = json.loads(run.V2_RESULT.read_text())
        self.native_rows = {
            (str(item["row_id"]), str(item["side"])): item
            for item in old["run"]["native_logits"]
        }

    def native(self, batch: producer.ModelBatch, *, capture: bool) -> producer.BatchOutput:
        self.native_calls += 1
        assert capture and batch.side == "donor"
        captured = {
            (row_id, f"attn:11:head:{head:02d}"): object()
            for row_id in batch.row_ids for head in range(9)
        }
        return producer.BatchOutput(tuple((0.0, 0.0) for _ in batch.row_ids), captured)

    def patched_heads(self, batch, *, layer, heads, donor_cache):
        self.patch_calls += 1
        assert layer == 11 and tuple(heads) == run.COMPLEMENT_HEADS
        assert all(
            (row_id, f"attn:11:head:{head:02d}") in donor_cache
            for row_id in batch.row_ids for head in heads
        )
        output = []
        by_id = {str(row["row_id"]): row for row in candidate.build_rows()}
        for row_id in batch.row_ids:
            base = self.native_rows[(row_id, "base")]
            donor = self.native_rows[(row_id, "donor")]
            base_margin = float(base["answer_logit"]) - float(base["foil_logit"])
            if by_id[row_id]["answer_changes"]:
                donor_margin = float(donor["answer_logit"]) - float(donor["foil_logit"])
                margin = base_margin - self.complement_recovery * (base_margin + donor_margin)
            else:
                margin = base_margin
            output.append((margin, 0.0))
        return producer.BatchOutput(tuple(output), {})


def test_dryrun_hash_binds_old_corners_and_prices_only_new_arm() -> None:
    plan = run.compile_dryrun()
    assert plan["authority_sha256"] == run.EXPECTED_AUTHORITY_SHA256
    assert plan["v2_result_sha256"] == run.EXPECTED_V2_SHA256
    assert plan["complement_heads"] == [0, 1, 2, 4, 5, 6, 7, 8]
    assert plan["maximum_price"] == {
        "forward_calls": 8,
        "example_evaluations": 256,
        "backward_calls": 0,
        "model_updates": 0,
        "raw_numeric_evidence_bytes": 1024,
    }
    assert plan["model_loaded"] is False and plan["gpu_accessed"] is False


def test_existing_head_and_full_corners_plus_zero_complement_give_clean_split() -> None:
    backend = FakeBackend(0.0)
    ticks = iter((10.0, 11.0))
    result = run.run_science(backend=backend, clock=lambda: next(ticks))
    assert backend.native_calls == 4 and backend.patch_calls == 4
    assert result["active_price"]["forward_calls"] == 8
    assert result["terminal"] == "clean_split"
    assert result["predictions"] == {
        "pred_a_exact_factorial_partition": True,
        "pred_b_clean_head11_3_split": True,
        "pred_c_distributed_or_interactive_complement": False,
    }
    assert len(result["cells"]) == 4 and len(result["evidence"]) == 128
    assert all(abs(item["complement"]) < 1e-12 for item in result["evidence"])


def test_large_joint_complement_is_classified_as_distributed() -> None:
    result = run.run_science(backend=FakeBackend(0.30))
    assert result["terminal"] == "distributed_or_interactive"
    assert result["predictions"]["pred_c_distributed_or_interactive_complement"] is True


def test_cli_has_explicit_dryrun_and_rejects_unknown_arguments(capsys) -> None:
    run.main(["--dry-run"])
    assert json.loads(capsys.readouterr().out)["model_loaded"] is False
    with pytest.raises(SystemExit):
        run.main(["--not-a-real-option"])
