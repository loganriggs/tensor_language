"""CPU tests for the narrative-tense attention-11 H3/complement factorial."""

from __future__ import annotations

import json

import pytest

import circuit_fast_screen_candidate_narrative_tense as candidate
import circuit_fast_screen_producer as producer
import run_narrative_tense_attn11_head3_complement_factorial as run


class FakeBackend:
    def __init__(self, head_recovery: float, complement_recovery: float) -> None:
        self.head_recovery = head_recovery
        self.complement_recovery = complement_recovery
        self.native_calls = 0
        self.patch_calls = 0
        old = json.loads(run.PARENT.read_text())
        self.native_rows = {
            (str(item["row_id"]), str(item["side"])): item
            for item in old["run"]["native_logits"]
        }
        self.rows = {str(row["row_id"]): row for row in candidate.build_rows()}

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
        assert layer == 11
        recovery = self.head_recovery if tuple(heads) == (3,) else self.complement_recovery
        output = []
        for row_id in batch.row_ids:
            base = self.native_rows[(row_id, "base")]
            donor = self.native_rows[(row_id, "donor")]
            base_margin = float(base["answer_logit"]) - float(base["foil_logit"])
            donor_margin = float(donor["answer_logit"]) - float(donor["foil_logit"])
            if self.rows[row_id]["answer_changes"]:
                margin = base_margin - recovery * (base_margin + donor_margin)
            else:
                margin = base_margin
            output.append((margin, 0.0))
        return producer.BatchOutput(tuple(output), {})


def test_dryrun_binds_frozen_closure_and_prices_only_new_arms() -> None:
    plan = run.compile_plan()
    assert plan["prior_art_sha256"] == run.EXPECTED_PRIOR_ART_SHA256
    assert plan["parent_result_sha256"] == run.EXPECTED_PARENT_SHA256
    assert plan["authority_sha256"] == run.EXPECTED_AUTHORITY_SHA256
    assert plan["corners"] == {
        "empty": "frozen native", "head3": "new", "other8": "new", "full": "frozen attn11"
    }
    assert plan["price"] == {
        "model_forwards": 12, "example_evaluations": 384,
        "backwards": 0, "parameter_updates": 0,
    }


def test_head3_passes_when_it_carries_the_frozen_effect_without_control_damage() -> None:
    backend = FakeBackend(head_recovery=0.40, complement_recovery=0.0)
    ticks = iter((2.0, 3.0))
    result = run.run_science(backend=backend, clock=lambda: next(ticks))
    assert backend.native_calls == 4 and backend.patch_calls == 8
    assert result["active_price"]["model_forwards"] == 12
    assert result["predictions"]["pred_b_shared_copular_service"] is True
    assert result["predictions"]["pred_c_task_split"] is False
    assert len(result["cells"]) == 4 and len(result["evidence"]) == 128


def test_weak_head3_produces_task_split_terminal() -> None:
    result = run.run_science(backend=FakeBackend(head_recovery=0.02, complement_recovery=0.30))
    assert result["terminal"] == "task_split"
    assert result["predictions"] == {
        "pred_a_instrument_live": True,
        "pred_b_shared_copular_service": False,
        "pred_c_task_split": True,
    }


def test_missing_capture_fails_exact_instrument() -> None:
    backend = FakeBackend(0.4, 0.0)
    original = backend.native

    def missing(batch, *, capture):
        out = original(batch, capture=capture)
        return producer.BatchOutput(out.answer_foil, {
            key: value for key, value in out.captured.items() if not key[1].endswith("head:08")
        })

    backend.native = missing  # type: ignore[method-assign]
    with pytest.raises(run.FactorialError, match="lacks one or more"):
        run.run_science(backend=backend)


def test_managed_environment_is_model_free(monkeypatch, capsys) -> None:
    monkeypatch.setenv("BQLIB_DRYRUN", "1")
    monkeypatch.setenv("BQLIB_NO_MODEL", "1")
    monkeypatch.setattr(producer.Bilin18TorchBackend, "load", lambda *_: pytest.fail("model loaded"))
    run.main([])
    assert json.loads(capsys.readouterr().out)["model_loaded"] is False


def test_malformed_preflight_flag_fails_closed(monkeypatch) -> None:
    monkeypatch.setenv("BQLIB_NO_MODEL", "true")
    with pytest.raises(run.FactorialError, match="absent or exactly 1"):
        run.main([])
