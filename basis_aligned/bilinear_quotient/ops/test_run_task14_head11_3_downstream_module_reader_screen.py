from __future__ import annotations

import math

import pytest

import circuit_fast_screen_producer as producer
import run_task14_head11_3_downstream_module_reader_screen as run


class FakeBackend:
    def __init__(self, *, amplifier: bool = True, replay_shift: float = 0.0) -> None:
        rows, parent = run._load()
        self.native_pairs, self.head_pairs = run._parent_maps(parent)
        self.families = {str(row["row_id"]): str(row["transform_id"]) for row in rows}
        self.amplifier = amplifier
        self.replay_shift = replay_shift

    def native(self, batch: producer.ModelBatch, *, capture: bool) -> producer.BatchOutput:
        pairs = []
        captured = {}
        for row_id in batch.row_ids:
            pair = self.native_pairs[(row_id, batch.side)]
            pairs.append((pair[0] + self.replay_shift, pair[1]))
            if capture:
                captured[(row_id, run.HEAD_SITE)] = object()
                for site in run.SITES:
                    captured[(row_id, site)] = object()
        return producer.BatchOutput(tuple(pairs), captured)

    def induce_and_restore(
        self, batch: producer.ModelBatch, *, restore_site: str,
        donor_cache, recipient_cache,
    ) -> producer.BatchOutput:
        pairs = []
        for row_id in batch.row_ids:
            head = self.head_pairs[row_id]
            if not self.amplifier or restore_site != "mlp:11" or self.families[row_id] not in {"A1", "A2"}:
                pairs.append(head)
                continue
            base = self.native_pairs[(row_id, "base")]
            donor = self.native_pairs[(row_id, "donor")]
            denominator = run._margin(donor) + run._margin(base)
            # All target rows have answer-changing counterfactuals.  Lower the
            # restored recovery by exactly 0.25 relative to head-only.
            head_recovery = (run._margin(base) - run._margin(head)) / denominator
            restored_recovery = head_recovery - 0.25
            restored_margin = run._margin(base) - restored_recovery * denominator
            pairs.append((restored_margin, 0.0))
        return producer.BatchOutput(tuple(pairs), {})


def test_dryrun_freezes_sites_price_and_opposing_bars() -> None:
    receipt = run.compile_dryrun()
    assert receipt["restore_sites"] == list(run.SITES)
    assert len(run.SITES) == 13
    assert receipt["maximum_price"] == {
        "forward_calls": 60,
        "example_evaluations": 1920,
        "backward_calls": 0,
        "model_updates": 0,
        "raw_numeric_evidence_bytes": 13312,
    }
    assert receipt["bars"]["no_single_module_mean_abs_max"] < receipt["bars"]["shared_amplifier_mean_loss_min"]


def test_fake_shared_amplifier_scores_exactly() -> None:
    result = run.run_science(backend=FakeBackend(amplifier=True), clock=lambda: 4.0)
    assert result["terminal"] == "shared_amplifier_screen"
    assert result["predictions"] == {
        "pred_a_native_replay": True,
        "pred_b_one_shared_amplifier": True,
        "pred_c_no_strong_single_module": False,
    }
    assert result["shared_amplifier_sites"] == ["mlp:11"]
    mlp11 = result["site_summaries"][0]
    assert mlp11["mean_signed_target_loss"] == pytest.approx(0.25)
    assert set(mlp11["target_cell_mean_signed_loss"]) == {
        "A1/fit_pp_near/plural_to_singular",
        "A1/fit_pp_near/singular_to_plural",
        "A2/fit_relative_placed_beside/plural_to_singular",
        "A2/fit_relative_placed_beside/singular_to_plural",
    }
    assert result["active_price"]["forward_calls"] == 60
    assert result["active_price"]["example_evaluations"] == 1920
    assert len(result["evidence"]) == 13 * 128


def test_fake_no_single_module_is_honest_null() -> None:
    result = run.run_science(backend=FakeBackend(amplifier=False), clock=lambda: 5.0)
    assert result["terminal"] == "no_strong_single_module_null"
    assert result["predictions"]["pred_c_no_strong_single_module"] is True
    assert result["shared_amplifier_sites"] == []
    assert all(
        math.isclose(item["mean_absolute_target_cell_effect"], 0.0, abs_tol=1e-12)
        for item in result["site_summaries"]
    )


def test_native_replay_failure_is_invalid_not_scientific_null() -> None:
    result = run.run_science(
        backend=FakeBackend(amplifier=False, replay_shift=1.0e-2), clock=lambda: 6.0,
    )
    assert result["terminal"] == "invalid"
    assert result["predictions"]["pred_a_native_replay"] is False


def test_pair_rejects_nonfinite_values() -> None:
    with pytest.raises(run.ReaderScreenError, match="malformed"):
        run._pair((float("nan"), 0.0))
