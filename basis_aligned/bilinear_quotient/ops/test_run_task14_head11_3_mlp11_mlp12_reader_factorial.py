from __future__ import annotations

import pytest

import circuit_fast_screen_producer as producer
import run_task14_head11_3_mlp11_mlp12_reader_factorial as run
import run_task14_head11_3_downstream_module_reader_screen as parent_reader


class FakeBackend:
    def __init__(self, *, interaction: float = 0.0, replay_shift: float = 0.0) -> None:
        rows, self.native_pairs, self.head, self.singles = run._load()
        self.family = {str(row["row_id"]): str(row["transform_id"]) for row in rows}
        self.interaction = interaction
        self.replay_shift = replay_shift
        self.scale = __import__("statistics").median(
            parent_reader._margin(self.native_pairs[(str(row["row_id"]), "donor")])
            + parent_reader._margin(self.native_pairs[(str(row["row_id"]), "base")])
            for row in rows if row["transform_id"] in {"A1", "A2"}
        )

    def native(self, batch: producer.ModelBatch, *, capture: bool):
        pairs, cache = [], {}
        for row_id in batch.row_ids:
            pair = self.native_pairs[(row_id, batch.side)]
            pairs.append((pair[0] + self.replay_shift, pair[1]))
            if capture:
                cache[(row_id, parent_reader.HEAD_SITE)] = object()
                for site in run.RESTORE_SITES:
                    cache[(row_id, site)] = object()
        return producer.BatchOutput(tuple(pairs), cache)

    def induce_and_restore_both(self, batch, *, donor_cache, recipient_cache):
        pairs = []
        for row_id in batch.row_ids:
            family = self.family[row_id]
            base, donor = self.native_pairs[(row_id, "base")], self.native_pairs[(row_id, "donor")]
            args = family, base, donor
            f0 = parent_reader._recovery(*args, self.head[row_id], self.scale)
            f11 = parent_reader._recovery(*args, self.singles["mlp:11"][row_id], self.scale)
            f12 = parent_reader._recovery(*args, self.singles["mlp:12"][row_id], self.scale)
            target_interaction = self.interaction if family in {"A1", "A2"} else 0.0
            fb = f11 + f12 - f0 + target_interaction
            base_margin = parent_reader._margin(base)
            if family in {"A1", "A2"}:
                margin = base_margin - fb * (parent_reader._margin(donor) + base_margin)
            else:
                margin = base_margin + fb * self.scale
            pairs.append((margin, 0.0))
        return producer.BatchOutput(tuple(pairs), {})


def test_dryrun_is_only_missing_corner() -> None:
    receipt = run.compile_dryrun()
    assert receipt["maximum_new_price"] == {
        "forward_calls": 12, "example_evaluations": 384,
        "backward_calls": 0, "model_updates": 0,
        "raw_numeric_evidence_bytes": 1024,
    }
    assert receipt["bars"]["additive_cell_abs_max"] < receipt["bars"]["grouped_cell_abs_min"]


def test_exact_additive_null() -> None:
    result = run.run_science(backend=FakeBackend(interaction=0.0), clock=lambda: 1.0)
    assert result["terminal"] == "additive_null"
    assert result["predictions"] == {
        "pred_a_native_replay": True,
        "pred_b_nonlinear_grouping": False,
        "pred_c_additive_use": True,
    }
    assert max(abs(x) for x in result["target_cell_mean_interaction"].values()) < 1e-6
    assert len(result["evidence"]) == 128


def test_planted_joint_interaction_is_grouping_screen() -> None:
    result = run.run_science(backend=FakeBackend(interaction=-0.2), clock=lambda: 2.0)
    assert result["terminal"] == "nonlinear_grouping_screen"
    assert result["predictions"]["pred_b_nonlinear_grouping"] is True
    assert all(value == pytest.approx(-0.2) for value in result["target_cell_mean_interaction"].values())


def test_replay_failure_is_invalid() -> None:
    result = run.run_science(backend=FakeBackend(replay_shift=0.01), clock=lambda: 3.0)
    assert result["terminal"] == "invalid"
    assert result["predictions"]["pred_a_native_replay"] is False
