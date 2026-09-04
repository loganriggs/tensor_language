from __future__ import annotations

import statistics

import pytest

import circuit_fast_screen_producer as producer
import run_task14_head11_3_downstream_module_reader_screen as reader
import run_task14_head11_3_mlp15_17_vs_mlp16_factorial as run


class FakeBackend:
    def __init__(self, *, nonlinear: bool = False, control_interaction: float = 0.0,
                 replay_shift: float = 0.0):
        rows, self.native_pairs, _head, self.prior = run._load()
        self.family = {str(row["row_id"]): str(row["transform_id"]) for row in rows}
        self.nonlinear = nonlinear
        self.control_interaction = control_interaction
        self.replay_shift = replay_shift
        self.scale = statistics.median(
            reader._margin(self.native_pairs[(str(row["row_id"]), "donor")])
            + reader._margin(self.native_pairs[(str(row["row_id"]), "base")])
            for row in rows if row["transform_id"] in {"A1", "A2"}
        )

    def native(self, batch: producer.ModelBatch, *, capture: bool) -> producer.BatchOutput:
        pairs = []
        cache = {}
        for row_id in batch.row_ids:
            pair = self.native_pairs[(row_id, batch.side)]
            pairs.append((pair[0] + self.replay_shift, pair[1]))
            if capture:
                cache[(row_id, reader.HEAD_SITE)] = object()
                for site in run.CORE:
                    cache[(row_id, site)] = object()
        return producer.BatchOutput(tuple(pairs), cache)

    def induce_and_restore(self, batch, *, restore_sites, donor_cache, recipient_cache):
        assert tuple(restore_sites) == run.CORE
        pairs = []
        for row_id in batch.row_ids:
            prior = self.prior[row_id]
            family = self.family[row_id]
            interaction = (
                -0.10 if self.nonlinear and family in {"A1", "A2"}
                else self.control_interaction if family not in {"A1", "A2"}
                else 0.0
            )
            recovery = (
                prior["all"] - prior["mlp16"] + prior["empty"] - interaction
                if self.nonlinear or interaction else prior["all"]
            )
            base = self.native_pairs[(row_id, "base")]
            donor = self.native_pairs[(row_id, "donor")]
            base_margin = reader._margin(base)
            margin = (
                base_margin - recovery * (reader._margin(donor) + base_margin)
                if family in {"A1", "A2"}
                else base_margin + recovery * self.scale
            )
            pairs.append((margin, 0.0))
        return producer.BatchOutput(tuple(pairs), {})


def test_dryrun_reuses_three_corners_and_opens_one_arm():
    dryrun = run.compile_dryrun()
    assert dryrun["new_arm"] == ["mlp:15", "mlp:17"]
    assert dryrun["reused_corners"] == ["F(empty)", "F(16)", "F(15,16,17)"]
    assert dryrun["maximum_new_price"] == {
        "forward_calls": 12, "example_evaluations": 384,
        "backward_calls": 0, "model_updates": 0,
        "raw_numeric_evidence_bytes": 1024,
    }


def test_mlp15_17_core_exactly_explains_combined_path():
    result = run.run_science(backend=FakeBackend(), clock=lambda: 1.0)
    assert result["terminal"] == "mlp15_17_core_path_screen"
    assert result["core_relative_l2_to_all"] == pytest.approx(0.0)
    assert result["predictions"]["pred_b_mlp15_17_explain_combined_path"] is True


def test_planted_mlp16_interaction():
    result = run.run_science(backend=FakeBackend(nonlinear=True), clock=lambda: 2.0)
    assert result["terminal"] == "mlp16_interaction_screen"
    assert result["predictions"]["pred_c_mlp16_interaction_dependent"] is True
    assert all(cell["interaction"] == pytest.approx(-0.10)
               for cell in result["target_cells"].values())


def test_control_failure_blocks_scientific_terminal():
    result = run.run_science(backend=FakeBackend(control_interaction=0.30), clock=lambda: 3.0)
    assert result["terminal"] == "inconclusive"
    assert set(result["control_mean_absolute_terms"]["P"]) == {
        "empty", "mlp16", "mlp15_17", "all", "all_loss", "mlp16_loss",
        "mlp15_17_loss", "interaction",
    }


def test_native_replay_failure_invalidates():
    result = run.run_science(backend=FakeBackend(replay_shift=0.01), clock=lambda: 4.0)
    assert result["terminal"] == "invalid"
