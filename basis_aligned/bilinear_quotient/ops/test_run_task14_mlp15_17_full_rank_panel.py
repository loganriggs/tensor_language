from __future__ import annotations

import json

import numpy as np
import pytest

import run_task14_mlp15_17_full_rank_panel as run


class FakeBackend:
    def __init__(self, mode: str = "broad", *, break_provenance: bool = False):
        if mode not in {"broad", "direction", "split_mismatch"}:
            raise ValueError(mode)
        self.mode = mode
        self.break_provenance = break_provenance
        _contract, _endpoints, _relations, self.parent = run._load_plan()

    def native(self, endpoints):
        states = []
        for endpoint in endpoints:
            row_id, side = endpoint.endpoint_id.rsplit(":", 1)
            pair = self.parent[(row_id, side)]
            scalar = (int(endpoint.endpoint_id[:2], 16) % 13) / 10
            vector = np.array([scalar, scalar + 0.2, scalar - 0.1], dtype=np.float32)
            states.append(run.NativeState(
                answer_foil=pair,
                full_logits=np.array([pair[0], pair[1], 0.0, 0.0], dtype=np.float32),
                head11_3=np.zeros(3, dtype=np.float32),
                x15=vector, z15=vector + 1,
                x17=vector + 2, z17=vector + 3,
            ))
        return tuple(states)

    @staticmethod
    def _pair_for_score(score: float, role: str):
        return (-score, 0.0) if role == "target" else (score, 0.0)

    def _effect(self, row, component):
        if row.role == "control":
            return 0.0
        mode = self.mode
        if mode == "split_mismatch":
            mode = "broad" if row.split == "FIT" else "direction"
        if mode == "broad":
            return -0.12 if component != "joint" else -0.20
        if row.target_class == "paired" and row.recipient_subject_state == 1:
            return -0.12 if component != "joint" else -0.20
        return -0.01

    def conditional(self, relations, *, condition, endpoints, native):
        pairs, logits, diagnostics, provenance = [], [], [], []
        for row in relations:
            base_pair = native[row.target_endpoint_id].answer_foil
            base_score = run._score(base_pair, row.role)
            head = 1.0 if row.role == "target" else 0.0
            head_score = base_score + head
            if condition == "H" or condition.endswith("rescue"):
                score = head_score
            elif condition == "MLP15_reset":
                score = head_score + self._effect(row, "mlp15")
            elif condition == "MLP17_reset":
                score = head_score + self._effect(row, "mlp17")
            else:
                score = head_score + self._effect(row, "joint")
            pair = self._pair_for_score(score, row.role)
            pairs.append(pair)
            logits.append(np.array([pair[0], pair[1], 0.0, 0.0], dtype=np.float32))
            diagnostic = {"endpoint_error": 0.0}
            if condition == "H":
                diagnostic.update({
                    "closure15": 0.0, "closure17": 0.0,
                    "z17_H": np.zeros(3, dtype=np.float32),
                })
            if condition in {"MLP15_reset", "joint_reset"}:
                diagnostic["z17_live"] = np.array(
                    [row.ordinal, row.recipient_subject_state, 1.0], dtype=np.float32,
                )
            diagnostics.append(diagnostic)
            events = run.EXPECTED_JOINT_EVENTS if condition == "joint_reset" else ()
            if self.break_provenance and condition == "joint_reset":
                events = events[:-1]
            provenance.append(events)
        return run.ConditionOutput(
            tuple(pairs), np.stack(logits), tuple(diagnostics), tuple(provenance),
        )


def test_dryrun_freezes_exact_price_and_full_vocab_baselines():
    receipt = run.compile_dryrun()
    assert receipt["endpoint_count"] == 128
    assert receipt["relations_by_split"] == {"FIT": 153, "SELECT": 145}
    assert receipt["maximum_price"]["forward_calls"] == 74
    assert receipt["maximum_price"]["example_evaluations"] == 2214
    assert receipt["maximum_price"]["raw_numeric_evidence_bytes"] == 35592
    assert receipt["full_vocab_baselines"]["joint_sufficiency"] \
        == "logits(joint_rescue)-logits(joint_reset)"


def test_fake_broad_response_requires_both_splits_and_all_direction_cells():
    result = run.run_science(backend=FakeBackend("broad"), clock=lambda: 2.0)
    assert result["terminal"] == "broad_task14_full_rank_response_supported"
    assert result["predictions"]["pred_b_broad_by_split"] == {"FIT": True, "SELECT": True}
    assert result["active_price"] == {
        "forward_calls": 74, "example_evaluations": 2214,
        "backward_calls": 0, "model_updates": 0,
        "raw_numeric_evidence_bytes": 35592,
    }
    assert result["target_metrics"]["FIT"]["joint"]["paired|1"]["beta"] \
        == pytest.approx(0.2)


def test_fake_direction_specific_terminal_uses_joint_effect_only():
    result = run.run_science(backend=FakeBackend("direction"), clock=lambda: 3.0)
    assert result["terminal"] == "direction_specific_compensatory_response_supported"
    assert result["predictions"]["pred_c_direction_specific_by_split"] \
        == {"FIT": True, "SELECT": True}
    assert result["target_metrics"]["FIT"]["joint"]["paired|1"]["q"] \
        >= run.DIRECTION_LIVE_Q_MIN
    assert result["target_metrics"]["FIT"]["joint"]["paired|-1"]["q"] \
        < run.DIRECTION_WEAK_Q_MAX


def test_fit_select_pattern_disagreement_is_mixed():
    result = run.run_science(backend=FakeBackend("split_mismatch"), clock=lambda: 4.0)
    assert result["split_decisions"]["FIT"]["pattern"] == "broad"
    assert result["split_decisions"]["SELECT"]["pattern"] == "direction_specific"
    assert result["terminal"] == "mixed_full_rank_response_screen"


def test_missing_joint_event_is_instrument_invalid():
    result = run.run_science(
        backend=FakeBackend("broad", break_provenance=True), clock=lambda: 5.0,
    )
    assert result["terminal"] == "instrument_invalid"
    assert result["instrument"]["joint_provenance_valid"] is False


def test_create_only_publication_refuses_overwrite(tmp_path):
    path = tmp_path / "result.json"
    run._write_create_only(path, {"terminal": "screen"})
    assert json.loads(path.read_text()) == {"terminal": "screen"}
    with pytest.raises(FileExistsError):
        run._write_create_only(path, {"terminal": "different"})
