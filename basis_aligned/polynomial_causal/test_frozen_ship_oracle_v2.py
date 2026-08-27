import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


PATH = Path(__file__).with_name("frozen_ship_oracle_v2.py")
SPEC = importlib.util.spec_from_file_location("frozen_ship_oracle_v2", PATH)
PIPELINE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(PIPELINE)


def gain(mean):
    return {"global": {"mean": mean, "ci95": [mean - 0.01, mean + 0.01]}}


def test_exact_fineweb_gate_replaces_interpolated_null_gate():
    result = {"site_decisions": {}, "paired_gains": {}}
    for site in range(3):
        key = str(site)
        result["site_decisions"][key] = {
            "full_oracle_ci95_lower_gt_zero": True,
            "content_positive_both_splits": True,
            "content_beats_matched_null95_heldout": True,
        }
        heldout = {"content": gain(0.10)}
        for index in range(20):
            heldout[f"null_{index:02d}"] = gain(0.05)
        result["paired_gains"][key] = {"heldout": heldout}
    # Site 1 has one tied null: interpolated quantile may pass, exact test must fail.
    result["paired_gains"]["1"]["heldout"]["null_19"]["global"]["mean"] = 0.10

    decisions = PIPELINE.exact_fineweb_decisions(result)

    assert result["training_license_sites"] == [0, 2]
    assert decisions["0"]["exact_twenty_null_test"]["exact_one_sided_p"] == 1 / 21
    assert decisions["1"]["exact_twenty_null_test"]["passes_5pct"] is False
    assert decisions["1"]["preliminary_interpolated_null95_gate"] is True


def test_cpu_tree_detaches_tensors_without_changing_structure():
    source = {"x": (torch.arange(4), [True, 3.0]), "name": "ship"}
    copied = PIPELINE.cpu_tree(source)
    assert copied["name"] == "ship"
    assert isinstance(copied["x"], tuple)
    assert torch.equal(copied["x"][0], source["x"][0])
    source["x"][0][0] = 99
    assert int(copied["x"][0][0]) == 0


def test_authoritative_source_explicitly_upgrades_preliminary_authority():
    source = PATH.read_text()
    assert '"authority": "canonical_fineweb"' in source
    assert '"authorized_for_scored_experiments": True' in source


def _fake_fingerprint(*_args, **_kwargs):
    return {
        "rows": 2,
        "positions": [64],
        "vocab_slice": [0, 1],
        "global_ce": 1.25,
        "full_logits_raw_sha256": "a" * 64,
        "sample_logits": torch.arange(4, dtype=torch.float32),
    }


def _redirect_frozen_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(PIPELINE, "FROZEN_STATE", tmp_path / "state.pt")
    monkeypatch.setattr(PIPELINE, "FROZEN_MANIFEST", tmp_path / "manifest.json")
    monkeypatch.setattr(PIPELINE, "FROZEN_LOCK", tmp_path / "state.lock")
    monkeypatch.setattr(PIPELINE, "baseline_fingerprint", _fake_fingerprint)


def _fake_ship():
    return SimpleNamespace(
        DEV="cpu",
        SHIP={"x": torch.arange(3, dtype=torch.float32)},
        CORR={
            "on": True,
            "b": torch.ones(2),
            "U": torch.eye(2),
            "V": torch.eye(2),
        },
    )


def test_frozen_pair_is_atomic_validated_nonoverwriting_and_restorable(monkeypatch, tmp_path):
    _redirect_frozen_paths(monkeypatch, tmp_path)
    ship = _fake_ship()
    twall = {0: {"q": torch.eye(2)}}
    receipt = {"authority": "test", "entries": {}}
    rows = torch.zeros(2, 257, dtype=torch.long)

    realization, manifest = PIPELINE.freeze_ship_realization(
        ship, twall, frozenset({0}), receipt, rows
    )
    assert PIPELINE.FROZEN_STATE.is_file()
    assert PIPELINE.FROZEN_MANIFEST.is_file()
    payload, validated = PIPELINE.validate_frozen_ship_pair(receipt)
    assert payload["ship_realization_sha256"] == realization
    assert validated == manifest
    assert not PIPELINE.FROZEN_LOCK.exists()
    with pytest.raises(RuntimeError, match="already exists"):
        PIPELINE.freeze_ship_realization(ship, twall, frozenset({0}), receipt, rows)

    ship.SHIP = {"wrong": torch.tensor(9)}
    ship.CORR = {"on": False, "b": None, "U": None, "V": None}
    twall.clear()
    restored, _, lifecycle = PIPELINE.obtain_ship_realization(
        ship, twall, frozenset({0}), receipt, rows
    )
    assert lifecycle == "restored"
    assert restored == realization
    assert set(ship.SHIP) == {"x"}
    assert set(twall) == {0}


def test_frozen_pair_inconsistency_and_shared_lock_fail_closed(monkeypatch, tmp_path):
    _redirect_frozen_paths(monkeypatch, tmp_path)
    PIPELINE.FROZEN_STATE.write_bytes(b"state only")
    with pytest.raises(RuntimeError, match="inconsistent canonical frozen-state pair"):
        PIPELINE.validate_frozen_ship_pair({})
    PIPELINE.FROZEN_STATE.unlink()
    PIPELINE.FROZEN_LOCK.mkdir()
    with pytest.raises(RuntimeError, match="already claimed"):
        PIPELINE.validate_frozen_ship_pair({})


def test_frozen_manifest_hash_or_receipt_mismatch_fails_closed(monkeypatch, tmp_path):
    _redirect_frozen_paths(monkeypatch, tmp_path)
    ship = _fake_ship()
    receipt = {"authority": "test", "entries": {}}
    PIPELINE.freeze_ship_realization(
        ship, {0: {"q": torch.eye(2)}}, frozenset({0}), receipt,
        torch.zeros(2, 257, dtype=torch.long),
    )
    with pytest.raises(RuntimeError, match="row receipt changed"):
        PIPELINE.validate_frozen_ship_pair({"authority": "different"})
    manifest = json.loads(PIPELINE.FROZEN_MANIFEST.read_text())
    manifest["artifact_sha256"] = "0" * 64
    PIPELINE.FROZEN_MANIFEST.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="artifact hash changed"):
        PIPELINE.validate_frozen_ship_pair(receipt)
