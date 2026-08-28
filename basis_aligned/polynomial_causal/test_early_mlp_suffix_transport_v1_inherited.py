from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import early_mlp_suffix_transport_v1_inherited as loader


def test_real_inherited_initialization_is_source_closed_and_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_load = torch.load
    calls: list[tuple[str, str | torch.device | None, bool | None]] = []

    def recording_load(path, *args, **kwargs):
        calls.append((Path(path).name, kwargs.get("map_location"), kwargs.get("weights_only")))
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(loader.torch, "load", recording_load)
    inherited = loader.load_canonical_initialization()
    bases = inherited.clone_bases()
    states = inherited.clone_affine_states()
    assert set(bases) == {0, 1}
    assert set(states) == {0, 1}
    assert len(inherited.authority.bindings) == 8
    assert calls == [
        ("early_mlp_state_complete_compiler_v21_programs.pt", "cpu", True),
        ("joint_early_mlp_pca_composition_authoritative_v3_bases.pt", "cpu", True),
    ]
    for site in (0, 1):
        assert bases[site].shape == (1152, 64)
        assert bases[site].device.type == "cpu"
        assert states[site]["left"].shape == (1152, 64)
        assert states[site]["right"].shape == (64, 64)
        assert set(states[site]) == {
            "grammar", "interface", "mean", "scale", "bias", "left", "right",
        }
        assert loader.raw_tensor_sha256(
            states[site]["left"] @ states[site]["right"]
        ) == inherited.authority.full_product_sha256[site]
    second = inherited.clone_affine_states()
    second_bases = inherited.clone_bases()
    states[0]["left"][0, 0] += 1
    bases[0][0, 0] += 1
    assert not torch.equal(states[0]["left"], second[0]["left"])
    assert not torch.equal(bases[0], second_bases[0])
    p0 = inherited.make_program("L")
    p1 = inherited.make_program("L")
    for left, right in zip(p0.parameters(), p1.parameters(), strict=True):
        assert left.data_ptr() != right.data_ptr()
    z = torch.randn(2, 3, 1152)
    state = inherited.clone_affine_states()[0]
    expected = ((z - state["mean"]) / state["scale"]) @ (
        state["left"] @ state["right"]
    ) + state["bias"]
    torch.testing.assert_close(p0.site0(z), expected)


def test_master_and_authority_are_independently_sealed() -> None:
    inherited = loader.load_canonical_initialization()
    with pytest.raises(AttributeError, match="sealed"):
        inherited.authority = inherited.authority
    masters = inherited._LoadedInitialization__bases
    masters[0][0, 0] += 1
    with pytest.raises(RuntimeError, match="master mutated"):
        inherited.clone_bases()

    clean = loader.load_canonical_initialization()
    states = clean._LoadedInitialization__states
    states[0]["grammar"] = "changed"
    with pytest.raises(RuntimeError, match="master mutated"):
        clean.clone_affine_states()


def test_pinned_file_verification_precedes_deserialization(tmp_path: Path) -> None:
    path = tmp_path / "one.bin"
    path.write_bytes(b"exact")
    pin = loader.ArtifactPin("one.bin", loader.file_sha256(path), 5)
    assert loader._verify_pinned_files(tmp_path, (pin,))["one.bin"]["bytes"] == 5
    path.write_bytes(b"wrong")
    with pytest.raises(RuntimeError, match="binding changed"):
        loader._verify_pinned_files(tmp_path, (pin,))


def test_historical_source_closure_rejects_count_and_hash_drift() -> None:
    authority = json.loads(
        (loader.BQ / "early_mlp_state_complete_compiler_v21_final_authority.json").read_text()
    )
    source_hashes = authority["source_hashes"]
    with pytest.raises(RuntimeError, match="path count"):
        loader.verify_historical_source_closure(
            loader.SOURCE_COMMIT, source_hashes, expected_count=59,
        )
    changed = dict(source_hashes)
    first = next(iter(changed))
    changed[first] = "0" * 64
    with pytest.raises(RuntimeError, match="content changed"):
        loader.verify_historical_source_closure(
            loader.SOURCE_COMMIT, changed, expected_count=60,
        )


def test_v21_negative_authority_cannot_be_relabelled_as_admitted() -> None:
    bindings = loader._verify_pinned_files(loader.BQ, loader.PINS + loader.TERMINAL_CHAIN_PINS)
    authority = json.loads(
        (loader.BQ / "early_mlp_state_complete_compiler_v21_final_authority.json").read_text()
    )
    receipt = json.loads(
        (loader.BQ / "early_mlp_state_complete_compiler_v21_programs_receipt.json").read_text()
    )
    changed = dict(authority)
    changed["package_admitted"] = True
    with pytest.raises(RuntimeError, match="authority metadata"):
        loader.validate_v21_metadata(changed, receipt, bindings)


def test_affine_loader_rejects_bad_scale_and_returns_independent_copies() -> None:
    bundle = torch.load(
        loader.BQ / "early_mlp_state_complete_compiler_v21_programs.pt",
        map_location="cpu", weights_only=True,
    )
    states = loader.validate_affine_initializations(bundle)
    original = bundle["programs"]["true"][0]["left"][0, 0].item()
    states[0]["left"][0, 0] += 1
    assert bundle["programs"]["true"][0]["left"][0, 0].item() == original
    bundle["programs"]["true"][1]["scale"][0] = 0
    with pytest.raises(RuntimeError, match="normalization changed"):
        loader.validate_affine_initializations(bundle)


def test_strict_json_rejects_duplicate_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"a": 1, "a": 2}')
    with pytest.raises(RuntimeError, match="duplicate JSON key"):
        loader._strict_json(path)


def test_capability_not_minted_if_metadata_snapshot_drifts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = loader._verify_pinned_files
    calls = 0

    def drifting_snapshot(root, pins):
        nonlocal calls
        calls += 1
        value = original(root, pins)
        if calls == 2:
            value = {key: dict(binding) for key, binding in value.items()}
            name = "early_mlp_state_complete_compiler_v21_final_manifest.json"
            value[name]["sha256"] = "0" * 64
        return value

    monkeypatch.setattr(loader, "_verify_pinned_files", drifting_snapshot)
    with pytest.raises(RuntimeError, match="snapshot changed while loading"):
        loader.load_canonical_initialization()
    assert calls == 2


def test_capability_not_minted_if_source_drifts_during_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = loader.verify_historical_source_closure
    calls = 0

    def drifting_source(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("current inherited source content changed: synthetic")
        return original(*args, **kwargs)

    monkeypatch.setattr(loader, "verify_historical_source_closure", drifting_source)
    with pytest.raises(RuntimeError, match="source content changed"):
        loader.load_canonical_initialization()
    assert calls == 3
