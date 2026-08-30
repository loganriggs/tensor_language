import pytest
import torch

import causal_response_tensor_v1_fit_bundle as bundle
from causal_response_tensor_v1_backend import CircuitSpec, ObservedResponseCollector
from test_bilin18_observed_model_facade import tiny_model


def _spec(tag, component, members, slice_positions, size):
    member = torch.zeros(size, dtype=torch.bool)
    member[members] = True
    slice_mask = torch.zeros(size, dtype=torch.bool)
    slice_mask[slice_positions] = True
    return CircuitSpec(tag, component, member, slice_mask)


def _preimage():
    torch.manual_seed(23)
    model = tiny_model()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(mean=0.0, std=0.03)
    rows = torch.randint(0, 32, (8, 5), dtype=torch.int64)
    documents = torch.arange(8, dtype=torch.int64)
    grid = 32
    a_slice = [position for position in range(grid) if position % 4 in (0, 1)]
    m_slice = [position for position in range(grid) if position % 4 in (2, 3)]
    specs = (
        _spec("a.one", "a1", list(range(0, grid, 4)), a_slice, grid),
        _spec("a.two", "a1", list(range(1, grid, 4)), a_slice, grid),
        _spec("m.one", "m2", list(range(2, grid, 4)), m_slice, grid),
        _spec("m.two", "m2", list(range(3, grid, 4)), m_slice, grid),
    )
    collector = ObservedResponseCollector(
        model, rows, documents, specs, require_production=False
    )
    return collector.fit_stage(torch.arange(4, dtype=torch.int64))


def _binding():
    return bundle.FitBundleBinding(
        authority_sha256="a" * 64,
        source_closure_sha256="b" * 64,
        census_state_diverse_sha256="c" * 64,
        curated_rows_sha256="d" * 64,
        battery_sha256="e" * 64,
        document_split_sha256="f" * 64,
        config_sha256="1" * 64,
        weights_sha256="2" * 64,
        model_state_sha256_before="3" * 64,
        model_state_sha256_after="3" * 64,
    )


def _payload():
    return bundle.build_fit_bundle_payload(
        _preimage(), _binding(), require_production=False
    )


def test_build_publish_reload_returns_sealed_cloned_program(tmp_path):
    preimage = _preimage()
    payload = bundle.build_fit_bundle_payload(
        preimage, _binding(), require_production=False
    )
    preimage["directions"].zero_()
    assert payload["directions"].abs().sum() > 0
    path = tmp_path / "fit_bundle.pt"
    capability = bundle.publish_fit_bundle(
        path,
        payload,
        expected_authority_sha256="a" * 64,
        require_production=False,
    )
    directions = capability.clone_direction_map()
    fit_documents = capability.clone_fit_document_ids()
    assert tuple(directions) == ("full", "residual")
    assert tuple(directions["full"]) == tuple(payload["source_tags"])
    assert fit_documents.tolist() == [0, 1, 2, 3]
    directions["full"][payload["source_tags"][0]].zero_()
    assert capability.clone_direction_map()["full"][payload["source_tags"][0]].norm() > 0


def test_capability_cannot_be_constructed_externally():
    with pytest.raises(RuntimeError, match="cannot be constructed"):
        bundle.FitProgramCapability(
            object(), directions=torch.zeros(2, 1, 2),
            fit_document_ids=torch.zeros(1, dtype=torch.int64),
            source_tags=("a",), source_components=("a1",),
            authority_sha256="a" * 64, artifact_sha256="b" * 64,
        )


@pytest.mark.parametrize("attack", ["direction", "response", "ledger", "support", "state"])
def test_semantic_validator_rejects_tampering(attack):
    payload = _payload()
    if attack == "direction":
        payload["directions"][0, 0, 0] += 0.25
        message = "unit normalized|does not replay"
    elif attack == "response":
        payload["fit_response"]["statistics"]["member_abs_sum"][0, 0, 0, 0] = -1
        message = "nonnegative"
    elif attack == "ledger":
        payload["call_ledger"]["projection_event_counts"][0, 0, 0] = 2
        message = "structured event ledger"
    elif attack == "support":
        tag = payload["source_tags"][0]
        payload["support_hashes"][tag]["member_mask_sha256"] = "not-a-hash"
        message = "support hash"
    else:
        payload["binding"]["model_state_sha256_after"] = "4" * 64
        message = "model state changed"
    # Model an attacker who also recomputes the internal digest map: semantic replay
    # must still reject the self-consistent corruption.
    payload["tensor_hashes"] = bundle._tensor_hash_map({
        key: value for key, value in payload.items() if key != "tensor_hashes"
    })
    with pytest.raises((ValueError, RuntimeError), match=message):
        bundle.validate_fit_bundle_payload(payload, require_production=False)


def test_tensor_digest_rejects_unrehashable_mutation():
    payload = _payload()
    payload["directions"][0, 0, 0] += 0.25
    with pytest.raises(RuntimeError, match="tensor digest"):
        bundle.validate_fit_bundle_payload(payload, require_production=False)


def test_publication_is_create_only(tmp_path):
    payload = _payload()
    path = tmp_path / "fit_bundle.pt"
    bundle.publish_fit_bundle(
        path, payload, expected_authority_sha256="a" * 64,
        require_production=False,
    )
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        bundle.publish_fit_bundle(
            path, payload, expected_authority_sha256="a" * 64,
            require_production=False,
        )
    assert path.read_bytes() == original


def test_failed_private_replay_never_publishes(monkeypatch, tmp_path):
    payload = _payload()
    path = tmp_path / "fit_bundle.pt"
    monkeypatch.setattr(
        bundle,
        "load_fit_program",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("injected replay")),
    )
    with pytest.raises(RuntimeError, match="injected replay"):
        bundle.publish_fit_bundle(
            path, payload, expected_authority_sha256="a" * 64,
            require_production=False,
        )
    assert not path.exists()
