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


def _binding(preimage):
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
        model_rows_sha256="4" * 64,
        fit_role_sha256=bundle.tensor_sha256(
            preimage["fit_response"]["row_indices"]
        ),
        fit_document_ids_sha256=bundle.tensor_sha256(
            preimage["fit_response"]["document_ids"]
        ),
        support_hashes_sha256=bundle.logical_sha256(preimage["support_hashes"]),
    )


def _payload():
    preimage = _preimage()
    return bundle.build_fit_bundle_payload(
        preimage, _binding(preimage), require_production=False
    )


def test_build_publish_replays_exact_bytes_without_minting_program(tmp_path):
    preimage = _preimage()
    payload = bundle.build_fit_bundle_payload(
        preimage, _binding(preimage), require_production=False
    )
    preimage["directions"].zero_()
    assert payload["directions"].abs().sum() > 0
    path = tmp_path / "fit_bundle.pt"
    artifact_sha256 = bundle.publish_fit_bundle(
        path,
        payload,
        expected_authority_sha256="a" * 64,
        require_production=False,
    )
    assert artifact_sha256 == bundle.file_sha256(path)
    assert bundle.semantic_replay_fit_bundle(
        path,
        expected_authority_sha256="a" * 64,
        expected_artifact_sha256=artifact_sha256,
        require_production=False,
    ) == artifact_sha256


def test_manifest_summary_is_derived_from_exact_bytes_and_contains_no_tensors(tmp_path):
    payload = _payload()
    path = tmp_path / "fit_bundle.pt"
    artifact_sha256 = bundle.publish_fit_bundle(
        path, payload, expected_authority_sha256="a" * 64,
        require_production=False,
    )
    summary = bundle.fit_bundle_manifest_summary(
        path,
        expected_authority_sha256="a" * 64,
        expected_artifact_sha256=artifact_sha256,
        require_production=False,
    )
    assert summary["binding"] == payload["binding"]
    assert summary["tensor_hashes"] == payload["tensor_hashes"]
    assert summary["ledger"]["outer_forwards"] == payload["call_ledger"][
        "outer_forwards"
    ]

    def contains_tensor(value):
        if type(value) is torch.Tensor:
            return True
        if type(value) is dict:
            return any(contains_tensor(item) for item in value.values())
        if type(value) is list:
            return any(contains_tensor(item) for item in value)
        return False

    assert not contains_tensor(summary)
    with path.open("ab") as sink:
        sink.write(b"mutation")
    with pytest.raises(RuntimeError, match="changed during manifest derivation"):
        bundle.fit_bundle_manifest_summary(
            path,
            expected_authority_sha256="a" * 64,
            expected_artifact_sha256=artifact_sha256,
            require_production=False,
        )


def test_no_eval_capability_surface_exists_before_receipt():
    assert not hasattr(bundle, "FitProgramCapability")
    assert not hasattr(bundle, "load_fit_program")


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("claim_boundary", "this bundle authorizes EVAL", "claim boundary"),
        (
            "sign_convention",
            "dCE = native CE - rank-one-projection intervention CE",
            "sign convention",
        ),
        ("off_mask", "all nonmember positions", "off-target mask"),
    ],
)
def test_scalar_scientific_contract_is_exact(field, replacement, message):
    payload = _payload()
    payload[field] = replacement
    # Scalar strings are deliberately not hidden behind tensor hashes; the semantic
    # validator itself must know and enforce their exact scientific meaning.
    with pytest.raises(RuntimeError, match=message):
        bundle.validate_fit_bundle_payload(payload, require_production=False)


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


def test_bundle_binding_rejects_a_different_fit_document_set():
    payload = _payload()
    payload["fit_response"]["document_ids"][-1] += 100
    payload["tensor_hashes"] = bundle._tensor_hash_map({
        key: value for key, value in payload.items() if key != "tensor_hashes"
    })
    with pytest.raises(RuntimeError, match="document role"):
        bundle.validate_fit_bundle_payload(payload, require_production=False)


def test_semantic_validator_enforces_preregistered_residual_cutoff():
    payload = _payload()
    tags = payload["source_tags"][:2]
    indices = [0, 1]
    epsilon = 1e-7
    first = torch.zeros(payload["model_width"], dtype=torch.float64)
    first[0] = 1
    second = first.clone()
    second[1] = epsilon
    second /= second.norm()
    masters = [first, second]
    for tag, index, master in zip(tags, indices, masters):
        count = payload["fit_counts"][tag]
        count["member_count"] = 1
        count["off_count"] = 1
        stats = payload["fit_write_statistics"][tag]
        stats["member_sum"] = master.clone()
        stats["member_mean"] = master.clone()
        stats["off_sum"] = torch.zeros_like(master)
        stats["off_mean"] = torch.zeros_like(master)
        payload["full_direction_norms"][tag] = float(master.norm())
        payload["directions"][0, index] = master.float()
    matrix = torch.stack(masters)
    shared, spectrum = bundle.leading_shared_direction(matrix)
    component = payload["source_components"][0]
    payload["shared_directions"][component] = shared.float()
    payload["singular_spectra"][component] = spectrum
    payload["relative_singular_gaps"][component] = float(
        (spectrum[0] - spectrum[1]) / spectrum[0]
    )
    for tag, index, master in zip(tags, indices, masters):
        remainder = master - (master @ shared) * shared
        payload["residual_norms"][tag] = float(remainder.norm())
        payload["directions"][1, index] = (remainder / remainder.norm()).float()
    payload["tensor_hashes"] = bundle._tensor_hash_map({
        key: value for key, value in payload.items() if key != "tensor_hashes"
    })
    assert max(payload["residual_norms"][tag] for tag in tags) <= 1e-6
    with pytest.raises(RuntimeError, match="residual direction is numerically absent"):
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


def test_semantic_reload_deserializes_the_exact_hashed_bytes(monkeypatch, tmp_path):
    payload = _payload()
    path = tmp_path / "fit_bundle.pt"
    artifact_sha256 = bundle.publish_fit_bundle(
        path, payload, expected_authority_sha256="a" * 64,
        require_production=False,
    )
    original_load = bundle.torch.load

    def require_bytes(source, *args, **kwargs):
        assert isinstance(source, bundle.io.BytesIO)
        return original_load(source, *args, **kwargs)

    monkeypatch.setattr(bundle.torch, "load", require_bytes)
    assert bundle.semantic_replay_fit_bundle(
        path, expected_authority_sha256="a" * 64,
        expected_artifact_sha256=artifact_sha256,
        require_production=False,
    ) == artifact_sha256


def test_failed_private_replay_never_publishes(monkeypatch, tmp_path):
    payload = _payload()
    path = tmp_path / "fit_bundle.pt"
    monkeypatch.setattr(
        bundle,
        "semantic_replay_fit_bundle",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("injected replay")),
    )
    with pytest.raises(RuntimeError, match="injected replay"):
        bundle.publish_fit_bundle(
            path, payload, expected_authority_sha256="a" * 64,
            require_production=False,
        )
    assert not path.exists()
