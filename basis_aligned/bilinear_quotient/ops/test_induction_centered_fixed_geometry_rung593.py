#!/usr/bin/env python3
# BQLANE: cpu
"""Model-free owner tests for the prospective R593 instrument repair."""

from __future__ import annotations

import ast
import importlib.util
import inspect
import json
from pathlib import Path
import types

import numpy as np
import pytest


MODULE = Path(__file__).with_name("induction_centered_fixed_geometry_rung593.py")
RUNTIME = Path(__file__).with_name("induction_centered_fixed_geometry_rung593_runtime.py")
SPEC = importlib.util.spec_from_file_location("r593_owner", MODULE)
assert SPEC and SPEC.loader
r593 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(r593)


@pytest.fixture(scope="module")
def authority():
    return r593.load_authority()[1]


@pytest.fixture(scope="module")
def fit(authority):
    return r593.build_phase_manifest(authority, "FIT")


def valid_endpoint_arrays(call, token, support):
    b = int(call["batch_size"])
    arrays = {
        "tokens.npy": token.copy(),
        "logits.npy": np.zeros((b, r593.VOCAB), dtype="<f4"),
        "factor_e.npy": np.zeros((b, 4, 2), dtype="<f4"),
        "factor_u.npy": np.zeros((b, 4, 2, r593.RESIDUAL), dtype="<f4"),
        "support.npy": support.copy(),
        "factorized_equality_term.npy": np.zeros((b, 4, r593.RESIDUAL), dtype="<f4"),
        "native_full_attention_write.npy": np.zeros((b, 4, r593.RESIDUAL), dtype="<f4"),
        "independent_full_native_write.npy": np.zeros((b, 4, r593.RESIDUAL), dtype="<f4"),
        "native_equality_term.npy": np.zeros((b, 4, r593.RESIDUAL), dtype="<f8"),
        "native_non_equality_remainder.npy": np.zeros((b, 4, r593.RESIDUAL), dtype="<f8"),
        "native_head_write.npy": np.zeros((b, 4, r593.RESIDUAL), dtype="<f8"),
    }
    return arrays


def test_exact_authority_support_records_and_censuses(authority) -> None:
    fit_summary = r593.support_summary(authority, "FIT")
    select_summary = r593.support_summary(authority, "SELECT")
    assert fit_summary == {
        "records": 13_824, "true": 5_760, "false": 8_064,
        "record_sha256": "ad2e827af9d7fada09327aa27c9465173aa283ee918599bfd5cb5ee107f79d6a",
    }
    assert select_summary == {
        "records": 6_912, "true": 2_880, "false": 4_032,
        "record_sha256": "b33ebe9b6d971dd1d09cd3ab797b888703c718457e548ed8f2b244a6698397c9",
    }
    records = r593.support_records(authority)
    assert len(records) == 20_736
    assert r593.content_sha256(records) == "25a8b2e9c4cf2175c37f8aa08e3fd5b127397b441ab2e30d609b125bf03dcceb"


def test_real_zero_and_one_support_examples_are_bound_in_each_call(authority, fit) -> None:
    descriptors = []
    hist = {0: 0, 1: 0}
    endpoints = {row["endpoint_id"]: row for row in authority["endpoints"]}
    for call in (row for row in fit["calls"] if row["call_kind"] == "endpoint"):
        tokens = fit["token_arrays"][call["token_record_id"]]
        specs = [endpoints[value] for value in call["authority_row_ids"]]
        support = r593.expected_support_mask(specs, tokens)
        assert r593.sha256_bytes(support.tobytes(order="C")) == call["expected_support_sha256"]
        assert int(support.sum()) == call["expected_support_true_count"]
        assert int((~support).sum()) == call["expected_support_false_count"]
        values, counts = np.unique(support[:, 0].sum(axis=-1), return_counts=True)
        for value, count in zip(values, counts):
            hist[int(value)] += int(count)
        descriptors.append({
            "call_index": call["chunk_index"],
            "true_count": call["expected_support_true_count"],
            "false_count": call["expected_support_false_count"],
            "mask_sha256": call["expected_support_sha256"],
        })
    assert hist == {0: 288, 1: 1_440}
    assert r593.content_sha256(descriptors) == r593.SUPPORT_DESCRIPTOR_SHA256["FIT"]


def test_support_exact_mask_passes_and_true_false_flips_fail(authority, fit) -> None:
    call = next(row for row in fit["calls"] if row["call_kind"] == "endpoint")
    tokens = fit["token_arrays"][call["token_record_id"]]
    by_id = {row["endpoint_id"]: row for row in authority["endpoints"]}
    specs = [by_id[value] for value in call["authority_row_ids"]]
    expected = r593.expected_support_mask(specs, tokens)
    arrays = valid_endpoint_arrays(call, tokens, expected)
    cached = {"expected_support": expected}
    assert r593.evaluate_completed_call(call, arrays, tokens, cached=cached)[0] is None

    true_coordinate = tuple(np.argwhere(expected)[0])
    false_coordinate = tuple(np.argwhere(~expected)[0])
    for coordinate in (true_coordinate, false_coordinate):
        mutated = {name: value.copy() for name, value in arrays.items()}
        mutated["support.npy"][coordinate] = ~mutated["support.npy"][coordinate]
        predicate, details = r593.evaluate_completed_call(call, mutated, tokens, cached=cached)
        assert predicate == "factor_transport_failed"
        assert details["support_mismatch_count"] == 1

    all_true = {name: value.copy() for name, value in arrays.items()}
    all_true["support.npy"].fill(True)
    assert r593.evaluate_completed_call(call, all_true, tokens, cached=cached)[0] == "factor_transport_failed"


def test_complete_phase_support_reconstruction_rejects_one_bit(authority, tmp_path: Path) -> None:
    endpoints = [row for row in authority["endpoints"] if row["split"] == "FIT"]
    expected = r593.expected_support_mask(endpoints, r593.fixed_tokens(endpoints, "token_ids"))
    np.save(tmp_path / "support.npy", expected, allow_pickle=False)
    observed = r593.validate_phase_support_evidence(tmp_path, authority, "FIT")
    assert observed["support_true_count"] == 5_760
    assert observed["support_false_count"] == 8_064
    mutated = expected.copy()
    mutated[0, 0, 0] = ~mutated[0, 0, 0]
    np.save(tmp_path / "support.npy", mutated, allow_pickle=False)
    with pytest.raises(RuntimeError, match="differs from authority"):
        r593.validate_phase_support_evidence(tmp_path, authority, "FIT")


def test_actual_scale_float64_partition_and_live_absolute_falsifier() -> None:
    rng = np.random.default_rng(593_1)
    pattern = rng.normal(size=(32, 30)).astype("<f4")
    values = (8 * rng.normal(size=(32, 30, 128))).astype("<f4")
    projection = (0.055 * rng.normal(size=(1152, 128))).astype("<f4")
    support = rng.random(size=(32, 30)) < 0.2
    equality, remainder, full = r593.high_precision_native_decomposition(
        pattern, values, projection, support
    )
    assert all(value.dtype == np.dtype("<f8") for value in (equality, remainder, full))
    clean = float(np.max(np.abs(equality + remainder - full)))
    assert clean < 1e-10
    head_rms = float(np.sqrt(np.mean(np.square(full))))
    assert 24 < head_rms < 32
    planted = remainder.copy()
    planted[0, 0] += 2e-5
    assert float(np.max(np.abs(equality + planted - full))) > r593.TOLERANCE


def test_runtime_contract_uses_three_independent_double_contractions() -> None:
    source = RUNTIME.read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert "pattern64 = pattern[local, head, query].double()" in source
    assert "value64 = value[local, :, head].double()" in source
    assert "weight64 = attention.c_proj.weight[:, head * 128:(head + 1) * 128].double()" in source
    assert 'equality_head = torch.einsum("k,kd->d", pattern64 * mask, value64)' in source
    assert 'remainder_head = torch.einsum("k,kd->d", pattern64 * ~mask, value64)' in source
    assert 'full_head = torch.einsum("k,kd->d", pattern64, value64)' in source
    assert "remainder_head = full_head - equality_head" not in source
    eager_torch = [
        node for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        and any(alias.name == "torch" or alias.name.startswith("torch.") for alias in node.names)
    ]
    assert eager_torch == []


def test_exact_float64_evidence_prices_and_capacity() -> None:
    phase_bytes = {}
    for phase in ("FIT", "SELECT"):
        phase_bytes[phase] = sum(
            int(np.prod(item["shape"])) * np.dtype(item["dtype"]).itemsize
            for name, item in r593.phase_evidence_schema(phase).items() if name.endswith(".npy")
        )
    assert phase_bytes == {"FIT": 5_501_463_552, "SELECT": 2_750_731_776}
    chunk = sum(
        sum(int(np.prod(shape)) * dtype.itemsize for dtype, shape in
            r593.mandatory_call_shapes({"batch_size": 32, "call_kind": kind}).values())
        for kind in r593.DIRECTED_KINDS
    )
    assert chunk == 43_440_640
    assert sum(phase_bytes.values()) + chunk == 8_295_635_968
    assert r593.INITIAL_MINIMUM_FREE_BYTES == 9_455_639_040
    assert r593.SELECT_MINIMUM_FREE_BYTES == 3_954_175_488
    assert phase_bytes["SELECT"] + chunk + 1_160_003_072 == r593.SELECT_MINIMUM_FREE_BYTES


def test_invalid_prefix_in_place_truncation_has_zero_extra_data_peak(authority, fit) -> None:
    select = r593.build_phase_manifest(authority, "SELECT")
    endpoint_names = set(r593.StreamingPhaseStore.ENDPOINT_MAP.values())
    phase_bytes = {}
    for bundle in (fit, select):
        schema = r593.canonical_array_schema(bundle)
        endpoint_bounds = [0]
        directed_bounds = [0]
        for call in bundle["calls"]:
            if call["call_kind"] == "endpoint":
                endpoint_bounds.append(endpoint_bounds[-1] + int(call["batch_size"]))
            elif call["call_kind"] == "native":
                directed_bounds.append(directed_bounds[-1] + int(call["batch_size"]))
        full = sum(int(np.prod(shape)) * dtype.itemsize for dtype, shape in schema.values())
        phase_bytes[bundle["phase"]] = full
        for endpoint_rows in endpoint_bounds:
            retained = sum(
                endpoint_rows * int(np.prod(shape[1:])) * dtype.itemsize
                for name, (dtype, shape) in schema.items() if name in endpoint_names
            )
            assert retained <= full
        for directed_rows in directed_bounds:
            retained = sum(
                (shape[0] if name in endpoint_names else directed_rows)
                * int(np.prod(shape[1:])) * dtype.itemsize
                for name, (dtype, shape) in schema.items()
            )
            assert retained <= full
    source = inspect.getsource(r593.StreamingPhaseStore._compact_canonical_file)
    assert "open_memmap" not in source
    assert "stream.truncate(" in source
    assert phase_bytes["FIT"] + phase_bytes["SELECT"] + r593.LARGEST_CURRENT_CHUNK_DATA_BYTES == r593.MAXIMUM_STREAMING_DATA_BYTES


@pytest.mark.parametrize(
    ("boundary", "threshold"),
    (("model", 9_455_639_040), ("SELECT", 3_954_175_488)),
)
def test_capacity_equality_and_one_byte_failure(boundary, threshold) -> None:
    stat = lambda _path, value=threshold: types.SimpleNamespace(f_bavail=value, f_frsize=1)
    assert r593.require_free_space(Path("."), boundary=boundary, statvfs_function=stat)["available_bytes"] == threshold
    below = lambda _path, value=threshold - 1: types.SimpleNamespace(f_bavail=value, f_frsize=1)
    with pytest.raises(RuntimeError, match=f"{threshold - 1} < {threshold}"):
        r593.require_free_space(Path("."), boundary=boundary, statvfs_function=below)


def test_dryrun_preserves_science_price_and_closure() -> None:
    observed = r593.build_dryrun()
    assert observed["phase_counts"] == r593.PHASE_COUNTS
    assert observed["registered_max_model_forwards"] == 961
    assert observed["model_forwards"] == observed["model_backwards"] == 0
    assert observed["model_weights_updated"] is False
    assert not any(observed[key] for key in ("select_opened", "final_opened", "ood_opened"))
    assert observed["high_precision_decomposition_fixture"]["identity_max_abs"] <= r593.TOLERANCE
    assert observed["high_precision_decomposition_fixture"]["planted_2e_minus_5_max_abs"] > r593.TOLERANCE


def test_new_namespaces_and_invalid_receipt_provenance(tmp_path: Path) -> None:
    assert all("rung593" in path.name for path in r593.PUBLIC_NAMESPACES)
    stage = tmp_path / "stage"
    (stage / "evidence").mkdir(parents=True)
    manifest = [{"call_id": "FIT:test"}]
    provenance = {
        "implementation_sha256": "a" * 64,
        "adapter_sha256": "b" * 64,
        "runtime_sha256": "c" * 64,
        "checkpoint_weights_sha256": "d" * 64,
    }
    observed = r593.publish_invalid_prefix(
        stage, manifest, manifest, "factor_transport_failed", {},
        public_root=tmp_path, provenance=provenance,
    )
    assert observed["provenance"] == provenance
    receipt = json.loads((tmp_path / r593.INVALID_RECEIPT.name).read_text(encoding="utf-8"))
    assert receipt["provenance"] == provenance
