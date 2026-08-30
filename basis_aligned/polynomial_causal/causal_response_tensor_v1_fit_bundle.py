"""Sealed, model-free FIT bundle for causal-response tensor v1.

This module does not load rows, the model, or EVAL data.  It validates the complete
output of ``ObservedResponseCollector.fit_stage``, publishes it create-only only after
a private semantic replay, and returns an opaque capability.  Authority acquisition,
parent reconstruction, model execution, terminal receipts, and EVAL publication remain
separate lifecycle responsibilities.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
from collections.abc import Callable
from typing import Any, Mapping

import torch

from causal_response_tensor_collection import (
    STATISTIC_NAMES,
    validate_response_tensors,
)
from causal_response_tensor_v1_backend import (
    PHASES,
    PRODUCTION_COMPONENT_ORDER,
    PRODUCTION_SPEC_ORDER_SHA256,
    leading_shared_direction,
    tensor_sha256,
)


FIT_ROWS = 496
FIT_DOCUMENTS = 343
PRODUCTION_SOURCES = 49
PRODUCTION_POSITIONS = 256
PRODUCTION_LAYERS = 18
PRODUCTION_BATCH_SIZE = 4


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def logical_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


@dataclass(frozen=True)
class FitBundleBinding:
    authority_sha256: str
    source_closure_sha256: str
    census_state_diverse_sha256: str
    curated_rows_sha256: str
    battery_sha256: str
    document_split_sha256: str
    config_sha256: str
    weights_sha256: str
    model_state_sha256_before: str
    model_state_sha256_after: str
    model_rows_sha256: str
    fit_role_sha256: str
    support_hashes_sha256: str

    def validate(self, *, require_production: bool) -> None:
        values = asdict(self)
        if not all(_is_sha256(value) for value in values.values()):
            raise ValueError("every FIT bundle binding must be a lowercase SHA-256")
        if self.model_state_sha256_before != self.model_state_sha256_after:
            raise RuntimeError("model state changed across FIT collection")
        if require_production:
            expected = {
                "census_state_diverse_sha256":
                    "c785f3d938091253535aa4f613ab2b4107bf297c8d615da4f7eab4f8282f5e0b",
                "curated_rows_sha256":
                    "faaf89f38ddf1471234a1d30d978213367a566a9927bb3c73b274ab32afaa9dd",
                "battery_sha256":
                    "86d7ac72eeb95f9ec80a3e92ef65e28c0df66a36b9291d2d1d2d01f7bb6c5030",
                "document_split_sha256":
                    "3cb829ce5c9627f787e804e4e2ca44098030c629933f14df2c3a7fb07283317c",
                "config_sha256":
                    "428042bfd807ba36f8b4326395440fbbebe52cd3d040212e6fef14a4fdf2d83c",
                "weights_sha256":
                    "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3",
            }
            if any(values[name] != digest for name, digest in expected.items()):
                raise RuntimeError("a frozen causal-response parent hash changed")


def _require_exact_tensor(
    value: object,
    *,
    dtype: torch.dtype,
    shape: tuple[int, ...] | None = None,
    label: str,
) -> torch.Tensor:
    if type(value) is not torch.Tensor or value.dtype != dtype or (
        value.device.type != "cpu" or not value.is_contiguous()
    ):
        raise TypeError(f"{label} must be an exact contiguous CPU {dtype} tensor")
    if shape is not None and tuple(value.shape) != shape:
        raise ValueError(f"{label} has shape {tuple(value.shape)}, expected {shape}")
    if dtype.is_floating_point and not bool(torch.isfinite(value).all()):
        raise ValueError(f"{label} contains a nonfinite value")
    return value


def _require_plain_dict(value: object, *, label: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError(f"{label} must be a plain dict")
    return value


def _validate_order(
    tags: object, components: object, spec_order_sha256: object,
    *, require_production: bool,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if type(tags) is not list or type(components) is not list or (
        not tags or len(tags) != len(components)
        or any(type(tag) is not str or not tag for tag in tags)
        or any(type(component) is not str or not component for component in components)
        or len(set(tags)) != len(tags)
    ):
        raise ValueError("FIT source ordering is malformed")
    serialized = "".join(
        f"{component}\t{tag}\n" for component, tag in zip(components, tags)
    ).encode()
    observed = hashlib.sha256(serialized).hexdigest()
    if spec_order_sha256 != observed:
        raise RuntimeError("FIT circuit-order hash does not replay")
    if require_production and (
        len(tags) != PRODUCTION_SOURCES
        or observed != PRODUCTION_SPEC_ORDER_SHA256
    ):
        raise RuntimeError("FIT circuit ordering differs from the frozen production order")
    return tuple(tags), tuple(components)


def _validate_response(
    response: object,
    *,
    source_count: int,
    row_count: int,
    document_count: int,
) -> None:
    response = _require_plain_dict(response, label="FIT response")
    expected_keys = {
        "schema", "role", "row_indices", "document_ids", "member_count",
        "off_count", "statistics", "baseline_ce_mean", "validation",
    }
    if set(response) != expected_keys or (
        response.get("schema") != "causal_response_tensor_v1_role_preimage"
        or response.get("role") != "FIT"
    ):
        raise RuntimeError("FIT response schema or role changed")
    rows = _require_exact_tensor(
        response["row_indices"], dtype=torch.int64, shape=(row_count,),
        label="FIT row indices",
    )
    documents = _require_exact_tensor(
        response["document_ids"], dtype=torch.int64, shape=(document_count,),
        label="FIT document IDs",
    )
    if torch.unique(rows).numel() != row_count or (
        torch.unique(documents).numel() != document_count
    ):
        raise RuntimeError("FIT row or document identity is duplicated")
    if rows.min() < 0 or documents.min() < 0 or not torch.equal(
        rows, torch.sort(rows).values
    ) or not torch.equal(documents, torch.sort(documents).values):
        raise RuntimeError("FIT row and document identities must be nonnegative and sorted")
    counts_shape = (source_count, document_count)
    member_count = _require_exact_tensor(
        response["member_count"], dtype=torch.int64, shape=counts_shape,
        label="FIT member count",
    )
    off_count = _require_exact_tensor(
        response["off_count"], dtype=torch.int64, shape=counts_shape,
        label="FIT off count",
    )
    statistics = _require_plain_dict(response["statistics"], label="FIT statistics")
    replay = validate_response_tensors(
        statistics,
        member_count,
        off_count,
        expected_prefix=(len(PHASES), source_count, source_count),
        tolerance=1e-8,
    )
    if response["validation"] != replay:
        raise RuntimeError("FIT response validation receipt does not replay")
    baseline = response["baseline_ce_mean"]
    if type(baseline) is not float or not math.isfinite(baseline):
        raise ValueError("FIT baseline CE mean is malformed")


def _validate_directions_and_statistics(
    payload: Mapping[str, Any], tags: tuple[str, ...], components: tuple[str, ...]
) -> None:
    source_count = len(tags)
    width = payload["model_width"]
    if type(width) is not int or width <= 0:
        raise ValueError("FIT model width is malformed")
    directions = _require_exact_tensor(
        payload["directions"], dtype=torch.float32,
        shape=(len(PHASES), source_count, width), label="FIT directions",
    )
    norms = directions.double().norm(dim=-1)
    if not torch.allclose(norms, torch.ones_like(norms), atol=1e-5, rtol=0):
        raise RuntimeError("FIT directions are not unit normalized")

    component_order = tuple(dict.fromkeys(components))
    shared = _require_plain_dict(payload["shared_directions"], label="shared directions")
    spectra = _require_plain_dict(payload["singular_spectra"], label="singular spectra")
    gaps = _require_plain_dict(payload["relative_singular_gaps"], label="singular gaps")
    if set(shared) != set(component_order) or set(spectra) != set(component_order) or (
        set(gaps) != set(component_order)
    ):
        raise RuntimeError("FIT component SVD topology changed")

    counts = _require_plain_dict(payload["fit_counts"], label="FIT counts")
    write_stats = _require_plain_dict(
        payload["fit_write_statistics"], label="FIT write statistics"
    )
    residual_norms = _require_plain_dict(payload["residual_norms"], label="residual norms")
    full_norms = _require_plain_dict(
        payload["full_direction_norms"], label="full direction norms"
    )
    if set(counts) != set(tags) or set(write_stats) != set(tags) or (
        set(residual_norms) != set(tags) or set(full_norms) != set(tags)
    ):
        raise RuntimeError("FIT per-circuit direction topology changed")

    full_masters: dict[str, torch.Tensor] = {}

    for index, tag in enumerate(tags):
        count = _require_plain_dict(counts[tag], label=f"FIT counts {tag}")
        stats = _require_plain_dict(write_stats[tag], label=f"FIT write statistics {tag}")
        if set(count) != {"member_count", "off_count"} or (
            type(count["member_count"]) is not int
            or type(count["off_count"]) is not int
            or count["member_count"] <= 0
            or count["off_count"] <= 0
        ):
            raise RuntimeError("FIT direction support counts are malformed")
        if set(stats) != {"member_sum", "off_sum", "member_mean", "off_mean"}:
            raise RuntimeError("FIT write-statistic schema changed")
        for name in stats:
            _require_exact_tensor(
                stats[name], dtype=torch.float64, shape=(width,),
                label=f"{tag} {name}",
            )
        if not torch.equal(
            stats["member_sum"] / count["member_count"], stats["member_mean"]
        ) or not torch.equal(
            stats["off_sum"] / count["off_count"], stats["off_mean"]
        ):
            raise RuntimeError("FIT write means do not replay from sums and counts")
        contrast = stats["member_mean"] - stats["off_mean"]
        contrast_norm = float(contrast.norm())
        if contrast_norm == 0:
            raise RuntimeError("FIT direction contrast is absent")
        if type(full_norms[tag]) is not float or full_norms[tag] != contrast_norm:
            raise RuntimeError("FIT full-direction norm does not replay")
        full_masters[tag] = contrast / contrast.norm()
        expected_full = full_masters[tag].float()
        if not torch.equal(expected_full, directions[0, index]):
            raise RuntimeError("FIT full direction does not replay")

    for component in component_order:
        indices = [index for index, value in enumerate(components) if value == component]
        matrix = torch.stack([full_masters[tags[index]] for index in indices])
        expected_shared_master, expected_spectrum = leading_shared_direction(matrix)
        shared_value = _require_exact_tensor(
            shared[component], dtype=torch.float32, shape=(width,),
            label=f"shared direction {component}",
        )
        spectrum = _require_exact_tensor(
            spectra[component], dtype=torch.float64,
            shape=(min(len(indices), width),), label=f"singular spectrum {component}",
        )
        if not torch.equal(shared_value, expected_shared_master.float()) or not torch.equal(
            spectrum, expected_spectrum
        ):
            raise RuntimeError("FIT component SVD does not replay")
        expected_gap = float((spectrum[0] - spectrum[1]) / spectrum[0])
        if type(gaps[component]) is not float or gaps[component] != expected_gap:
            raise RuntimeError("FIT component singular gap does not replay")
        for index in indices:
            full_master = full_masters[tags[index]]
            remainder = full_master - (
                full_master @ expected_shared_master
            ) * expected_shared_master
            norm = float(remainder.norm())
            tag = tags[index]
            if type(residual_norms[tag]) is not float or residual_norms[tag] != norm:
                raise RuntimeError("FIT residual norm does not replay")
            expected_residual = (remainder / remainder.norm()).float()
            if not torch.equal(expected_residual, directions[1, index]):
                raise RuntimeError("FIT residual direction does not replay")


def _validate_ledger(
    ledger: object,
    *,
    tags: tuple[str, ...],
    components: tuple[str, ...],
    rows: int,
    batch_size: int,
    layer_count: int,
) -> None:
    ledger = _require_plain_dict(ledger, label="FIT call ledger")
    expected_keys = {
        "outer_forwards", "attention_native_calls", "mlp_native_calls",
        "attention_calls_by_site", "mlp_calls_by_site", "projection_calls",
        "capture_calls", "projection_phases", "projection_source_tags",
        "projection_source_components", "projection_batch_indices",
        "projection_event_counts", "capture_components",
        "capture_batch_indices", "capture_event_counts",
    }
    if set(ledger) != expected_keys:
        raise RuntimeError("FIT call-ledger schema changed")
    batches = math.ceil(rows / batch_size)
    expected_outer = batches * (2 + len(PHASES) * len(tags))
    if ledger["outer_forwards"] != expected_outer or (
        ledger["attention_native_calls"] != expected_outer * layer_count
        or ledger["mlp_native_calls"] != expected_outer * layer_count
    ):
        raise RuntimeError("FIT outer/native call ledger does not close")
    expected_by_site = {str(site): expected_outer for site in range(layer_count)}
    if ledger["attention_calls_by_site"] != expected_by_site or (
        ledger["mlp_calls_by_site"] != expected_by_site
    ):
        raise RuntimeError("FIT per-site native call ledger does not close")
    capture_components = tuple(dict.fromkeys(components))
    if ledger["projection_phases"] != list(PHASES) or (
        ledger["projection_source_tags"] != list(tags)
        or ledger["projection_source_components"] != list(components)
        or ledger["projection_batch_indices"] != list(range(batches))
        or ledger["capture_components"] != list(capture_components)
        or ledger["capture_batch_indices"] != list(range(batches))
    ):
        raise RuntimeError("FIT structured event axes changed")
    projection_counts = _require_exact_tensor(
        ledger["projection_event_counts"], dtype=torch.int64,
        shape=(len(PHASES), len(tags), batches),
        label="FIT projection event counts",
    )
    capture_counts = _require_exact_tensor(
        ledger["capture_event_counts"], dtype=torch.int64,
        shape=(len(capture_components), batches),
        label="FIT capture event counts",
    )
    if not bool(torch.all(projection_counts == 1)) or not bool(
        torch.all(capture_counts == 1)
    ):
        raise RuntimeError("FIT structured event ledger does not close")
    expected_projection_by_component = {
        component: components.count(component) * len(PHASES) * batches
        for component in capture_components
    }
    expected_capture_by_component = {
        component: batches for component in capture_components
    }
    if ledger["projection_calls"] != expected_projection_by_component or (
        ledger["capture_calls"] != expected_capture_by_component
    ):
        raise RuntimeError("FIT aggregate event ledger does not close")


FIT_PAYLOAD_KEYS = {
    "schema", "claim_boundary", "binding", "sign_convention", "off_mask",
    "phases", "source_tags", "source_components", "target_tags",
    "model_layer_count", "model_width", "batch_size", "spec_order_sha256",
    "support_hashes", "directions", "shared_directions", "fit_counts",
    "fit_write_statistics", "full_direction_norms", "singular_spectra",
    "relative_singular_gaps",
    "residual_norms", "fit_response", "call_ledger",
    "tensor_hashes", "forbidden_payload_contract",
}

FORBIDDEN_PAYLOAD_CONTRACT = {
    "raw_tokens": False,
    "targets": False,
    "component_activations": False,
    "component_writes": False,
    "logits": False,
    "eval_rows": False,
    "eval_outcomes": False,
    "aggregate_fit_write_sums_and_means_only": True,
}


def _tensor_hash_map(value: object, *, prefix: str = "root") -> dict[str, str]:
    result: dict[str, str] = {}
    if type(value) is torch.Tensor:
        result[prefix] = tensor_sha256(value)
    elif type(value) is dict:
        for key in sorted(value):
            if type(key) is not str:
                raise TypeError("FIT bundle dictionary keys must be strings")
            result.update(_tensor_hash_map(value[key], prefix=f"{prefix}.{key}"))
    elif type(value) is list:
        for index, item in enumerate(value):
            result.update(_tensor_hash_map(item, prefix=f"{prefix}[{index}]"))
    return result


def build_fit_bundle_payload(
    preimage: Mapping[str, Any],
    binding: FitBundleBinding,
    *,
    require_production: bool = True,
) -> dict[str, Any]:
    """Clone and validate an internal backend preimage into a serializable bundle."""
    if type(preimage) is not dict or preimage.get("schema") != (
        "causal_response_tensor_v1_fit_preimage"
    ):
        raise TypeError("FIT backend preimage is malformed")
    binding.validate(require_production=require_production)
    if "_direction_preimage" not in preimage:
        raise RuntimeError("FIT backend preimage lacks its private direction proof")
    expected_preimage_keys = (
        FIT_PAYLOAD_KEYS
        - {"binding", "tensor_hashes", "forbidden_payload_contract"}
    ) | {"_direction_preimage"}
    if set(preimage) != expected_preimage_keys:
        raise RuntimeError("FIT backend preimage schema changed")
    # torch.save owns tensor storage after this deep clone; no caller alias survives.
    payload = {
        key: _clone_tree(item)
        for key, item in preimage.items()
        if key != "_direction_preimage"
    }
    payload["schema"] = "causal_response_tensor_v1_fit_bundle"
    payload["claim_boundary"] = (
        "FIT-only program and causal-response bundle. It contains no raw tokens, "
        "targets, activations, logits, EVAL rows, or EVAL outcomes."
    )
    payload["binding"] = asdict(binding)
    payload["forbidden_payload_contract"] = dict(FORBIDDEN_PAYLOAD_CONTRACT)
    payload["tensor_hashes"] = _tensor_hash_map(payload)
    validate_fit_bundle_payload(payload, require_production=require_production)
    return payload


def _clone_tree(value: Any) -> Any:
    if type(value) is torch.Tensor:
        return value.clone()
    if type(value) is dict:
        return {key: _clone_tree(item) for key, item in value.items()}
    if type(value) is list:
        return [_clone_tree(item) for item in value]
    if type(value) in (str, int, float, bool) or value is None:
        return value
    raise TypeError(f"FIT bundle contains unsupported value type {type(value)!r}")


def validate_fit_bundle_payload(
    payload: Mapping[str, Any], *, require_production: bool = True
) -> None:
    payload = _require_plain_dict(payload, label="FIT bundle")
    if set(payload) != FIT_PAYLOAD_KEYS or payload.get("schema") != (
        "causal_response_tensor_v1_fit_bundle"
    ):
        raise RuntimeError("FIT bundle schema changed")
    binding_raw = _require_plain_dict(payload["binding"], label="FIT binding")
    if set(binding_raw) != set(FitBundleBinding.__dataclass_fields__):
        raise RuntimeError("FIT binding schema changed")
    binding = FitBundleBinding(**binding_raw)
    binding.validate(require_production=require_production)
    if payload["forbidden_payload_contract"] != FORBIDDEN_PAYLOAD_CONTRACT:
        raise RuntimeError("FIT forbidden-payload contract changed")
    tensor_hashes = _require_plain_dict(payload["tensor_hashes"], label="tensor hashes")
    replay_hashes = _tensor_hash_map({
        key: value for key, value in payload.items() if key != "tensor_hashes"
    })
    if tensor_hashes != replay_hashes:
        raise RuntimeError("FIT tensor digest map does not replay")
    if payload["phases"] != list(PHASES) or payload["target_tags"] != (
        payload["source_tags"]
    ):
        raise RuntimeError("FIT phase or target ordering changed")
    tags, components = _validate_order(
        payload["source_tags"], payload["source_components"],
        payload["spec_order_sha256"], require_production=require_production,
    )
    if type(payload["model_layer_count"]) is not int or (
        type(payload["batch_size"]) is not int
    ):
        raise TypeError("FIT model/batch dimensions must be exact integers")
    if require_production and (
        payload["model_layer_count"] != PRODUCTION_LAYERS
        or payload["batch_size"] != PRODUCTION_BATCH_SIZE
    ):
        raise RuntimeError("FIT production model or batch dimension changed")
    support = _require_plain_dict(payload["support_hashes"], label="support hashes")
    if list(support) != list(tags):
        raise RuntimeError("FIT support-hash ordering changed")
    for tag in tags:
        item = _require_plain_dict(support[tag], label=f"support hashes {tag}")
        if set(item) != {"member_mask_sha256", "slice_mask_sha256"} or not all(
            _is_sha256(value) for value in item.values()
        ):
            raise RuntimeError("FIT support hash is malformed")
    if logical_sha256(support) != binding.support_hashes_sha256:
        raise RuntimeError("FIT support hashes do not match the authority binding")
    row_count = FIT_ROWS if require_production else int(
        payload["fit_response"]["row_indices"].numel()
    )
    document_count = FIT_DOCUMENTS if require_production else int(
        payload["fit_response"]["document_ids"].numel()
    )
    _validate_response(
        payload["fit_response"], source_count=len(tags), row_count=row_count,
        document_count=document_count,
    )
    if tensor_sha256(payload["fit_response"]["row_indices"]) != (
        binding.fit_role_sha256
    ):
        raise RuntimeError("FIT row role does not match the authority binding")
    if require_production and payload["model_width"] != 1_152:
        raise RuntimeError("FIT production model width changed")
    _validate_directions_and_statistics(payload, tags, components)
    _validate_ledger(
        payload["call_ledger"], tags=tags, components=components, rows=row_count,
        batch_size=payload["batch_size"], layer_count=payload["model_layer_count"],
    )


def semantic_replay_fit_bundle(
    path: Path,
    *,
    expected_authority_sha256: str,
    expected_artifact_sha256: str | None = None,
    require_production: bool = True,
) -> str:
    """Validate the exact stable bytes and return only their digest, never a program."""
    if not _is_sha256(expected_authority_sha256):
        raise ValueError("expected FIT authority hash is malformed")
    before = file_sha256(path)
    raw = path.read_bytes()
    after = file_sha256(path)
    digest = hashlib.sha256(raw).hexdigest()
    if before != after or digest != before:
        raise RuntimeError("FIT bundle changed during semantic reload")
    if expected_artifact_sha256 is not None and (
        not _is_sha256(expected_artifact_sha256)
        or digest != expected_artifact_sha256
    ):
        raise RuntimeError("FIT bundle artifact hash differs from the receipt binding")
    payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    validate_fit_bundle_payload(payload, require_production=require_production)
    if payload["binding"]["authority_sha256"] != expected_authority_sha256:
        raise RuntimeError("FIT bundle authority binding changed")
    return digest


def publish_fit_bundle(
    path: Path,
    payload: Mapping[str, Any],
    *,
    expected_authority_sha256: str,
    require_production: bool = True,
    before_link: Callable[[], None] | None = None,
) -> str:
    """Prepare, fsync, and replay privately before create-only publication."""
    path = path.resolve()
    validate_fit_bundle_payload(payload, require_production=require_production)
    if payload["binding"]["authority_sha256"] != expected_authority_sha256:
        raise RuntimeError("FIT publication authority does not match the bundle")
    if require_production and not callable(before_link):
        raise RuntimeError("production FIT publication requires an adjacent guard")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        # No terminal claim or final path exists before this complete semantic replay.
        temporary_sha256 = semantic_replay_fit_bundle(
            temporary,
            expected_authority_sha256=expected_authority_sha256,
            require_production=require_production,
        )
        if before_link is not None:
            before_link()
        os.link(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        return semantic_replay_fit_bundle(
            path,
            expected_authority_sha256=expected_authority_sha256,
            expected_artifact_sha256=temporary_sha256,
            require_production=require_production,
        )
    finally:
        temporary.unlink(missing_ok=True)
