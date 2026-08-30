"""Canonical, model-free reconstruction of causal-response v1 FIT inputs.

There is deliberately no public production executor or Python pseudo-capability here.
The source-closed lifecycle calls the private helper only after frozen authority, and
passes an adjacent authority guard that is checked before and after parent access.  The
helper never loads bilin18, computes a forward pass, or reads a response outcome.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from collections.abc import Callable
from typing import Any, NamedTuple

import torch

from causal_response_tensor_split import SPLIT_SEED, document_side
from causal_response_tensor_v1_backend import (
    CircuitSpec,
    PRODUCTION_COMPONENT_ORDER,
    PRODUCTION_SPEC_ORDER_SHA256,
    tensor_sha256,
)


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BQ = ROOT / "basis_aligned" / "bilinear_quotient"

CENSUS = BQ / "census_state_diverse.pt"
CURATED = BQ / "curated_rows.pt"
BATTERY = BQ / "circuits" / "BATTERY.json"
SPLIT = HERE / "causal_response_tensor_document_split.json"

PARENT_SHA256S = {
    "census_state_diverse":
        "c785f3d938091253535aa4f613ab2b4107bf297c8d615da4f7eab4f8282f5e0b",
    "curated_rows":
        "faaf89f38ddf1471234a1d30d978213367a566a9927bb3c73b274ab32afaa9dd",
    "battery":
        "86d7ac72eeb95f9ec80a3e92ef65e28c0df66a36b9291d2d1d2d01f7bb6c5030",
    "split":
        "3cb829ce5c9627f787e804e4e2ca44098030c629933f14df2c3a7fb07283317c",
}
MODEL_ROWS_SHA256 = "1786a30bc0d27d26324486e582a539cc292428c2f3f4f1ed7594014390a437ce"
FIT_ROLE_SHA256 = "6873c2a279bf73fe17c38d72ac25003f4741825efc271ff91b6b783615cdd815"
FIT_DOCUMENT_IDS_SHA256 = (
    "0f514805a7615e5ef3fe862eb8bf37bebfe8c57b8b7e781fbb25907c729b808d"
)
SUPPORT_HASHES_SHA256 = (
    "a8e033d981e82b5e39404ed5ee705119897e1d5d5a1cceaf80ea12c0b711a5aa"
)


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


def _stable_bytes(path: Path, expected_sha256: str) -> bytes:
    before = file_sha256(path)
    if before != expected_sha256:
        raise RuntimeError(f"frozen FIT parent hash changed: {path}")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256 or (
        file_sha256(path) != expected_sha256
    ):
        raise RuntimeError(f"frozen FIT parent changed while reading: {path}")
    return raw


def _load_torch_parent(path: Path, expected_sha256: str) -> dict[str, Any]:
    value = torch.load(
        io.BytesIO(_stable_bytes(path, expected_sha256)),
        map_location="cpu",
        weights_only=True,
    )
    if type(value) is not dict:
        raise RuntimeError(f"frozen FIT tensor parent is not a plain dict: {path}")
    return value


def _load_json_parent(path: Path, expected_sha256: str) -> dict[str, Any]:
    value = json.loads(_stable_bytes(path, expected_sha256))
    if type(value) is not dict:
        raise RuntimeError(f"frozen FIT JSON parent is not a plain dict: {path}")
    return value


def _require_tensor(
    value: object, *, dtype: torch.dtype, shape: tuple[int, ...], label: str
) -> torch.Tensor:
    if type(value) is not torch.Tensor or value.dtype != dtype or (
        value.device.type != "cpu" or not value.is_contiguous()
        or tuple(value.shape) != shape
    ):
        raise RuntimeError(f"{label} tensor contract changed")
    return value


def _index_mask(indices: torch.Tensor, *, size: int, label: str) -> torch.Tensor:
    if type(indices) is not torch.Tensor or indices.dtype != torch.int64 or (
        indices.device.type != "cpu" or not indices.is_contiguous()
        or indices.ndim != 1 or indices.numel() == 0
    ):
        raise RuntimeError(f"{label} indices are malformed")
    if indices.min() < 0 or indices.max() >= size or (
        torch.unique(indices).numel() != indices.numel()
    ):
        raise RuntimeError(f"{label} indices are duplicated or out of range")
    mask = torch.zeros(size, dtype=torch.bool)
    mask[indices] = True
    return mask


def _validate_split_metadata(split: dict[str, Any]) -> None:
    expected_keys = {
        "schema", "claim_boundary", "split_rule", "split_seed", "parents",
        "rows_exactly_match", "rows", "positions_per_row", "unique_source_documents",
        "fit_source_documents", "eval_source_documents",
        "cross_role_document_overlap", "components", "total_usable_circuits",
        "minimum_member_document_support_each_role", "by_component", "runtime_seconds",
    }
    parents = split.get("parents")
    if set(split) != expected_keys or split.get("schema") != (
        "causal_response_tensor_document_split_v1"
    ) or split.get("split_seed") != SPLIT_SEED or split.get("rows") != 1_000 or (
        split.get("positions_per_row") != 256
        or split.get("unique_source_documents") != 688
        or split.get("fit_source_documents") != 343
        or split.get("eval_source_documents") != 345
        or split.get("cross_role_document_overlap") != 0
        or split.get("rows_exactly_match") is not True
        or split.get("components") != list(PRODUCTION_COMPONENT_ORDER)
        or split.get("total_usable_circuits") != 49
        or split.get("minimum_member_document_support_each_role") != 149
        or type(parents) is not dict
        or parents.get("census_state_sha256") != PARENT_SHA256S["census_state_diverse"]
        or parents.get("curated_rows_sha256") != PARENT_SHA256S["curated_rows"]
        or parents.get("battery_sha256") != PARENT_SHA256S["battery"]
    ):
        raise RuntimeError("frozen FIT split metadata changed")


class FitCollectorInputs(NamedTuple):
    rows: torch.Tensor
    row_document_ids: torch.Tensor
    fit_row_indices: torch.Tensor
    specs: tuple[CircuitSpec, ...]
    parent_sha256s: dict[str, str]
    model_rows_sha256: str
    fit_role_sha256: str
    fit_document_ids_sha256: str
    spec_order_sha256: str
    support_hashes: dict[str, dict[str, str]]


def _reconstruct_production_fit_inputs_after_authority(
    authority_guard: Callable[[], None],
) -> FitCollectorInputs:
    """Reconstruct frozen FIT inputs between two lifecycle authority checks."""
    if not callable(authority_guard):
        raise TypeError("FIT reconstruction requires a lifecycle authority guard")
    authority_guard()
    state = _load_torch_parent(CENSUS, PARENT_SHA256S["census_state_diverse"])
    curated = _load_torch_parent(CURATED, PARENT_SHA256S["curated_rows"])
    battery = _load_json_parent(BATTERY, PARENT_SHA256S["battery"])
    split = _load_json_parent(SPLIT, PARENT_SHA256S["split"])
    _validate_split_metadata(split)

    if set(state) != {"rows", "basev", "leaves"} or set(curated) != {
        "rows", "docid"
    } or set(battery) != {
        "schema_version", "generated", "method", "seed", "state",
        "grid_positions", "note", "by_tag", "unusable",
    }:
        raise RuntimeError("frozen FIT parent schema changed")
    rows = _require_tensor(
        state["rows"], dtype=torch.int64, shape=(1_000, 513), label="census rows"
    )
    curated_rows = _require_tensor(
        curated["rows"], dtype=torch.int64, shape=(1_000, 513), label="curated rows"
    )
    documents = _require_tensor(
        curated["docid"], dtype=torch.int64, shape=(1_000,), label="document IDs"
    )
    _require_tensor(
        state["basev"], dtype=torch.float32, shape=(256_000,), label="base vector"
    )
    if not torch.equal(rows, curated_rows) or torch.unique(documents).numel() != 688:
        raise RuntimeError("census/curated rows or source-document identity changed")

    fit_document_set = {
        int(document) for document in torch.unique(documents).tolist()
        if document_side(int(document), SPLIT_SEED) == "FIT"
    }
    fit_rows = torch.tensor([
        index for index, document in enumerate(documents.tolist())
        if int(document) in fit_document_set
    ], dtype=torch.int64)
    if fit_rows.shape != (496,) or len(fit_document_set) != 343 or (
        torch.unique(documents[fit_rows]).numel() != 343
    ):
        raise RuntimeError("canonical FIT role does not match the frozen split")

    leaves_raw = state["leaves"]
    by_tag = battery["by_tag"]
    if type(leaves_raw) is not list or type(by_tag) is not dict:
        raise RuntimeError("circuit inventory parent topology changed")
    leaves: dict[str, dict[str, Any]] = {}
    for leaf in leaves_raw:
        if type(leaf) is not dict or set(leaf) != {
            "tag", "repl", "class_r2", "n_members", "top_probes",
            "member", "score", "slice",
        } or type(leaf["tag"]) is not str or leaf["tag"] in leaves:
            raise RuntimeError("census leaf schema or tag identity changed")
        leaves[leaf["tag"]] = leaf

    specs: list[CircuitSpec] = []
    by_component = split["by_component"]
    if type(by_component) is not dict or set(by_component) != set(
        PRODUCTION_COMPONENT_ORDER
    ):
        raise RuntimeError("split component inventory changed")
    position_documents = documents[:, None].expand(-1, 256).reshape(-1)
    eval_document_set = set(map(int, torch.unique(documents).tolist())) - fit_document_set
    for component in PRODUCTION_COMPONENT_ORDER:
        tags = sorted(
            tag for tag, entry in by_tag.items()
            if type(entry) is dict and entry.get("best_mean") == component and tag in leaves
        )
        component_receipt = by_component[component]
        if type(component_receipt) is not dict or (
            component_receipt.get("usable_circuit_count") != len(tags)
            or set(component_receipt.get("circuits", {})) != set(tags)
        ):
            raise RuntimeError("split circuit inventory changed")
        for tag in tags:
            leaf = leaves[tag]
            member = _index_mask(leaf["member"], size=256_000, label=f"{tag} member")
            slice_mask = _index_mask(
                leaf["slice"], size=256_000, label=f"{tag} slice"
            )
            if not torch.all(slice_mask[member]) or bool(slice_mask.all()):
                raise RuntimeError("circuit member/slice relation changed")
            member_documents = set(map(int, position_documents[member].tolist()))
            receipt = component_receipt["circuits"][tag]
            if type(receipt) is not dict or receipt != {
                "member_positions": int(member.sum()),
                "member_unique_documents": len(member_documents),
                "fit_member_documents": len(member_documents & fit_document_set),
                "eval_member_documents": len(member_documents & eval_document_set),
            }:
                raise RuntimeError("split circuit support receipt does not replay")
            specs.append(CircuitSpec(tag, component, member, slice_mask))

    serialized = "".join(
        f"{spec.component}\t{spec.tag}\n" for spec in specs
    ).encode()
    order_sha = hashlib.sha256(serialized).hexdigest()
    if len(specs) != 49 or order_sha != PRODUCTION_SPEC_ORDER_SHA256:
        raise RuntimeError("canonical FIT circuit order does not replay")
    support_hashes = {
        spec.tag: {
            "member_mask_sha256": tensor_sha256(spec.member_mask),
            "slice_mask_sha256": tensor_sha256(spec.slice_mask),
        }
        for spec in specs
    }
    role_sha = tensor_sha256(fit_rows)
    # The frozen census scores the first 256 next-token positions from columns
    # 0..256; columns 257..512 are outside every registered mask.
    model_rows = rows[:, :257].contiguous()
    fit_document_ids = torch.unique(documents[fit_rows], sorted=True)
    observed_identities = {
        "model_rows_sha256": tensor_sha256(model_rows),
        "fit_role_sha256": role_sha,
        "fit_document_ids_sha256": tensor_sha256(fit_document_ids),
        "support_hashes_sha256": logical_sha256(support_hashes),
    }
    expected_identities = {
        "model_rows_sha256": MODEL_ROWS_SHA256,
        "fit_role_sha256": FIT_ROLE_SHA256,
        "fit_document_ids_sha256": FIT_DOCUMENT_IDS_SHA256,
        "support_hashes_sha256": SUPPORT_HASHES_SHA256,
    }
    if observed_identities != expected_identities:
        raise RuntimeError("canonical FIT input identity does not match amendment 2")
    result = FitCollectorInputs(
        model_rows.clone(), documents.clone(), fit_rows.clone(),
        tuple(CircuitSpec(
        spec.tag, spec.component, spec.member_mask.clone(), spec.slice_mask.clone()
        ) for spec in specs),
        dict(PARENT_SHA256S), observed_identities["model_rows_sha256"], role_sha,
        observed_identities["fit_document_ids_sha256"], order_sha,
        {tag: dict(value) for tag, value in support_hashes.items()},
    )
    authority_guard()
    return result
