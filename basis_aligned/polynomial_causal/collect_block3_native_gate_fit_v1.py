#!/usr/bin/env python3
"""Receipt-last fit-statistic collector for the block-3 native-gate subset assay."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time
from typing import Any, Mapping

import torch
import torch.nn.functional as F

ROOT = Path("/workspace/tensor_language")
sys.path.insert(0, str(ROOT))

import bilin18_observed_model_facade as facade
from grouped_block_coefficient_screen import balance_product_gauge
import native_gate_subset as subset


HERE = ROOT / "basis_aligned" / "polynomial_causal"
BQ = ROOT / "basis_aligned" / "bilinear_quotient"
PREREGISTRATION = HERE / "BLOCK3_NATIVE_GATE_SUBSET_V1_PREREGISTRATION.md"
FIT_ROWS = BQ / ".rowcache/fineweb_n480_skip80.pt"
ROW_RECEIPT = BQ / ".rowcache/fineweb_oracle_v2_receipt.json"
ROW_RECEIPT_SHA256 = "815b21618c2e477e8cbda17ce94bf01862017a9936e4ee03acaa6cd7256cba16"
FIT_ROWS_FILE_SHA256 = "2acf75382486988a1e124a1a575ef3230af43aa1b1507d80dee02eefc7bba496"
FIT_ROWS_RAW_SHA256 = "343d92ce07f78572e3233120d3361814c63f69fa76e97e58b62d1d6c8f24497f"
AUTHORITY = HERE / "block3_native_gate_fit_v1_authority.json"
PAYLOAD = HERE / "block3_native_gate_fit_v1_payload.pt"
RECEIPT = HERE / "block3_native_gate_fit_v1_receipt.json"
FAILURE = HERE / "block3_native_gate_fit_v1_failure.json"
LOCK = Path("/workspace/runs/.block3_native_gate_fit_v1.lock")
SOURCE_PATHS = (
    "basis_aligned/polynomial_causal/BLOCK3_NATIVE_GATE_SUBSET_V1_PREREGISTRATION.md",
    "basis_aligned/polynomial_causal/collect_block3_native_gate_fit_v1.py",
    "basis_aligned/polynomial_causal/test_collect_block3_native_gate_fit_v1.py",
    "basis_aligned/polynomial_causal/fit_block3_native_gate_subset_v1.py",
    "basis_aligned/polynomial_causal/test_fit_block3_native_gate_subset_v1.py",
    "basis_aligned/polynomial_causal/native_gate_subset.py",
    "basis_aligned/polynomial_causal/test_native_gate_subset.py",
    "basis_aligned/polynomial_causal/grouped_block_coefficient_screen.py",
    "basis_aligned/polynomial_causal/test_grouped_block_coefficient_screen.py",
    "basis_aligned/polynomial_causal/bilin18_observed_model_facade.py",
    "jacclust/__init__.py",
    "jacclust/tt_model.py",
)
LAYER = 3
WIDTH = 1152
GATES = 4608
ROW_COUNT = 480
ROW_WIDTH = 513
MODEL_TOKEN_COUNT = 256
POSITION_START = 64
POSITION_STOP = 256
TOKENS_PER_ROW = POSITION_STOP - POSITION_START
PREFILTER = 1024
BATCH_SIZE = 8
PASSES = 2
DEVICE = "cuda"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(
        value.detach().cpu().contiguous().numpy().tobytes(order="C")
    ).hexdigest()


def logical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def create_json(path: Path, value: Mapping[str, Any]) -> None:
    """Direct O_EXCL publication: a race can fail, never overwrite."""

    data = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def create_torch(path: Path, value: Mapping[str, Any]) -> None:
    """Direct O_EXCL torch publication; a partial file spends the namespace."""

    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            torch.save(dict(value), handle)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


@dataclass(frozen=True, slots=True)
class RunClaim:
    path: Path
    descriptor: int
    device: int
    inode: int
    nonce: str

    def verify(self) -> None:
        descriptor_stat = os.fstat(self.descriptor)
        path_stat = os.stat(self.path)
        if (descriptor_stat.st_dev, descriptor_stat.st_ino) != (
            self.device, self.inode
        ) or (path_stat.st_dev, path_stat.st_ino) != (self.device, self.inode) or (
            self.path.read_text(encoding="ascii") != self.nonce + "\n"
        ):
            raise RuntimeError("collector run claim was replaced or altered")

    def release(self) -> None:
        self.verify()
        os.close(self.descriptor)
        self.path.unlink()
        _fsync_directory(self.path.parent)


def acquire_claim(path: Path) -> RunClaim:
    path.parent.mkdir(parents=True, exist_ok=True)
    nonce = secrets.token_hex(32)
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
    os.write(descriptor, (nonce + "\n").encode("ascii"))
    os.fsync(descriptor)
    stat = os.fstat(descriptor)
    claim = RunClaim(path, descriptor, stat.st_dev, stat.st_ino, nonce)
    claim.verify()
    _fsync_directory(path.parent)
    return claim


def source_closure() -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
        text=True, stdout=subprocess.PIPE,
    ).stdout.strip()
    hashes: dict[str, str] = {}
    for relative in SOURCE_PATHS:
        blob = subprocess.run(
            ["git", "show", f"{commit}:{relative}"], cwd=ROOT,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        if blob.returncode != 0:
            raise RuntimeError(f"source is not committed at frozen HEAD: {relative}")
        committed_sha256 = hashlib.sha256(blob.stdout).hexdigest()
        live_sha256 = file_sha256(ROOT / relative)
        if live_sha256 != committed_sha256:
            raise RuntimeError(f"live source differs from frozen commit: {relative}")
        hashes[relative] = committed_sha256
    remote = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "origin/main"], cwd=ROOT,
    )
    if remote.returncode != 0:
        raise RuntimeError("collector HEAD is not pushed to origin/main")
    payload = {"commit": commit, "paths": hashes}
    return {**payload, "sha256": logical_sha256(payload)}


def verify_source_closure(source: Mapping[str, Any]) -> None:
    if not isinstance(source, Mapping) or set(source) != {"commit", "paths", "sha256"} or (
        logical_sha256({"commit": source["commit"], "paths": source["paths"]}) != source["sha256"]
    ) or set(source["paths"]) != set(SOURCE_PATHS):
        raise RuntimeError("source closure is malformed")
    for relative, expected in source["paths"].items():
        if file_sha256(ROOT / relative) != expected:
            raise RuntimeError(f"source changed during collection: {relative}")


def validate_row_provenance() -> dict[str, Any]:
    before = file_sha256(ROW_RECEIPT)
    if before != ROW_RECEIPT_SHA256:
        raise RuntimeError("canonical row receipt bytes changed")
    raw = json.loads(ROW_RECEIPT.read_text())
    if raw.get("authority") != "pinned_local_ordered_manifest" or raw.get(
        "authorized_for_scored_experiments"
    ) is not True or raw.get("entries", {}).get("n480_skip80", {}).get(
        "tensor_raw_sha256"
    ) != FIT_ROWS_RAW_SHA256:
        raise RuntimeError("canonical fit row authority changed")
    names = ("n480_skip80", "n192_skip7000", "n192_skip11000")
    expected_counts = {"n480_skip80": 480, "n192_skip7000": 192, "n192_skip11000": 192}
    documents: dict[str, frozenset[str]] = {}
    provenance_hashes: dict[str, str] = {}
    for name in names:
        records = raw.get("document_provenance", {}).get("sets", {}).get(name)
        if not isinstance(records, list) or len(records) != expected_counts[name] or any(
            not isinstance(record, dict) or not isinstance(record.get("document_id"), str) or not (
                record["document_id"]
            ) for record in records
        ):
            raise RuntimeError(f"canonical provenance is malformed for {name}")
        documents[name] = frozenset(record["document_id"] for record in records)
        provenance_hashes[name] = logical_sha256(records)
    intersections = {
        "fit_validation": len(documents["n480_skip80"] & documents["n192_skip7000"]),
        "fit_replication": len(documents["n480_skip80"] & documents["n192_skip11000"]),
        "validation_replication": len(documents["n192_skip7000"] & documents["n192_skip11000"]),
    }
    if any(intersections.values()) or file_sha256(ROW_RECEIPT) != before:
        raise RuntimeError("row provenance roles overlap or receipt changed during read")
    return {
        "receipt_path": str(ROW_RECEIPT.resolve()),
        "receipt_sha256": before,
        "authority": raw["authority"],
        "ordered_manifest_sha256": raw["ordered_manifest_local_parquet_identity_gate"][
            "ordered_manifest_sha256"
        ],
        "entry_provenance_sha256s": provenance_hashes,
        "document_counts": {name: len(documents[name]) for name in names},
        "pairwise_document_intersections": intersections,
        "disjointness_sha256": logical_sha256({
            name: sorted(documents[name]) for name in names
        }),
    }


def validate_rows() -> tuple[torch.Tensor, dict[str, Any]]:
    provenance = validate_row_provenance()
    before = file_sha256(FIT_ROWS)
    if before != FIT_ROWS_FILE_SHA256:
        raise RuntimeError("fit row serialized bytes changed")
    rows = torch.load(FIT_ROWS, map_location="cpu", weights_only=True)
    if not torch.is_tensor(rows) or rows.dtype != torch.long or tuple(rows.shape) != (
        ROW_COUNT, ROW_WIDTH
    ) or not rows.is_contiguous() or tensor_sha256(rows) != FIT_ROWS_RAW_SHA256 or (
        file_sha256(FIT_ROWS) != before
    ):
        raise RuntimeError("fit row tensor changed")
    return rows, provenance


def verify_rows_unchanged(binding: Mapping[str, Any], rows: torch.Tensor) -> None:
    if validate_row_provenance() != dict(binding) or file_sha256(FIT_ROWS) != (
        FIT_ROWS_FILE_SHA256
    ) or tensor_sha256(rows) != FIT_ROWS_RAW_SHA256:
        raise RuntimeError("row authority or tensor changed during collection")


def _feature_bank(
    balanced_left: torch.Tensor, balanced_right: torch.Tensor,
    u: torch.Tensor, v: torch.Tensor,
) -> dict[str, torch.Tensor]:
    lu = F.linear(u, balanced_left)
    ru = F.linear(u, balanced_right)
    lv = F.linear(v, balanced_left)
    rv = F.linear(v, balanced_right)
    return {"uu": lu * ru, "uv": lu * rv, "vu": lv * ru, "vv": lv * rv}


def first_pass_statistics(
    u: torch.Tensor, v: torch.Tensor, features: Mapping[str, torch.Tensor],
    down: torch.Tensor,
) -> dict[str, torch.Tensor | int]:
    if u.shape != v.shape or u.ndim != 2 or u.shape[1] != WIDTH or set(
        features
    ) != set(subset.TERM_NAMES):
        raise ValueError("first-pass typed batch is malformed")
    z = u + v
    return {
        "count": len(u),
        "u_second": u.T @ u,
        "v_second": v.T @ v,
        "z_second": z.T @ z,
        "energy": subset.contribution_energy(features, down),
    }


def second_pass_statistics(
    features: Mapping[str, torch.Tensor], down: torch.Tensor,
    indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    if indices.ndim != 1 or indices.dtype != torch.long or len(indices) != PREFILTER:
        raise ValueError("second-pass prefilter changed")
    ordered = [features[name].reshape(-1, GATES) for name in subset.TERM_NAMES]
    selected = torch.cat([value[:, indices] for value in ordered], dim=0)
    writes = torch.cat([F.linear(value, down) for value in ordered], dim=0)
    gram, cross = subset.sufficient_statistics(selected, writes)
    _unused, permuted_cross = subset.sufficient_statistics(selected, writes.flip(0))
    return gram, cross, permuted_cross, writes.square().sum(), len(selected)


@dataclass(slots=True)
class MeasuredCalls:
    attention: list[int]
    mlp: list[int]
    outer_returned: int = 0
    explicit_typed_down_banks: int = 0

    @classmethod
    def empty(cls) -> "MeasuredCalls":
        return cls(attention=[0] * 18, mlp=[0] * 18)

    def receipt(self) -> dict[str, Any]:
        return {
            "attention_calls_by_site": {str(i): value for i, value in enumerate(self.attention)},
            "mlp_calls_by_site": {str(i): value for i, value in enumerate(self.mlp)},
            "outer_returned": self.outer_returned,
            "explicit_typed_down_banks": self.explicit_typed_down_banks,
        }


def model_parameter_metadata_sha256(model: torch.nn.Module) -> str:
    payload = []
    for name, parameter in model.named_parameters():
        payload.append({
            "name": name,
            "shape": list(parameter.shape),
            "dtype": str(parameter.dtype),
            "device": str(parameter.device),
            "requires_grad": parameter.requires_grad,
            "version": parameter._version,
            "data_ptr": parameter.data_ptr(),
        })
    return logical_sha256(payload)


def block0_through_3_state_sha256(model: torch.nn.Module) -> str:
    """Content hash every tensor that can affect the collected block-3 trajectory."""

    digest = hashlib.sha256()
    modules = [("transformer.wte", model.transformer.wte)] + [
        (f"transformer.h.{site}", model.transformer.h[site])
        for site in range(LAYER + 1)
    ]
    seen: set[int] = set()
    for module_prefix, module in modules:
        tensors = list(module.named_parameters(recurse=True)) + list(
            module.named_buffers(recurse=True)
        )
        for relative, value in sorted(tensors, key=lambda item: item[0]):
            identity = id(value)
            if identity in seen:
                continue
            seen.add(identity)
            name = f"{module_prefix}.{relative}" if relative else module_prefix
            contiguous = value.detach().cpu().contiguous()
            header = json.dumps({
                "name": name,
                "shape": list(contiguous.shape),
                "dtype": str(contiguous.dtype),
            }, sort_keys=True, separators=(",", ":")).encode()
            digest.update(len(header).to_bytes(8, "little"))
            digest.update(header)
            digest.update(contiguous.numpy().tobytes(order="C"))
    return digest.hexdigest()


def block3_factor_sha256s(model: torch.nn.Module) -> dict[str, str]:
    mlp = model.transformer.h[LAYER].mlp
    return {
        "left": tensor_sha256(mlp.Left.weight),
        "right": tensor_sha256(mlp.Right.weight),
        "down": tensor_sha256(mlp.Down.weight),
        "bias": tensor_sha256(mlp.Down_bias),
    }


@torch.no_grad()
def block3_typed_variables(
    model: torch.nn.Module, tokens: torch.Tensor, calls: MeasuredCalls | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if tokens.dtype != torch.long or tokens.ndim != 2 or tokens.shape[1] != (
        MODEL_TOKEN_COUNT
    ):
        raise ValueError("collector tokens changed")
    x = F.rms_norm(model.transformer.wte(tokens), (WIDTH,))
    x0 = x
    first_value = None
    for site in range(LAYER + 1):
        block = model.transformer.h[site]
        h = block.lambdas[0] * x + block.lambdas[1] * x0
        attention_write, first_value = block.attn(F.rms_norm(h, (WIDTH,)), first_value)
        if calls is not None:
            calls.attention[site] += 1
        post = h + attention_write
        z = F.rms_norm(post, (WIDTH,))
        if site == LAYER:
            epsilon = torch.finfo(post.dtype).eps
            gamma = torch.rsqrt(post.square().mean(dim=-1, keepdim=True) + epsilon)
            u = gamma * h
            v = gamma * attention_write
            replay = float((u + v - z).abs().max())
            selection = (..., slice(POSITION_START, POSITION_STOP), slice(None))
            if calls is not None:
                calls.outer_returned += 1
            return u[selection], v[selection], z[selection], replay
        x = post + block.mlp(z)
        if calls is not None:
            calls.mlp[site] += 1
    raise AssertionError("block-3 collector loop did not return")


def _authority(
    source: Mapping[str, Any], checkpoint: facade.CheckpointReceipt,
    row_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema": "block3_native_gate_fit_v1_authority",
        "status": "frozen_before_any_block3_fit_tensor_outcome",
        "authorized_for_evaluation": False,
        "authorized_for_global_ledger_credit": False,
        "source_closure": dict(source),
        "preregistration_sha256": file_sha256(PREREGISTRATION),
        "fit_rows": {
            "path": str(FIT_ROWS.resolve()),
            "file_sha256": FIT_ROWS_FILE_SHA256,
            "raw_sha256": FIT_ROWS_RAW_SHA256,
            "shape": [ROW_COUNT, ROW_WIDTH],
            "provenance": dict(row_provenance),
        },
        "checkpoint": asdict(checkpoint),
        "layer": LAYER,
        "width": WIDTH,
        "gates": GATES,
        "positions_half_open": [POSITION_START, POSITION_STOP],
        "fit_token_count": ROW_COUNT * TOKENS_PER_ROW,
        "passes": PASSES,
        "prefilter": PREFILTER,
        "batch_size": BATCH_SIZE,
        "device": DEVICE,
        "selection_after_first_pass": "stable_descending_contribution_energy_gate_index_tiebreak",
        "fit_label_permutation": "reverse_stacked_typed_write_rows_within_each_fixed_physical_batch",
        "retained_full_gate_matrix": False,
        "retained_logits": False,
    }
    return {**payload, "authority_sha256": logical_sha256(payload)}


def run() -> dict[str, Any]:
    namespace = (AUTHORITY, PAYLOAD, RECEIPT, FAILURE, LOCK)
    if any(path.exists() for path in namespace):
        raise RuntimeError("fit collector output namespace is not pristine")
    claim = acquire_claim(LOCK)
    started = time.time()
    try:
        source = source_closure()
        rows, row_provenance = validate_rows()
        checkpoint = facade.validate_snapshot(verify_weights_sha256=True)
        expected_authority = _authority(source, checkpoint, row_provenance)
        claim.verify()
        create_json(AUTHORITY, expected_authority)
        if json.loads(AUTHORITY.read_text()) != expected_authority:
            raise RuntimeError("published pre-outcome authority did not replay")
        verify_source_closure(source)
        verify_rows_unchanged(row_provenance, rows)

        model, loaded_receipt = facade.load_bilin18(
            device=DEVICE, dtype=torch.float32, verify_weights_sha256=True,
        )
        if loaded_receipt != checkpoint:
            raise RuntimeError("loaded checkpoint receipt differs from authority")
        model_metadata_before = model_parameter_metadata_sha256(model)
        trajectory_content_before = block0_through_3_state_sha256(model)
        factor_sha256s_before = block3_factor_sha256s(model)
        block = model.transformer.h[LAYER]
        left, right, down = (
            block.mlp.Left.weight.detach().float(),
            block.mlp.Right.weight.detach().float(),
            block.mlp.Down.weight.detach().float(),
        )
        balanced_left, balanced_right, _ = balance_product_gauge(left, right)
        accum = {
            "u_second": torch.zeros(WIDTH, WIDTH, dtype=torch.float64),
            "v_second": torch.zeros(WIDTH, WIDTH, dtype=torch.float64),
            "z_second": torch.zeros(WIDTH, WIDTH, dtype=torch.float64),
            "energy": torch.zeros(GATES, dtype=torch.float64),
        }
        count = 0
        replay_max = 0.0
        batch_count = math.ceil(ROW_COUNT / BATCH_SIZE)
        calls = MeasuredCalls.empty()
        for start in range(0, ROW_COUNT, BATCH_SIZE):
            claim.verify()
            tokens = rows[start:start + BATCH_SIZE, :MODEL_TOKEN_COUNT].to(DEVICE)
            u, v, z, replay = block3_typed_variables(model, tokens, calls)
            flat_u, flat_v = u.reshape(-1, WIDTH), v.reshape(-1, WIDTH)
            features = _feature_bank(balanced_left, balanced_right, flat_u, flat_v)
            stats = first_pass_statistics(flat_u, flat_v, features, down)
            for name in accum:
                accum[name].add_(stats[name].detach().cpu().double())
            count += int(stats["count"])
            replay_max = max(replay_max, replay)
        if count != ROW_COUNT * TOKENS_PER_ROW or replay_max > 2e-6:
            raise RuntimeError("first-pass count or exact RMS replay failed")
        prefilter_indices = torch.argsort(
            accum["energy"], descending=True, stable=True,
        )[:PREFILTER].to(DEVICE)

        gram = torch.zeros(PREFILTER, PREFILTER, dtype=torch.float64)
        cross = torch.zeros(PREFILTER, WIDTH, dtype=torch.float64)
        permuted_cross = torch.zeros(PREFILTER, WIDTH, dtype=torch.float64)
        write_energy = torch.zeros((), dtype=torch.float64)
        stacked_count = 0
        replay_second = 0.0
        for start in range(0, ROW_COUNT, BATCH_SIZE):
            claim.verify()
            tokens = rows[start:start + BATCH_SIZE, :MODEL_TOKEN_COUNT].to(DEVICE)
            u, v, _z, replay = block3_typed_variables(model, tokens, calls)
            features = _feature_bank(
                balanced_left, balanced_right,
                u.reshape(-1, WIDTH), v.reshape(-1, WIDTH),
            )
            batch_gram, batch_cross, batch_permuted, batch_write_energy, batch_stacked = second_pass_statistics(
                features, down, prefilter_indices,
            )
            gram.add_(batch_gram.detach().cpu().double())
            cross.add_(batch_cross.detach().cpu().double())
            permuted_cross.add_(batch_permuted.detach().cpu().double())
            write_energy.add_(batch_write_energy.detach().cpu().double())
            stacked_count += batch_stacked
            calls.explicit_typed_down_banks += len(subset.TERM_NAMES)
            replay_second = max(replay_second, replay)
        expected_stacked = count * len(subset.TERM_NAMES)
        if stacked_count != expected_stacked or replay_second > 2e-6:
            raise RuntimeError("second-pass count or exact RMS replay failed")

        expected_calls = MeasuredCalls.empty()
        for site in range(4):
            expected_calls.attention[site] = PASSES * batch_count
        for site in range(3):
            expected_calls.mlp[site] = PASSES * batch_count
        expected_calls.outer_returned = PASSES * batch_count
        expected_calls.explicit_typed_down_banks = len(subset.TERM_NAMES) * batch_count
        if calls.receipt() != expected_calls.receipt():
            raise RuntimeError("measured physical call census differs from protocol")
        model_metadata_after = model_parameter_metadata_sha256(model)
        trajectory_content_after = block0_through_3_state_sha256(model)
        factor_sha256s_after = block3_factor_sha256s(model)
        if model_metadata_after != model_metadata_before or trajectory_content_after != (
            trajectory_content_before
        ) or factor_sha256s_after != factor_sha256s_before:
            raise RuntimeError("model parameters or block-3 factors changed during fit collection")

        payload = {
            "schema": "block3_native_gate_fit_v1_payload",
            "authority_sha256": expected_authority["authority_sha256"],
            "count": count,
            "stacked_typed_count": stacked_count,
            "u_second_moment": accum["u_second"] / count,
            "v_second_moment": accum["v_second"] / count,
            "z_second_moment": accum["z_second"] / count,
            "contribution_energy": accum["energy"],
            "prefilter_indices": prefilter_indices.detach().cpu(),
            "prefilter_gram": gram,
            "prefilter_cross": cross,
            "prefilter_permuted_cross": permuted_cross,
            "native_typed_write_energy": write_energy,
            "rms_replay_max_abs": max(replay_max, replay_second),
            "fit_call_ledger": {
                "passes": PASSES,
                **calls.receipt(),
                "native_mlp3_calls": calls.mlp[LAYER],
                "retained_logits": 0,
                "retained_full_gate_matrix": False,
            },
            "model_integrity": {
                "parameter_metadata_before_sha256": model_metadata_before,
                "parameter_metadata_after_sha256": model_metadata_after,
                "block0_through_3_content_before_sha256": trajectory_content_before,
                "block0_through_3_content_after_sha256": trajectory_content_after,
                "block3_factor_before_sha256s": factor_sha256s_before,
                "block3_factor_after_sha256s": factor_sha256s_after,
            },
        }
        # Revalidate every mutable external input immediately before publishing data.
        claim.verify()
        verify_source_closure(source)
        verify_rows_unchanged(row_provenance, rows)
        if facade.validate_snapshot(verify_weights_sha256=True) != checkpoint or (
            json.loads(AUTHORITY.read_text()) != expected_authority
        ):
            raise RuntimeError("source input or authority changed before payload publication")
        create_torch(PAYLOAD, payload)
        payload_file_sha256 = file_sha256(PAYLOAD)
        replay_payload = torch.load(PAYLOAD, map_location="cpu", weights_only=True)
        if replay_payload.get("authority_sha256") != expected_authority[
            "authority_sha256"
        ] or replay_payload.get("count") != count or replay_payload.get(
            "fit_call_ledger"
        ) != payload["fit_call_ledger"]:
            raise RuntimeError("published fit payload did not replay")
        receipt_payload = {
            "schema": "block3_native_gate_fit_v1_receipt",
            "status": "fit_sufficient_statistics_complete_no_evaluation_opened",
            "authority_sha256": expected_authority["authority_sha256"],
            "authority_file_sha256": file_sha256(AUTHORITY),
            "payload_file_sha256": payload_file_sha256,
            "row_receipt_sha256": ROW_RECEIPT_SHA256,
            "row_disjointness_sha256": row_provenance["disjointness_sha256"],
            "source_closure_sha256": source["sha256"],
            "checkpoint_weights_sha256": checkpoint.weights_sha256,
            "model_parameter_metadata_sha256": model_metadata_after,
            "block0_through_3_content_sha256": trajectory_content_after,
            "block3_factor_sha256s": factor_sha256s_after,
            "measured_call_ledger": calls.receipt(),
            "fit_rows_raw_sha256": FIT_ROWS_RAW_SHA256,
            "count": count,
            "stacked_typed_count": stacked_count,
            "rms_replay_max_abs": max(replay_max, replay_second),
            "elapsed_seconds": time.time() - started,
        }
        # Receipt is terminal and last. Repeat all integrity checks after payload load.
        claim.verify()
        verify_source_closure(source)
        verify_rows_unchanged(row_provenance, rows)
        if facade.validate_snapshot(verify_weights_sha256=True) != checkpoint or (
            json.loads(AUTHORITY.read_text()) != expected_authority
        ) or file_sha256(PAYLOAD) != payload_file_sha256 or (
            model_parameter_metadata_sha256(model) != model_metadata_before
        ) or block0_through_3_state_sha256(model) != trajectory_content_before or (
            block3_factor_sha256s(model) != factor_sha256s_before
        ):
            raise RuntimeError("terminal integrity replay failed before receipt")
        create_json(RECEIPT, receipt_payload)
        return receipt_payload
    except BaseException as error:
        if not FAILURE.exists() and not RECEIPT.exists():
            try:
                claim.verify()
                create_json(FAILURE, {
                    "schema": "block3_native_gate_fit_v1_failure",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "authority_exists": AUTHORITY.exists(),
                    "payload_exists": PAYLOAD.exists(),
                    "receipt_exists": RECEIPT.exists(),
                    "elapsed_seconds": time.time() - started,
                })
            except BaseException:
                pass
        raise
    finally:
        claim.release()


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
