#!/usr/bin/env python3
"""Collect masked MLP2 product moments and frozen local control supports."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time
from typing import Any, Mapping

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import bilin18_observed_model_facade as facade


PREREG = HERE / "MLP2_CMR_V1_PREREGISTRATION.md"
ADDENDUM = HERE / "MLP2_CMR_V1_FIT_MEAN_ADDENDUM.md"
TOKEN_ROWS = HERE / "mlp2_cmr_v1_token_rows.pt"
TOKEN_MANIFEST = HERE / "mlp2_cmr_v1_token_rows_manifest.json"
TOKEN_RECEIPT = HERE / "mlp2_cmr_v1_token_rows_receipt.json"
AUTHORITY = HERE / "mlp2_cmr_v1_fit_mean_authority.json"
BUNDLE = HERE / "mlp2_cmr_v1_fit_mean_bundle.pt"
RESULT = HERE / "mlp2_cmr_v1_fit_mean_result.json"
RECEIPT = HERE / "mlp2_cmr_v1_fit_mean_receipt.json"
FAILURE = HERE / "mlp2_cmr_v1_fit_mean_failure.json"
LOCK = HERE / ".mlp2_cmr_v1_fit_mean.lock"
TOKEN_ROWS_SHA256 = "3ed0192993095f7de70ab7f1350d091b6c1d8c4c7d0583fd5f0f6441556e4aa6"
TOKEN_MANIFEST_SHA256 = "8b8f3155a21b73af8b89278b9f09c60bf82fd965a7723e046e191415c5d57bb4"
TOKEN_RECEIPT_SHA256 = "47113c255bf47f9d1c7369639fab39664c71f93134099babadcce9d89a011e85"
SITE = 2
DOCUMENTS = 192
BATCH = 4
SEQUENCE = 256
HIDDEN = 4608
K = 512
EXPECTED_OBSERVATIONS = 30_801
SOURCE_CLOSURE = (
    PREREG, ADDENDUM, Path(__file__).resolve(),
    HERE / "test_collect_mlp2_cmr_v1_fit_mean.py",
    Path(facade.__file__).resolve(), ROOT / "jacclust/tt_model.py",
)


class _StopAfterMLP2Capture(Exception):
    """Private non-error exit after the only authorized model quantity exists."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def finalize_moments(
    total: torch.Tensor, square_total: torch.Tensor, count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if total.ndim != 1 or square_total.shape != total.shape or count <= 0:
        raise ValueError("moments sufficient statistics are malformed")
    mean = total.double() / count
    second = square_total.double() / count
    variance = (second - mean.square()).clamp_min(0)
    return mean, variance, second


def select_top(score: torch.Tensor, k: int = K) -> torch.Tensor:
    if score.ndim != 1 or not 0 < k <= score.numel() or not bool(torch.isfinite(score).all()):
        raise ValueError("selection score is malformed")
    return torch.argsort(score.double(), descending=True, stable=True)[:k]


def support_jaccard(first: torch.Tensor, second: torch.Tensor) -> float:
    a, b = set(first.tolist()), set(second.tolist())
    return len(a & b) / max(len(a | b), 1)


def write_create_only(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as target:
            target.write(data)
            target.flush()
            os.fsync(target.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def committed_source() -> tuple[str, dict[str, str]]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    subprocess.run(["git", "merge-base", "--is-ancestor", commit, "origin/main"], cwd=ROOT, check=True)
    hashes = {}
    for path in SOURCE_CLOSURE:
        relative = path.relative_to(ROOT)
        blob = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)
        digest = hashlib.sha256(blob).hexdigest()
        if file_sha256(path) != digest:
            raise RuntimeError(f"fit source differs from committed bytes: {relative}")
        hashes[str(relative)] = digest
    return commit, hashes


def protected_inputs() -> dict[str, str]:
    actual = {
        "token_rows": file_sha256(TOKEN_ROWS),
        "token_manifest": file_sha256(TOKEN_MANIFEST),
        "token_receipt": file_sha256(TOKEN_RECEIPT),
    }
    expected = {
        "token_rows": TOKEN_ROWS_SHA256,
        "token_manifest": TOKEN_MANIFEST_SHA256,
        "token_receipt": TOKEN_RECEIPT_SHA256,
    }
    if actual != expected:
        raise RuntimeError("MLP2 CMR token parents changed")
    receipt = json.loads(TOKEN_RECEIPT.read_text())
    if receipt.get("authorized_for_token_inputs") is not True or \
            receipt.get("authorized_for_model_forward") is not False:
        raise RuntimeError("token parent authority changed")
    return actual


def publish_torch_create_only(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}")
    try:
        torch.save(value, temporary)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@torch.no_grad()
def collect() -> tuple[dict[str, Any], dict[str, Any], Any]:
    started = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows_bundle = torch.load(TOKEN_ROWS, map_location="cpu", weights_only=True)
    fit = rows_bundle["FIT_MEAN"]
    rows = fit["rows"]
    masks = fit["eligible_mask"]
    if tuple(rows.shape) != (DOCUMENTS, SEQUENCE + 1) or \
            tuple(masks.shape) != (DOCUMENTS, SEQUENCE) or int(masks.sum()) != EXPECTED_OBSERVATIONS:
        raise RuntimeError("FIT_MEAN token support changed")
    model, checkpoint = facade.load_bilin18(device=device, dtype=torch.bfloat16)
    total = torch.zeros(HIDDEN, dtype=torch.float64)
    square_total = torch.zeros(HIDDEN, dtype=torch.float64)
    count = 0
    forward_calls = 0
    captured_calls = 0

    for start in range(0, DOCUMENTS, BATCH):
        tokens = rows[start:start + BATCH, :-1].to(device).contiguous()
        mask = masks[start:start + BATCH].to(device)
        capture: list[torch.Tensor] = []

        def attention(event: facade.AttentionEvent):
            return event.block.attn(event.state, event.first_value)

        def mlp(event: facade.EarlyMLPEvent):
            nonlocal captured_calls
            if event.site == SITE:
                product = event.block.mlp.Left(event.state) * event.block.mlp.Right(event.state)
                selected = product[mask].float()
                capture.extend((selected.sum(0).double().cpu(), selected.square().sum(0).double().cpu()))
                captured_calls += 1
                raise _StopAfterMLP2Capture
            return event.block.mlp(event.state)

        try:
            facade.forward_with_dispatch(model, tokens, attention, mlp)
        except _StopAfterMLP2Capture:
            pass
        else:
            raise RuntimeError("observed-model prefix did not reach MLP2 capture")
        forward_calls += 1
        if len(capture) != 2:
            raise RuntimeError("MLP2 product capture call count changed")
        total += capture[0]
        square_total += capture[1]
        count += int(mask.sum())
        del capture
    if count != EXPECTED_OBSERVATIONS or forward_calls != 48 or captured_calls != 48:
        raise RuntimeError("FIT_MEAN call or support ledger changed")

    mean, variance, second = finalize_moments(total, square_total, count)
    mlp2 = model.transformer.h[SITE].mlp
    left_norm2 = mlp2.Left.weight.detach().float().cpu().square().sum(1).double()
    right_norm2 = mlp2.Right.weight.detach().float().cpu().square().sum(1).double()
    down_norm2 = mlp2.Down.weight.detach().float().cpu().square().sum(0).double()
    scores = {
        "LOCAL": variance * down_norm2,
        "RMS": second * down_norm2,
        "MASS": left_norm2 * right_norm2 * down_norm2,
    }
    supports = {name: select_top(score) for name, score in scores.items()}
    random_generator = torch.Generator().manual_seed(20260829)
    random_score = torch.rand(HIDDEN, generator=random_generator, dtype=torch.float64)
    supports["RANDOM"] = select_top(random_score)

    gauge_generator = torch.Generator().manual_seed(20260830)
    log_s = 6 * torch.rand(HIDDEN, generator=gauge_generator, dtype=torch.float64) - 3
    log_t = 6 * torch.rand(HIDDEN, generator=gauge_generator, dtype=torch.float64) - 3
    s2, t2 = (2 * log_s).exp(), (2 * log_t).exp()
    p2 = s2 * t2
    gauged = {
        "LOCAL": (variance * p2) * (down_norm2 / p2),
        "RMS": (second * p2) * (down_norm2 / p2),
        "MASS": (left_norm2 * s2) * (right_norm2 * t2) * (down_norm2 / p2),
    }
    gauge_audit = {
        name: {
            "top512_jaccard": support_jaccard(supports[name], select_top(gauged[name])),
            "maximum_relative_score_error": float(
                ((gauged[name] - scores[name]).abs() / scores[name].abs().clamp_min(1e-30)).max()
            ),
        } for name in scores
    }
    bundle = {
        "schema": "mlp2_cmr_v1_fit_mean_bundle",
        "count": count,
        "mean": mean,
        "variance": variance,
        "second_moment": second,
        "left_norm2": left_norm2,
        "right_norm2": right_norm2,
        "down_norm2": down_norm2,
        "scores": scores,
        "supports": supports,
    }
    overlaps = {
        f"{left}_{right}": support_jaccard(supports[left], supports[right])
        for index, left in enumerate(supports)
        for right in tuple(supports)[index + 1:]
    }
    summary = {
        "schema": "mlp2_cmr_v1_fit_mean_result",
        "status": "fit_mean_and_local_controls_complete_no_suffix_or_validation",
        "checkpoint": checkpoint.__dict__,
        "fit_observations": count,
        "forward_calls": forward_calls,
        "captured_mlp2_calls": captured_calls,
        "score_summaries": {
            name: {
                "minimum": float(score.min()),
                "median": float(score.median()),
                "maximum": float(score.max()),
                "sum": float(score.sum()),
            } for name, score in scores.items()
        },
        "top512_support_overlaps": overlaps,
        "gauge_audit": gauge_audit,
        "program_price": {
            "retained_products": K,
            "stored_scalar_values": 3456 * K + 1152,
            "fraction_native_mlp2_values": (3456 * K + 1152) / (3456 * HIDDEN + 1152),
        },
        "runtime_seconds": time.time() - started,
        "role_use": {
            "FIT_MEAN": "opened_for_product_moments",
            "FIT_SELECTOR": "not_used_in_model",
            "VALIDATION": "not_used_in_model",
            "REPLICATION": "not_used_in_model",
        },
        "outcome_access": {
            "targets_accessed": False,
            "loss_or_accuracy_computed": False,
            "logits_constructed": False,
            "raw_logits_published": False,
            "raw_products_published": False,
            "suffix_selector_fitted": False,
            "finite_candidate_evaluated": False,
        },
    }
    return bundle, summary, checkpoint


def main() -> None:
    namespace = (AUTHORITY, BUNDLE, RESULT, RECEIPT, FAILURE)
    if any(path.exists() for path in namespace):
        raise RuntimeError("refusing to overwrite MLP2 CMR FIT_MEAN namespace")
    lock_fd = os.open(LOCK, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        commit, source_hashes = committed_source()
        parents = protected_inputs()
        authority = {
            "schema_version": 1,
            "experiment_id": "bilin18_mlp2_cmr_v1_fit_mean",
            "status": "authority_frozen_before_checkpoint_or_model_access",
            "source_commit": commit,
            "source_hashes": source_hashes,
            "parents": parents,
            "authorized_role": "FIT_MEAN",
            "authorized_forward_calls": 48,
            "authorized_capture": "masked MLP2 product sums and squared sums only",
            "forbidden": ["targets", "loss", "KL", "accuracy", "logits", "raw products", "FIT_SELECTOR", "VALIDATION", "REPLICATION"],
        }
        write_create_only(AUTHORITY, json.dumps(authority, indent=2, sort_keys=True).encode() + b"\n")
        authority_hash = file_sha256(AUTHORITY)
        try:
            bundle, result, _ = collect()
            result.update({
                "authority_sha256": authority_hash,
                "source_commit": commit,
                "source_hashes": source_hashes,
                "parents": parents,
                "tensor_hashes": {
                    "mean": tensor_sha256(bundle["mean"]),
                    "variance": tensor_sha256(bundle["variance"]),
                    "second_moment": tensor_sha256(bundle["second_moment"]),
                    "scores": {name: tensor_sha256(value) for name, value in bundle["scores"].items()},
                    "supports": {name: tensor_sha256(value) for name, value in bundle["supports"].items()},
                },
            })
            publish_torch_create_only(BUNDLE, bundle)
            result["bundle_sha256"] = file_sha256(BUNDLE)
            write_create_only(RESULT, json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
            replay = torch.load(BUNDLE, map_location="cpu", weights_only=True)
            if tensor_sha256(replay["mean"]) != result["tensor_hashes"]["mean"]:
                raise RuntimeError("FIT_MEAN bundle semantic replay failed")
            receipt = {
                "schema_version": 1,
                "experiment_id": "bilin18_mlp2_cmr_v1_fit_mean",
                "status": "fit_mean_complete_receipt_last",
                "authority_sha256": authority_hash,
                "bundle_sha256": file_sha256(BUNDLE),
                "result_sha256": file_sha256(RESULT),
                "source_commit": commit,
                "source_hashes": source_hashes,
                "parents": parents,
                "fit_observations": result["fit_observations"],
                "authorized_for_suffix_selector": True,
                "authorized_for_validation": False,
                "authorized_for_replication": False,
                "scientific_claim": "none_fit_artifact_only",
            }
            if file_sha256(AUTHORITY) != authority_hash or protected_inputs() != parents:
                raise RuntimeError("FIT_MEAN parents changed before receipt")
            write_create_only(RECEIPT, json.dumps(receipt, indent=2, sort_keys=True).encode() + b"\n")
            print(json.dumps(result, indent=2, sort_keys=True))
        except BaseException as exc:
            if not FAILURE.exists():
                failure = {
                    "schema_version": 1,
                    "experiment_id": "bilin18_mlp2_cmr_v1_fit_mean",
                    "status": "failed_after_authority",
                    "authority_sha256": authority_hash,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "bundle_exists": BUNDLE.exists(),
                    "result_exists": RESULT.exists(),
                    "receipt_exists": RECEIPT.exists(),
                }
                write_create_only(FAILURE, json.dumps(failure, indent=2, sort_keys=True).encode() + b"\n")
            raise
    finally:
        os.close(lock_fd)
        LOCK.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
