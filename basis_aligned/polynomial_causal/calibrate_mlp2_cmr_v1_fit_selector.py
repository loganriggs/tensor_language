#!/usr/bin/env python3
"""Freeze FIT_SELECTOR-only margin thresholds and target-frequency reference."""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time
from typing import Any

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for source_root in (ROOT, HERE):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

import bilin18_observed_model_facade as facade
import project_mlp2_cmr_v1_fit_selector_rows as projection


PREREG = HERE / "MLP2_CMR_V1_PREREGISTRATION.md"
ADDENDUM = HERE / "MLP2_CMR_V1_MARGIN_FREQUENCY_ADDENDUM.md"
COPY_PREREG = HERE / "COPY_SOURCE_EDGE_DISCOVERY_PREREGISTRATION.md"
ROLE_ROWS = HERE / "mlp2_cmr_v1_fit_selector_rows.pt"
ROLE_MANIFEST = HERE / "mlp2_cmr_v1_fit_selector_rows_manifest.json"
ROLE_RECEIPT = HERE / "mlp2_cmr_v1_fit_selector_rows_receipt.json"
SUFFIX_BUNDLE = HERE / "mlp2_cmr_v1_suffix_v2_bundle.pt"
SUFFIX_RESULT = HERE / "mlp2_cmr_v1_suffix_v2_result.json"
SUFFIX_RECEIPT = HERE / "mlp2_cmr_v1_suffix_v2_receipt.json"
CORRECTION = HERE / "mlp2_cmr_v1_suffix_v2_overlap_correction.json"
CORRECTION_RECEIPT = HERE / "mlp2_cmr_v1_suffix_v2_overlap_correction_receipt.json"
AUTHORITY = HERE / "mlp2_cmr_v1_fit_selector_calibration_authority.json"
BUNDLE = HERE / "mlp2_cmr_v1_fit_selector_calibration_bundle.pt"
RESULT = HERE / "mlp2_cmr_v1_fit_selector_calibration_result.json"
RECEIPT = HERE / "mlp2_cmr_v1_fit_selector_calibration_receipt.json"
FAILURE = HERE / "mlp2_cmr_v1_fit_selector_calibration_failure.json"
LOCK = HERE / ".mlp2_cmr_v1_fit_selector_calibration.lock"

ROLE_ROWS_SHA256 = "08a508d6e1526800347d94c6637c84a662c220d84ef30bc674bf6b905ab67798"
ROLE_MANIFEST_SHA256 = "6073c2fd38ad3287c6b7349f2d99aae41c0e98655961255fc18ba4c7c4b745a2"
ROLE_RECEIPT_SHA256 = "6a0dad2f7df3dd17d20fc16df15c03b47c3ef0da30fd65c1cc2149d762709a21"
SUFFIX_BUNDLE_SHA256 = "cb3f8d3caecab86881eba825785cabd58c1b7ac8e2aa1eb93b459168cff17ce1"
SUFFIX_RESULT_SHA256 = "ab08dc0f0a71b5daf21228991b9e78a272aa74d226d97189ac414a546dc16f62"
SUFFIX_RECEIPT_SHA256 = "b61c7308409ec64dc05601206bda21e1f4e24097871ba8dff0c92bc84e761e1f"
CORRECTION_SHA256 = "ffd5a826962f09ffec0af6c842eaf0bf64530423b827f6239556bc43db9d7ff4"
CORRECTION_RECEIPT_SHA256 = "dd557dc6366503bea2f3f7649d6312abc8a89857bb277ea4e844d8822e4e968a"

DOCUMENTS = 192
BATCH = 4
CALLS = 48
SEQUENCE = 256
SCORE_START = 64
ELIGIBLE_POSITIONS = 31_505
WINDOW = 128
VOCAB = facade.LOGIT_VOCAB
FREQUENCY_BOUNDARIES = (1, 2, 4, 8, 16, 32, 64, 128)
MARGIN_QUANTILES = (
    0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40,
    0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999,
)
DYADIC_EXPONENTS = tuple(range(-10, 6))

SOURCE_CLOSURE = (
    PREREG, ADDENDUM, COPY_PREREG, Path(__file__).resolve(),
    HERE / "test_calibrate_mlp2_cmr_v1_fit_selector.py",
    Path(projection.__file__).resolve(),
    HERE / "test_project_mlp2_cmr_v1_fit_selector_rows.py",
    Path(facade.__file__).resolve(), ROOT / "jacclust/tt_model.py",
)


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


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_create_only(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as target:
            target.write(data)
            target.flush()
            os.fsync(target.fileno())
        os.link(temporary, path)
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def publish_torch_create_only(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}")
    try:
        torch.save(value, temporary)
        with temporary.open("rb") as source:
            os.fsync(source.fileno())
        os.link(temporary, path)
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def committed_source() -> tuple[str, dict[str, str]]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "origin/main"],
        cwd=ROOT, check=True,
    )
    hashes: dict[str, str] = {}
    for path in SOURCE_CLOSURE:
        relative = path.relative_to(ROOT)
        blob = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)
        digest = hashlib.sha256(blob).hexdigest()
        if file_sha256(path) != digest:
            raise RuntimeError(f"calibration source differs from committed bytes: {relative}")
        hashes[str(relative)] = digest
    return commit, hashes


def protected_inputs() -> tuple[dict[str, str], dict[str, bytes]]:
    expected = {
        "role_rows": ROLE_ROWS_SHA256,
        "role_manifest": ROLE_MANIFEST_SHA256,
        "role_receipt": ROLE_RECEIPT_SHA256,
        "suffix_bundle": SUFFIX_BUNDLE_SHA256,
        "suffix_result": SUFFIX_RESULT_SHA256,
        "suffix_receipt": SUFFIX_RECEIPT_SHA256,
        "correction": CORRECTION_SHA256,
        "correction_receipt": CORRECTION_RECEIPT_SHA256,
    }
    paths = {
        "role_rows": ROLE_ROWS,
        "role_manifest": ROLE_MANIFEST,
        "role_receipt": ROLE_RECEIPT,
        "suffix_bundle": SUFFIX_BUNDLE,
        "suffix_result": SUFFIX_RESULT,
        "suffix_receipt": SUFFIX_RECEIPT,
        "correction": CORRECTION,
        "correction_receipt": CORRECTION_RECEIPT,
    }
    captured = {name: path.read_bytes() for name, path in paths.items()}
    actual = {
        name: hashlib.sha256(payload).hexdigest()
        for name, payload in captured.items()
    }
    if actual != expected:
        raise RuntimeError("MLP2 calibration protected parent changed")
    role_manifest = json.loads(captured["role_manifest"])
    role_receipt = json.loads(captured["role_receipt"])
    suffix_receipt = json.loads(captured["suffix_receipt"])
    correction_receipt = json.loads(captured["correction_receipt"])
    if (
        role_manifest.get("output_sha256") != ROLE_ROWS_SHA256
        or role_manifest.get("contains_roles") != ["FIT_SELECTOR"]
        or role_receipt.get("output_sha256") != ROLE_ROWS_SHA256
        or role_receipt.get("manifest_sha256") != ROLE_MANIFEST_SHA256
        or role_receipt.get("authorized_for_fit_selector_calibration_input") is not True
        or role_receipt.get("authorized_for_validation") is not False
        or role_receipt.get("authorized_for_replication") is not False
        or suffix_receipt.get("authorized_for_validation") is not True
        or correction_receipt.get("authorized_for_validation") is not True
        or correction_receipt.get("supersedes_only") != "support_overlap_summary"
        or (
        suffix_receipt.get("authorized_for_replication") is not False
        )
    ):
        raise RuntimeError("MLP2 suffix receipt authority boundary changed")
    return actual, captured


def validate_claim(nonce: str, inode: tuple[int, int]) -> None:
    descriptor = os.open(LOCK, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        stat = os.fstat(descriptor)
        payload = os.read(descriptor, 4096)
    finally:
        os.close(descriptor)
    if (stat.st_dev, stat.st_ino) != inode or json.loads(payload).get("nonce") != nonce:
        raise RuntimeError("MLP2 calibration claim changed")


class _CalibrationCapability:
    __slots__ = ("nonce", "inode", "authority_sha256", "consumed")

    def __init__(self, nonce: str, inode: tuple[int, int], authority_sha256: str) -> None:
        self.nonce = nonce
        self.inode = inode
        self.authority_sha256 = authority_sha256
        self.consumed = False


def mint_capability(nonce: str, inode: tuple[int, int], authority_sha256: str) -> _CalibrationCapability:
    validate_claim(nonce, inode)
    if not AUTHORITY.is_file() or file_sha256(AUTHORITY) != authority_sha256:
        raise RuntimeError("MLP2 calibration authority is absent or changed")
    authority = json.loads(AUTHORITY.read_bytes())
    if authority.get("status") != "authority_frozen_before_calibration_model_access" or (
        authority.get("authorized_role") != "FIT_SELECTOR"
        or authority.get("authorized_forward_calls") != CALLS
        or authority.get("authorized_backward_calls") != 0
    ):
        raise RuntimeError("MLP2 calibration authority semantics changed")
    return _CalibrationCapability(nonce, inode, authority_sha256)


def final_guard(
    source_hashes: dict[str, str], parents: dict[str, str], nonce: str,
    inode: tuple[int, int], authority_hash: str, bundle_hash: str,
    result_hash: str,
) -> None:
    validate_claim(nonce, inode)
    for relative, expected in source_hashes.items():
        if file_sha256(ROOT / relative) != expected:
            raise RuntimeError(f"calibration source changed during execution: {relative}")
    current, _ = protected_inputs()
    if current != parents or file_sha256(AUTHORITY) != authority_hash or file_sha256(
        BUNDLE
    ) != bundle_hash or file_sha256(RESULT) != result_hash or FAILURE.exists() or (
        RECEIPT.exists()
    ):
        raise RuntimeError("MLP2 calibration terminal snapshot changed")


def target_frequency_reference(
    rows: torch.Tensor, eligible: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if rows.shape != (DOCUMENTS, SEQUENCE + 1) or rows.dtype != torch.long or (
        eligible.shape != (DOCUMENTS, SEQUENCE) or eligible.dtype != torch.bool
    ):
        raise ValueError("calibration token rows or eligibility mask are malformed")
    targets = rows[:, 1:]
    selected = targets[eligible]
    if selected.numel() != ELIGIBLE_POSITIONS or bool((selected < 0).any()) or bool(
        (selected >= facade.TOKENIZER_VOCAB).any()
    ):
        raise ValueError("calibration eligible targets changed")
    counts = torch.bincount(selected, minlength=VOCAB).long().contiguous()
    boundaries = torch.tensor(FREQUENCY_BOUNDARIES, dtype=torch.long)
    bins = torch.bucketize(counts.index_select(0, selected), boundaries, right=True)
    return counts, torch.bincount(bins, minlength=len(FREQUENCY_BOUNDARIES) + 1).long()


def nearest_repeat_cells(rows: torch.Tensor, eligible: torch.Tensor) -> dict[str, torch.Tensor]:
    if rows.shape != (DOCUMENTS, SEQUENCE + 1) or eligible.shape != (
        DOCUMENTS, SEQUENCE
    ):
        raise ValueError("copy-cell token rows or eligibility mask are malformed")
    inputs, targets = rows[:, :-1], rows[:, 1:]
    positions = torch.arange(SEQUENCE)
    source = torch.full((DOCUMENTS, SEQUENCE), -1, dtype=torch.long)
    expanded_positions = positions.expand(DOCUMENTS, -1)
    for distance in range(1, WINDOW + 1):
        candidate = (positions - distance).clamp_min(0)
        candidate_token = inputs.gather(1, candidate.expand(DOCUMENTS, -1))
        choose = (
            (source < 0) & (positions >= distance).unsqueeze(0)
            & (inputs == candidate_token)
        )
        source[choose] = expanded_positions[choose] - distance
    has_repeat = source >= 0
    successor = (source.clamp_min(0) + 1).clamp_max(SEQUENCE - 1)
    successor_token = inputs.gather(1, successor)
    copy_positive = eligible & has_repeat & (targets == successor_token)
    repeat_negative = eligible & has_repeat & ~copy_positive
    nonrepeat = eligible & ~has_repeat
    if not torch.equal(copy_positive | repeat_negative | nonrepeat, eligible) or bool(
        (copy_positive & repeat_negative).any()
        or (copy_positive & nonrepeat).any()
        or (repeat_negative & nonrepeat).any()
    ):
        raise RuntimeError("copy/repeat cells do not partition eligible positions")
    return {
        "copy_positive": copy_positive,
        "repeat_negative": repeat_negative,
        "nonrepeat": nonrepeat,
    }


def epsilon_grid(margins: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if margins.ndim != 1 or margins.dtype != torch.float64 or margins.numel() != (
        ELIGIBLE_POSITIONS
    ) or not bool(torch.isfinite(margins).all()) or bool((margins < 0).any()):
        raise ValueError("native margin vector is malformed")
    quantiles = torch.quantile(
        margins, torch.tensor(MARGIN_QUANTILES, dtype=torch.float64),
    )
    candidates = [2.0 ** exponent for exponent in DYADIC_EXPONENTS]
    candidates.extend(float(value / 2.0) for value in quantiles)
    grid = torch.tensor(sorted({value for value in candidates if value > 0}), dtype=torch.float64)
    if grid.ndim != 1 or not bool(torch.isfinite(grid).all()) or bool(
        (grid <= 0).any()
    ) or not bool((grid[1:] > grid[:-1]).all()):
        raise RuntimeError("native margin epsilon grid is malformed")
    return quantiles.contiguous(), grid.contiguous()


def collect(token_rows_bytes: bytes) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    started = time.time()
    if hashlib.sha256(token_rows_bytes).hexdigest() != TOKEN_ROWS_SHA256:
        raise RuntimeError("captured FIT_SELECTOR token bytes changed")
    token_bundle = torch.load(
        io.BytesIO(token_rows_bytes), map_location="cpu", weights_only=True,
    )
    role = token_bundle["FIT_SELECTOR"]
    rows = role["rows"].contiguous()
    eligible = role["eligible_mask"].contiguous()
    if int(eligible.sum()) != ELIGIBLE_POSITIONS:
        raise RuntimeError("FIT_SELECTOR eligible-position count changed")
    frequency, frequency_bin_counts = target_frequency_reference(rows, eligible)
    cells = nearest_repeat_cells(rows, eligible)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = facade.load_bilin18(device=device, dtype=torch.bfloat16)
    margin_parts: list[torch.Tensor] = []
    attention_calls = mlp_calls = forward_calls = 0
    with torch.inference_mode():
        for start in range(0, DOCUMENTS, BATCH):
            tokens = rows[start:start + BATCH, :-1].to(device)

            def attention(event: facade.AttentionEvent):
                nonlocal attention_calls
                attention_calls += 1
                return event.block.attn(event.state, event.first_value)

            def mlp(event: facade.EarlyMLPEvent):
                nonlocal mlp_calls
                mlp_calls += 1
                return event.block.mlp(event.state)

            logits = facade.forward_with_dispatch(model, tokens, attention, mlp)
            mask = eligible[start:start + BATCH].to(device)
            top2 = torch.topk(logits, 2, dim=-1).values
            margin_parts.append((top2[..., 0] - top2[..., 1])[mask].cpu().double())
            forward_calls += 1
            del logits, top2
            print(f"MLP2 calibration batch {forward_calls}/{CALLS}", flush=True)
    margins = torch.cat(margin_parts).contiguous()
    quantiles, grid = epsilon_grid(margins)
    if forward_calls != CALLS or attention_calls != 18 * CALLS or mlp_calls != 18 * CALLS:
        raise RuntimeError("MLP2 calibration call ledger changed")
    cell_counts = {name: int(mask.sum()) for name, mask in cells.items()}
    if sum(cell_counts.values()) != ELIGIBLE_POSITIONS or int(frequency.sum()) != (
        ELIGIBLE_POSITIONS
    ) or int(frequency_bin_counts.sum()) != ELIGIBLE_POSITIONS:
        raise RuntimeError("MLP2 calibration count ledger changed")
    summary = {
        "schema": "mlp2_cmr_v1_fit_selector_calibration_result",
        "status": "fit_selector_calibration_complete_no_validation_or_replication",
        "checkpoint": checkpoint.__dict__,
        "documents": DOCUMENTS,
        "eligible_positions": ELIGIBLE_POSITIONS,
        "forward_calls": forward_calls,
        "backward_calls": 0,
        "attention_calls": attention_calls,
        "mlp_calls": mlp_calls,
        "margin_quantiles": {
            format(q, ".6g"): float(value)
            for q, value in zip(MARGIN_QUANTILES, quantiles.tolist())
        },
        "margin_minimum": float(margins.min()),
        "margin_maximum": float(margins.max()),
        "margin_mean": float(margins.mean()),
        "epsilon_grid": grid.tolist(),
        "epsilon_grid_count": int(grid.numel()),
        "frequency_boundaries": list(FREQUENCY_BOUNDARIES),
        "fit_frequency_bin_counts": frequency_bin_counts.tolist(),
        "copy_cell_counts": cell_counts,
        "tensor_hashes": {
            "fit_token_counts": tensor_sha256(frequency),
            "margin_quantiles": tensor_sha256(quantiles),
            "epsilon_grid": tensor_sha256(grid),
        },
        "runtime_seconds": time.time() - started,
        "validation_opened": False,
        "replication_opened": False,
        "finite_candidate_constructed": False,
        "raw_logits_published": False,
    }
    bundle = {
        "schema": "mlp2_cmr_v1_fit_selector_calibration_bundle",
        "fit_token_counts": frequency,
        "frequency_boundaries": torch.tensor(FREQUENCY_BOUNDARIES, dtype=torch.long),
        "margin_quantiles": quantiles,
        "epsilon_grid": grid,
    }
    del model, margins
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summary, bundle


def main() -> None:
    forbidden_outputs = (AUTHORITY, BUNDLE, RESULT, RECEIPT, FAILURE, LOCK)
    if any(path.exists() for path in forbidden_outputs):
        raise RuntimeError("MLP2 calibration output already exists")
    source_commit, source_hashes = committed_source()
    parents, parent_bytes = protected_inputs()
    write_create_only(LOCK, canonical_json_bytes({
        "experiment_id": "bilin18_mlp2_cmr_v1_fit_selector_calibration",
        "nonce": secrets.token_hex(32),
    }))
    authority = {
        "schema_version": 1,
        "experiment_id": "bilin18_mlp2_cmr_v1_fit_selector_calibration",
        "status": "authority_frozen_before_calibration_model_access",
        "source_commit": source_commit,
        "source_hashes": source_hashes,
        "parents": parents,
        "authorized_role": "FIT_SELECTOR",
        "authorized_forward_calls": CALLS,
        "authorized_backward_calls": 0,
        "authorized_outputs": [BUNDLE.name, RESULT.name, RECEIPT.name],
        "forbidden": [
            "VALIDATION", "REPLICATION", "finite MLP2 candidate",
            "next-token loss", "accuracy", "raw logit publication",
        ],
    }
    write_create_only(AUTHORITY, canonical_json_bytes(authority))
    try:
        summary, bundle = collect(parent_bytes["token_rows"])
        final_guard(source_hashes, parents)
        publish_torch_create_only(BUNDLE, bundle)
        bundle_hash = file_sha256(BUNDLE)
        summary["authority_sha256"] = file_sha256(AUTHORITY)
        summary["bundle_sha256"] = bundle_hash
        write_create_only(RESULT, canonical_json_bytes(summary))
        final_guard(source_hashes, parents)
        receipt = {
            "schema_version": 1,
            "experiment_id": authority["experiment_id"],
            "status": "fit_selector_calibration_complete_receipt_last",
            "authority_sha256": file_sha256(AUTHORITY),
            "bundle_sha256": bundle_hash,
            "result_sha256": file_sha256(RESULT),
            "lock_sha256": file_sha256(LOCK),
            "source_commit": source_commit,
            "source_hashes": source_hashes,
            "parents": parents,
            "authorized_for_validation_implementation": True,
            "authorized_for_validation_execution": False,
            "authorized_for_replication": False,
        }
        write_create_only(RECEIPT, canonical_json_bytes(receipt))
    except BaseException as error:
        failure = {
            "schema_version": 1,
            "experiment_id": authority["experiment_id"],
            "status": "failed_after_authority",
            "authority_sha256": file_sha256(AUTHORITY),
            "error_type": type(error).__name__,
            "error": str(error),
            "validation_opened": False,
            "replication_opened": False,
        }
        write_create_only(FAILURE, canonical_json_bytes(failure))
        raise


if __name__ == "__main__":
    main()
