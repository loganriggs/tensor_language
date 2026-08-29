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
from typing import Any, Callable

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


def write_create_only(
    path: Path, data: bytes, *, before_link: Callable[[], None] | None = None,
) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as target:
            target.write(data)
            target.flush()
            os.fsync(target.fileno())
        if before_link is not None:
            before_link()
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


def terminal_prelink_guard(
    target: Path, opposite: Path, protected_guard: Callable[[], None],
) -> None:
    protected_guard()
    if target.exists() or opposite.exists():
        raise RuntimeError("MLP2 calibration terminal namespace changed before link")


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
    role_parents = {
        "combined": "3ed0192993095f7de70ab7f1350d091b6c1d8c4c7d0583fd5f0f6441556e4aa6",
        "combined_manifest": "8b8f3155a21b73af8b89278b9f09c60bf82fd965a7723e046e191415c5d57bb4",
        "combined_receipt": "47113c255bf47f9d1c7369639fab39664c71f93134099babadcce9d89a011e85",
    }
    suffix_parents = {
        "fit_bundle": "043bb52b9580d9c9c342460e5bb80ff579db01486b3b6c6672bf5fba77e46f8e",
        "fit_receipt": "9dc14d909a1b4aafd33c67dc7a3d066db4ccc9cb83c7059fe7aaf499ca9e5efa",
        "fit_result": "65c1ee33f0399d6489cae0227442d479a9d59b9be98f619d92423cfd39fc7833",
        "previous_authority": "d204e9adeef3d65a1d6f38ed76071aa38c921bd2884ed136ac6e37f4696c7296",
        "previous_failure": "eea77b4e7fa9fd6ed35dce31ea43a72f8cf2d21d8c2e76a94396a81547f6d8a2",
        "token_receipt": role_parents["combined_receipt"],
        "token_rows": role_parents["combined"],
    }
    correction_parents = {
        "bundle": SUFFIX_BUNDLE_SHA256,
        "receipt": SUFFIX_RECEIPT_SHA256,
        "result": SUFFIX_RESULT_SHA256,
    }
    if (
        role_manifest.get("output_sha256") != ROLE_ROWS_SHA256
        or role_manifest.get("contains_roles") != ["FIT_SELECTOR"]
        or role_manifest.get("parents") != role_parents
        or role_receipt.get("output_sha256") != ROLE_ROWS_SHA256
        or role_receipt.get("manifest_sha256") != ROLE_MANIFEST_SHA256
        or role_receipt.get("parents") != role_parents
        or role_receipt.get("summary") != role_manifest.get("summary")
        or role_receipt.get("authorized_for_fit_selector_calibration_input") is not True
        or role_receipt.get("authorized_for_validation") is not False
        or role_receipt.get("authorized_for_replication") is not False
        or suffix_receipt.get("bundle_sha256") != SUFFIX_BUNDLE_SHA256
        or suffix_receipt.get("result_sha256") != SUFFIX_RESULT_SHA256
        or suffix_receipt.get("parents") != suffix_parents
        or suffix_receipt.get("authorized_for_validation") is not True
        or suffix_receipt.get("authorized_for_replication") is not False
        or correction_receipt.get(
            "authorized_for_validation_with_original_selector_receipt"
        ) is not True
        or correction_receipt.get("authorized_for_replication") is not False
        or correction_receipt.get("supersedes_only")
        != "mlp2_cmr_v1_suffix_v2_result.json:support_overlaps"
        or correction_receipt.get("parents") != correction_parents
        or correction_receipt.get("result_sha256") != CORRECTION_SHA256
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


def _capability_system():
    seal = object()
    minted = False

    class Capability:
        __slots__ = ("nonce", "inode", "authority_sha256", "consumed")

        def __init__(
            self, provided_seal: object, nonce: str, inode: tuple[int, int],
            authority_sha256: str,
        ) -> None:
            if provided_seal is not seal:
                raise TypeError("MLP2 calibration capability is not directly constructible")
            self.nonce = nonce
            self.inode = inode
            self.authority_sha256 = authority_sha256
            self.consumed = False

        def __copy__(self):
            raise TypeError("MLP2 calibration capability cannot be copied")

        def __deepcopy__(self, memo):
            raise TypeError("MLP2 calibration capability cannot be copied")

    def mint(nonce: str, inode: tuple[int, int], authority_sha256: str) -> Capability:
        nonlocal minted
        if minted:
            raise RuntimeError("MLP2 calibration capability was already minted")
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
        minted = True
        return Capability(seal, nonce, inode, authority_sha256)

    def consume(capability: Capability) -> None:
        if type(capability) is not Capability or capability.consumed:
            raise RuntimeError("fresh MLP2 calibration capability required")
        validate_claim(capability.nonce, capability.inode)
        if not AUTHORITY.is_file() or file_sha256(
            AUTHORITY
        ) != capability.authority_sha256:
            raise RuntimeError("MLP2 calibration capability authority changed")
        capability.consumed = True

    return Capability, mint, consume


_CalibrationCapability, _mint_capability, _consume_capability = _capability_system()


def guard_inputs(
    source_hashes: dict[str, str], parents: dict[str, str], nonce: str,
    inode: tuple[int, int], authority_hash: str,
) -> None:
    validate_claim(nonce, inode)
    for relative, expected in source_hashes.items():
        if file_sha256(ROOT / relative) != expected:
            raise RuntimeError(f"calibration source changed during execution: {relative}")
    current, _ = protected_inputs()
    if current != parents or not AUTHORITY.is_file() or file_sha256(
        AUTHORITY
    ) != authority_hash or FAILURE.exists() or RECEIPT.exists():
        raise RuntimeError("MLP2 calibration protected input snapshot changed")


def final_guard(
    source_hashes: dict[str, str], parents: dict[str, str], nonce: str,
    inode: tuple[int, int], authority_hash: str, bundle_hash: str,
    result_hash: str,
) -> None:
    guard_inputs(source_hashes, parents, nonce, inode, authority_hash)
    if not BUNDLE.is_file() or file_sha256(BUNDLE) != bundle_hash or not (
        RESULT.is_file()
    ) or file_sha256(RESULT) != result_hash:
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


def _collect(
    role_rows_bytes: bytes, capability: _CalibrationCapability,
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.time()
    _consume_capability(capability)
    if hashlib.sha256(role_rows_bytes).hexdigest() != ROLE_ROWS_SHA256:
        raise RuntimeError("captured FIT_SELECTOR role-only bytes changed")
    role = torch.load(
        io.BytesIO(role_rows_bytes), map_location="cpu", weights_only=True,
    )
    role_summary = projection.validate_role(role)
    rows = role["rows"].contiguous()
    eligible = role["eligible_mask"].contiguous()
    frequency, frequency_bin_counts = target_frequency_reference(rows, eligible)
    cells = nearest_repeat_cells(rows, eligible)

    if not torch.cuda.is_available():
        raise RuntimeError("MLP2 calibration requires CUDA; CPU fallback is forbidden")
    if torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 5090":
        raise RuntimeError("MLP2 calibration requires the registered RTX 5090")
    device = torch.device("cuda:0")
    model, checkpoint = facade.load_bilin18(device=device, dtype=torch.bfloat16)
    checkpoint_after_load = facade.validate_snapshot()
    if checkpoint_after_load != checkpoint:
        raise RuntimeError("bilin18 checkpoint changed across strict model load")
    facade.validate_production_model(model)
    margin_parts: list[torch.Tensor] = []
    attention_calls_by_site = [0] * 18
    mlp_calls_by_site = [0] * 18
    forward_calls = forward_returns = 0
    with torch.inference_mode():
        for start in range(0, DOCUMENTS, BATCH):
            tokens = rows[start:start + BATCH, :-1].to(device)

            def attention(event: facade.AttentionEvent):
                if event.site < 0 or event.site >= 18:
                    raise RuntimeError("attention site outside registered ledger")
                attention_calls_by_site[event.site] += 1
                return event.block.attn(event.state, event.first_value)

            def mlp(event: facade.EarlyMLPEvent):
                if event.site < 0 or event.site >= 18:
                    raise RuntimeError("MLP site outside registered ledger")
                mlp_calls_by_site[event.site] += 1
                return event.block.mlp(event.state)

            forward_calls += 1
            logits = facade.forward_with_dispatch(model, tokens, attention, mlp)
            forward_returns += 1
            mask = eligible[start:start + BATCH].to(device)
            top2 = torch.topk(logits, 2, dim=-1).values
            margin_parts.append((top2[..., 0] - top2[..., 1])[mask].cpu().double())
            del logits, top2
            print(f"MLP2 calibration batch {forward_calls}/{CALLS}", flush=True)
    torch.cuda.synchronize(device)
    margins = torch.cat(margin_parts).contiguous()
    quantiles, grid = epsilon_grid(margins)
    if forward_calls != CALLS or forward_returns != CALLS or attention_calls_by_site != (
        [CALLS] * 18
    ) or mlp_calls_by_site != ([CALLS] * 18):
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
        "checkpoint_after_load": checkpoint_after_load.__dict__,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0),
        "model_dtype": str(torch.bfloat16),
        "strict_state_dict_load": True,
        "role_summary": role_summary,
        "documents": DOCUMENTS,
        "eligible_positions": ELIGIBLE_POSITIONS,
        "forward_calls": forward_calls,
        "forward_returns": forward_returns,
        "backward_calls": 0,
        "attention_calls": sum(attention_calls_by_site),
        "mlp_calls": sum(mlp_calls_by_site),
        "attention_calls_by_site": attention_calls_by_site,
        "mlp_calls_by_site": mlp_calls_by_site,
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
            **{
                f"fit_{name}_mask": tensor_sha256(mask)
                for name, mask in cells.items()
            },
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
    del model, margins, role
    torch.cuda.empty_cache()
    return summary, bundle


def validate_output_semantics(
    replay: Any, result: Any, role_rows_bytes: bytes,
) -> None:
    expected_keys = {
        "schema", "fit_token_counts", "frequency_boundaries",
        "margin_quantiles", "epsilon_grid",
    }
    if not isinstance(replay, dict) or set(replay) != expected_keys or replay.get(
        "schema"
    ) != (
        "mlp2_cmr_v1_fit_selector_calibration_bundle"
    ):
        raise RuntimeError("MLP2 calibration bundle schema replay failed")
    frequency = replay["fit_token_counts"]
    boundaries = replay["frequency_boundaries"]
    quantiles = replay["margin_quantiles"]
    grid = replay["epsilon_grid"]
    if not torch.is_tensor(frequency) or frequency.shape != (VOCAB,) or (
        frequency.dtype != torch.long
    ) or not torch.is_tensor(boundaries) or boundaries.dtype != torch.long or not (
        torch.equal(boundaries, torch.tensor(FREQUENCY_BOUNDARIES, dtype=torch.long))
    ) or not torch.is_tensor(quantiles) or quantiles.shape != (
        (len(MARGIN_QUANTILES),)
    ) or quantiles.dtype != torch.float64 or not bool(torch.isfinite(quantiles).all()) or (
        bool((quantiles < 0).any()) or not bool((quantiles[1:] >= quantiles[:-1]).all())
    ) or not torch.is_tensor(grid) or (
        grid.dtype != torch.float64 or grid.ndim != 1 or grid.numel() < len(DYADIC_EXPONENTS)
        or not bool(torch.isfinite(grid).all()) or bool((grid <= 0).any())
        or not bool((grid[1:] > grid[:-1]).all())
    ):
        raise RuntimeError("MLP2 calibration bundle tensor contract failed")
    if not set(2.0 ** exponent for exponent in DYADIC_EXPONENTS) <= set(grid.tolist()):
        raise RuntimeError("MLP2 calibration dyadic epsilon grid changed")
    expected_grid_values = [2.0 ** exponent for exponent in DYADIC_EXPONENTS]
    expected_grid_values.extend(
        float(value / 2.0) for value in quantiles.tolist() if value > 0
    )
    expected_grid = torch.tensor(
        sorted(set(expected_grid_values)), dtype=torch.float64,
    )
    if not torch.equal(grid, expected_grid):
        raise RuntimeError("MLP2 calibration epsilon grid is not the exact frozen union")
    role = torch.load(io.BytesIO(role_rows_bytes), map_location="cpu", weights_only=True)
    role_summary = projection.validate_role(role)
    expected_frequency, frequency_bins = target_frequency_reference(
        role["rows"], role["eligible_mask"],
    )
    cells = nearest_repeat_cells(role["rows"], role["eligible_mask"])
    if not torch.equal(frequency, expected_frequency):
        raise RuntimeError("MLP2 calibration token-only semantic replay failed")
    if not isinstance(result, dict) or result.get("schema") != (
        "mlp2_cmr_v1_fit_selector_calibration_result"
    ) or result.get("status") != (
        "fit_selector_calibration_complete_no_validation_or_replication"
    ) or result.get("role_summary") != role_summary or result.get("documents") != DOCUMENTS or (
        result.get("eligible_positions") != ELIGIBLE_POSITIONS
    ):
        raise RuntimeError("MLP2 calibration result role replay failed")
    if result.get("forward_calls") != CALLS or result.get("forward_returns") != CALLS or (
        result.get("backward_calls") != 0
        or result.get("attention_calls") != 18 * CALLS
        or result.get("mlp_calls") != 18 * CALLS
        or result.get("attention_calls_by_site") != [CALLS] * 18
        or result.get("mlp_calls_by_site") != [CALLS] * 18
    ):
        raise RuntimeError("MLP2 calibration result call-ledger replay failed")
    expected_cell_counts = {name: int(mask.sum()) for name, mask in cells.items()}
    expected_hashes = {
        "fit_token_counts": tensor_sha256(frequency),
        "margin_quantiles": tensor_sha256(quantiles),
        "epsilon_grid": tensor_sha256(grid),
        **{f"fit_{name}_mask": tensor_sha256(mask) for name, mask in cells.items()},
    }
    expected_quantile_summary = {
        format(q, ".6g"): float(value)
        for q, value in zip(MARGIN_QUANTILES, quantiles.tolist())
    }
    if result.get("fit_frequency_bin_counts") != frequency_bins.tolist() or result.get(
        "copy_cell_counts"
    ) != expected_cell_counts or result.get("tensor_hashes") != expected_hashes or result.get(
        "margin_quantiles"
    ) != expected_quantile_summary or result.get("epsilon_grid") != grid.tolist() or result.get(
        "epsilon_grid_count"
    ) != int(grid.numel()) or result.get("frequency_boundaries") != list(
        FREQUENCY_BOUNDARIES
    ):
        raise RuntimeError("MLP2 calibration result tensor-summary replay failed")
    if result.get("checkpoint") != result.get("checkpoint_after_load") or result.get(
        "checkpoint", {}
    ).get("weights_sha256") != facade.WEIGHTS_SHA256 or result.get("device") != (
        "cuda:0"
    ) or result.get("device_name") != "NVIDIA GeForce RTX 5090" or result.get(
        "model_dtype"
    ) != str(torch.bfloat16) or result.get("strict_state_dict_load") is not True:
        raise RuntimeError("MLP2 calibration runtime/checkpoint replay failed")
    if result.get("validation_opened") is not False or result.get(
        "replication_opened"
    ) is not False or result.get("finite_candidate_constructed") is not False or result.get(
        "raw_logits_published"
    ) is not False:
        raise RuntimeError("MLP2 calibration forbidden-outcome replay failed")
    scalar_values = [
        result.get("margin_minimum"), result.get("margin_mean"),
        result.get("margin_maximum"), result.get("runtime_seconds"),
    ]
    if any(not isinstance(value, (int, float)) for value in scalar_values) or not bool(
        torch.isfinite(torch.tensor(scalar_values, dtype=torch.float64)).all()
    ) or not (
        0 <= result["margin_minimum"] <= result["margin_mean"] <= result["margin_maximum"]
        and result["runtime_seconds"] > 0
        and result["margin_minimum"] <= float(quantiles.min())
        and float(quantiles.max()) <= result["margin_maximum"]
    ):
        raise RuntimeError("MLP2 calibration scalar-summary replay failed")


def main() -> None:
    forbidden_outputs = (AUTHORITY, BUNDLE, RESULT, RECEIPT, FAILURE, LOCK)
    if any(path.exists() for path in forbidden_outputs):
        raise RuntimeError("MLP2 calibration output already exists")
    source_commit, source_hashes = committed_source()
    parents, parent_bytes = protected_inputs()
    experiment_id = "bilin18_mlp2_cmr_v1_fit_selector_calibration"
    nonce = secrets.token_hex(32)
    authority: dict[str, Any] | None = None
    inode: tuple[int, int] | None = None
    try:
        write_create_only(LOCK, canonical_json_bytes({
            "experiment_id": experiment_id,
            "nonce": nonce,
        }))
        stat = LOCK.stat(follow_symlinks=False)
        inode = (stat.st_dev, stat.st_ino)
        validate_claim(nonce, inode)
        authority = {
            "schema_version": 1,
            "experiment_id": experiment_id,
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
        authority_hash = file_sha256(AUTHORITY)
        capability = _mint_capability(nonce, inode, authority_hash)
        summary, bundle = _collect(parent_bytes["role_rows"], capability)
        if not capability.consumed:
            raise RuntimeError("MLP2 calibration capability was not consumed")
        guard_inputs(source_hashes, parents, nonce, inode, authority_hash)
        publish_torch_create_only(BUNDLE, bundle)
        bundle_hash_before_load = file_sha256(BUNDLE)
        replay_bundle = torch.load(BUNDLE, map_location="cpu", weights_only=True)
        bundle_hash = file_sha256(BUNDLE)
        if bundle_hash != bundle_hash_before_load:
            raise RuntimeError("MLP2 calibration bundle changed across semantic load")
        summary["authority_sha256"] = authority_hash
        summary["bundle_sha256"] = bundle_hash
        write_create_only(RESULT, canonical_json_bytes(summary))
        result_bytes = RESULT.read_bytes()
        result_hash_before_load = hashlib.sha256(result_bytes).hexdigest()
        replay_result = json.loads(result_bytes)
        if file_sha256(RESULT) != result_hash_before_load or replay_result != summary:
            raise RuntimeError("MLP2 calibration result semantic replay failed")
        validate_output_semantics(
            replay_bundle, replay_result, parent_bytes["role_rows"],
        )
        result_hash = result_hash_before_load
        final_guard(
            source_hashes, parents, nonce, inode, authority_hash, bundle_hash,
            result_hash,
        )
        receipt = {
            "schema_version": 1,
            "experiment_id": experiment_id,
            "status": "fit_selector_calibration_complete_receipt_last",
            "authority_sha256": authority_hash,
            "bundle_sha256": bundle_hash,
            "result_sha256": result_hash,
            "lock_sha256": file_sha256(LOCK),
            "source_commit": source_commit,
            "source_hashes": source_hashes,
            "parents": parents,
            "authorized_for_validation_implementation": True,
            "authorized_for_validation_execution": False,
            "authorized_for_replication": False,
        }
        def receipt_prelink_guard() -> None:
            terminal_prelink_guard(
                RECEIPT, FAILURE,
                lambda: final_guard(
                    source_hashes, parents, nonce, inode, authority_hash, bundle_hash,
                    result_hash,
                ),
            )
            validate_claim(nonce, inode)

        write_create_only(
            RECEIPT, canonical_json_bytes(receipt), before_link=receipt_prelink_guard,
        )
    except BaseException as error:
        if not RECEIPT.exists() and not FAILURE.exists():
            failure = {
                "schema_version": 1,
                "experiment_id": experiment_id,
                "status": (
                    "failed_after_authority" if AUTHORITY.is_file()
                    else "failed_before_authority"
                ),
                "authority_sha256": (
                    file_sha256(AUTHORITY) if AUTHORITY.is_file() else None
                ),
                "error_type": type(error).__name__,
                "error": str(error),
                "validation_opened": False,
                "replication_opened": False,
            }
            def failure_prelink_guard() -> None:
                def protected_failure_guard() -> None:
                    if inode is not None:
                        validate_claim(nonce, inode)
                    if authority is not None and (
                        not AUTHORITY.is_file() or file_sha256(AUTHORITY) != hashlib.sha256(
                            canonical_json_bytes(authority)
                        ).hexdigest()
                    ):
                        raise RuntimeError("MLP2 calibration authority changed before failure")

                terminal_prelink_guard(
                    FAILURE, RECEIPT, protected_failure_guard,
                )

            try:
                write_create_only(
                    FAILURE, canonical_json_bytes(failure),
                    before_link=failure_prelink_guard,
                )
            except BaseException:
                pass
        raise


if __name__ == "__main__":
    main()
