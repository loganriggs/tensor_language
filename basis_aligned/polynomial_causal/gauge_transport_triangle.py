#!/usr/bin/env python3
"""Fail-closed causal transport screen for the L8 -> L11 -> L14 triangle.

This is Stage 1a of ``PRICED_GAUGE_TRANSPORT_SPEC.md``.  It answers, in order:

1. can the native callback harness reproduce the deployed forward exactly;
2. is the frozen U14 interface sufficient when given the *true* L14 response;
3. can a response map fitted on isotropic L8 perturbations predict an unseen
   donor-direction intervention;
4. do independently fitted L8->L11 and L11->L14 maps compose without consuming
   the true intermediate response?

Passing this screen does NOT license an interface.  It only licenses the expensive
20-null, behavior-cell, gauge-price, and alternate-background extension.  The script
requires the content-addressed FineWeb v2 receipt, opens no network stream, uses
disjoint basis/response-fit/evaluation rows, and patches every prediction into the
baseline L14 state.

Frozen before execution:
  * sites 8/11/14; post-block raw residuals; live downstream RMSNorm;
  * rank 64 within independently fit rank-256 token-deviation supports;
  * relative ridge 1e-3, no intercept; four isotropic perturbation draws;
  * scale chosen on the first sixteen response-fit rows from a fixed log grid so
    median per-row early-intervention KL lies in [0.01, 0.20];
  * response-fit uses the remaining rows; heldout family is a position-matched
    donor direction normalized to the frozen perturbation norm;
  * output scoring is over causal suffixes, using aggregate KL ratios on capped
    logits and centered relative RMSE on raw pre-softcap logits.
"""

from __future__ import annotations

import json
import hashlib
import math
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
BQ = HERE.parent / "bilinear_quotient"
QK = HERE.parent / "qk_mdl"
sys.path.insert(0, str(BQ))
sys.path.insert(0, str(QK))

from gauge_transport import fit_delta_ridge, response_metrics, response_r2  # noqa: E402
from source_global_preflight import require_defined_globals  # noqa: E402


D = 1152
K = 64
SUPPORT_RANK = 256
SITES = (8, 11, 14)
SEQ = 256
MIN_POSITION = 64
BATCH = 4
FIT_DRAWS = 4
RIDGE = 1e-3
SEED = 8675309
BASIS_SPEC = (96, 80)  # Legacy address retained only for old contract tests.
FIT_SPEC = (96, 1200)  # V2 execution uses the explicit roles below.
EVAL_SPEC = (192, 11000)
ROLE_SIZES = {"basis": 96, "fit": 96, "evaluation": 192}
ROW_TOKEN_LENGTH = 513
CALIBRATION_ROWS = 16
MAP_FIT_ROWS = 64
MAP_VALIDATION_ROWS = 16
SCALE_MULTIPLIERS = tuple(10.0 ** exponent for exponent in range(-2, 3))
KL_BAND = (0.01, 0.20)
POSITION_SHUFFLE_NRE_FLOOR = 0.25
PRICE_QUANTIZATION_STEP = 1e-4
OUT = HERE / "gauge_transport_triangle_results.json"
STATE_OUT = HERE / "gauge_transport_triangle_state.pt"
RUN_AUTHORITY = HERE / "gauge_transport_triangle_v1_execution_authority.json"
RUN_RECEIPT = HERE / "gauge_transport_triangle_v1_execution_receipt.json"
RUN_FAILURE = HERE / "gauge_transport_triangle_v1_execution_failure.json"
ROW_AUTHORITY = HERE / "gauge_transport_triangle_unique_rows_v2_authority.json"
ROW_MANIFEST = HERE / "gauge_transport_triangle_unique_rows_v2_manifest.json"
ROW_RECEIPT = HERE / "gauge_transport_triangle_unique_rows_v2_receipt.json"
ROW_ARTIFACT = HERE / "gauge_transport_triangle_unique_rows_v2_rows.pt"
ROW_FAILURE = HERE / "gauge_transport_triangle_unique_rows_v2_failure.json"
ROW_AUTHORITY_FILE_SHA256 = "6e10911a458da61ebf0cb0db09637d6d9eefa404fb138da26148e98eb532041f"
ROW_MANIFEST_FILE_SHA256 = "f86b9df20f0fc2ae5cef0e8f31ce4f02ca120821dfcab8de47d2a3166f5f5f1e"
ROW_RECEIPT_FILE_SHA256 = "3f92d8b3aa5e89e6059a010338521bffa0cf440e0815d9d67e1b65aa58a8e102"
ROW_ARTIFACT_FILE_SHA256 = "102b79726b7132a6438b4080272fee1774499ac4fc83c4aa025fa86439b4074d"
ROW_AUTHORITY_SHA256 = "99226c959912b22701c2df085029d9e082fda3af95c482a0d7483a319b368c3c"
ROW_MANIFEST_SHA256 = "f781231f1eca2a77a10bebb767eddc17a579b9f103e930fc0ba816bdfc1d68e2"
ROW_RECEIPT_SHA256 = "bfd6eeb3f7f8f5ce57ceb2fca6109f5b50f02c007725099c1082163ba0f81468"
ROW_SELECTION_PLAN_SHA256 = "0d66f060a43959c94afc14691b4a19730147c942da94807f919513fb8c421629"
RUN_SOURCE_FILES = (
    "basis_aligned/polynomial_causal/gauge_transport_triangle.py",
    "basis_aligned/polynomial_causal/gauge_transport.py",
    "basis_aligned/polynomial_causal/PRICED_GAUGE_TRANSPORT_SPEC.md",
    "basis_aligned/polynomial_causal/GAUGE_TRANSPORT_TRIANGLE_V1_EXECUTION_PREREGISTRATION.md",
    "basis_aligned/polynomial_causal/test_gauge_transport_triangle.py",
    "basis_aligned/polynomial_causal/test_gauge_transport_triangle_contract_audit.py",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=HERE.parents[1], text=True,
    ).strip()


def validate_execution_authority(value: dict[str, Any]) -> None:
    body = {key: item for key, item in value.items() if key != "authority_sha256"}
    if value.get("schema") != "gauge_transport_triangle_v1_execution_authority" or (
        value.get("status") != "source_closed_go"
        or value.get("authority_sha256") != canonical_sha256(body)
    ):
        raise RuntimeError("triangle execution authority identity changed")
    if tuple(value.get("source_files", ())) != RUN_SOURCE_FILES or set(
        value.get("source_sha256s", {})
    ) != set(RUN_SOURCE_FILES):
        raise RuntimeError("triangle execution source set changed")
    for relative in RUN_SOURCE_FILES:
        if file_sha256(HERE.parents[1] / relative) != value["source_sha256s"][relative]:
            raise RuntimeError(f"triangle execution source changed: {relative}")
    if value.get("row_artifact_file_sha256") != ROW_ARTIFACT_FILE_SHA256 or (
        value.get("row_receipt_file_sha256") != ROW_RECEIPT_FILE_SHA256
        or value.get("model_weights_sha256") is None
    ):
        raise RuntimeError("triangle execution input binding changed")
    if value.get("terminal_outputs") != {
        "result": OUT.name,
        "state": STATE_OUT.name,
        "receipt": RUN_RECEIPT.name,
        "failure": RUN_FAILURE.name,
    }:
        raise RuntimeError("triangle execution terminal namespace changed")


def require_source_closed_runner_lifecycle() -> dict[str, Any]:
    """Require a pushed immutable authority and a fresh create-only terminal namespace."""
    if not RUN_AUTHORITY.exists():
        raise RuntimeError("triangle execution authority is absent")
    authority = json.loads(RUN_AUTHORITY.read_text())
    validate_execution_authority(authority)
    commit = str(authority.get("source_commit", ""))
    if not commit:
        raise RuntimeError("triangle execution authority lacks source commit")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=HERE.parents[1], check=True,
    )
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "origin/main"],
        cwd=HERE.parents[1], check=True,
    )
    status = _git("status", "--porcelain", "--", *RUN_SOURCE_FILES,
                  str(RUN_AUTHORITY.relative_to(HERE.parents[1])))
    if status:
        raise RuntimeError(f"triangle execution sources are dirty: {status}")
    for terminal in (OUT, STATE_OUT, RUN_RECEIPT, RUN_FAILURE):
        if terminal.exists():
            raise RuntimeError(f"triangle terminal namespace is not fresh: {terminal.name}")
    return authority


def create_only_json(path: Path, value: dict[str, Any]) -> None:
    data = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def create_only_torch(path: Path, value: dict[str, Any]) -> None:
    with path.open("xb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())


def publish_execution(
    authority: dict[str, Any], output: dict[str, Any], state: dict[str, Any] | None,
) -> None:
    output["execution_authority_sha256"] = authority["authority_sha256"]
    create_only_json(OUT, output)
    if state is not None:
        create_only_torch(STATE_OUT, state)
    result_hash = file_sha256(OUT)
    state_hash = file_sha256(STATE_OUT) if STATE_OUT.exists() else None
    replay = json.loads(OUT.read_text())
    if replay != output or replay.get("execution_authority_sha256") != authority[
        "authority_sha256"
    ]:
        raise RuntimeError("triangle result semantic replay failed")
    receipt = {
        "schema": "gauge_transport_triangle_v1_execution_receipt",
        "status": "complete_receipt_last",
        "authority_sha256": authority["authority_sha256"],
        "result_file_sha256": result_hash,
        "state_file_sha256": state_hash,
        "failure_absent": not RUN_FAILURE.exists(),
        "result_status": output.get("config", {}).get("status"),
        "screen_passed": output.get("screen_passed"),
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    create_only_json(RUN_RECEIPT, receipt)


def composite_tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def load_pinned_json(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    before = file_sha256(path)
    serialized = path.read_bytes()
    after = file_sha256(path)
    if before != expected_sha256 or after != before or hashlib.sha256(
        serialized
    ).hexdigest() != before:
        raise RuntimeError(f"triangle {label} changed during pinned read")
    value = json.loads(serialized)
    if not isinstance(value, dict):
        raise RuntimeError(f"triangle {label} is not a JSON object")
    return value


def validate_v2_row_metadata(
    authority: dict[str, Any], manifest: dict[str, Any], receipt: dict[str, Any],
) -> None:
    authority_body = {key: value for key, value in authority.items() if key != "authority_sha256"}
    manifest_body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    receipt_body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if authority.get("schema") != "gauge_transport_triangle_unique_rows_v2_authority" or (
        authority.get("authority_sha256") != ROW_AUTHORITY_SHA256
        or canonical_sha256(authority_body) != ROW_AUTHORITY_SHA256
    ):
        raise RuntimeError("triangle row authority identity changed")
    if manifest.get("schema") != "gauge_transport_triangle_unique_rows_v2_manifest" or (
        manifest.get("manifest_sha256") != ROW_MANIFEST_SHA256
        or canonical_sha256(manifest_body) != ROW_MANIFEST_SHA256
    ):
        raise RuntimeError("triangle row manifest identity changed")
    if receipt.get("schema") != "gauge_transport_triangle_unique_rows_v2_receipt" or (
        receipt.get("status") != "complete_v2_unique_document_rows_receipt_last"
        or receipt.get("receipt_sha256") != ROW_RECEIPT_SHA256
        or canonical_sha256(receipt_body) != ROW_RECEIPT_SHA256
    ):
        raise RuntimeError("triangle row receipt identity changed")
    if receipt.get("failure_absent") is not True or ROW_FAILURE.exists():
        raise RuntimeError("triangle row receipt/failure exclusivity changed")
    if receipt.get("triangle_runner_authorized_by_this_receipt") is not False or (
        manifest.get("triangle_runner_authorized_by_this_manifest") is not False
    ):
        raise RuntimeError("row artifacts cannot themselves authorize the triangle runner")
    if manifest.get("role_sizes") != ROLE_SIZES or manifest.get(
        "unique_document_count"
    ) != sum(ROLE_SIZES.values()):
        raise RuntimeError("triangle row role allocation changed")
    if not (
        authority.get("selection_plan", {}).get("selection_plan_sha256")
        == manifest.get("selection_plan_sha256")
        == receipt.get("selection_plan_sha256")
        == ROW_SELECTION_PLAN_SHA256
    ):
        raise RuntimeError("triangle row selection identity changed")
    if not (
        receipt.get("authority_file_sha256") == ROW_AUTHORITY_FILE_SHA256
        and receipt.get("manifest_file_sha256") == ROW_MANIFEST_FILE_SHA256
        and receipt.get("rows_file_sha256") == ROW_ARTIFACT_FILE_SHA256
        and receipt.get("authority_sha256") == ROW_AUTHORITY_SHA256
        and receipt.get("manifest_sha256") == ROW_MANIFEST_SHA256
        and receipt.get("role_tensor_composite_sha256s")
        == manifest.get("role_tensor_composite_sha256s")
    ):
        raise RuntimeError("triangle row terminal hash chain changed")


def validate_v2_row_payload(
    payload: dict[str, Any], authority: dict[str, Any], manifest: dict[str, Any],
    receipt: dict[str, Any],
) -> dict[str, torch.Tensor]:
    if set(payload) != {
        "schema", "authority_sha256", "selection_plan_sha256", "roles", "records",
    } or payload.get("schema") != "gauge_transport_triangle_unique_rows_v2_rows" or (
        payload.get("authority_sha256") != ROW_AUTHORITY_SHA256
        or payload.get("selection_plan_sha256") != ROW_SELECTION_PLAN_SHA256
        or set(payload.get("roles", {})) != set(ROLE_SIZES)
        or set(payload.get("records", {})) != set(ROLE_SIZES)
    ):
        raise RuntimeError("triangle row payload header changed")
    documents = []
    for role, size in ROLE_SIZES.items():
        tensor = payload["roles"][role]
        records = payload["records"][role]
        if not torch.is_tensor(tensor) or tensor.dtype != torch.long or tuple(tensor.shape) != (
            size, ROW_TOKEN_LENGTH,
        ):
            raise RuntimeError(f"triangle row tensor changed: {role}")
        if records != authority["selection_plan"]["roles"][role] or len(records) != size:
            raise RuntimeError(f"triangle row provenance changed: {role}")
        if composite_tensor_sha256(tensor) != receipt[
            "role_tensor_composite_sha256s"
        ][role] or canonical_sha256(records) != manifest["role_record_sha256s"][role]:
            raise RuntimeError(f"triangle row semantic hash changed: {role}")
        documents.extend(record["document_id"] for record in records)
    if len(documents) != len(set(documents)):
        raise RuntimeError("triangle row documents are not globally unique")
    return dict(payload["roles"])


def load_unique_v2_rows() -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    authority = load_pinned_json(
        ROW_AUTHORITY, ROW_AUTHORITY_FILE_SHA256, "row authority",
    )
    manifest = load_pinned_json(ROW_MANIFEST, ROW_MANIFEST_FILE_SHA256, "row manifest")
    receipt = load_pinned_json(ROW_RECEIPT, ROW_RECEIPT_FILE_SHA256, "row receipt")
    validate_v2_row_metadata(authority, manifest, receipt)
    before = file_sha256(ROW_ARTIFACT)
    payload = torch.load(ROW_ARTIFACT, map_location="cpu", weights_only=True)
    after = file_sha256(ROW_ARTIFACT)
    if before != ROW_ARTIFACT_FILE_SHA256 or after != before:
        raise RuntimeError("triangle row tensor artifact changed during pinned load")
    rows = validate_v2_row_payload(payload, authority, manifest, receipt)
    if file_sha256(ROW_ARTIFACT) != before or ROW_FAILURE.exists():
        raise RuntimeError("triangle row terminal state changed after semantic load")
    return receipt, rows


def suffix_mask(positions: torch.Tensor, length: int = SEQ) -> torch.Tensor:
    if positions.ndim != 1 or positions.dtype != torch.long:
        raise ValueError("positions must be a rank-1 torch.long tensor")
    if bool(((positions < 0) | (positions >= length)).any()):
        raise ValueError("positions lie outside the sequence")
    index = torch.arange(length, device=positions.device)
    return index.unsqueeze(0) >= positions.unsqueeze(1)


def choose_scale(calibration: list[dict[str, float]]) -> dict[str, Any]:
    """Freeze the in-band scale nearest the geometric center; fail if none exists."""
    if not calibration:
        raise ValueError("scale calibration is empty")
    low, high = KL_BAND
    target = math.sqrt(low * high)
    in_band = [row for row in calibration if low <= row["median_suffix_kl"] <= high]
    if not in_band:
        return {"passed": False, "selected": None, "target": target}
    selected = min(
        in_band,
        key=lambda row: abs(math.log(max(row["median_suffix_kl"], 1e-30) / target)),
    )
    return {"passed": True, "selected": selected, "target": target}


def screen_decisions(metrics: dict[str, dict[str, float]]) -> dict[str, bool]:
    full = metrics["full_oracle"]
    projected = metrics["projected_oracle"]
    direct = metrics["direct"]
    chain = metrics["chain"]
    return {
        "full_oracle_exact": (
            full["e_out"] <= 1e-3
            and full["centered_raw_logit_relative_rmse"] <= 1e-3
        ),
        "projected_u14_sufficient": projected["e_out"] <= 0.25,
        "direct_response_transport": (
            direct["coordinate_response_r2"] >= 0.75 and direct["e_out"] <= 0.25
        ),
        "chain_composes": (
            chain["e_out"] <= 0.35
            and chain["e_out"] <= direct["e_out"] + 0.10
            and chain["coordinate_response_r2"]
            >= 0.75 * max(direct["coordinate_response_r2"], 0.0)
        ),
    }


def require_document_disjoint_receipt(receipt: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Require row-level FineWeb document/chunk provenance for the triangle claim."""
    provenance = receipt.get("document_provenance")
    if not isinstance(provenance, dict) or provenance.get("schema_version") != 1:
        raise RuntimeError(
            "triangle requires document_provenance schema 1; the oracle row receipt "
            "may not be reused from hashes/skips alone"
        )
    sets = provenance.get("sets")
    if not isinstance(sets, dict):
        raise RuntimeError("triangle receipt document_provenance lacks sets")
    required = (BASIS_SPEC, FIT_SPEC, EVAL_SPEC)
    output = {}
    seen_documents = {}
    for n, skip in required:
        key = f"n{n}_skip{skip}"
        rows = sets.get(key)
        if not isinstance(rows, list) or len(rows) != n:
            raise RuntimeError(f"triangle provenance has invalid row set {key}")
        documents = []
        for row in rows:
            if not isinstance(row, dict) or "document_id" not in row or "chunk_id" not in row:
                raise RuntimeError(f"triangle provenance row in {key} lacks document/chunk id")
            document = str(row["document_id"])
            documents.append(document)
            previous = seen_documents.get(document)
            if previous is not None and previous != key:
                raise RuntimeError(
                    f"FineWeb document {document} crosses triangle splits {previous} and {key}"
                )
            seen_documents[document] = key
        if len(set(documents)) != len(documents):
            raise RuntimeError(f"triangle headline requires one sequence per document in {key}")
        output[key] = rows
    return output


@torch.no_grad()
def native_forward(
    model,
    idx: torch.Tensor,
    *,
    patch_layer: int | None = None,
    patch_delta: torch.Tensor | None = None,
    capture_sites: tuple[int, ...] = (),
    return_logits: bool = True,
):
    """Native block loop with a single additive raw post-block callback."""
    if (patch_layer is None) != (patch_delta is None):
        raise ValueError("patch_layer and patch_delta must be provided together")
    x = F.rms_norm(model.transformer.wte(idx), (D,))
    x0 = x
    v1 = None
    captures = {}
    for layer, block in enumerate(model.transformer.h):
        x, v1 = block(x, v1, x0)
        if layer == patch_layer:
            if patch_delta.shape != x.shape:
                raise ValueError(f"patch shape {patch_delta.shape} != residual shape {x.shape}")
            x = x + patch_delta.to(device=x.device, dtype=x.dtype)
        if layer in capture_sites:
            captures[layer] = x.detach().float()
    if not return_logits:
        return None, None, captures
    normalized = F.rms_norm(x, (D,))
    raw = model.lm_head(normalized).float()
    capped = 30.0 * torch.tanh(raw / 30.0)
    return raw, capped, captures


@torch.no_grad()
def cp_gauge_canary(model) -> dict[str, float | bool]:
    """Verify an actual bilin18 CP scale/sign/permutation rewrite without mutation."""
    mlp = model.transformer.h[0].mlp
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(SEED + 17)
    x = torch.randn(32, D, generator=generator, device=device)
    original = mlp(x).float()
    left = mlp.Left.weight.float()
    right = mlp.Right.weight.float()
    down = mlp.Down.weight.float()
    rank = left.shape[0]
    permutation = torch.randperm(rank, generator=generator, device=device)
    alpha = torch.exp(torch.linspace(-1.0, 1.0, rank, device=device))
    beta = torch.exp(torch.linspace(0.7, -0.7, rank, device=device))
    alpha[::2] *= -1
    beta[1::2] *= -1
    rewritten_left = left[permutation] * alpha[:, None]
    rewritten_right = right[permutation] * beta[:, None]
    rewritten_down = down[:, permutation] / (alpha * beta)[None, :]
    product = F.linear(x, rewritten_left) * F.linear(x, rewritten_right)
    rewritten = F.linear(product, rewritten_down, mlp.Down_bias.float()).float()
    relative = math.sqrt(
        float((rewritten - original).double().square().sum())
        / max(float(original.double().square().sum()), 1e-30)
    )
    return {"relative_rmse": relative, "passed": relative <= 1e-5}


def sparse_physical_delta(
    coordinates: torch.Tensor,
    basis: torch.Tensor,
    positions: torch.Tensor,
    length: int = SEQ,
) -> torch.Tensor:
    if coordinates.ndim != 2 or coordinates.shape[1] != basis.shape[1]:
        raise ValueError("coordinate and basis dimensions disagree")
    if len(positions) != len(coordinates):
        raise ValueError("one intervention position is required per row")
    delta = torch.zeros(
        len(coordinates), length, basis.shape[0],
        device=coordinates.device, dtype=coordinates.dtype,
    )
    delta[torch.arange(len(coordinates), device=coordinates.device), positions] = (
        coordinates @ basis.T
    )
    return delta


@torch.no_grad()
def capture_rows(model, rows: torch.Tensor) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
    captures = {site: [] for site in SITES}
    tokens = []
    for start in range(0, len(rows), BATCH):
        block = rows[start:start + BATCH, :SEQ + 1]
        idx = block[:, :-1].to(next(model.parameters()).device)
        _, _, states = native_forward(model, idx, capture_sites=SITES, return_logits=False)
        for site in SITES:
            captures[site].append(states[site].cpu())
        tokens.append(idx.cpu())
    return {site: torch.cat(parts) for site, parts in captures.items()}, torch.cat(tokens)


@torch.no_grad()
def local_token_deviation_support(
    residual: torch.Tensor,
    tokens: torch.Tensor,
    vocab: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    values = residual.reshape(-1, D).to(device)
    token = tokens.reshape(-1).long().to(device)
    counts = torch.bincount(token, minlength=vocab).float()
    means = torch.zeros(vocab, D, device=device)
    means.index_add_(0, token, values)
    means /= counts.clamp_min(1).unsqueeze(1)
    deviation = values - means[token]
    deviation -= deviation.mean(0)
    covariance = deviation.T @ deviation / max(len(deviation) - 1, 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance.double())
    order = eigenvalues.argsort(descending=True)[:SUPPORT_RANK]
    support = eigenvectors[:, order].float().contiguous()
    basis = support[:, :K].contiguous()
    natural_scale = math.sqrt(float(eigenvalues[order].clamp_min(0).mean()))
    return basis, support, natural_scale


def deterministic_positions(n: int, draw: int, *, offset: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(SEED + 1000 * offset + draw)
    return torch.randint(MIN_POSITION, SEQ, (n,), generator=generator, dtype=torch.long)


def deterministic_coordinates(n: int, draw: int, amplitude: float, *, offset: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(SEED + 10000 * offset + draw)
    direction = torch.randn(n, K, generator=generator)
    direction /= torch.linalg.vector_norm(direction, dim=1, keepdim=True).clamp_min(1e-12)
    return direction * (amplitude * math.sqrt(K))


def deterministic_support_coordinates(
    n: int, draw: int, amplitude: float, *, offset: int
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(SEED + 10000 * offset + draw)
    direction = torch.randn(n, SUPPORT_RANK, generator=generator)
    direction /= torch.linalg.vector_norm(direction, dim=1, keepdim=True).clamp_min(1e-12)
    return direction * (amplitude * math.sqrt(SUPPORT_RANK))


@torch.no_grad()
def per_row_suffix_kl(
    early_logits: torch.Tensor,
    baseline_logits: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    log_early = F.log_softmax(early_logits.double(), dim=-1)
    kl = (log_early.exp() * (log_early - F.log_softmax(baseline_logits.double(), dim=-1))).sum(-1)
    mask = suffix_mask(positions.to(kl.device), kl.shape[1])
    return (kl * mask).sum(1) / mask.sum(1)


@torch.no_grad()
def calibrate_scale(model, rows: torch.Tensor, support8: torch.Tensor, natural_scale: float):
    device = next(model.parameters()).device
    rows = rows[:CALIBRATION_ROWS, :SEQ + 1]
    idx = rows[:, :-1].to(device)
    _, baseline, _ = native_forward(model, idx)
    positions = deterministic_positions(len(rows), 0, offset=7).to(device)
    table = []
    for multiplier in SCALE_MULTIPLIERS:
        amplitude = natural_scale * multiplier
        coordinates = deterministic_support_coordinates(
            len(rows), 0, amplitude, offset=71 + int(round(math.log10(multiplier)))
        ).to(device)
        delta = sparse_physical_delta(coordinates, support8, positions)
        _, early, _ = native_forward(model, idx, patch_layer=8, patch_delta=delta)
        median = float(per_row_suffix_kl(early, baseline, positions).median())
        table.append({
            "multiplier": multiplier,
            "amplitude": amplitude,
            "median_suffix_kl": median,
        })
        del delta, early
    return table, choose_scale(table)


@torch.no_grad()
def position_shuffle_canary(
    model, rows: torch.Tensor, support8: torch.Tensor, amplitude: float
) -> dict[str, float | bool]:
    device = next(model.parameters()).device
    idx = rows[:CALIBRATION_ROWS, :SEQ].to(device)
    _, baseline, _ = native_forward(model, idx)
    positions = deterministic_positions(len(idx), 3, offset=703).to(device)
    shuffled_positions = (
        MIN_POSITION + (positions - MIN_POSITION + 53) % (SEQ - MIN_POSITION)
    )
    coordinates = deterministic_support_coordinates(
        len(idx), 3, amplitude, offset=704
    ).to(device)
    correct_delta = sparse_physical_delta(coordinates, support8, positions)
    shuffled_delta = sparse_physical_delta(coordinates, support8, shuffled_positions)
    _, correct, _ = native_forward(model, idx, patch_layer=8, patch_delta=correct_delta)
    _, shuffled, _ = native_forward(model, idx, patch_layer=8, patch_delta=shuffled_delta)
    mask = suffix_mask(positions, SEQ)
    reference = (correct - baseline)[mask]
    prediction = (shuffled - baseline)[mask]
    nre = response_metrics(reference, prediction)["nre"]
    return {
        "response_nre": nre,
        "registered_floor": POSITION_SHUFFLE_NRE_FLOOR,
        "passed": nre >= POSITION_SHUFFLE_NRE_FLOOR,
    }


def expanded_physical_price_canary(
    bases: dict[int, torch.Tensor], maps: dict[str, torch.Tensor]
) -> dict[str, Any]:
    """Gauge-check the safe expanded-ambient linear-codec price upper bound."""
    pricing_root = Path("/workspace/theseus-bench/research/tensor_program_pricing")
    sys.path.insert(0, str(pricing_root))
    from pricing import price_bits

    device = bases[8].device
    generator = torch.Generator(device=device).manual_seed(SEED + 505)
    gauges = {}
    for site in SITES:
        q, _ = torch.linalg.qr(
            torch.randn(K, K, generator=generator, device=device), mode="reduced"
        )
        gauges[site] = q
    rows = {}
    maximum_response_drift = 0.0
    maximum_price_drift = 0.0
    for name, source, destination in (
        ("8_11", 8, 11), ("8_14", 8, 14), ("11_14", 11, 14)
    ):
        weight = maps[name].to(device)
        physical = bases[source] @ weight @ bases[destination].T
        source_basis_prime = bases[source] @ gauges[source]
        destination_basis_prime = bases[destination] @ gauges[destination]
        weight_prime = gauges[source].T @ weight @ gauges[destination]
        physical_prime = source_basis_prime @ weight_prime @ destination_basis_prime.T
        drift = math.sqrt(
            float((physical_prime - physical).double().square().sum())
            / max(float(physical.double().square().sum()), 1e-30)
        )
        program = {"nodes": [{"name": name, "op": "linear", "weight": physical,
                                "rank": K}]}
        program_prime = {"nodes": [{"name": name, "op": "linear",
                                      "weight": physical_prime, "rank": K}]}
        bits = price_bits(program, PRICE_QUANTIZATION_STEP)
        bits_prime = price_bits(program_prime, PRICE_QUANTIZATION_STEP)
        price_drift = abs(bits_prime - bits) / max(bits, 1)
        rows[name] = {
            "physical_response_relative_drift": drift,
            "expanded_price_bits": bits,
            "gauged_expanded_price_bits": bits_prime,
            "price_relative_drift": price_drift,
        }
        maximum_response_drift = max(maximum_response_drift, drift)
        maximum_price_drift = max(maximum_price_drift, price_drift)
    return {
        "maps": rows,
        "maximum_response_relative_drift": maximum_response_drift,
        "maximum_price_relative_drift": maximum_price_drift,
        "passed": maximum_response_drift <= 1e-5 and maximum_price_drift <= 0.01,
        "pricing_scope": "expanded ambient SVD upper bound; shared-basis DAG price not certified",
    }


@torch.no_grad()
def fit_triangle_maps(
    model,
    rows: torch.Tensor,
    bases: dict[int, torch.Tensor],
    supports: dict[int, torch.Tensor],
    amplitude: float,
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    device = next(model.parameters()).device
    source_8, destination_11, destination_14 = [], [], []
    source_11, destination_14_from_11 = [], []
    rows = rows[CALIBRATION_ROWS:CALIBRATION_ROWS + MAP_FIT_ROWS]
    for start in range(0, len(rows), BATCH):
        idx = rows[start:start + BATCH, :SEQ].to(device)
        _, _, baseline = native_forward(model, idx, capture_sites=(11, 14), return_logits=False)
        global_start = CALIBRATION_ROWS + start
        for draw in range(FIT_DRAWS):
            positions = deterministic_positions(len(rows), draw, offset=global_start).to(device)
            positions = positions[start:start + len(idx)]
            support_coord8 = deterministic_support_coordinates(
                len(rows), draw, amplitude, offset=200 + global_start
            )[start:start + len(idx)].to(device)
            coord8 = support_coord8 @ (supports[8].T @ bases[8])
            support_coord11 = deterministic_support_coordinates(
                len(rows), draw, amplitude, offset=400 + global_start
            )[start:start + len(idx)].to(device)
            coord11 = support_coord11 @ (supports[11].T @ bases[11])
            for sign in (-1.0, 1.0):
                delta8 = sparse_physical_delta(sign * support_coord8, supports[8], positions)
                _, _, early8 = native_forward(
                    model, idx, patch_layer=8, patch_delta=delta8,
                    capture_sites=(11, 14), return_logits=False,
                )
                row_index = torch.arange(len(idx), device=device)
                dc11 = (early8[11] - baseline[11])[row_index, positions] @ bases[11]
                dc14 = (early8[14] - baseline[14])[row_index, positions] @ bases[14]
                source_8.append((sign * coord8).cpu())
                destination_11.append(dc11.cpu())
                destination_14.append(dc14.cpu())

                delta11 = sparse_physical_delta(sign * support_coord11, supports[11], positions)
                _, _, early11 = native_forward(
                    model, idx, patch_layer=11, patch_delta=delta11,
                    capture_sites=(14,), return_logits=False,
                )
                dc14_from_11 = (early11[14] - baseline[14])[row_index, positions] @ bases[14]
                source_11.append((sign * coord11).cpu())
                destination_14_from_11.append(dc14_from_11.cpu())
                del delta8, delta11, early8, early11

    x8 = torch.cat(source_8)
    y11 = torch.cat(destination_11)
    y14 = torch.cat(destination_14)
    x11 = torch.cat(source_11)
    y14_from_11 = torch.cat(destination_14_from_11)
    maps = {
        "8_11": fit_delta_ridge(x8, y11, relative_ridge=RIDGE).float(),
        "8_14": fit_delta_ridge(x8, y14, relative_ridge=RIDGE).float(),
        "11_14": fit_delta_ridge(x11, y14_from_11, relative_ridge=RIDGE).float(),
    }
    counts = {"l8_examples": len(x8), "l11_examples": len(x11)}
    return maps, counts


@torch.no_grad()
def validate_triangle_maps(
    model,
    rows: torch.Tensor,
    bases: dict[int, torch.Tensor],
    supports: dict[int, torch.Tensor],
    maps: dict[str, torch.Tensor],
    amplitude: float,
) -> dict[str, float]:
    """Row-grouped diagnostic only; validation outcomes select no hyperparameter."""
    device = next(model.parameters()).device
    rows = rows[
        CALIBRATION_ROWS + MAP_FIT_ROWS:
        CALIBRATION_ROWS + MAP_FIT_ROWS + MAP_VALIDATION_ROWS
    ]
    idx = rows[:, :SEQ].to(device)
    _, _, baseline = native_forward(model, idx, capture_sites=(11, 14), return_logits=False)
    positions = deterministic_positions(len(rows), 0, offset=808).to(device)
    row_index = torch.arange(len(rows), device=device)
    output: dict[str, list[torch.Tensor]] = {
        "true11": [], "pred11": [], "true14_direct": [], "pred14_direct": [],
        "true14_mid": [], "pred14_mid": [],
    }
    support8 = deterministic_support_coordinates(len(rows), 0, amplitude, offset=809).to(device)
    support11 = deterministic_support_coordinates(len(rows), 0, amplitude, offset=810).to(device)
    coord8 = support8 @ (supports[8].T @ bases[8])
    coord11 = support11 @ (supports[11].T @ bases[11])
    for sign in (-1.0, 1.0):
        delta8 = sparse_physical_delta(sign * support8, supports[8], positions)
        _, _, early8 = native_forward(
            model, idx, patch_layer=8, patch_delta=delta8,
            capture_sites=(11, 14), return_logits=False,
        )
        true11 = (early8[11] - baseline[11])[row_index, positions] @ bases[11]
        true14 = (early8[14] - baseline[14])[row_index, positions] @ bases[14]
        output["true11"].append(true11.cpu())
        output["pred11"].append((sign * coord8 @ maps["8_11"].to(device)).cpu())
        output["true14_direct"].append(true14.cpu())
        output["pred14_direct"].append((sign * coord8 @ maps["8_14"].to(device)).cpu())

        delta11 = sparse_physical_delta(sign * support11, supports[11], positions)
        _, _, early11 = native_forward(
            model, idx, patch_layer=11, patch_delta=delta11,
            capture_sites=(14,), return_logits=False,
        )
        true14_mid = (early11[14] - baseline[14])[row_index, positions] @ bases[14]
        output["true14_mid"].append(true14_mid.cpu())
        output["pred14_mid"].append((sign * coord11 @ maps["11_14"].to(device)).cpu())
    return {
        "r2_8_11": response_r2(torch.cat(output["true11"]), torch.cat(output["pred11"])),
        "r2_8_14": response_r2(
            torch.cat(output["true14_direct"]), torch.cat(output["pred14_direct"])
        ),
        "r2_11_14": response_r2(
            torch.cat(output["true14_mid"]), torch.cat(output["pred14_mid"])
        ),
        "selects_hyperparameters": False,
    }


def empty_arm_accumulator() -> dict[str, float]:
    return {"kl_error": 0.0, "kl_target": 0.0, "raw_error": 0.0, "raw_target": 0.0}


@torch.no_grad()
def accumulate_output_arm(
    accumulator: dict[str, float],
    baseline_raw: torch.Tensor,
    baseline_capped: torch.Tensor,
    early_raw: torch.Tensor,
    early_capped: torch.Tensor,
    transported_raw: torch.Tensor,
    transported_capped: torch.Tensor,
    positions: torch.Tensor,
) -> None:
    mask = suffix_mask(positions.to(early_capped.device), early_capped.shape[1])
    base_c = baseline_capped[mask].double()
    early_c = early_capped[mask].double()
    trans_c = transported_capped[mask].double()
    log_early = F.log_softmax(early_c, dim=-1)
    probability = log_early.exp()
    accumulator["kl_error"] += float(
        (probability * (log_early - F.log_softmax(trans_c, dim=-1))).sum()
    )
    accumulator["kl_target"] += float(
        (probability * (log_early - F.log_softmax(base_c, dim=-1))).sum()
    )
    base_r = baseline_raw[mask].double()
    early_r = early_raw[mask].double()
    trans_r = transported_raw[mask].double()
    base_r -= base_r.mean(-1, keepdim=True)
    early_r -= early_r.mean(-1, keepdim=True)
    trans_r -= trans_r.mean(-1, keepdim=True)
    accumulator["raw_error"] += float((trans_r - early_r).square().sum())
    accumulator["raw_target"] += float((early_r - base_r).square().sum())


def finish_output_arm(accumulator: dict[str, float]) -> dict[str, float]:
    if accumulator["kl_target"] <= 0 or accumulator["raw_target"] <= 0:
        raise RuntimeError("output intervention denominator is zero")
    return {
        "e_out": accumulator["kl_error"] / accumulator["kl_target"],
        "centered_raw_logit_relative_rmse": math.sqrt(
            accumulator["raw_error"] / accumulator["raw_target"]
        ),
        "early_vs_baseline_kl_sum": accumulator["kl_target"],
        "early_vs_transport_kl_sum": accumulator["kl_error"],
    }


@torch.no_grad()
def evaluate_triangle(
    model,
    rows: torch.Tensor,
    bases: dict[int, torch.Tensor],
    supports: dict[int, torch.Tensor],
    maps: dict[str, torch.Tensor],
    amplitude: float,
) -> dict[str, dict[str, float]]:
    device = next(model.parameters()).device
    half = len(rows) // 2
    source_rows = rows[:half]
    target_rows = rows[half:2 * half]
    accumulators = {
        name: empty_arm_accumulator()
        for name in (
            "full_l11_oracle", "projected_l11_oracle", "predicted_l11",
            "full_oracle", "projected_oracle", "direct", "chain",
        )
    }
    true_coordinates, direct_coordinates, chain_coordinates = [], [], []
    for start in range(0, half, BATCH):
        source_idx = source_rows[start:start + BATCH, :SEQ].to(device)
        target_idx = target_rows[start:start + BATCH, :SEQ].to(device)
        if len(source_idx) != len(target_idx):
            continue
        _, _, source_state = native_forward(
            model, source_idx, capture_sites=(8,), return_logits=False
        )
        baseline_raw, baseline_capped, baseline_state = native_forward(
            model, target_idx, capture_sites=(8, 11, 14)
        )
        positions = deterministic_positions(half, 0, offset=900)[start:start + len(target_idx)].to(device)
        row_index = torch.arange(len(target_idx), device=device)
        donor_physical = (
            source_state[8][row_index, positions] - baseline_state[8][row_index, positions]
        )
        donor_physical = (donor_physical @ supports[8]) @ supports[8].T
        donor_physical /= torch.linalg.vector_norm(
            donor_physical, dim=1, keepdim=True
        ).clamp_min(1e-12)
        donor_physical *= amplitude * math.sqrt(SUPPORT_RANK)
        dc8 = donor_physical @ bases[8]
        delta8 = torch.zeros(len(target_idx), SEQ, D, device=device)
        delta8[row_index, positions] = donor_physical
        early_raw, early_capped, early_state = native_forward(
            model, target_idx, patch_layer=8, patch_delta=delta8, capture_sites=(11, 14)
        )
        physical11 = early_state[11] - baseline_state[11]
        physical14 = early_state[14] - baseline_state[14]
        true_dc14 = physical14[row_index, positions] @ bases[14]
        predicted_direct = dc8 @ maps["8_14"].to(device)
        predicted_chain = dc8 @ maps["8_11"].to(device) @ maps["11_14"].to(device)
        true_coordinates.append(true_dc14.cpu())
        direct_coordinates.append(predicted_direct.cpu())
        chain_coordinates.append(predicted_chain.cpu())

        projected14 = (physical14 @ bases[14]) @ bases[14].T
        projected11 = (physical11 @ bases[11]) @ bases[11].T
        predicted11_coordinates = dc8 @ maps["8_11"].to(device)
        predicted11 = sparse_physical_delta(predicted11_coordinates, bases[11], positions)
        direct14 = sparse_physical_delta(predicted_direct, bases[14], positions)
        chain14 = sparse_physical_delta(predicted_chain, bases[14], positions)
        arm_deltas = {
            "full_l11_oracle": (11, physical11),
            "projected_l11_oracle": (11, projected11),
            "predicted_l11": (11, predicted11),
            "full_oracle": (14, physical14),
            "projected_oracle": (14, projected14),
            "direct": (14, direct14),
            "chain": (14, chain14),
        }
        for name, (patch_layer, delta_value) in arm_deltas.items():
            arm_raw, arm_capped, _ = native_forward(
                model, target_idx, patch_layer=patch_layer, patch_delta=delta_value
            )
            accumulate_output_arm(
                accumulators[name], baseline_raw, baseline_capped,
                early_raw, early_capped, arm_raw, arm_capped, positions,
            )
            del arm_raw, arm_capped
        del source_state, baseline_state, early_state, physical14, projected14

    result = {name: finish_output_arm(value) for name, value in accumulators.items()}
    true = torch.cat(true_coordinates)
    result["direct"]["coordinate_response_r2"] = response_r2(
        true, torch.cat(direct_coordinates)
    )
    result["chain"]["coordinate_response_r2"] = response_r2(
        true, torch.cat(chain_coordinates)
    )
    result["full_oracle"]["coordinate_response_r2"] = 1.0
    result["projected_oracle"]["coordinate_response_r2"] = 1.0
    for name in ("full_l11_oracle", "projected_l11_oracle", "predicted_l11"):
        result[name]["coordinate_response_r2"] = None
    return result


@torch.no_grad()
def harness_canaries(model, row: torch.Tensor) -> dict[str, Any]:
    from tier2_model import reference_forward

    device = next(model.parameters()).device
    idx = row[:2, :SEQ].to(device)
    native_raw, native_capped, _ = native_forward(model, idx)
    reference = reference_forward(model, idx, "bf16").float()
    reference_relative_rmse = math.sqrt(
        float((native_capped - reference).double().square().sum())
        / max(float(reference.double().square().sum()), 1e-30)
    )
    zero = torch.zeros(len(idx), SEQ, D, device=device)
    zero_raw, zero_capped, _ = native_forward(
        model, idx, patch_layer=8, patch_delta=zero
    )
    zero_relative_rmse = math.sqrt(
        float((zero_capped - native_capped).double().square().sum())
        / max(float(native_capped.double().square().sum()), 1e-30)
    )
    return {
        "native_vs_reference_relative_rmse": reference_relative_rmse,
        "zero_patch_relative_rmse": zero_relative_rmse,
        "passed": reference_relative_rmse <= 1e-5 and zero_relative_rmse <= 1e-7,
        "raw_zero_patch_max_abs": float((zero_raw - native_raw).abs().max()),
        "cp_gauge_rewrite": cp_gauge_canary(model),
    }


def scientific_run(
    authority: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    require_defined_globals([Path(__file__), Path(__file__).with_name("gauge_transport.py")])
    start = time.time()
    receipt, rows = load_unique_v2_rows()
    from bilin18_observed_model_facade import load_bilin18

    model, model_receipt = load_bilin18(verify_weights_sha256=True)
    if model_receipt.weights_sha256 != authority["model_weights_sha256"]:
        raise RuntimeError("triangle checkpoint differs from execution authority")
    device = next(model.parameters()).device
    canaries = harness_canaries(model, rows["basis"])
    canaries["passed"] = canaries["passed"] and bool(canaries["cp_gauge_rewrite"]["passed"])
    if not canaries["passed"]:
        raise RuntimeError(f"native intervention harness canary failed: {canaries}")

    basis_states, basis_tokens = capture_rows(model, rows["basis"])
    bases, supports, natural_scales = {}, {}, {}
    vocab = int(model.lm_head.weight.shape[0])
    for site in SITES:
        bases[site], supports[site], natural_scales[site] = local_token_deviation_support(
            basis_states[site], basis_tokens, vocab, device
        )
        del basis_states[site]
        torch.cuda.empty_cache()

    calibration, scale_decision = calibrate_scale(
        model, rows["fit"], supports[8], natural_scales[8]
    )
    if not scale_decision["passed"]:
        output = {
            "config": {
                "status": "scale_calibration_failed_before_response_fit",
                "model": "bilin18",
                "model_revision": model_receipt.revision,
                "model_config_sha256": model_receipt.config_sha256,
                "model_weights_sha256": model_receipt.weights_sha256,
                "row_receipt_file_sha256": ROW_RECEIPT_FILE_SHA256,
            },
            "canaries": canaries,
            "scale_calibration": calibration,
            "scale_decision": scale_decision,
            "runtime_s": round(time.time() - start, 1),
        }
        return output, None
    amplitude = float(scale_decision["selected"]["amplitude"])
    shuffle_canary = position_shuffle_canary(
        model, rows["fit"], supports[8], amplitude
    )
    if not shuffle_canary["passed"]:
        raise RuntimeError(f"position-shuffle negative control is underpowered: {shuffle_canary}")
    maps, fit_counts = fit_triangle_maps(
        model, rows["fit"], bases, supports, amplitude
    )
    map_validation = validate_triangle_maps(
        model, rows["fit"], bases, supports, maps, amplitude
    )
    price_canary = expanded_physical_price_canary(bases, maps)
    if not price_canary["passed"]:
        raise RuntimeError(f"physical transport gauge/price canary failed: {price_canary}")
    metrics = evaluate_triangle(
        model, rows["evaluation"], bases, supports, maps, amplitude
    )
    decisions = screen_decisions(metrics)
    screen_passed = all(decisions.values())
    output = {
        "config": {
            "model": "bilin18",
            "model_revision": model_receipt.revision,
            "model_config_sha256": model_receipt.config_sha256,
            "model_weights_sha256": model_receipt.weights_sha256,
            "sites": list(SITES),
            "residual_semantics": "raw post-block; downstream RMSNorm live",
            "rank": K,
            "support_rank": SUPPORT_RANK,
            "basis_spec": list(BASIS_SPEC),
            "response_fit_spec": list(FIT_SPEC),
            "evaluation_spec": list(EVAL_SPEC),
            "sequence_length": SEQ,
            "fit_draws": FIT_DRAWS,
            "response_rows": {
                "scale_calibration": CALIBRATION_ROWS,
                "map_fit": MAP_FIT_ROWS,
                "row_group_validation": MAP_VALIDATION_ROWS,
            },
            "relative_ridge": RIDGE,
            "seed": SEED,
            "row_receipt_schema": receipt["schema"],
            "row_receipt_file_sha256": ROW_RECEIPT_FILE_SHA256,
            "row_selection_plan_sha256": ROW_SELECTION_PLAN_SHA256,
            "status": (
                "preliminary_screen_passed_requires_20_null_behavior_gauge_price_extension"
                if screen_passed else
                "preliminary_screen_failed_no_interface_license"
            ),
        },
        "canaries": canaries,
        "position_shuffle_canary": shuffle_canary,
        "expanded_physical_price_canary": price_canary,
        "natural_coordinate_scales": natural_scales,
        "scale_calibration": calibration,
        "scale_decision": scale_decision,
        "fit_counts": fit_counts,
        "row_group_map_validation": map_validation,
        "metrics": metrics,
        "decisions": decisions,
        "screen_passed": screen_passed,
        "not_licensed": "No interface claim until all 20 matched nulls, behavior cells, complete gauges, price, and alternate backgrounds pass.",
        "runtime_s": round(time.time() - start, 1),
    }
    state = {
        "config": output["config"],
        "bases": {site: basis.cpu() for site, basis in bases.items()},
        "supports": {site: support.cpu() for site, support in supports.items()},
        "maps": maps,
    }
    return output, state


def main() -> None:
    authority = require_source_closed_runner_lifecycle()
    try:
        output, state = scientific_run(authority)
        publish_execution(authority, output, state)
    except BaseException as error:
        if not RUN_FAILURE.exists() and not RUN_RECEIPT.exists():
            failure = {
                "schema": "gauge_transport_triangle_v1_execution_failure",
                "status": "failed_before_receipt",
                "authority_sha256": authority["authority_sha256"],
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "partial_result_sha256": file_sha256(OUT) if OUT.exists() else None,
                "partial_state_sha256": file_sha256(STATE_OUT) if STATE_OUT.exists() else None,
            }
            failure["failure_sha256"] = canonical_sha256(failure)
            create_only_json(RUN_FAILURE, failure)
        raise
    print(json.dumps(output, indent=2, allow_nan=False), flush=True)
    print(f"wrote {OUT} and receipt {RUN_RECEIPT} ({output['runtime_s']}s)", flush=True)


if __name__ == "__main__":
    main()
