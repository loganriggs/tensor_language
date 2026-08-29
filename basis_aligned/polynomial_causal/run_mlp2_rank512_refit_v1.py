#!/usr/bin/env python3
"""Train and evaluate the preregistered free-factor MLP2 rank-512 students."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for source_root in (ROOT, HERE):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

import bilin18_observed_model_facade as facade
from mlp2_cmr_v1_physical_program import PhysicalRetainedBilinearMLP, zero_mlp_write
import prepare_mlp2_rank512_refit_v1_rows as row_life

PREREG = HERE / "MLP2_RANK512_REFIT_V1_PREREGISTRATION.md"
FREEZER = HERE / "prepare_mlp2_rank512_refit_v1_rows.py"
TEST = HERE / "test_mlp2_rank512_refit_v1.py"
RUNNER = Path(__file__).resolve()
SOURCE_PATHS = row_life.SOURCE_PATHS

BQ = HERE.parent / "bilinear_quotient"
ROWS_RECEIPT = BQ / "mlp2_rank512_refit_v1_rows_receipt.json"
MEAN_BUNDLE = HERE / "mlp2_cmr_v1_fit_mean_bundle.pt"
MEAN_RECEIPT = HERE / "mlp2_cmr_v1_fit_mean_receipt.json"
MEAN_AUTHORITY = HERE / "mlp2_cmr_v1_fit_mean_authority.json"
MEAN_RESULT = HERE / "mlp2_cmr_v1_fit_mean_result.json"
BUNDLE = HERE / "mlp2_rank512_refit_v1_bundle.pt"
LEDGER = HERE / "mlp2_rank512_refit_v1_ledger.pt"
RESULT = HERE / "mlp2_rank512_refit_v1_result.json"
FAILURE = HERE / "mlp2_rank512_refit_v1_failure.json"
AUTHORITY = HERE / "mlp2_rank512_refit_v1_execution_authority.json"
RECEIPT = HERE / "mlp2_rank512_refit_v1_receipt.json"
LOCK = Path("/workspace/runs/.mlp2_rank512_refit_v1.lock")

SITE = 2
WIDTH = 1152
RANK = 512
SCORING = slice(64, 256)
TRAIN_DOCUMENTS = 160
DEV_DOCUMENTS = 32
SEED = 2026082921
BOOTSTRAP_SEED = 2026082922
BOOTSTRAPS = 10_000
ARMS = ("NATIVE", "ZERO", "LOCAL512", "DOWN512", "FULL512", "RANDOM512")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def committed_sources() -> tuple[str, dict[str, str]]:
    commit = git("rev-parse", "HEAD")
    if commit != git("rev-parse", "origin/main"):
        raise RuntimeError("runner requires synchronized HEAD/origin")
    hashes = {}
    for path in SOURCE_PATHS:
        relative = str(path.relative_to(ROOT))
        blob = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)
        digest = hashlib.sha256(blob).hexdigest()
        if file_sha256(path) != digest:
            raise RuntimeError(f"uncommitted runner source: {relative}")
        hashes[relative] = digest
    return commit, hashes


def stable_bytes(path: Path, expected: str | None = None) -> tuple[bytes, str]:
    before = file_sha256(path)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    after = file_sha256(path)
    if before != digest or after != before or (expected is not None and digest != expected):
        raise RuntimeError(f"artifact changed during stable read: {path}")
    return raw, digest


def stable_json(path: Path, expected: str | None = None) -> tuple[dict[str, Any], str]:
    raw, digest = stable_bytes(path, expected)
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON parent has wrong schema: {path}")
    return value, digest


def stable_torch(path: Path, expected: str | None = None) -> tuple[Any, str]:
    raw, digest = stable_bytes(path, expected)
    return torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True), digest


def validate_row_receipt(value: dict[str, Any], sources: dict[str, str]) -> None:
    expected_keys = {
        "schema", "status", "source_commit", "source_hashes", "independent_audit",
        "selection", "roles", "entries", "provenance", "disjointness",
        "ordered_manifest_gate", "registry_hashes", "prior_tensor_hashes",
        "waiver_proofs", "nonrow_proofs", "outcome_access",
    }
    if set(value) != expected_keys or value.get("schema") != (
        "mlp2_rank512_refit_v1_rows"
    ) or value.get("status") != "fresh_roles_frozen_before_any_model_or_training_access" \
            or value.get("source_hashes") != sources or value.get("selection") != {
                "start_document_index": 100_000, "documents_per_role": 192,
                "token_length": 257, "scored_slice": [64, 256],
            } or value.get("roles") != {
                "TRAIN": {"authorized_for_training": True,
                          "authorized_for_evaluation": False},
                "EVALUATION": {"authorized_for_training": False,
                               "authorized_for_evaluation": True},
            } or value.get("outcome_access") != {
                "model_loaded": False, "training_run": False,
            } or not value.get("disjointness") or not all(value["disjointness"].values()):
        raise RuntimeError("canonical MLP2 refit row receipt changed")
    audit, audit_hash = row_life.validate_independent_audit(sources)
    audit_entry = value.get("independent_audit", {})
    if audit_entry.get("file_sha256") != audit_hash or (
        audit_entry.get("audited_source_commit") != audit["audited_source_commit"]
    ) or audit_entry.get("reviewer") != audit["reviewer"]:
        raise RuntimeError("row receipt independent-audit join changed")
    if set(value.get("entries", {})) != {"TRAIN", "EVALUATION"} or (
        set(value.get("provenance", {})) != {"TRAIN", "EVALUATION"}
    ):
        raise RuntimeError("row role family changed")
    documents: set[str] = set()
    indices: set[int] = set()
    for role in ("TRAIN", "EVALUATION"):
        entry = value["entries"][role]
        records = value["provenance"][role]
        if set(entry) != {"path", "file_sha256", "tensor_sha256", "shape", "dtype"} \
                or entry["shape"] != [192, 257] or entry["dtype"] != "torch.int64" \
                or len(records) != 192:
            raise RuntimeError(f"row {role} entry/provenance changed")
        role_docs = [record.get("document_id") for record in records]
        role_indices = [record.get("dataset_document_index") for record in records]
        if any(not isinstance(item, str) or not item for item in role_docs) or any(
            type(item) is not int for item in role_indices
        ) or len(set(role_docs)) != 192 or len(set(role_indices)) != 192 or (
            documents.intersection(role_docs) or indices.intersection(role_indices)
        ):
            raise RuntimeError("row role provenance overlap or schema changed")
        documents.update(role_docs); indices.update(role_indices)


def validate_mean_parent(receipt: dict[str, Any], bundle: Any) -> None:
    if receipt.get("status") != "fit_mean_complete_receipt_last" or (
        receipt.get("experiment_id") != "bilin18_mlp2_cmr_v1_fit_mean"
    ) or receipt.get("scientific_claim") != "none_fit_artifact_only" or (
        receipt.get("fit_observations") != 30_801
    ) or receipt.get("authorized_for_validation") is not False or (
        receipt.get("authorized_for_replication") is not False
    ) or receipt.get("authorized_for_suffix_selector") is not True or (
        receipt.get("authority_sha256") != file_sha256(MEAN_AUTHORITY)
    ) or receipt.get("result_sha256") != file_sha256(MEAN_RESULT):
        raise RuntimeError("fit-mean receipt/authority join changed")
    authority, _ = stable_json(MEAN_AUTHORITY, receipt["authority_sha256"])
    if authority.get("status") != "authority_frozen_before_checkpoint_or_model_access" \
            or authority.get("experiment_id") != receipt["experiment_id"] or (
                authority.get("parents") != receipt.get("parents")
            ) or authority.get("source_commit") != receipt.get("source_commit") or (
                authority.get("source_hashes") != receipt.get("source_hashes")
            ) or authority.get("authorized_role") != "FIT_MEAN":
        raise RuntimeError("fit-mean authority semantics changed")
    commit = receipt.get("source_commit")
    hashes = receipt.get("source_hashes")
    if not isinstance(commit, str) or not isinstance(hashes, dict):
        raise RuntimeError("fit-mean source binding malformed")
    for relative, expected in hashes.items():
        blob = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)
        if hashlib.sha256(blob).hexdigest() != expected:
            raise RuntimeError("fit-mean historical source binding changed")
    expected_bundle_keys = {
        "schema", "count", "mean", "variance", "second_moment", "left_norm2",
        "right_norm2", "down_norm2", "scores", "supports",
    }
    if not isinstance(bundle, dict) or set(bundle) != expected_bundle_keys or (
        bundle.get("schema") != "mlp2_cmr_v1_fit_mean_bundle"
    ) or bundle.get("count") != 30_801 or set(bundle.get("scores", {})) != {
        "LOCAL", "RMS", "MASS"
    } or set(bundle.get("supports", {})) != {"LOCAL", "RMS", "MASS", "RANDOM"}:
        raise RuntimeError("fit-mean bundle schema changed")
    for key in ("mean", "variance", "second_moment", "left_norm2", "right_norm2",
                "down_norm2"):
        tensor = bundle[key]
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != (4608,) \
                or not tensor.is_floating_point() or not torch.isfinite(tensor).all():
            raise RuntimeError(f"fit-mean tensor changed: {key}")
    for key, tensor in bundle["scores"].items():
        if tensor.dtype != torch.float64 or tuple(tensor.shape) != (4608,) \
                or not torch.isfinite(tensor).all():
            raise RuntimeError(f"fit-mean score changed: {key}")
    for key, support in bundle["supports"].items():
        if support.dtype != torch.long or tuple(support.shape) != (512,) \
                or torch.unique(support).numel() != 512 or int(support.min()) < 0 \
                or int(support.max()) >= 4608:
            raise RuntimeError(f"fit-mean support changed: {key}")


def protected_snapshot(authority: dict[str, Any], sources: dict[str, str]) -> dict[str, Any]:
    if committed_sources()[1] != sources:
        raise RuntimeError("protected source closure changed")
    receipt, rows_hash = stable_json(ROWS_RECEIPT, authority["parents"]["rows_receipt"])
    validate_row_receipt(receipt, sources)
    mean_receipt, mean_receipt_hash = stable_json(
        MEAN_RECEIPT, authority["parents"]["mean_receipt"])
    mean_bundle, mean_bundle_hash = stable_torch(
        MEAN_BUNDLE, authority["parents"]["mean_bundle"])
    validate_mean_parent(mean_receipt, mean_bundle)
    row_hashes = {
        role: file_sha256(Path(receipt["entries"][role]["path"]))
        for role in ("TRAIN", "EVALUATION")
    }
    if row_hashes != {
        "TRAIN": authority["parents"]["train_rows"],
        "EVALUATION": authority["parents"]["evaluation_rows"],
    }:
        raise RuntimeError("protected row bytes changed")
    snapshot = Path(facade.DEFAULT_SNAPSHOT)
    return {
        "source_hashes": sources,
        "audit_sha256": file_sha256(row_life.AUDIT),
        "rows_receipt_sha256": rows_hash, "row_hashes": row_hashes,
        "mean_receipt_sha256": mean_receipt_hash,
        "mean_bundle_sha256": mean_bundle_hash,
        "mean_authority_sha256": file_sha256(MEAN_AUTHORITY),
        "mean_result_sha256": file_sha256(MEAN_RESULT),
        "checkpoint_config_sha256": file_sha256(snapshot / "config.json"),
        "checkpoint_weights_sha256": file_sha256(snapshot / "pytorch_model.bin"),
        "checkpoint_weights_bytes": (snapshot / "pytorch_model.bin").stat().st_size,
    }


def verify_protected_snapshot(
    expected: dict[str, Any], authority: dict[str, Any], sources: dict[str, str],
    claim: row_life.RunClaim,
) -> None:
    row_life.require_claim(claim, LOCK)
    if protected_snapshot(authority, sources) != expected:
        raise RuntimeError("protected MLP2 refit inputs changed during execution")
    row_life.require_claim(claim, LOCK)


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path: Path, value: Any, *, pre_link_check=None) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{secrets.token_hex(8)}")
    try:
        with temporary.open("x") as handle:
            json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if pre_link_check is not None:
            pre_link_check()
        os.link(temporary, path); fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_torch(path: Path, value: Any, *, pre_link_check=None) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{secrets.token_hex(8)}")
    try:
        torch.save(value, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        if pre_link_check is not None:
            pre_link_check()
        os.link(temporary, path); fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


class RankBilinear(nn.Module):
    def __init__(self, left: torch.Tensor, right: torch.Tensor,
                 down: torch.Tensor, bias: torch.Tensor) -> None:
        super().__init__()
        if left.shape != (RANK, WIDTH) or right.shape != left.shape or (
            down.shape != (WIDTH, RANK) or bias.shape != (WIDTH,)
        ):
            raise ValueError("rank-512 topology changed")
        self.left = nn.Parameter(left.detach().float().clone())
        self.right = nn.Parameter(right.detach().float().clone())
        self.down = nn.Parameter(down.detach().float().clone())
        self.bias = nn.Parameter(bias.detach().float().clone())

    @classmethod
    def from_physical(cls, program: PhysicalRetainedBilinearMLP) -> "RankBilinear":
        return cls(program.left, program.right, program.down, program.folded_bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        dtype = state.dtype
        left = F.linear(state, self.left.to(dtype=dtype))
        right = F.linear(state, self.right.to(dtype=dtype))
        return F.linear(left * right, self.down.to(dtype=dtype), self.bias.to(dtype=dtype))

    def price(self) -> dict[str, int]:
        return {
            "input_width": WIDTH, "output_width": WIDTH, "products": RANK,
            "coefficient_count": sum(p.numel() for p in self.parameters()),
            "stored_scalar_values": sum(p.numel() for p in self.parameters()),
            "stored_bytes_float32": 4 * sum(p.numel() for p in self.parameters()),
            "support_metadata_values": 0,
            "dense_matrix_multiplies_per_token": 3,
            "stored_dtype": "torch.float32",
            "execution_dtype": "state_dtype_bfloat16_in_deployment",
            "native_mlp_calls_per_forward": 0,
        }


@torch.no_grad()
def canonicalize_minimum_norm(model: RankBilinear) -> dict[str, float]:
    canary_generator = torch.Generator(device="cpu").manual_seed(SEED + 9)
    canary = torch.randn(2, 3, WIDTH, generator=canary_generator).to(model.left.device)
    before = model(canary).detach().clone()
    l = torch.linalg.vector_norm(model.left, dim=1).clamp_min(1e-20)
    r = torch.linalg.vector_norm(model.right, dim=1).clamp_min(1e-20)
    d = torch.linalg.vector_norm(model.down, dim=0).clamp_min(1e-20)
    equalizer = torch.sqrt(r / l)
    model.left.mul_(equalizer[:, None])
    model.right.div_(equalizer[:, None])
    q = torch.sqrt(l * r)
    scale = (d / q).pow(1.0 / 3.0)
    model.left.mul_(scale[:, None])
    model.right.mul_(scale[:, None])
    model.down.div_(scale.square()[None, :])
    after = model(canary)
    norms = torch.cat((
        torch.linalg.vector_norm(model.left, dim=1),
        torch.linalg.vector_norm(model.right, dim=1),
        torch.linalg.vector_norm(model.down, dim=0),
    ))
    return {
        "canary_max_abs_error": float((before - after).abs().max()),
        "active_factor_norm_max_min_ratio": float(norms.max() / norms.min().clamp_min(1e-20)),
    }


@torch.no_grad()
def cancellation_ratio(model: RankBilinear, state: torch.Tensor) -> float:
    # Center each scalar product across the whole dev distribution.  Bias and the
    # mean product write are therefore absent from both numerator and denominator.
    product_sum = torch.zeros(RANK, device=model.left.device)
    count = 0
    for start in range(0, state.shape[0], 1024):
        x = state[start:start + 1024].to(model.left.device).float()
        product = F.linear(x, model.left) * F.linear(x, model.right)
        product_sum += product.sum(0)
        count += product.shape[0]
    product_mean = product_sum / count
    total_singleton = 0.0
    total_write = 0.0
    for start in range(0, state.shape[0], 1024):
        x = state[start:start + 1024].to(model.left.device).float()
        product = F.linear(x, model.left) * F.linear(x, model.right) - product_mean
        down_norm2 = model.down.square().sum(0)
        total_singleton += float((product.square() * down_norm2).sum())
        variable = F.linear(product, model.down)
        total_write += float(variable.square().sum())
    return total_singleton / max(total_write, 1e-30)


def capture_native_training(model: nn.Module, rows: torch.Tensor, device: torch.device):
    states, writes = [], []
    calls = 0
    with torch.inference_mode():
        for start in range(0, rows.shape[0], 4):
            tokens = rows[start:start + 4, :-1].to(device).contiguous()
            captured = []

            def attention(event: facade.AttentionEvent):
                return event.block.attn(event.state, event.first_value)

            def mlp(event: facade.EarlyMLPEvent):
                nonlocal calls
                write = event.block.mlp(event.state)
                if event.site == SITE:
                    captured.append((
                        event.state[:, SCORING].detach().float().cpu(),
                        write[:, SCORING].detach().float().cpu(),
                    ))
                    calls += 1
                return write

            logits = facade.forward_with_dispatch(model, tokens, attention, mlp)
            if len(captured) != 1:
                raise RuntimeError("native MLP2 capture count changed")
            states.append(captured[0][0]); writes.append(captured[0][1])
            del logits, tokens, captured
    if calls != 48:
        raise RuntimeError("native MLP2 training capture census changed")
    return torch.cat(states).reshape(-1, WIDTH), torch.cat(writes).reshape(-1, WIDTH), calls


@torch.no_grad()
def dev_loss(model: RankBilinear, x: torch.Tensor, y: torch.Tensor) -> float:
    loss = 0.0
    for start in range(0, x.shape[0], 1024):
        xb = x[start:start + 1024].to(model.left.device).float()
        yb = y[start:start + 1024].to(model.left.device).float()
        loss += float((model(xb) - yb).square().sum())
    return loss / y.numel()


def train_candidate(
    model: RankBilinear, train_x: torch.Tensor, train_y: torch.Tensor,
    dev_x: torch.Tensor, dev_y: torch.Tensor, *, mode: str,
    steps: int, learning_rate: float, seed: int,
) -> tuple[RankBilinear, list[dict[str, float]], dict[str, Any]]:
    if mode == "down":
        model.left.requires_grad_(False); model.right.requires_grad_(False)
    elif mode != "full":
        raise ValueError("unknown trainability mode")
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    best_loss = dev_loss(model, dev_x, dev_y)
    significant_best = best_loss
    best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    curve = [{"step": 0, "dev_mse": best_loss}]
    stale = 0
    completed = 0
    model.train()
    for step in range(1, steps + 1):
        index = torch.randint(0, train_x.shape[0], (1024,), generator=generator)
        xb = train_x[index].to(model.left.device).float()
        yb = train_y[index].to(model.left.device).float()
        optimizer.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(xb), yb)
        loss.backward()
        optimizer.step()
        completed = step
        if step % 25 == 0:
            observed = dev_loss(model, dev_x, dev_y)
            curve.append({"step": step, "dev_mse": observed, "train_batch_mse": float(loss)})
            retain, significant = checkpoint_decision(
                observed, best_loss, significant_best,
            )
            if retain:
                best_loss = observed
                best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if significant:
                significant_best = observed
                stale = 0
            else:
                stale += 1
            if stale >= 8:
                break
    model.load_state_dict(best)
    model.left.requires_grad_(True); model.right.requires_grad_(True)
    target_mean = train_y.mean(0)
    denominator = float((dev_y - target_mean).square().sum())
    nrmse = math.sqrt(best_loss * dev_y.numel() / max(denominator, 1e-30))
    return model, curve, {
        "steps_completed": completed, "optimizer_steps": completed,
        "dev_evaluations": len(curve), "best_dev_mse": best_loss,
        "best_dev_nrmse": nrmse, "stopped_early": completed < steps,
    }


def checkpoint_decision(
    observed: float, literal_best: float, significant_best: float,
) -> tuple[bool, bool]:
    if not all(math.isfinite(value) and value >= 0 for value in (
        observed, literal_best, significant_best,
    )):
        raise ValueError("checkpoint losses must be finite and nonnegative")
    return observed < literal_best, observed < significant_best * 0.999


def program_state(model: RankBilinear) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}


def build_from_state(value: dict[str, torch.Tensor], device: torch.device) -> RankBilinear:
    model = RankBilinear(value["left"], value["right"], value["down"], value["bias"])
    model.load_state_dict(value)
    return model.to(device).eval()


def validate_bundle(value: Any, expected_parents: dict[str, str],
                    expected_sources: dict[str, str],
                    expected_commit: str | None = None) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "schema", "programs", "price", "training", "gauges", "parents",
        "source_commit", "source_hashes", "evaluation_opened",
    } or value["schema"] != "mlp2_rank512_refit_v1_bundle" or (
        value["evaluation_opened"] is not False
    ) or value["parents"] != expected_parents or value["source_hashes"] != expected_sources \
            or (expected_commit is not None and value["source_commit"] != expected_commit):
        raise RuntimeError("serialized candidate bundle schema or bindings changed")
    if set(value["programs"]) != {"DOWN512", "FULL512", "RANDOM512"}:
        raise RuntimeError("serialized candidate program set changed")
    expected_shapes = {
        "left": (RANK, WIDTH), "right": (RANK, WIDTH),
        "down": (WIDTH, RANK), "bias": (WIDTH,),
    }
    for arm, state in value["programs"].items():
        if not isinstance(state, dict) or set(state) != set(expected_shapes):
            raise RuntimeError(f"serialized {arm} state schema changed")
        for key, shape in expected_shapes.items():
            tensor = state[key]
            if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.float32 \
                    or tuple(tensor.shape) != shape or not torch.isfinite(tensor).all():
                raise RuntimeError(f"serialized {arm}.{key} changed")
    reference = RankBilinear(*(value["programs"]["FULL512"][key]
                              for key in ("left", "right", "down", "bias")))
    if value["price"] != reference.price():
        raise RuntimeError("serialized literal price changed")
    if set(value["training"]) != set(value["programs"]) or (
        set(value["gauges"]) != set(value["programs"])
    ):
        raise RuntimeError("serialized training/gauge metadata changed")
    return value


@torch.no_grad()
def deployment_replay(
    before: RankBilinear, after: RankBilinear, device: torch.device,
) -> float:
    generator = torch.Generator(device="cpu").manual_seed(SEED + 10)
    state = torch.randn(2, 7, WIDTH, generator=generator).to(
        device=device, dtype=torch.bfloat16,
    )
    error = float((before(state).float() - after(state).float()).abs().max())
    if error != 0.0:
        raise RuntimeError("serialized deployment-precision candidate changed")
    return error


def reduce_document(native: torch.Tensor, candidate: torch.Tensor,
                    targets: torch.Tensor) -> torch.Tensor:
    native = native[:, SCORING].float()
    candidate = candidate[:, SCORING].float()
    targets = targets[:, SCORING]
    native_logp = F.log_softmax(native, dim=-1)
    candidate_logp = F.log_softmax(candidate, dim=-1)
    probability = native_logp.exp()
    native_nll = -native_logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1).sum(1)
    candidate_nll = -candidate_logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1).sum(1)
    kl = (probability * (native_logp - candidate_logp)).sum((-1, -2))
    delta = candidate - native
    centered_delta = delta - delta.mean(-1, keepdim=True)
    centered_native = native - native.mean(-1, keepdim=True)
    sse = centered_delta.square().sum((-1, -2))
    energy = centered_native.square().sum((-1, -2))
    agreement = (candidate.argmax(-1) == native.argmax(-1)).sum(1)
    native_correct = (native.argmax(-1) == targets).sum(1)
    candidate_correct = (candidate.argmax(-1) == targets).sum(1)
    count = torch.full_like(native_nll, targets.shape[1])
    return torch.stack((
        native_nll, candidate_nll, kl, sse, energy,
        agreement.float(), native_correct.float(), candidate_correct.float(), count,
    ), 1).double().cpu()


def summarize(ledger: torch.Tensor, prefix: int) -> dict[str, float]:
    value = ledger[:prefix].sum(0)
    count = float(value[8])
    return {
        "dce": float((value[1] - value[0]) / count),
        "teacher_kl": float(value[2] / count),
        "centered_logit_nrmse": math.sqrt(float(value[3] / value[4])),
        "top1_agreement": float(value[5] / count),
        "native_accuracy": float(value[6] / count),
        "candidate_accuracy": float(value[7] / count),
    }


def bootstrap_improvements(ledgers: dict[str, torch.Tensor]) -> dict[str, Any]:
    if set(ledgers) != set(ARMS) or any(
        not isinstance(value, torch.Tensor) or value.dtype != torch.float64
        or tuple(value.shape) != (192, 9) or not torch.isfinite(value).all()
        or (value[:, 8] <= 0).any()
        for value in ledgers.values()
    ):
        raise RuntimeError("bootstrap ledger schema or finiteness changed")
    generator = torch.Generator().manual_seed(BOOTSTRAP_SEED)
    indices = torch.randint(0, 192, (BOOTSTRAPS, 192), generator=generator)
    output = {}
    full = ledgers["FULL512"]
    full_doc_dce = (full[:, 1] - full[:, 0]) / full[:, 8]
    full_doc_kl = full[:, 2] / full[:, 8]
    for control in ("LOCAL512", "ZERO"):
        value = ledgers[control]
        control_dce = (value[:, 1] - value[:, 0]) / value[:, 8]
        control_kl = value[:, 2] / value[:, 8]
        kl_reduction = control_kl[indices].mean(1) - full_doc_kl[indices].mean(1)
        dce_reduction = (
            control_dce[indices].mean(1).abs() - full_doc_dce[indices].mean(1).abs()
        )
        output[control] = {
            "kl_reduction_point": float(control_kl.mean() - full_doc_kl.mean()),
            "kl_reduction_bonferroni_lcb": float(torch.quantile(
                kl_reduction, 0.0125, interpolation="linear")),
            "abs_dce_reduction_point": float(control_dce.mean().abs() - full_doc_dce.mean().abs()),
            "abs_dce_reduction_bonferroni_lcb": float(torch.quantile(
                dce_reduction, 0.0125, interpolation="linear")),
        }
    return output


def evaluate(model: nn.Module, rows: torch.Tensor, programs: dict[str, nn.Module],
             device: torch.device):
    ledgers = {arm: [] for arm in ARMS}
    calls = {arm: {
        "outer_calls": 0, "outer_returns": 0,
        "attention_sites": {str(site): 0 for site in range(18)},
        "native_mlp_sites": {str(site): 0 for site in range(18)},
        "candidate_mlp2": 0,
    } for arm in ARMS}
    with torch.inference_mode():
        for start in range(0, rows.shape[0], 4):
            batch = rows[start:start + 4]
            tokens, targets = batch[:, :-1].to(device), batch[:, 1:].to(device)

            def forward(arm: str):
                def attention(event: facade.AttentionEvent):
                    calls[arm]["attention_sites"][str(event.site)] += 1
                    return event.block.attn(event.state, event.first_value)
                def mlp(event: facade.EarlyMLPEvent):
                    if event.site != SITE:
                        calls[arm]["native_mlp_sites"][str(event.site)] += 1
                        return event.block.mlp(event.state)
                    if arm == "NATIVE":
                        calls[arm]["native_mlp_sites"][str(SITE)] += 1
                        return event.block.mlp(event.state)
                    calls[arm]["candidate_mlp2"] += 1
                    if arm == "ZERO":
                        return zero_mlp_write(event.state)
                    return programs[arm](event.state)
                calls[arm]["outer_calls"] += 1
                output = facade.forward_with_dispatch(model, tokens, attention, mlp)
                calls[arm]["outer_returns"] += 1
                return output

            native = forward("NATIVE")
            ledgers["NATIVE"].append(reduce_document(native, native, targets))
            for arm in ARMS[1:]:
                candidate = forward(arm)
                ledgers[arm].append(reduce_document(native, candidate, targets))
                del candidate
            del native, tokens, targets
    packed = {arm: torch.cat(values) for arm, values in ledgers.items()}
    expected = expected_evaluation_census()
    if calls != expected or any(value.shape != (192, 9) for value in packed.values()):
        raise RuntimeError("evaluation call or sufficient-statistic census changed")
    return packed, calls


def expected_evaluation_census() -> dict[str, Any]:
    expected: dict[str, Any] = {}
    for arm in ARMS:
        native_sites = {str(site): 48 for site in range(18)}
        if arm != "NATIVE":
            native_sites[str(SITE)] = 0
        expected[arm] = {
            "outer_calls": 48, "outer_returns": 48,
            "attention_sites": {str(site): 48 for site in range(18)},
            "native_mlp_sites": native_sites,
            "candidate_mlp2": 0 if arm == "NATIVE" else 48,
        }
    return expected


def validate_ledger(value: Any, *, bundle_hash: str, evaluation_hash: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "schema", "fields", "arms", "bundle_sha256", "evaluation_rows_sha256",
        "call_census", "checkpoint", "deployment_replay_max_abs_error",
    } or value["schema"] != "mlp2_rank512_refit_v1_document_ledger" or (
        value["fields"] != (
            "native_nll", "candidate_nll", "teacher_kl", "centered_logit_sse",
            "native_centered_logit_energy", "top1_agreement", "native_correct",
            "candidate_correct", "count",
        )
    ) or value["bundle_sha256"] != bundle_hash or (
        value["evaluation_rows_sha256"] != evaluation_hash
    ) or value["call_census"] != expected_evaluation_census() or (
        set(value["deployment_replay_max_abs_error"]) != {
            "DOWN512", "FULL512", "RANDOM512"
        }
    ) or any(error != 0.0 for error in value["deployment_replay_max_abs_error"].values()):
        raise RuntimeError("serialized MLP2 refit ledger semantics changed")
    bootstrap_improvements(value["arms"])
    return value


def derive_result(
    ledger: dict[str, Any], bundle: dict[str, Any], *, bundle_hash: str,
    ledger_hash: str, runtime_seconds: float,
) -> dict[str, Any]:
    ledgers = ledger["arms"]
    summaries = {
        arm: {str(prefix): summarize(value, prefix) for prefix in (48, 96, 192)}
        for arm, value in ledgers.items()
    }
    improvements = bootstrap_improvements(ledgers)
    final = summaries["FULL512"]["192"]
    stability = all(
        abs(final[key] - summaries["FULL512"]["96"][key]) <= 0.01
        for key in ("dce", "teacher_kl", "centered_logit_nrmse", "top1_agreement")
    )
    training = bundle["training"]
    gauges = bundle["gauges"]
    absolute_gates = {
        "abs_dce_at_most_0p02": abs(final["dce"]) <= 0.02,
        "kl_at_most_0p02": final["teacher_kl"] <= 0.02,
        "logit_nrmse_at_most_0p10": final["centered_logit_nrmse"] <= 0.10,
        "top1_at_least_0p90": final["top1_agreement"] >= 0.90,
        "prefix_stability": stability,
        "local_dev_nrmse_at_most_0p25": training["FULL512"]["fit"]["best_dev_nrmse"] <= 0.25,
        "gauge_and_cancellation": gauges["FULL512"]["pathology_pass"],
    }
    down_final = summaries["DOWN512"]["192"]
    local_final = summaries["LOCAL512"]["192"]
    relative_gates = {
        "full_beats_down_kl_20pct": final["teacher_kl"] <= 0.8 * down_final["teacher_kl"],
        "full_beats_down_abs_dce_20pct": abs(final["dce"]) <= 0.8 * abs(down_final["dce"]),
        "full_beats_local_kl_50pct": final["teacher_kl"] <= 0.5 * local_final["teacher_kl"],
        "full_beats_local_abs_dce_50pct": abs(final["dce"]) <= 0.5 * abs(local_final["dce"]),
        "simultaneous_lcbs_positive": all(
            value[key] > 0 for value in improvements.values()
            for key in ("kl_reduction_bonferroni_lcb", "abs_dce_reduction_bonferroni_lcb")
        ),
    }
    return {
        "schema": "mlp2_rank512_refit_v1_result",
        "status": (
            "absolute_pass_exploratory_replication_required"
            if all(absolute_gates.values()) else
            "optimization_failure" if not absolute_gates["local_dev_nrmse_at_most_0p25"]
            else "finite_consequence_failure"
        ),
        "claim_boundary": "native_trajectory_in_distribution_only_no_strict_ledger_move",
        "checkpoint": ledger["checkpoint"], "documents": 192,
        "execution_counts": {
            "training_capture_outer_calls": 48,
            "training_capture_attention_calls": 864,
            "training_capture_native_mlp_calls": 864,
            "optimizer_steps": {arm: training[arm]["fit"]["optimizer_steps"]
                                for arm in ("DOWN512", "FULL512", "RANDOM512")},
            "dev_evaluations": {arm: training[arm]["fit"]["dev_evaluations"]
                                for arm in ("DOWN512", "FULL512", "RANDOM512")},
            "canonicalization_canary_calls": 3,
        },
        "price": bundle["price"],
        "physical_program_types": {
            "LOCAL512": "PhysicalRetainedBilinearMLP",
            "DOWN512": "RankBilinear", "FULL512": "RankBilinear",
            "RANDOM512": "RankBilinear",
        },
        "deployment_replay_max_abs_error": ledger["deployment_replay_max_abs_error"],
        "training": training, "gauges": gauges,
        "summaries": summaries, "bootstrap_improvements": improvements,
        "absolute_gates": absolute_gates, "relative_gates": relative_gates,
        "call_census": ledger["call_census"],
        "artifacts": {"bundle_sha256": bundle_hash, "ledger_sha256": ledger_hash},
        "parents": bundle["parents"], "source_commit": bundle["source_commit"],
        "source_hashes": bundle["source_hashes"],
        "bootstrap": {"seed": BOOTSTRAP_SEED, "draws": BOOTSTRAPS,
                      "rng": "torch.Generator_cpu", "quantile": "linear",
                      "bonferroni_tail_probability": 0.0125},
        "evaluation_opened_after_semantic_bundle_reload": True,
        "runtime_seconds": runtime_seconds,
    }


def prepare_execution_authority(claim: row_life.RunClaim) -> dict[str, Any]:
    """Bind code and parents before TRAIN bytes or the checkpoint may be opened."""
    commit, sources = committed_sources()
    audit, audit_hash = row_life.validate_independent_audit(sources)
    rows_receipt, rows_hash = stable_json(ROWS_RECEIPT)
    mean_receipt, mean_receipt_hash = stable_json(MEAN_RECEIPT)
    validate_row_receipt(rows_receipt, sources)
    mean_hash = mean_receipt.get("bundle_sha256")
    mean_bundle, loaded_mean_hash = stable_torch(MEAN_BUNDLE, mean_hash)
    validate_mean_parent(mean_receipt, mean_bundle)
    if not isinstance(mean_hash, str) or loaded_mean_hash != mean_hash:
        raise RuntimeError("mean bundle binding changed")
    entries = rows_receipt.get("entries", {})
    if set(entries) != {"TRAIN", "EVALUATION"} or any(
        not isinstance(entries[role].get("file_sha256"), str)
        for role in entries
    ):
        raise RuntimeError("row role entries changed")
    authority = {
        "schema": "mlp2_rank512_refit_v1_execution_authority",
        "status": "spent_before_train_or_model_access",
        "source_commit": commit, "source_hashes": sources,
        "independent_audit_sha256": audit_hash,
        "independent_audit_reviewer": audit["reviewer"],
        "parents": {
            "rows_receipt": rows_hash,
            "train_rows": entries["TRAIN"]["file_sha256"],
            "evaluation_rows": entries["EVALUATION"]["file_sha256"],
            "mean_receipt": mean_receipt_hash, "mean_bundle": mean_hash,
        },
        "outcome_access_before_authority": {
            "train_rows_opened": False, "evaluation_rows_opened": False,
            "mean_bundle_semantically_validated": True, "checkpoint_loaded": False,
            "model_forward_calls": 0,
        },
    }
    def authority_guard() -> None:
        row_life.require_claim(claim, LOCK)
        if any(path.exists() for path in (AUTHORITY, BUNDLE, LEDGER, RESULT, RECEIPT, FAILURE)):
            raise RuntimeError("MLP2 refit namespace raced execution authority")
        # Re-bind the small parent manifests at the actual publication boundary.
        stable_json(ROWS_RECEIPT, rows_hash)
        stable_json(MEAN_RECEIPT, mean_receipt_hash)
        stable_bytes(MEAN_BUNDLE, mean_hash)
        row_life.require_claim(claim, LOCK)
        if AUTHORITY.exists() or RECEIPT.exists() or FAILURE.exists():
            raise RuntimeError("MLP2 refit terminal raced execution authority")
    atomic_json(AUTHORITY, authority, pre_link_check=authority_guard)
    return authority


def run(claim: row_life.RunClaim) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    started = time.time()
    commit, sources = committed_sources()
    authority, authority_hash = stable_json(AUTHORITY)
    if authority.get("status") != "spent_before_train_or_model_access" or (
        authority.get("source_hashes") != sources
    ):
        raise RuntimeError("execution authority changed")
    protected = protected_snapshot(authority, sources)
    verify_protected_snapshot(protected, authority, sources, claim)
    receipt, rows_receipt_hash = stable_json(
        ROWS_RECEIPT, authority["parents"]["rows_receipt"])
    validate_row_receipt(receipt, sources)
    train_entry = receipt["entries"]["TRAIN"]
    eval_entry = receipt["entries"]["EVALUATION"]
    train_path = Path(train_entry["path"])
    train_rows, train_hash = stable_torch(train_path, train_entry["file_sha256"])
    if train_hash != authority["parents"]["train_rows"] or not isinstance(
        train_rows, torch.Tensor
    ) or train_rows.dtype != torch.long or tuple(train_rows.shape) != (192, 257) or (
        row_life.tensor_sha256(train_rows) != train_entry["tensor_sha256"]
    ):
        raise RuntimeError("TRAIN row shape changed")

    mean_receipt, mean_receipt_hash = stable_json(
        MEAN_RECEIPT, authority["parents"]["mean_receipt"])
    mean_bundle, mean_bundle_hash = stable_torch(
        MEAN_BUNDLE, authority["parents"]["mean_bundle"])
    if mean_receipt.get("bundle_sha256") != mean_bundle_hash:
        raise RuntimeError("mean parent hash changed")
    validate_mean_parent(mean_receipt, mean_bundle)

    device = torch.device("cuda")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.set_float32_matmul_precision("high")
    model, checkpoint = facade.load_bilin18(device=device, dtype=torch.bfloat16)
    verify_protected_snapshot(protected, authority, sources, claim)
    mean = mean_bundle["mean"].to(device)
    local_support = mean_bundle["supports"]["LOCAL"].to(device)
    native_mlp = model.transformer.h[SITE].mlp
    local_program = PhysicalRetainedBilinearMLP.from_native(
        native_mlp, mean, local_support,
    ).to(device)
    random_support = torch.randperm(4608, generator=torch.Generator().manual_seed(SEED + 1))[:RANK].to(device)
    random_program = PhysicalRetainedBilinearMLP.from_native(
        native_mlp, mean, random_support,
    ).to(device)
    states, writes, capture_calls = capture_native_training(model, train_rows, device)
    split = TRAIN_DOCUMENTS * 192
    train_x, dev_x = states[:split], states[split:]
    train_y, dev_y = writes[:split], writes[split:]

    down = RankBilinear.from_physical(local_program).to(device)
    down, down_curve, down_fit = train_candidate(
        down, train_x, train_y, dev_x, dev_y, mode="down", steps=600,
        learning_rate=1e-3, seed=SEED + 2,
    )
    full = copy.deepcopy(down).to(device)
    full, full_curve, full_fit = train_candidate(
        full, train_x, train_y, dev_x, dev_y, mode="full", steps=1200,
        learning_rate=3e-4, seed=SEED + 3,
    )
    random = RankBilinear.from_physical(random_program).to(device)
    random, random_curve, random_fit = train_candidate(
        random, train_x, train_y, dev_x, dev_y, mode="full", steps=1200,
        learning_rate=3e-4, seed=SEED + 4,
    )
    gauges = {}
    for name, candidate in (("DOWN512", down), ("FULL512", full), ("RANDOM512", random)):
        gauges[name] = canonicalize_minimum_norm(candidate)
        gauges[name]["dev_cancellation_ratio"] = cancellation_ratio(candidate, dev_x)
        gauges[name]["pathology_pass"] = (
            gauges[name]["canary_max_abs_error"] <= 1e-4
            and gauges[name]["active_factor_norm_max_min_ratio"] <= 1e4
            and gauges[name]["dev_cancellation_ratio"] <= 100
        )
    bundle = {
        "schema": "mlp2_rank512_refit_v1_bundle",
        "programs": {name: program_state(value) for name, value in (
            ("DOWN512", down), ("FULL512", full), ("RANDOM512", random),
        )},
        "price": full.price(),
        "training": {
            "DOWN512": {"fit": down_fit, "curve": down_curve},
            "FULL512": {"fit": full_fit, "curve": full_curve},
            "RANDOM512": {"fit": random_fit, "curve": random_curve},
        },
        "gauges": gauges,
        "parents": {"execution_authority": authority_hash,
                    "rows_receipt": rows_receipt_hash,
                    "mean_bundle": mean_bundle_hash,
                    "mean_receipt": mean_receipt_hash},
        "source_commit": commit, "source_hashes": sources,
        "evaluation_opened": False,
    }
    verify_protected_snapshot(protected, authority, sources, claim)

    def bundle_guard() -> None:
        verify_protected_snapshot(protected, authority, sources, claim)
        if BUNDLE.exists() or RECEIPT.exists() or FAILURE.exists():
            raise RuntimeError("MLP2 refit artifact raced bundle publication")
    atomic_torch(BUNDLE, bundle, pre_link_check=bundle_guard)
    reloaded_bundle, bundle_hash = stable_torch(BUNDLE)
    reloaded_bundle = validate_bundle(
        reloaded_bundle, bundle["parents"], sources,
        expected_commit=authority["source_commit"],
    )

    # Evaluation token bytes are first opened only after the immutable candidate exists.
    verify_protected_snapshot(protected, authority, sources, claim)
    eval_rows, eval_hash = stable_torch(
        Path(eval_entry["path"]), eval_entry["file_sha256"])
    if eval_hash != authority["parents"]["evaluation_rows"] or not isinstance(
        eval_rows, torch.Tensor
    ) or eval_rows.dtype != torch.long or tuple(eval_rows.shape) != (192, 257) or (
        row_life.tensor_sha256(eval_rows) != eval_entry["tensor_sha256"]
    ):
        raise RuntimeError("EVALUATION row shape changed")
    programs = {
        "LOCAL512": local_program,
        "DOWN512": build_from_state(reloaded_bundle["programs"]["DOWN512"], device),
        "FULL512": build_from_state(reloaded_bundle["programs"]["FULL512"], device),
        "RANDOM512": build_from_state(reloaded_bundle["programs"]["RANDOM512"], device),
    }
    deployment_replays = {
        "DOWN512": deployment_replay(down, programs["DOWN512"], device),
        "FULL512": deployment_replay(full, programs["FULL512"], device),
        "RANDOM512": deployment_replay(random, programs["RANDOM512"], device),
    }
    ledgers, calls = evaluate(model, eval_rows, programs, device)
    verify_protected_snapshot(protected, authority, sources, claim)
    ledger = {
        "schema": "mlp2_rank512_refit_v1_document_ledger",
        "fields": ("native_nll", "candidate_nll", "teacher_kl", "centered_logit_sse",
                   "native_centered_logit_energy", "top1_agreement", "native_correct",
                   "candidate_correct", "count"),
        "arms": ledgers,
        "bundle_sha256": bundle_hash,
        "evaluation_rows_sha256": eval_hash,
        "call_census": calls, "checkpoint": checkpoint.__dict__,
        "deployment_replay_max_abs_error": deployment_replays,
    }
    atomic_torch(LEDGER, ledger,
                 pre_link_check=lambda: verify_protected_snapshot(
                     protected, authority, sources, claim))
    reloaded_ledger, ledger_hash = stable_torch(LEDGER)
    reloaded_ledger = validate_ledger(
        reloaded_ledger, bundle_hash=bundle_hash, evaluation_hash=eval_hash,
    )
    result = derive_result(
        reloaded_ledger, reloaded_bundle, bundle_hash=bundle_hash,
        ledger_hash=ledger_hash, runtime_seconds=time.time() - started,
    )
    verify_protected_snapshot(protected, authority, sources, claim)
    return result, bundle, protected


def main() -> None:
    namespace = (AUTHORITY, BUNDLE, LEDGER, RESULT, RECEIPT, FAILURE, LOCK)
    if any(path.exists() for path in namespace):
        raise RuntimeError("MLP2 rank512 refit namespace already exists")
    claim = row_life.acquire_claim(LOCK)
    try:
        authority = prepare_execution_authority(claim)
        row_life.require_claim(claim, LOCK)
        result, _, protected = run(claim)

        def result_guard() -> None:
            verify_protected_snapshot(
                protected, authority, authority["source_hashes"], claim,
            )
            if RESULT.exists() or RECEIPT.exists() or FAILURE.exists():
                raise RuntimeError("MLP2 refit terminal raced result publication")
        atomic_json(RESULT, result, pre_link_check=result_guard)
        reloaded_result, result_hash = stable_json(RESULT)
        reloaded_bundle, bundle_hash = stable_torch(BUNDLE)
        reloaded_bundle = validate_bundle(
            reloaded_bundle, result["parents"], authority["source_hashes"],
            expected_commit=authority["source_commit"],
        )
        reloaded_ledger, ledger_hash = stable_torch(LEDGER)
        reloaded_ledger = validate_ledger(
            reloaded_ledger, bundle_hash=bundle_hash,
            evaluation_hash=authority["parents"]["evaluation_rows"],
        )
        recomputed = derive_result(
            reloaded_ledger, reloaded_bundle, bundle_hash=bundle_hash,
            ledger_hash=ledger_hash, runtime_seconds=result["runtime_seconds"],
        )
        if reloaded_result != recomputed or reloaded_result != result:
            raise RuntimeError("serialized MLP2 refit result does not replay")
        receipt = {
            "schema": "mlp2_rank512_refit_v1_receipt",
            "status": "result_complete_receipt_last",
            "authority_sha256": file_sha256(AUTHORITY),
            "bundle_sha256": bundle_hash,
            "ledger_sha256": ledger_hash,
            "result_sha256": result_hash,
            "source_commit": authority["source_commit"],
            "source_hashes": authority["source_hashes"],
            "evaluation_opened": True,
        }

        def success_guard() -> None:
            verify_protected_snapshot(
                protected, authority, authority["source_hashes"], claim,
            )
            if RECEIPT.exists() or FAILURE.exists():
                raise RuntimeError("competing MLP2 refit terminal artifact appeared")
            for path, expected in (
                (AUTHORITY, receipt["authority_sha256"]),
                (BUNDLE, receipt["bundle_sha256"]),
                (LEDGER, receipt["ledger_sha256"]),
                (RESULT, receipt["result_sha256"]),
            ):
                stable_bytes(path, expected)
            row_life.require_claim(claim, LOCK)
            if RECEIPT.exists() or FAILURE.exists():
                raise RuntimeError("competing MLP2 refit terminal raced success")
        atomic_json(RECEIPT, receipt, pre_link_check=success_guard)
        reloaded_receipt, _ = stable_json(RECEIPT)
        if reloaded_receipt != receipt:
            raise RuntimeError("serialized MLP2 refit receipt changed")
        print(json.dumps(result, sort_keys=True, indent=2))
    except BaseException as exc:
        failure = {
            "schema": "mlp2_rank512_refit_v1_failure",
            "status": "terminal_failure_no_receipt", "error": repr(exc),
            "authority_sha256": file_sha256(AUTHORITY) if AUTHORITY.is_file() else None,
            "parent_hashes": authority.get("parents", {}) if "authority" in locals() else {},
            "source_hashes": authority.get("source_hashes", {}) if "authority" in locals() else {},
            "protected_snapshot": protected if "protected" in locals() else None,
            "artifact_hashes": {path.name: file_sha256(path) for path in (
                BUNDLE, LEDGER, RESULT,
            ) if path.is_file()},
            "bundle_exists": BUNDLE.exists(), "ledger_exists": LEDGER.exists(),
            "result_exists": RESULT.exists(), "receipt_exists": RECEIPT.exists(),
            "evaluation_may_have_opened": BUNDLE.exists(),
            "lock_inode": claim.inode,
            "lock_nonce_sha256": hashlib.sha256(claim.nonce.encode()).hexdigest(),
        }
        if not RECEIPT.exists() and not FAILURE.exists():
            def failure_guard() -> None:
                row_life.require_claim(claim, LOCK)
                if RECEIPT.exists() or FAILURE.exists():
                    raise RuntimeError("competing MLP2 refit terminal raced failure")
            atomic_json(FAILURE, failure, pre_link_check=failure_guard)
        raise
    finally:
        row_life.release_claim(claim, LOCK)


if __name__ == "__main__":
    main()
