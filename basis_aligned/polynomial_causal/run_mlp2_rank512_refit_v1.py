#!/usr/bin/env python3
"""Train and evaluate the preregistered free-factor MLP2 rank-512 students."""

from __future__ import annotations

import copy
import hashlib
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

PREREG = HERE / "MLP2_RANK512_REFIT_V1_PREREGISTRATION.md"
FREEZER = HERE / "prepare_mlp2_rank512_refit_v1_rows.py"
TEST = HERE / "test_mlp2_rank512_refit_v1.py"
RUNNER = Path(__file__).resolve()
SOURCE_PATHS = (PREREG, FREEZER, RUNNER, TEST)

BQ = HERE.parent / "bilinear_quotient"
ROWS_RECEIPT = BQ / "mlp2_rank512_refit_v1_rows_receipt.json"
MEAN_BUNDLE = HERE / "mlp2_cmr_v1_fit_mean_bundle.pt"
MEAN_RECEIPT = HERE / "mlp2_cmr_v1_fit_mean_receipt.json"
BUNDLE = HERE / "mlp2_rank512_refit_v1_bundle.pt"
LEDGER = HERE / "mlp2_rank512_refit_v1_ledger.pt"
RESULT = HERE / "mlp2_rank512_refit_v1_result.json"
FAILURE = HERE / "mlp2_rank512_refit_v1_failure.json"
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


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{secrets.token_hex(8)}")
    try:
        with temporary.open("x") as handle:
            json.dump(value, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_torch(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{secrets.token_hex(8)}")
    try:
        torch.save(value, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.link(temporary, path)
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
            "stored_scalar_values": sum(p.numel() for p in self.parameters()),
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
    total_singleton = 0.0
    total_write = 0.0
    for start in range(0, state.shape[0], 1024):
        x = state[start:start + 1024].to(model.left.device).float()
        product = F.linear(x, model.left) * F.linear(x, model.right)
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
            if observed < best_loss * 0.999:
                best_loss = observed
                best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
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
        "steps_completed": completed, "best_dev_mse": best_loss,
        "best_dev_nrmse": nrmse, "stopped_early": completed < steps,
    }


def program_state(model: RankBilinear) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}


def build_from_state(value: dict[str, torch.Tensor], device: torch.device) -> RankBilinear:
    model = RankBilinear(value["left"], value["right"], value["down"], value["bias"])
    model.load_state_dict(value)
    return model.to(device).eval()


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
            "kl_reduction_bonferroni_lcb": float(torch.quantile(kl_reduction, 0.0125)),
            "abs_dce_reduction_point": float(control_dce.mean().abs() - full_doc_dce.mean().abs()),
            "abs_dce_reduction_bonferroni_lcb": float(torch.quantile(dce_reduction, 0.0125)),
        }
    return output


def evaluate(model: nn.Module, rows: torch.Tensor, programs: dict[str, nn.Module],
             device: torch.device):
    ledgers = {arm: [] for arm in ARMS}
    calls = {arm: {"forward": 0, "native_mlp2": 0, "candidate_mlp2": 0} for arm in ARMS}
    with torch.inference_mode():
        for start in range(0, rows.shape[0], 4):
            batch = rows[start:start + 4]
            tokens, targets = batch[:, :-1].to(device), batch[:, 1:].to(device)

            def forward(arm: str):
                def attention(event: facade.AttentionEvent):
                    return event.block.attn(event.state, event.first_value)
                def mlp(event: facade.EarlyMLPEvent):
                    if event.site != SITE:
                        return event.block.mlp(event.state)
                    if arm == "NATIVE":
                        calls[arm]["native_mlp2"] += 1
                        return event.block.mlp(event.state)
                    calls[arm]["candidate_mlp2"] += 1
                    if arm == "ZERO":
                        return zero_mlp_write(event.state)
                    return programs[arm](event.state)
                calls[arm]["forward"] += 1
                return facade.forward_with_dispatch(model, tokens, attention, mlp)

            native = forward("NATIVE")
            ledgers["NATIVE"].append(reduce_document(native, native, targets))
            for arm in ARMS[1:]:
                candidate = forward(arm)
                ledgers[arm].append(reduce_document(native, candidate, targets))
                del candidate
            del native, tokens, targets
    packed = {arm: torch.cat(values) for arm, values in ledgers.items()}
    expected = {
        "NATIVE": {"forward": 48, "native_mlp2": 48, "candidate_mlp2": 0},
        **{arm: {"forward": 48, "native_mlp2": 0, "candidate_mlp2": 48}
           for arm in ARMS[1:]},
    }
    if calls != expected or any(value.shape != (192, 9) for value in packed.values()):
        raise RuntimeError("evaluation call or sufficient-statistic census changed")
    return packed, calls


def run() -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.time()
    commit, sources = committed_sources()
    receipt = json.loads(ROWS_RECEIPT.read_text())
    if receipt.get("status") != "fresh_roles_frozen_before_any_model_or_training_access" or (
        receipt.get("roles", {}).get("TRAIN", {}).get("authorized_for_training") is not True
        or receipt.get("roles", {}).get("EVALUATION", {}).get("authorized_for_training") is not False
        or not all(receipt.get("disjointness", {}).values())
    ):
        raise RuntimeError("fresh row receipt does not authorize this lifecycle")
    train_entry = receipt["entries"]["TRAIN"]
    eval_entry = receipt["entries"]["EVALUATION"]
    train_path = Path(train_entry["path"])
    if file_sha256(train_path) != train_entry["file_sha256"]:
        raise RuntimeError("TRAIN rows changed")
    train_rows = torch.load(train_path, map_location="cpu", weights_only=True)
    if train_rows.shape != (192, 257):
        raise RuntimeError("TRAIN row shape changed")

    device = torch.device("cuda")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.set_float32_matmul_precision("high")
    model, checkpoint = facade.load_bilin18(device=device, dtype=torch.bfloat16)
    mean_bundle = torch.load(MEAN_BUNDLE, map_location="cpu", weights_only=True)
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
        "parents": {"rows_receipt": file_sha256(ROWS_RECEIPT),
                    "mean_bundle": file_sha256(MEAN_BUNDLE),
                    "mean_receipt": file_sha256(MEAN_RECEIPT)},
        "source_commit": commit, "source_hashes": sources,
        "evaluation_opened": False,
    }
    atomic_torch(BUNDLE, bundle)
    bundle_hash = file_sha256(BUNDLE)

    # Evaluation token bytes are first opened only after the immutable candidate exists.
    if file_sha256(Path(eval_entry["path"])) != eval_entry["file_sha256"]:
        raise RuntimeError("EVALUATION rows changed")
    eval_rows = torch.load(eval_entry["path"], map_location="cpu", weights_only=True)
    if eval_rows.shape != (192, 257):
        raise RuntimeError("EVALUATION row shape changed")
    programs = {
        "LOCAL512": local_program,
        "DOWN512": build_from_state(bundle["programs"]["DOWN512"], device),
        "FULL512": build_from_state(bundle["programs"]["FULL512"], device),
        "RANDOM512": build_from_state(bundle["programs"]["RANDOM512"], device),
    }
    ledgers, calls = evaluate(model, eval_rows, programs, device)
    ledger = {
        "schema": "mlp2_rank512_refit_v1_document_ledger",
        "fields": ("native_nll", "candidate_nll", "teacher_kl", "centered_logit_sse",
                   "native_centered_logit_energy", "top1_agreement", "native_correct",
                   "candidate_correct", "count"),
        "arms": ledgers,
        "bundle_sha256": bundle_hash,
        "evaluation_rows_sha256": eval_entry["file_sha256"],
    }
    atomic_torch(LEDGER, ledger)
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
    absolute_gates = {
        "abs_dce_at_most_0p02": abs(final["dce"]) <= 0.02,
        "kl_at_most_0p02": final["teacher_kl"] <= 0.02,
        "logit_nrmse_at_most_0p10": final["centered_logit_nrmse"] <= 0.10,
        "top1_at_least_0p90": final["top1_agreement"] >= 0.90,
        "prefix_stability": stability,
        "local_dev_nrmse_at_most_0p25": full_fit["best_dev_nrmse"] <= 0.25,
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
    result = {
        "schema": "mlp2_rank512_refit_v1_result",
        "status": (
            "absolute_pass_exploratory_replication_required"
            if all(absolute_gates.values()) else
            "optimization_failure" if not absolute_gates["local_dev_nrmse_at_most_0p25"]
            else "finite_consequence_failure"
        ),
        "claim_boundary": "native_trajectory_in_distribution_only_no_strict_ledger_move",
        "checkpoint": checkpoint.__dict__, "documents": 192,
        "training_capture_calls": capture_calls,
        "price": full.price(), "training": bundle["training"], "gauges": gauges,
        "summaries": summaries, "bootstrap_improvements": improvements,
        "absolute_gates": absolute_gates, "relative_gates": relative_gates,
        "call_census": calls,
        "artifacts": {"bundle_sha256": bundle_hash, "ledger_sha256": file_sha256(LEDGER)},
        "parents": bundle["parents"], "source_commit": commit, "source_hashes": sources,
        "evaluation_opened_after_bundle": True,
        "runtime_seconds": time.time() - started,
    }
    return result, bundle


def main() -> None:
    namespace = (BUNDLE, LEDGER, RESULT, FAILURE, LOCK)
    if any(path.exists() for path in namespace):
        raise RuntimeError("MLP2 rank512 refit namespace already exists")
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(LOCK, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        result, _ = run()
        atomic_json(RESULT, result)
        print(json.dumps(result, sort_keys=True, indent=2))
    except BaseException as exc:
        failure = {
            "schema": "mlp2_rank512_refit_v1_failure", "error": repr(exc),
            "bundle_exists": BUNDLE.exists(), "ledger_exists": LEDGER.exists(),
            "result_exists": RESULT.exists(), "replication_opened": False,
        }
        if not FAILURE.exists():
            atomic_json(FAILURE, failure)
        raise
    finally:
        os.close(fd)
        LOCK.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
