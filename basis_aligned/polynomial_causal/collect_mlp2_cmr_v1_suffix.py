#!/usr/bin/env python3
"""Collect frozen centered MLP2 suffix responses and publish selector supports."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time
from typing import Any, Sequence

import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for source_root in (ROOT, HERE):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

import bilin18_observed_model_facade as facade
import mlp2_cmr_v1_suffix_math as suffix_math
import mlp_global_gate_response as gate_math
import tensor_bilin18_tangent_collector as tangent


PREREG = HERE / "MLP2_CMR_V1_PREREGISTRATION.md"
ADDENDUM = HERE / "MLP2_CMR_V1_SUFFIX_ADDENDUM.md"
DISCREPANCY = HERE / "mlp2_cmr_v1_random_control_discrepancy.json"
RECOVERY = HERE / "MLP2_CMR_V1_SUFFIX_V2_RECOVERY.md"
TOKEN_ROWS = HERE / "mlp2_cmr_v1_token_rows.pt"
TOKEN_RECEIPT = HERE / "mlp2_cmr_v1_token_rows_receipt.json"
FIT_BUNDLE = HERE / "mlp2_cmr_v1_fit_mean_bundle.pt"
FIT_RESULT = HERE / "mlp2_cmr_v1_fit_mean_result.json"
FIT_RECEIPT = HERE / "mlp2_cmr_v1_fit_mean_receipt.json"
PREVIOUS_AUTHORITY = HERE / "mlp2_cmr_v1_suffix_authority.json"
PREVIOUS_FAILURE = HERE / "mlp2_cmr_v1_suffix_failure.json"
AUTHORITY = HERE / "mlp2_cmr_v1_suffix_v2_authority.json"
BUNDLE = HERE / "mlp2_cmr_v1_suffix_v2_bundle.pt"
RESULT = HERE / "mlp2_cmr_v1_suffix_v2_result.json"
RECEIPT = HERE / "mlp2_cmr_v1_suffix_v2_receipt.json"
FAILURE = HERE / "mlp2_cmr_v1_suffix_v2_failure.json"
LOCK = HERE / ".mlp2_cmr_v1_suffix_v2.lock"

TOKEN_ROWS_SHA256 = "3ed0192993095f7de70ab7f1350d091b6c1d8c4c7d0583fd5f0f6441556e4aa6"
TOKEN_RECEIPT_SHA256 = "47113c255bf47f9d1c7369639fab39664c71f93134099babadcce9d89a011e85"
FIT_BUNDLE_SHA256 = "043bb52b9580d9c9c342460e5bb80ff579db01486b3b6c6672bf5fba77e46f8e"
FIT_RESULT_SHA256 = "65c1ee33f0399d6489cae0227442d479a9d59b9be98f619d92423cfd39fc7833"
FIT_RECEIPT_SHA256 = "9dc14d909a1b4aafd33c67dc7a3d066db4ccc9cb83c7059fe7aaf499ca9e5efa"
PREVIOUS_AUTHORITY_SHA256 = "d204e9adeef3d65a1d6f38ed76071aa38c921bd2884ed136ac6e37f4696c7296"
PREVIOUS_FAILURE_SHA256 = "eea77b4e7fa9fd6ed35dce31ea43a72f8cf2d21d8c2e76a94396a81547f6d8a2"

SITE = 2
SOURCE_DOCUMENTS = 192
LIVE_DOCUMENTS = 191
BATCH = 4
CALLS = 48
SEQUENCE = 256
SCORE_START = 64
SCORE_STOP = 256
ELIGIBLE_POSITIONS = 31_505
HIDDEN = 4608
K = 512
TARGET_RANK = 256
PROBE_SEEDS = tuple(range(2026090201, 2026090209))
DERANGEMENT_SEED = 2026090209
HASH_RANDOM_SEED = 20260829

SOURCE_CLOSURE = (
    PREREG, ADDENDUM, DISCREPANCY, RECOVERY, Path(__file__).resolve(),
    HERE / "test_collect_mlp2_cmr_v1_suffix.py",
    HERE / "mlp2_cmr_v1_suffix_math.py",
    HERE / "test_mlp2_cmr_v1_suffix_math.py",
    HERE / "mlp_global_gate_response.py",
    HERE / "tensor_bilin18_tangent_collector.py",
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


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()


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


def publish_torch_create_only(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}")
    try:
        torch.save(value, temporary)
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
            raise RuntimeError(f"suffix source differs from committed bytes: {relative}")
        hashes[str(relative)] = digest
    return commit, hashes


def protected_inputs() -> dict[str, str]:
    actual = {
        "token_rows": file_sha256(TOKEN_ROWS),
        "token_receipt": file_sha256(TOKEN_RECEIPT),
        "fit_bundle": file_sha256(FIT_BUNDLE),
        "fit_result": file_sha256(FIT_RESULT),
        "fit_receipt": file_sha256(FIT_RECEIPT),
        "previous_authority": file_sha256(PREVIOUS_AUTHORITY),
        "previous_failure": file_sha256(PREVIOUS_FAILURE),
    }
    expected = {
        "token_rows": TOKEN_ROWS_SHA256,
        "token_receipt": TOKEN_RECEIPT_SHA256,
        "fit_bundle": FIT_BUNDLE_SHA256,
        "fit_result": FIT_RESULT_SHA256,
        "fit_receipt": FIT_RECEIPT_SHA256,
        "previous_authority": PREVIOUS_AUTHORITY_SHA256,
        "previous_failure": PREVIOUS_FAILURE_SHA256,
    }
    if actual != expected:
        raise RuntimeError("MLP2 suffix protected parent changed")
    receipt = json.loads(FIT_RECEIPT.read_text())
    if receipt.get("authorized_for_suffix_selector") is not True or (
        receipt.get("authorized_for_validation") is not False
        or receipt.get("authorized_for_replication") is not False
    ):
        raise RuntimeError("FIT_MEAN authority boundary changed")
    return actual


def live_document_indices(mask: torch.Tensor) -> tuple[int, ...]:
    if mask.shape != (SOURCE_DOCUMENTS, SEQUENCE) or mask.dtype != torch.bool:
        raise ValueError("FIT_SELECTOR mask is malformed")
    result = tuple(torch.nonzero(mask.any(1), as_tuple=False).squeeze(1).tolist())
    if len(result) != LIVE_DOCUMENTS:
        raise ValueError("FIT_SELECTOR live-document count changed")
    return result


def batch_plan(live: Sequence[int], start: int) -> tuple[tuple[int, ...], int]:
    indices = tuple(live[start:start + BATCH])
    if not indices or len(indices) > BATCH or len(set(indices)) != len(indices):
        raise ValueError("suffix batch slice is malformed")
    real = len(indices)
    return indices + (indices[0],) * (BATCH - real), real


def _score_and_support(response: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
    balanced = gate_math.context_balance(response)
    score = gate_math.ridge_leverage_scores(balanced, TARGET_RANK)
    return score, gate_math.select_top(score, K)


def _gauge_and_permutation_audit(
    mean: torch.Tensor, variance: torch.Tensor, down: torch.Tensor,
    suffix_score: torch.Tensor, suffix_support: Sequence[int],
) -> dict[str, Any]:
    std, orientation, permutation = suffix_math.canonical_derangement(
        mean, variance, down, DERANGEMENT_SEED,
    )
    random_support = suffix_math.canonical_hash_random_support(
        mean, variance, down, K, HASH_RANDOM_SEED,
    )
    pattern = torch.tensor([2.0, -4.0, 0.5, -0.25, 8.0, -2.0, 0.125], dtype=torch.float64)
    scales = pattern.repeat((HIDDEN + len(pattern) - 1) // len(pattern))[:HIDDEN]
    std2, orientation2, permutation2 = suffix_math.canonical_derangement(
        mean * scales, variance * scales.square(), down / scales, DERANGEMENT_SEED,
    )
    random2 = suffix_math.canonical_hash_random_support(
        mean * scales, variance * scales.square(), down / scales, K, HASH_RANDOM_SEED,
    )
    canonical_down = down * std[None, :] * orientation[None, :]
    canonical_down2 = down / scales * std2[None, :] * orientation2[None, :]
    general_generator = torch.Generator().manual_seed(2026090211)
    general_scales = torch.exp(
        3 * torch.randn(HIDDEN, generator=general_generator, dtype=torch.float64)
    )
    general_scales[::2].neg_()
    general_std, general_orientation, _ = suffix_math.canonical_derangement(
        mean * general_scales, variance * general_scales.square(),
        down / general_scales, DERANGEMENT_SEED,
    )
    general_canonical_down = (
        down / general_scales * general_std[None, :] * general_orientation[None, :]
    )
    generator = torch.Generator().manual_seed(2026090210)
    order = tuple(torch.randperm(HIDDEN, generator=generator).tolist())
    index = torch.tensor(order)
    _, _, permuted_derangement = suffix_math.canonical_derangement(
        mean[index], variance[index], down[:, index], DERANGEMENT_SEED,
    )
    permuted_random = suffix_math.canonical_hash_random_support(
        mean[index], variance[index], down[:, index], K, HASH_RANDOM_SEED,
    )
    permuted_score = suffix_score[index]
    permuted_suffix = gate_math.select_top(permuted_score, K)
    return {
        "dyadic_reciprocal": {
            "derangement_exact": permutation2 == permutation,
            "hash_random_exact": random2 == random_support,
            "canonical_down_max_abs_error": float((canonical_down2 - canonical_down).abs().max()),
        },
        "general_reciprocal_functional": {
            "canonical_down_max_relative_error": float(
                ((general_canonical_down - canonical_down).abs()
                 / canonical_down.abs().clamp_min(1e-30)).max()
            ),
            "hash_byte_replay_required": False,
        },
        "channel_permutation": {
            "derangement_equivariant": permuted_derangement == suffix_math.mapped_permutation(
                permutation, order,
            ),
            "hash_random_equivariant": {order[i] for i in permuted_random} == set(random_support),
            "suffix_support_equivariant": {order[i] for i in permuted_suffix} == set(suffix_support),
        },
        "derangement_sha256": canonical_sha256(list(permutation)),
        "hash_random_support_sha256": canonical_sha256(list(random_support)),
    }


def collect() -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    token_bundle = torch.load(TOKEN_ROWS, map_location="cpu", weights_only=True)
    role = token_bundle["FIT_SELECTOR"]
    rows, masks, document_indices = role["rows"], role["eligible_mask"], role["document_indices"]
    if rows.shape != (SOURCE_DOCUMENTS, SEQUENCE + 1) or document_indices.shape != (
        SOURCE_DOCUMENTS,
    ) or int(masks.sum()) != ELIGIBLE_POSITIONS:
        raise RuntimeError("FIT_SELECTOR token support changed")
    live = live_document_indices(masks)
    fit = torch.load(FIT_BUNDLE, map_location="cpu", weights_only=True)
    mean, variance = fit["mean"].double(), fit["variance"].double()
    if mean.shape != (HIDDEN,) or variance.shape != (HIDDEN,):
        raise RuntimeError("FIT_MEAN channel statistics changed")

    model, checkpoint = facade.load_bilin18(device=device, dtype=torch.bfloat16)
    down = model.transformer.h[SITE].mlp.Down.weight.detach().cpu().double().contiguous()
    standard_deviation, orientation, permutation = suffix_math.canonical_derangement(
        mean, variance, down, DERANGEMENT_SEED,
    )
    native_parts: list[torch.Tensor] = []
    deranged_parts: list[torch.Tensor] = []
    target_hashes: list[str] = []
    calls = mlp2_calls = backward_calls = scored_positions = 0
    native_baseline_checked = False
    native_baseline_max_abs_error = float("nan")

    for start in range(0, len(live), BATCH):
        planned, real = batch_plan(live, start)
        selected = torch.tensor(planned, dtype=torch.long)
        tokens = rows[selected, :-1].to(device).contiguous()
        mask = masks[selected].clone()
        if real < BATCH:
            mask[real:] = False
        mask_device = mask[:, SCORE_START:SCORE_STOP].to(device)
        ids = [f"fineweb-document-index:{int(document_indices[i])}" for i in planned]
        for dummy in range(real, BATCH):
            ids[dummy] = f"dummy-padding:{start}:{dummy}"
        alpha = torch.ones(BATCH, HIDDEN, device=device, dtype=torch.bfloat16, requires_grad=True)
        beta = torch.zeros_like(alpha, requires_grad=True)

        def attention(event: facade.AttentionEvent):
            return event.block.attn(event.state, event.first_value)

        def mlp(event: facade.EarlyMLPEvent):
            nonlocal mlp2_calls, native_baseline_checked, native_baseline_max_abs_error
            if event.site != SITE:
                return event.block.mlp(event.state)
            product = event.block.mlp.Left(event.state) * event.block.mlp.Right(event.state)
            mlp2_calls += 1
            write = suffix_math.centered_dual_write(
                product, mean, event.block.mlp.Down.weight, event.block.mlp.Down_bias,
                alpha, beta, standard_deviation, orientation, permutation,
            )
            if not native_baseline_checked:
                reference = event.block.mlp(event.state)
                native_baseline_max_abs_error = float(
                    (write.detach() - reference.detach()).abs().max()
                )
                if not torch.equal(write.detach(), reference.detach()):
                    raise RuntimeError("centered dual leaf changed the native MLP2 baseline")
                native_baseline_checked = True
            return write

        logits = facade.forward_with_dispatch(model, tokens, attention, mlp)
        targets = tangent.stateless_categorical_fisher_targets(
            logits, ids, PROBE_SEEDS, score_start=SCORE_START, score_stop=SCORE_STOP,
        )
        target_hashes.append(tensor_sha256(targets))
        log_probabilities = F.log_softmax(logits[:, SCORE_START:SCORE_STOP], dim=-1)
        target_device = targets.to(device)
        batch_native = torch.empty(real, len(PROBE_SEEDS), HIDDEN, dtype=torch.float64)
        batch_deranged = torch.empty_like(batch_native)
        for probe in range(len(PROBE_SEEDS)):
            selected_log_probability = torch.gather(
                log_probabilities, -1, target_device[probe].unsqueeze(-1),
            ).squeeze(-1)
            scalar = (selected_log_probability * mask_device).sum()
            native_gradient, deranged_gradient = torch.autograd.grad(
                scalar, (alpha, beta), retain_graph=probe + 1 < len(PROBE_SEEDS),
                create_graph=False, allow_unused=False,
            )
            native_cpu = native_gradient[:real].detach().cpu().double()
            deranged_cpu = deranged_gradient[:real].detach().cpu().double()
            if not bool(torch.isfinite(native_cpu).all() and torch.isfinite(deranged_cpu).all()):
                raise RuntimeError("MLP2 suffix response is nonfinite")
            batch_native[:, probe] = native_cpu
            batch_deranged[:, probe] = deranged_cpu
            backward_calls += 1
        native_parts.append(batch_native)
        deranged_parts.append(batch_deranged)
        calls += 1
        scored_positions += int(mask[:real].sum())
        del logits, targets, log_probabilities, target_device, alpha, beta
        torch.cuda.empty_cache()
        print(f"MLP2 suffix batch {calls}/{CALLS}", flush=True)

    native = torch.cat(native_parts, 0).contiguous()
    deranged = torch.cat(deranged_parts, 0).contiguous()
    expected_shape = (LIVE_DOCUMENTS, len(PROBE_SEEDS), HIDDEN)
    if native.shape != expected_shape or deranged.shape != expected_shape or (
        calls != CALLS or mlp2_calls != CALLS or backward_calls != CALLS * len(PROBE_SEEDS)
        or scored_positions != ELIGIBLE_POSITIONS or len(target_hashes) != CALLS
        or not native_baseline_checked or native_baseline_max_abs_error != 0
    ):
        raise RuntimeError("MLP2 suffix execution ledger changed")
    del model
    torch.cuda.empty_cache()

    suffix_score, suffix_support = _score_and_support(native)
    deranged_score, deranged_support = _score_and_support(deranged)
    first_score, first_support = _score_and_support(native[:, :4])
    second_score, second_support = _score_and_support(native[:, 4:])
    hash_random = suffix_math.canonical_hash_random_support(
        mean, variance, down, K, HASH_RANDOM_SEED,
    )
    supports = {
        "SUFFIX": torch.tensor(suffix_support, dtype=torch.int64),
        "DERANGED": torch.tensor(deranged_support, dtype=torch.int64),
        "LOCAL": fit["supports"]["LOCAL"].long(),
        "RMS": fit["supports"]["RMS"].long(),
        "MASS": fit["supports"]["MASS"].long(),
        "HASH_RANDOM": torch.tensor(hash_random, dtype=torch.int64),
    }
    if any(value.shape != (K,) or len(set(value.tolist())) != K for value in supports.values()):
        raise RuntimeError("MLP2 suffix support bundle is malformed")
    overlaps = {
        f"{left}_{right}": suffix_math.support_jaccard(supports[left], supports[right])
        for position, left in enumerate(supports)
        for right in tuple(supports)[position + 1:]
    }
    audit = _gauge_and_permutation_audit(mean, variance, down, suffix_score, suffix_support)
    if not all(audit[section][key] is True for section, key in (
        ("dyadic_reciprocal", "derangement_exact"),
        ("dyadic_reciprocal", "hash_random_exact"),
        ("channel_permutation", "derangement_equivariant"),
        ("channel_permutation", "hash_random_equivariant"),
        ("channel_permutation", "suffix_support_equivariant"),
    )) or audit["dyadic_reciprocal"]["canonical_down_max_abs_error"] != 0:
        raise RuntimeError("MLP2 suffix gauge/permutation audit failed")
    if audit["general_reciprocal_functional"]["canonical_down_max_relative_error"] > 1e-12:
        raise RuntimeError("MLP2 suffix general reciprocal functional replay failed")

    bundle = {
        "schema": "mlp2_cmr_v1_suffix_bundle",
        "scores": {"SUFFIX": suffix_score, "DERANGED": deranged_score},
        "supports": supports,
    }
    summary = {
        "schema": "mlp2_cmr_v1_suffix_result",
        "status": "fit_selector_complete_no_validation_or_replication",
        "checkpoint": checkpoint.__dict__,
        "documents": LIVE_DOCUMENTS,
        "eligible_positions": ELIGIBLE_POSITIONS,
        "probe_seeds": list(PROBE_SEEDS),
        "response_shape": list(expected_shape),
        "target_rank": TARGET_RANK,
        "forward_calls": calls,
        "backward_calls": backward_calls,
        "mlp2_centered_dual_calls": mlp2_calls,
        "native_baseline_bit_exact": native_baseline_checked,
        "native_baseline_max_abs_error": native_baseline_max_abs_error,
        "target_hashes": target_hashes,
        "support_overlaps": overlaps,
        "probe_half_stability": {
            "top512_jaccard": suffix_math.support_jaccard(first_support, second_support),
            "score_spearman": suffix_math.spearman(first_score, second_score),
            "first_half_support_sha256": canonical_sha256(list(first_support)),
            "second_half_support_sha256": canonical_sha256(list(second_support)),
            "nonpromotive": True,
        },
        "score_summaries": {
            name: {
                "minimum": float(score.min()), "median": float(score.median()),
                "maximum": float(score.max()), "sum": float(score.sum()),
            } for name, score in (("SUFFIX", suffix_score), ("DERANGED", deranged_score))
        },
        "gauge_and_permutation_audit": audit,
        "role_use": {
            "FIT_MEAN": "parent_only_no_model_access",
            "FIT_SELECTOR": "opened_for_centered_suffix_responses",
            "VALIDATION": "not_used_in_model",
            "REPLICATION": "not_used_in_model",
        },
        "outcome_access": {
            "raw_responses_published": False,
            "raw_targets_published": False,
            "raw_logits_published": False,
            "loss_or_accuracy_computed": False,
            "finite_candidate_evaluated": False,
            "validation_opened": False,
            "replication_opened": False,
        },
        "runtime_seconds": time.time() - started,
    }
    return bundle, summary


def main() -> None:
    namespace = (AUTHORITY, BUNDLE, RESULT, RECEIPT, FAILURE)
    if any(path.exists() for path in namespace):
        raise RuntimeError("refusing to overwrite MLP2 suffix namespace")
    lock_fd = os.open(LOCK, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        commit, source_hashes = committed_source()
        parents = protected_inputs()
        authority = {
            "schema_version": 1,
            "experiment_id": "bilin18_mlp2_cmr_v1_suffix_v2",
            "status": "authority_frozen_before_fit_selector_checkpoint_or_model_access",
            "source_commit": commit,
            "source_hashes": source_hashes,
            "parents": parents,
            "authorized_role": "FIT_SELECTOR",
            "authorized_forward_calls": CALLS,
            "authorized_backward_calls": CALLS * len(PROBE_SEEDS),
            "authorized_probe_seeds": list(PROBE_SEEDS),
            "forbidden": ["VALIDATION", "REPLICATION", "next-token loss", "accuracy", "finite candidate", "raw response publication", "raw target publication", "raw logit publication"],
        }
        write_create_only(AUTHORITY, json.dumps(authority, indent=2, sort_keys=True).encode() + b"\n")
        authority_hash = file_sha256(AUTHORITY)
        try:
            bundle, result = collect()
            result.update({
                "authority_sha256": authority_hash,
                "source_commit": commit,
                "source_hashes": source_hashes,
                "parents": parents,
                "tensor_hashes": {
                    "scores": {name: tensor_sha256(value) for name, value in bundle["scores"].items()},
                    "supports": {name: tensor_sha256(value) for name, value in bundle["supports"].items()},
                },
            })
            publish_torch_create_only(BUNDLE, bundle)
            result["bundle_sha256"] = file_sha256(BUNDLE)
            write_create_only(RESULT, json.dumps(result, indent=2, sort_keys=True).encode() + b"\n")
            replay = torch.load(BUNDLE, map_location="cpu", weights_only=True)
            if any(tensor_sha256(replay["supports"][name]) != digest for name, digest in result[
                "tensor_hashes"
            ]["supports"].items()):
                raise RuntimeError("MLP2 suffix bundle semantic replay failed")
            receipt = {
                "schema_version": 1,
                "experiment_id": "bilin18_mlp2_cmr_v1_suffix_v2",
                "status": "fit_selector_complete_receipt_last",
                "authority_sha256": authority_hash,
                "bundle_sha256": file_sha256(BUNDLE),
                "result_sha256": file_sha256(RESULT),
                "source_commit": commit,
                "source_hashes": source_hashes,
                "parents": parents,
                "authorized_for_validation": True,
                "authorized_for_replication": False,
                "scientific_claim": "none_fit_selector_artifact_only",
            }
            if file_sha256(AUTHORITY) != authority_hash or protected_inputs() != parents:
                raise RuntimeError("MLP2 suffix parents changed before receipt")
            write_create_only(RECEIPT, json.dumps(receipt, indent=2, sort_keys=True).encode() + b"\n")
            print(json.dumps(result, indent=2, sort_keys=True))
        except BaseException as exc:
            if not FAILURE.exists():
                failure = {
                    "schema_version": 1,
                    "experiment_id": "bilin18_mlp2_cmr_v1_suffix_v2",
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
