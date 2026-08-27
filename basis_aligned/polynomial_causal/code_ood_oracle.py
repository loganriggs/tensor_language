#!/usr/bin/env python3
"""Conditional optimizer-free code-OOD oracle for FineWeb-licensed ship sites.

This file is intentionally standalone from the live shared ship implementation.
It is a callback for an authoritative FineWeb pipeline that hands it the exact
same in-memory ship realization. Independent ship reconstruction is forbidden.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
BQ = HERE.parent / "bilinear_quotient"
CORPUS = HERE / "code_oracle_corpus_v2.pt"
MANIFEST = HERE / "code_oracle_corpus_v2_manifest.json"
FINEWEB_RESULT = BQ / "ship_content_oracle_screen_results.json"
FACTORS = HERE / "content_product_frontier_factors.pt"
OUT = HERE / "code_ood_oracle_results.json"

D = 1152
T = 256
RANK = 64
SUPPORT_RANK = 256
NULLS = 20
SEED = 1618033
CONTENT_LAYERS = (8, 10, 12)
CELLS = ("global", "copy", "novel_freq", "novel_rare")
BOOTSTRAP_DRAWS = 2000
NULL_SCALE_RANGE = (0.1, 10.0)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes(order="C")).hexdigest()


def tensor_tree_sha256(value: Any) -> str:
    """Canonical hash for the exact derived ship realization handed between stages."""
    digest = hashlib.sha256()

    def update(item: Any) -> None:
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"tensor\0")
            digest.update(str(tensor.dtype).encode() + b"\0")
            digest.update(json.dumps(list(tensor.shape)).encode() + b"\0")
            digest.update(tensor.numpy().tobytes(order="C"))
        elif isinstance(item, dict):
            digest.update(b"dict\0")
            for key in sorted(item, key=lambda key_: str(key_)):
                update(str(key))
                update(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(("list" if isinstance(item, list) else "tuple").encode() + b"\0")
            for child in item:
                update(child)
        elif isinstance(item, (str, int, float, bool)) or item is None:
            digest.update(type(item).__name__.encode() + b"\0")
            digest.update(repr(item).encode() + b"\0")
        else:
            raise TypeError(f"unsupported ship-state value for hashing: {type(item)}")

    update(value)
    return digest.hexdigest()


def load_frozen_corpus() -> tuple[torch.Tensor, dict[str, Any]]:
    payload = torch.load(CORPUS, map_location="cpu", weights_only=False)
    manifest = json.loads(MANIFEST.read_text())
    if payload.get("manifest") != manifest:
        raise RuntimeError("embedded code-corpus manifest differs from tracked manifest")
    rows = payload.get("rows")
    if not isinstance(rows, torch.Tensor) or tuple(rows.shape) != (480, 257):
        raise RuntimeError(f"invalid frozen code corpus shape: {getattr(rows, 'shape', None)}")
    if rows.dtype != torch.long:
        raise RuntimeError(f"invalid frozen code corpus dtype: {rows.dtype}")
    if manifest.get("splits") != {
        "basis": [0, 96], "discovery": [96, 288], "heldout": [288, 480]
    }:
        raise RuntimeError("frozen code corpus split boundaries changed")
    if tensor_sha256(rows) != manifest.get("tensor_raw_sha256"):
        raise RuntimeError("frozen code corpus tensor hash mismatch")
    return rows.contiguous(), manifest


def load_fineweb_license(path: Path = FINEWEB_RESULT) -> tuple[list[int], dict[str, Any]]:
    """Fail closed before importing any GPU module or searching a second domain."""
    if not path.exists():
        raise RuntimeError(f"FineWeb oracle result is absent: {path}")
    result = json.loads(path.read_text())
    config = result.get("config")
    if not isinstance(config, dict) or config.get("status") != "authoritative_frozen_ship_v2":
        raise RuntimeError("FineWeb result is not an authoritative frozen-ship v2 result")
    ship_hash = config.get("ship_realization_sha256")
    if not isinstance(ship_hash, str) or len(ship_hash) != 64:
        raise RuntimeError("FineWeb result lacks an exact ship-realization hash")
    listed = result.get("training_license_sites")
    decisions = result.get("site_decisions")
    if not isinstance(listed, list) or not isinstance(decisions, dict):
        raise RuntimeError("FineWeb oracle result lacks license fields")
    sites = []
    for value in listed:
        if type(value) is not int or value not in (0, 1, 2) or value in sites:
            raise RuntimeError(f"invalid FineWeb licensed site: {value!r}")
        row = decisions.get(str(value))
        if not isinstance(row, dict) or not all(
            row.get(gate) is True for gate in (
                "full_oracle_ci95_lower_gt_zero",
                "content_positive_both_splits",
                "content_beats_matched_null95_heldout",
            )
        ):
            raise RuntimeError(f"FineWeb license list disagrees with site {value} gates")
        sites.append(value)
    if not sites:
        raise RuntimeError("FineWeb oracle licensed no site; code search is forbidden")
    return sites, result


def top_eigenbasis(values: torch.Tensor, rank: int, center: bool) -> torch.Tensor:
    """Exact right-PCA basis from a D x D Gram matrix, with no randomized SVD."""
    if values.ndim != 2 or rank <= 0 or rank > min(values.shape):
        raise ValueError(f"invalid PCA shape/rank: {tuple(values.shape)}, {rank}")
    work = values.float()
    if center:
        work = work - work.mean(0)
    gram = work.T @ work
    eigenvalues, vectors = torch.linalg.eigh(gram)
    basis = vectors[:, -rank:].flip(1).contiguous()
    if float(eigenvalues[-rank]) <= 0.0:
        raise RuntimeError("requested PCA support is not full rank")
    return basis


def token_conditional_content_basis(
    layer_values: dict[int, torch.Tensor],
    tokens: torch.Tensor,
    rank: int = RANK,
) -> torch.Tensor:
    """Frozen content recipe: average per-token deviations at residual L8/10/12."""
    if tuple(sorted(layer_values)) != CONTENT_LAYERS:
        raise ValueError(f"expected content layers {CONTENT_LAYERS}")
    flat_tokens = tokens.reshape(-1).long()
    n = len(flat_tokens)
    width = next(iter(layer_values.values())).shape[-1]
    device = next(iter(layer_values.values())).device
    flat_tokens = flat_tokens.to(device)
    vocab = max(50257, int(flat_tokens.max()) + 1)
    counts = torch.bincount(flat_tokens, minlength=vocab).float()
    pooled = torch.zeros(n, width, device=device, dtype=torch.float32)
    for layer in CONTENT_LAYERS:
        values = layer_values[layer].reshape(-1, width).float()
        if len(values) != n or values.device != device:
            raise ValueError("content layer values do not align with tokens/device")
        means = torch.zeros(vocab, width, device=device)
        means.index_add_(0, flat_tokens, values)
        means /= counts.clamp_min(1).unsqueeze(1)
        pooled += values - means[flat_tokens]
    pooled /= len(CONTENT_LAYERS)
    basis = top_eigenbasis(pooled, rank, center=True)
    return torch.linalg.qr(basis, mode="reduced").Q.contiguous()


def projection_rms(residual: torch.Tensor, basis: torch.Tensor | None, scale: float = 1.0) -> float:
    if basis is None:
        square_sum = residual.double().square().sum()
    else:
        square_sum = (residual.float() @ basis.float()).double().square().sum()
    return math.sqrt(float(square_sum) * scale * scale / residual.numel())


def build_projection_arms(
    residual: torch.Tensor,
    prose_basis: torch.Tensor,
    code_basis: torch.Tensor,
    site: int,
    rank: int = RANK,
    support_rank: int = SUPPORT_RANK,
    nulls: int = NULLS,
) -> dict[str, dict[str, Any]]:
    """Build exact arms and one shared Haar family, separately RMS-scaled."""
    support = top_eigenbasis(residual, support_rank, center=False)
    local = support[:, :rank].contiguous()
    prose = torch.linalg.qr(prose_basis.float(), mode="reduced").Q[:, :rank].contiguous()
    code = torch.linalg.qr(code_basis.float(), mode="reduced").Q[:, :rank].contiguous()
    arms: dict[str, dict[str, Any]] = {
        "full": {"basis": None, "scale": 1.0},
        "prose_content": {"basis": prose, "scale": 1.0},
        "code_content": {"basis": code, "scale": 1.0},
        "local_pca": {"basis": local, "scale": 1.0},
    }
    target_rms = {
        name: projection_rms(residual, arms[name]["basis"])
        for name in ("prose_content", "code_content")
    }
    generator = torch.Generator(device=residual.device).manual_seed(
        SEED + 10000 * site
    )
    for null_index in range(nulls):
        coordinates = torch.randn(
            support_rank, rank, device=residual.device, generator=generator
        )
        haar = torch.linalg.qr(coordinates, mode="reduced").Q
        basis = (support @ haar).contiguous()
        raw_rms = projection_rms(residual, basis)
        for content_name in ("prose_content", "code_content"):
            scale = target_rms[content_name] / max(raw_rms, 1e-30)
            if not math.isfinite(scale) or not (
                NULL_SCALE_RANGE[0] <= scale <= NULL_SCALE_RANGE[1]
            ):
                raise RuntimeError(
                    f"site {site} null {null_index} invalid RMS scale {scale}"
                )
            arms[f"{content_name}_null_{null_index:02d}"] = {
                "basis": basis,
                "scale": scale,
                "raw_fit_correction_rms": raw_rms,
                "matched_to": content_name,
                "shared_null_index": null_index,
            }
    for row in arms.values():
        row["fit_correction_rms"] = projection_rms(
            residual, row["basis"], row["scale"]
        )
    return arms


def fit_lexical_residual_table(
    residual: torch.Tensor, sampled_tokens: torch.Tensor, vocab: int = 50257
) -> torch.Tensor:
    """Basis-only token-mean residual with global-mean fallback for unseen tokens."""
    tokens = sampled_tokens.reshape(-1).long().to(residual.device)
    if len(tokens) != len(residual):
        raise ValueError("sampled tokens do not align with captured residual")
    sums = torch.zeros(vocab, residual.shape[1], device=residual.device)
    counts = torch.bincount(tokens, minlength=vocab).float()
    sums.index_add_(0, tokens, residual.float())
    global_mean = residual.float().mean(0)
    return torch.where(
        counts[:, None] > 0,
        sums / counts.clamp_min(1)[:, None],
        global_mean[None, :],
    ).contiguous()


def rare_vocabulary(rows: torch.Tensor) -> torch.Tensor:
    counts = torch.zeros(50257)
    targets = rows[:, 1:]
    valid = torch.ones_like(targets, dtype=torch.bool)
    valid[:, :64] = False
    counts.index_add_(0, targets.reshape(-1), valid.reshape(-1).float())
    threshold = counts.sort(descending=True).values[500]
    return counts < threshold


def row_masks(idx: torch.Tensor, targets: torch.Tensor, rare: torch.Tensor) -> dict[str, torch.Tensor]:
    valid = torch.ones_like(targets, dtype=torch.bool)
    valid[:, :64] = False
    copy = torch.zeros_like(valid)
    for lag in range(64):
        past = torch.roll(idx, lag, dims=1)
        if lag:
            past[:, :lag] = -1
        copy |= past == targets
    copy &= valid
    rare_targets = rare.to(targets.device)[targets] & valid
    return {
        "global": valid,
        "copy": copy,
        "novel_freq": valid & ~copy & ~rare_targets,
        "novel_rare": valid & ~copy & rare_targets,
    }


def summarize_row_stats(row_sums: dict[str, list[float]], row_counts: dict[str, list[int]]) -> dict[str, Any]:
    return {
        "ce": {
            cell: sum(row_sums[cell]) / max(sum(row_counts[cell]), 1)
            for cell in CELLS
        },
        "counts": {cell: int(sum(row_counts[cell])) for cell in CELLS},
        "row_sums": row_sums,
        "row_counts": row_counts,
    }


def _cluster_reduce(
    values: list[float], counts: list[int] | list[float], clusters: list[str]
) -> tuple[torch.Tensor, torch.Tensor]:
    if not (len(values) == len(counts) == len(clusters)):
        raise ValueError("row values/counts/clusters do not align")
    names = list(dict.fromkeys(clusters))
    lookup = {name: index for index, name in enumerate(names)}
    sums = torch.zeros(len(names), dtype=torch.float64)
    denominators = torch.zeros(len(names), dtype=torch.float64)
    for value, count, cluster in zip(values, counts, clusters):
        index = lookup[cluster]
        sums[index] += float(value)
        denominators[index] += float(count)
    return sums, denominators


def _cluster_draws(n_clusters: int, seed: int, draws: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(n_clusters, (draws, n_clusters), generator=generator)


def _interval(values: torch.Tensor) -> list[float | None]:
    finite = values[torch.isfinite(values)]
    if not len(finite):
        return [None, None]
    return [float(torch.quantile(finite, 0.025)), float(torch.quantile(finite, 0.975))]


def paired_cell_gain(
    baseline: dict[str, Any], arm: dict[str, Any], cell: str, clusters: list[str],
    seed: int, draws: int = BOOTSTRAP_DRAWS,
) -> dict[str, Any]:
    if baseline["row_counts"][cell] != arm["row_counts"][cell]:
        raise RuntimeError(f"paired masks changed in cell {cell}")
    base_sum, counts = _cluster_reduce(
        baseline["row_sums"][cell], baseline["row_counts"][cell], clusters
    )
    arm_sum, arm_counts = _cluster_reduce(
        arm["row_sums"][cell], arm["row_counts"][cell], clusters
    )
    if not torch.equal(counts, arm_counts):
        raise RuntimeError(f"paired cluster masks changed in cell {cell}")
    indices = _cluster_draws(len(counts), seed, draws)
    denominator = counts[indices].sum(1)
    boot = (base_sum[indices].sum(1) - arm_sum[indices].sum(1)) / denominator
    boot[denominator == 0] = float("nan")
    return {
        "mean": baseline["ce"][cell] - arm["ce"][cell],
        "ci95": _interval(boot),
        "clusters": len(counts),
        "finite_bootstrap_draws": int(torch.isfinite(boot).sum()),
    }


def bootstrap_correction_rms(
    row_mse: list[float], clusters: list[str], seed: int,
    draws: int = BOOTSTRAP_DRAWS,
) -> dict[str, Any]:
    sums, counts = _cluster_reduce(row_mse, [1] * len(row_mse), clusters)
    indices = _cluster_draws(len(counts), seed, draws)
    boot = (sums[indices].sum(1) / counts[indices].sum(1)).clamp_min(0).sqrt()
    return {
        "mean": math.sqrt(sum(row_mse) / len(row_mse)),
        "ci95": _interval(boot),
        "clusters": len(counts),
    }


def bootstrap_fraction_of_full(
    baseline: dict[str, Any], arm: dict[str, Any], full: dict[str, Any],
    clusters: list[str], seed: int, draws: int = BOOTSTRAP_DRAWS,
) -> dict[str, Any]:
    cell = "global"
    base_sum, counts = _cluster_reduce(
        baseline["row_sums"][cell], baseline["row_counts"][cell], clusters
    )
    arm_sum, _ = _cluster_reduce(
        arm["row_sums"][cell], arm["row_counts"][cell], clusters
    )
    full_sum, _ = _cluster_reduce(
        full["row_sums"][cell], full["row_counts"][cell], clusters
    )
    indices = _cluster_draws(len(counts), seed, draws)
    arm_gain = (base_sum[indices] - arm_sum[indices]).sum(1)
    full_gain = (base_sum[indices] - full_sum[indices]).sum(1)
    full_gain_interval = _interval(full_gain)
    full_positive = (
        full_gain_interval[0] is not None and full_gain_interval[0] > 0.0
    )
    ratios = arm_gain / full_gain if full_positive else torch.full_like(full_gain, float("nan"))
    point_denominator = baseline["ce"][cell] - full["ce"][cell]
    point = ((baseline["ce"][cell] - arm["ce"][cell]) / point_denominator
             if point_denominator > 0 else None)
    return {
        "mean": point,
        "ci95": _interval(ratios) if full_positive else [None, None],
        "status": ("defined_full_gain_ci_lower_gt_zero" if full_positive
                   else "undefined_full_gain_ci_includes_zero"),
        "full_gain_sum_ci95": full_gain_interval,
        "positive_full_gain_bootstrap_draws": int(torch.isfinite(ratios).sum()),
        "clusters": len(counts),
    }


def paired_arm_gain_difference(
    prose: dict[str, Any], code: dict[str, Any], clusters: list[str], seed: int,
    draws: int = BOOTSTRAP_DRAWS,
) -> dict[str, Any]:
    """Code gain minus prose gain, equal to prose CE minus code CE."""
    cell = "global"
    prose_sum, counts = _cluster_reduce(
        prose["row_sums"][cell], prose["row_counts"][cell], clusters
    )
    code_sum, code_counts = _cluster_reduce(
        code["row_sums"][cell], code["row_counts"][cell], clusters
    )
    if not torch.equal(counts, code_counts):
        raise RuntimeError("prose/code paired masks changed")
    indices = _cluster_draws(len(counts), seed, draws)
    denominator = counts[indices].sum(1)
    boot = (prose_sum[indices].sum(1) - code_sum[indices].sum(1)) / denominator
    return {
        "mean": prose["ce"][cell] - code["ce"][cell],
        "ci95": _interval(boot),
        "clusters": len(counts),
    }


def exact_null_test(content_gain: float, null_gains: list[float]) -> dict[str, Any]:
    exceedances = sum(value >= content_gain for value in null_gains)
    return {
        "content_gain": content_gain,
        "null_gains": null_gains,
        "nulls_at_least_content": exceedances,
        "exact_one_sided_p": (1 + exceedances) / (1 + len(null_gains)),
        "passes_5pct": len(null_gains) == 20 and exceedances == 0,
    }


def classify_site(
    gains: dict[str, dict[str, dict[str, Any]]],
    null_tests: dict[str, dict[str, Any]],
    code_minus_prose: dict[str, Any],
) -> dict[str, Any]:
    held = gains["heldout"]
    disc = gains["discovery"]
    full = held["full"]["global"]
    threshold = max(0.02, 0.10 * full["mean"])

    def content_pass(name: str) -> bool:
        return (
            disc[name]["global"]["mean"] > 0.0
            and held[name]["global"]["mean"] > 0.0
            and null_tests[name]["passes_5pct"]
            and held[name]["global"]["mean"] >= threshold
        )

    full_pass = full["ci95"][0] > 0.0
    prose_pass = full_pass and content_pass("prose_content")
    code_pass = full_pass and content_pass("code_content")
    local = held["local_pca"]["global"]
    local_pass = (
        disc["local_pca"]["global"]["mean"] > 0.0
        and local["ci95"][0] > 0.0
        and local["mean"] >= threshold
    )
    lexical = held["lexical_mean"]["global"]
    lexical_pass = (
        disc["lexical_mean"]["global"]["mean"] > 0.0
        and lexical["ci95"][0] > 0.0
        and lexical["mean"] >= threshold
    )
    difference_ci = code_minus_prose["ci95"]
    if not full_pass:
        label = "compensatory-only site"
    elif prose_pass and difference_ci[1] is not None and difference_ci[1] <= 0.02:
        label = "shared prose coordinate"
    elif code_pass and difference_ci[0] is not None and difference_ci[0] >= 0.02:
        label = "domain-typed coordinate"
    elif not prose_pass and not code_pass and lexical_pass:
        label = "non-contextual lexical residual"
    elif not prose_pass and not code_pass and (full_pass or local_pass):
        label = "non-content residual"
    else:
        label = "inconclusive content coordinate"
    return {
        "classification": label,
        "full_oracle_ci95_lower_gt_zero": full_pass,
        "prose_content_passes": prose_pass,
        "code_content_passes": code_pass,
        "local_pca_passes": local_pass,
        "lexical_mean_passes": lexical_pass,
        "minimum_content_gain": threshold,
        "exact_null_tests": null_tests,
        "code_minus_prose_gain": code_minus_prose,
        "licenses_learned_code_predictor": label in (
            "shared prose coordinate", "domain-typed coordinate"
        ),
    }


@torch.no_grad()
def capture_clean_content(sa: Any, rows: torch.Tensor) -> dict[int, torch.Tensor]:
    values = {layer: [] for layer in CONTENT_LAYERS}
    for start in range(0, len(rows), 8):
        idx = rows[start:start + 8, :-1].to(sa.DEV).contiguous()
        x = F.rms_norm(sa.m.transformer.wte(idx), (D,))
        x0 = x
        v1 = None
        for layer, block in enumerate(sa.m.transformer.h):
            x, v1 = block(x, v1, x0)
            if layer in CONTENT_LAYERS:
                values[layer].append(x.detach().float().cpu())
    return {layer: torch.cat(parts) for layer, parts in values.items()}


def make_instrumented_oracle(sa: Any, metric_sink: dict[str, Any]):
    def add(site: int, block: Any, z: torch.Tensor, mo: torch.Tensor) -> torch.Tensor:
        should_capture = (
            sa.ORACLE_CORR["capture"] is not None
            and site in sa.ORACLE_CORR["capture"]
        )
        should_inject = sa.ORACLE_CORR["on"] and sa.ORACLE_CORR["site"] == site
        if not should_capture and not should_inject:
            return mo
        original = block.mlp(z).float()
        residual = original - mo.float()
        if should_capture:
            sa.ORACLE_CORR["capture"][site].append(
                residual[:, 64::3].detach().cpu()
            )
        if not should_inject:
            return mo
        lexical_table = sa.ORACLE_CORR.get("lexical_table")
        basis = sa.ORACLE_CORR.get("basis")
        if lexical_table is not None:
            tokens = metric_sink.get("tokens")
            if tokens is None or tuple(tokens.shape) != residual.shape[:2]:
                raise RuntimeError("lexical arm lacks aligned live tokens")
            delta = lexical_table[tokens].view_as(residual)
        elif basis is None:
            delta = residual
        else:
            flat = residual.reshape(-1, D)
            delta = ((flat @ basis) @ basis.T).view_as(residual)
        delta = sa.ORACLE_CORR["scale"] * delta
        if metric_sink.get("row_mse") is not None:
            mse = delta[:, 64:].double().square().mean(dim=(1, 2))
            metric_sink["row_mse"].extend(mse.detach().cpu().tolist())
        corrected = mo + delta.to(mo.dtype)
        if basis is None and lexical_table is None and sa.ORACLE_CORR["scale"] == 1.0:
            if not torch.allclose(corrected.float(), original, atol=2e-4, rtol=2e-4):
                raise RuntimeError("full oracle does not reconstruct exact original MLP output")
        return corrected
    return add


@torch.no_grad()
def score_rows(
    sa: Any, rows: torch.Tensor, twall: dict, all_attention: frozenset[int],
    all_mlps: frozenset[int], rare: torch.Tensor,
    metric_sink: dict[str, Any],
) -> dict[str, Any]:
    row_sums = {cell: [] for cell in CELLS}
    row_counts = {cell: [] for cell in CELLS}
    metric_sink["row_mse"] = [] if sa.ORACLE_CORR["on"] else None
    for start in range(0, len(rows), 8):
        batch = rows[start:start + 8].to(sa.DEV)
        idx, targets = batch[:, :-1].contiguous(), batch[:, 1:].contiguous()
        metric_sink["tokens"] = idx
        logits = sa.fwd_arm(idx, all_attention, twall, all_mlps).float()
        ce = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none"
        ).view_as(targets)
        masks = row_masks(idx, targets, rare)
        for cell, select in masks.items():
            row_sums[cell].extend((ce * select).sum(1).detach().cpu().tolist())
            row_counts[cell].extend(select.sum(1).detach().cpu().tolist())
    result = summarize_row_stats(row_sums, row_counts)
    result["row_correction_mse"] = metric_sink["row_mse"]
    metric_sink["row_mse"] = None
    metric_sink["tokens"] = None
    return result


def serializable_arm_metrics(arms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        name: {key: value for key, value in row.items()
               if key not in ("basis", "lexical_table")}
        for name, row in arms.items()
    }


def split_clusters(manifest: dict[str, Any], split: str) -> list[str]:
    clusters = [row["path"] for row in manifest["row_provenance"][split]]
    start, end = manifest["splits"][split]
    if len(clusters) != end - start:
        raise RuntimeError(f"manifest cluster labels do not close for {split}")
    return clusters


@torch.no_grad()
def run_code_oracle(
    sa: Any, twall: dict, all_attention: frozenset[int], start_time: float,
    licensed_sites: list[int], fineweb_result: dict[str, Any],
    corpus_rows: torch.Tensor, corpus_manifest: dict[str, Any],
) -> None:
    ship_realization_sha256 = tensor_tree_sha256({
        "TWALL": twall,
        "SHIP": sa.SHIP,
        "CORR": {key: sa.CORR[key] for key in ("on", "b", "U", "V")},
        "all_attention": sorted(all_attention),
    })
    expected_ship_hash = fineweb_result["config"]["ship_realization_sha256"]
    if ship_realization_sha256 != expected_ship_hash:
        raise RuntimeError(
            "code stage ship realization differs from authoritative FineWeb stage"
        )
    all_mlps = frozenset(range(18))
    basis_rows = corpus_rows[0:96]
    discovery_rows = corpus_rows[96:288]
    heldout_rows = corpus_rows[288:480]
    factors = torch.load(FACTORS, map_location="cpu", weights_only=False)
    prose_raw_cpu = factors["sites"]["0"]["content_basis"][:, :RANK].float()
    prose_raw = prose_raw_cpu.to(sa.DEV)
    prose_basis = torch.linalg.qr(prose_raw.float(), mode="reduced").Q
    prose_basis_hashes = {}
    for site in (0, 1, 2):
        candidate = factors["sites"][str(site)]["content_basis"][:, :RANK].float()
        candidate_q = torch.linalg.qr(candidate.to(sa.DEV), mode="reduced").Q
        projector_error = float(
            (candidate_q @ candidate_q.T - prose_basis @ prose_basis.T).abs().max()
        )
        if projector_error > 2e-4:
            raise RuntimeError(f"prose content projectors differ at factor site {site}")
        prose_basis_hashes[str(site)] = tensor_sha256(candidate)

    clean_values = capture_clean_content(sa, basis_rows)
    code_basis = token_conditional_content_basis(
        {layer: value.to(sa.DEV) for layer, value in clean_values.items()},
        basis_rows[:, :-1].to(sa.DEV),
    )
    del clean_values
    torch.cuda.empty_cache()

    metric_sink: dict[str, Any] = {"row_mse": None, "tokens": None}
    sa.add_oracle_correction = make_instrumented_oracle(sa, metric_sink)
    sa.CONTENT_CORR["on"] = False
    sa.ORACLE_CORR.update({
        "on": False, "capture": {site: [] for site in licensed_sites}
    })
    for start in range(0, len(basis_rows), 8):
        idx = basis_rows[start:start + 8, :-1].to(sa.DEV).contiguous()
        sa.fwd_arm(idx, all_attention, twall, all_mlps)
    captured = {
        site: torch.cat(sa.ORACLE_CORR["capture"][site]).reshape(-1, D).to(sa.DEV)
        for site in licensed_sites
    }
    sa.ORACLE_CORR["capture"] = None

    arms = {
        site: build_projection_arms(captured[site], prose_basis, code_basis, site)
        for site in licensed_sites
    }
    sampled_tokens = basis_rows[:, :-1][:, 64::3].reshape(-1).to(sa.DEV)
    for site in licensed_sites:
        lexical_table = fit_lexical_residual_table(captured[site], sampled_tokens)
        lexical_fit = lexical_table[sampled_tokens]
        arms[site]["lexical_mean"] = {
            "basis": None,
            "lexical_table": lexical_table,
            "scale": 1.0,
            "fit_correction_rms": math.sqrt(
                float(lexical_fit.double().square().mean())
            ),
        }
    fit_metrics = {
        str(site): serializable_arm_metrics(site_arms)
        for site, site_arms in arms.items()
    }
    del captured
    torch.cuda.empty_cache()

    rare = rare_vocabulary(discovery_rows)
    evaluations: dict[str, Any] = {}
    for split, rows in (("discovery", discovery_rows), ("heldout", heldout_rows)):
        sa.ORACLE_CORR["on"] = False
        baseline = score_rows(
            sa, rows, twall, all_attention, all_mlps, rare, metric_sink
        )
        evaluations[split] = {"ship_baseline": baseline, "sites": {}}
        for site in licensed_sites:
            site_scores = {}
            for arm_name, arm in arms[site].items():
                sa.ORACLE_CORR.update({
                    "on": True, "site": site, "basis": arm["basis"],
                    "scale": arm["scale"],
                    "lexical_table": arm.get("lexical_table"),
                })
                site_scores[arm_name] = score_rows(
                    sa, rows, twall, all_attention, all_mlps, rare, metric_sink
                )
                print(f"code oracle {split} site={site} arm={arm_name} done", flush=True)
            evaluations[split]["sites"][str(site)] = site_scores

    paired_gains: dict[str, Any] = {}
    decisions: dict[str, Any] = {}
    for site in licensed_sites:
        key = str(site)
        paired_gains[key] = {}
        for split_index, split in enumerate(("discovery", "heldout")):
            base = evaluations[split]["ship_baseline"]
            clusters = split_clusters(corpus_manifest, split)
            paired_gains[key][split] = {}
            for arm_index, (arm_name, scored) in enumerate(
                evaluations[split]["sites"][key].items()
            ):
                arm_gain = {
                    cell: paired_cell_gain(
                        base, scored, cell, clusters,
                        SEED + 100000 * site + 10000 * split_index
                        + 100 * arm_index + cell_index,
                    )
                    for cell_index, cell in enumerate(CELLS)
                }
                arm_gain["correction_rms"] = bootstrap_correction_rms(
                    scored["row_correction_mse"], clusters,
                    SEED + 200000 * site + 10000 * split_index + arm_index,
                )
                paired_gains[key][split][arm_name] = arm_gain
        for split_index, split in enumerate(("discovery", "heldout")):
            base = evaluations[split]["ship_baseline"]
            full = evaluations[split]["sites"][key]["full"]
            clusters = split_clusters(corpus_manifest, split)
            for arm_index, (arm_name, scored) in enumerate(
                evaluations[split]["sites"][key].items()
            ):
                paired_gains[key][split][arm_name]["fraction_of_full_global_gain"] = (
                    bootstrap_fraction_of_full(
                        base, scored, full, clusters,
                        SEED + 300000 * site + 10000 * split_index + arm_index,
                    )
                )
        null_tests = {}
        for content_name in ("prose_content", "code_content"):
            null_values = [
                paired_gains[key]["heldout"][
                    f"{content_name}_null_{index:02d}"
                ]["global"]["mean"]
                for index in range(NULLS)
            ]
            null_tests[content_name] = exact_null_test(
                paired_gains[key]["heldout"][content_name]["global"]["mean"],
                null_values,
            )
        heldout_clusters = split_clusters(corpus_manifest, "heldout")
        code_minus_prose = paired_arm_gain_difference(
            evaluations["heldout"]["sites"][key]["prose_content"],
            evaluations["heldout"]["sites"][key]["code_content"],
            heldout_clusters,
            SEED + 400000 * site,
        )
        decisions[key] = classify_site(
            paired_gains[key], null_tests, code_minus_prose
        )

    output = {
        "config": {
            "model": "bilin18",
            "ship": "identical current K=3072 ship with incumbent MLP2 glue live",
            "source_fineweb_licensed_sites": licensed_sites,
            "projection_rank": RANK,
            "matched_null_support_rank": SUPPORT_RANK,
            "matched_nulls_per_content_arm": NULLS,
            "null_gate": "exact one-sided Monte Carlo: content beats all 20",
            "null_scale_valid_range": list(NULL_SCALE_RANGE),
            "content_layers": list(CONTENT_LAYERS),
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "seed": SEED,
            "splits": corpus_manifest["splits"],
            "copy_definition": "target recurs at context distance 1 through 64",
            "frequency_vocab": "frozen from code discovery rows and reused on heldout",
            "uncertainty_unit": "file-cluster bootstrap; rows within a file resampled together",
            "status": "optimizer-free conditional code-OOD singleton oracle",
        },
        "provenance": {
            "source_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=HERE, check=True,
                capture_output=True, text=True,
            ).stdout.strip(),
            "script_sha256": file_sha256(Path(__file__)),
            "fineweb_result_sha256": file_sha256(FINEWEB_RESULT),
            "factors_sha256": file_sha256(FACTORS),
            "corpus_manifest_sha256": file_sha256(MANIFEST),
            "corpus_tensor_raw_sha256": corpus_manifest["tensor_raw_sha256"],
            "ship_realization_sha256": ship_realization_sha256,
            "prose_basis_tensor_sha256_by_factor_site": prose_basis_hashes,
        },
        "arms": {
            "full": "exact live original MLP output minus deployed plank output",
            "prose_content": "full residual projected through frozen prose content basis",
            "code_content": "full residual projected through basis-split code content basis",
            "local_pca": "full residual projected through basis-split top residual PCs",
            "lexical_mean": "basis-only token-mean missing residual, with global-mean unseen-token fallback",
            "matched_nulls": "same 20 Haar directions inside local top-256 support, RMS-scaled separately to each content arm",
        },
        "fineweb_license": {
            "training_license_sites": licensed_sites,
            "site_decisions": {
                str(site): fineweb_result["site_decisions"][str(site)]
                for site in licensed_sites
            },
        },
        "fit_correction_metrics": fit_metrics,
        "evaluations": evaluations,
        "paired_gains": paired_gains,
        "site_decisions": decisions,
        "learned_code_predictor_license_sites": [
            int(site) for site, row in decisions.items()
            if row["licenses_learned_code_predictor"]
        ],
        "interpretation_guardrail": (
            "A code pass classifies the coordinate interface for missing original "
            "computation. It does not license deployment before alternate-background "
            "and intervention-family transfer tests."
        ),
        "runtime_s": round(time.time() - start_time, 1),
    }
    if OUT.exists():
        raise RuntimeError(f"refusing to overwrite existing result: {OUT}")
    OUT.write_text(json.dumps(output, indent=2) + "\n")
    sa.ORACLE_CORR["on"] = False
    print(json.dumps({
        "site_decisions": decisions,
        "learned_code_predictor_license_sites": output[
            "learned_code_predictor_license_sites"
        ],
    }, indent=2), flush=True)
    print(f"wrote {OUT} ({output['runtime_s']}s)", flush=True)


def main() -> None:
    raise SystemExit(
        "Independent execution is forbidden: code_ood_oracle.run_code_oracle must "
        "receive the exact authoritative FineWeb ship realization in memory."
    )


if __name__ == "__main__":
    main()
