#!/usr/bin/env python3
"""Rung452: freeze uncertainty-separated MLP-PCA comparisons from rung450 only."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

import torch


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BQ = HERE.parent / "bilinear_quotient"
SOURCE = Path(__file__).resolve()
RESULT = BQ / "simplicity_mlp_pca_complete_candidate_consequences_results.json"
NATIVE = BQ / "simplicity_mlp0_complete_unablated.pt"
CONDITIONS = {
    "unablated": BQ / "simplicity_mlp_pca_complete_unablated.pt",
    "knockout": BQ / "simplicity_mlp_pca_complete_knockout.pt",
    "partner": BQ / "simplicity_mlp_pca_complete_partner.pt",
}
OUT = BQ / "simplicity_mlp_pca_reliability_spec_v1.json"
IDS = (
    "mlp_pca_p0_8_r256", "mlp_pca_p0_17_r256", "mlp_pca_p8_17_r256",
    "mlp_pca_p8_17_r384", "mlp_pca_p8_17_r512", "mlp_pca_grad32", "mlp_pca_grad64",
)
HASHES = {
    RESULT: "0f99365bdb9a21fb4674cc5695d89435127bd3225de7767e4d4d177a6191344e",
    NATIVE: "e3fa3c373b11ea455bf843cb555e6beb4cd68451bdda97f82a8393692096dd59",
    CONDITIONS["unablated"]: "04d43269360f7616e047231f26d9e0b5aa2db0e0954f37796952e930598401af",
    CONDITIONS["knockout"]: "3b5d36062b37ac2aded25b0ed57b9f877504e702b08ca25d03714bc45b13e1f3",
    CONDITIONS["partner"]: "786584aa55929b0edf797b1962cfaf143e095a2d28fc8e40dd039cbf5f5fb9f6",
}
SEED = 452
BOOTSTRAPS = 2000
ALPHA = .05
DOCUMENTS = 96
TOKENS = 256


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def committed_source() -> tuple[str, str]:
    relative = str(SOURCE.relative_to(REPO))
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    subprocess.run(["git", "merge-base", "--is-ancestor", commit, "origin/main"], cwd=REPO, check=True)
    blob = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=REPO)
    digest = hashlib.sha256(blob).hexdigest()
    if sha256(SOURCE) != digest:
        raise RuntimeError("rung452 freezer is not the committed HEAD blob")
    return commit, digest


def document_sums(numerator: torch.Tensor, denominator: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return (numerator.double().square().reshape(DOCUMENTS, TOKENS).sum(1),
            denominator.double().square().reshape(DOCUMENTS, TOKENS).sum(1))


def ratios(stats: dict[str, tuple[torch.Tensor, torch.Tensor]], samples: torch.Tensor) -> torch.Tensor:
    values = []
    for name in IDS:
        numerator, denominator = stats[name]
        values.append((numerator[samples].sum(1) / denominator[samples].sum(1).clamp_min(1e-24)).sqrt())
    return torch.stack(values, 1)


def freeze_pairs(values: torch.Tensor) -> list[dict[str, object]]:
    pairs = []
    for left in range(len(IDS)):
        for right in range(left + 1, len(IDS)):
            difference = values[:, left] - values[:, right]
            low, high = torch.quantile(difference, torch.tensor([ALPHA / 2, 1 - ALPHA / 2])).tolist()
            if low > 0 or high < 0:
                pairs.append({
                    "left": IDS[left], "right": IDS[right],
                    "expected_sign_left_minus_right": 1 if low > 0 else -1,
                    "bootstrap_difference_mean": float(difference.mean()),
                    "bootstrap_difference_ci95": [low, high],
                })
    return pairs


def write_create_only(payload: dict[str, object]) -> None:
    descriptor = os.open(OUT, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w") as sink:
        json.dump(payload, sink, indent=2, sort_keys=True)
        sink.write("\n"); sink.flush(); os.fsync(sink.fileno())


def main() -> None:
    if OUT.exists():
        raise RuntimeError("rung452 output namespace already exists")
    for path, digest in HASHES.items():
        if sha256(path) != digest:
            raise RuntimeError(f"frozen hash mismatch: {path}")
    commit, source_hash = committed_source()
    old = json.loads(RESULT.read_text())
    if old["rung"] != 450 or tuple(old["arms"]) != IDS or old["sealed_opened"]:
        raise RuntimeError("rung450 result identity changed")
    data = {name: torch.load(path, map_location="cpu", weights_only=True)
            for name, path in CONDITIONS.items()}
    native_aux = torch.load(NATIVE, map_location="cpu", weights_only=True)["native_aux"]
    native = native_aux["native_ce"].float(); native_ko = native_aux["native_ko_ce"].float()
    partner = native_aux["partner_ce"].float()
    if native.numel() != DOCUMENTS * TOKENS:
        raise RuntimeError("rung450 document geometry changed")

    removal_stats = {}; composition_stats = {}; full = {"removal": {}, "composition": {}}
    for name in IDS:
        p = data["unablated"]["candidate_ce"][name].float()
        pko = data["knockout"]["candidate_ce"][name].float()
        pq = data["partner"]["candidate_ce"][name].float()
        # Match rung450's promotion points exactly: each effect is formed in float32,
        # promoted to float64, and only then are the two effects subtracted.
        native_effect = (native_ko - native).double()
        candidate_effect = (pko - p).double()
        removal_stats[name] = document_sums(candidate_effect - native_effect, native_effect)
        physical = (pq - native).double()
        additive = ((p - native) + (partner - native)).double()
        composition_stats[name] = document_sums(physical - additive, additive)
        full["removal"][name] = float((removal_stats[name][0].sum() /
                                        removal_stats[name][1].sum()).sqrt())
        full["composition"][name] = float((composition_stats[name][0].sum() /
                                            composition_stats[name][1].sum()).sqrt())
        if abs(full["removal"][name] - old["arms"][name]["full"]["removal_normalized_error"]) > 1e-10:
            raise RuntimeError(f"removal reconstruction mismatch: {name}")
        if abs(full["composition"][name] - old["arms"][name]["full"]["composition_normalized_error"]) > 1e-10:
            raise RuntimeError(f"composition reconstruction mismatch: {name}")

    generator = torch.Generator().manual_seed(SEED)
    samples = torch.randint(DOCUMENTS, (BOOTSTRAPS, DOCUMENTS), generator=generator)
    boot = {"removal": ratios(removal_stats, samples),
            "composition": ratios(composition_stats, samples)}
    separated = {metric: freeze_pairs(values) for metric, values in boot.items()}
    if len(separated["removal"]) != 13 or len(separated["composition"]) != 16:
        raise RuntimeError("uncertainty-separated pair count changed")
    summaries = {
        metric: {name: {"bootstrap_mean": float(values[:, index].mean()),
                        "bootstrap_std": float(values[:, index].std(unbiased=True))}
                 for index, name in enumerate(IDS)}
        for metric, values in boot.items()
    }
    payload = {
        "schema": "simplicity_mlp_pca_reliability_spec_v1", "status": "frozen_before_new_outcomes",
        "rung": 452, "source_commit": commit, "source_sha256": source_hash,
        "inputs": {str(path): digest for path, digest in HASHES.items()},
        "method": {"resampling_unit": "document", "documents": DOCUMENTS, "tokens_per_document": TOKENS,
                   "bootstrap_resamples": BOOTSTRAPS, "seed": SEED, "interval": "percentile_95",
                   "separated_if": "CI(left_minus_right) excludes zero", "shared_resamples_across_arms": True},
        "candidate_ids": list(IDS), "original_full_values": full, "bootstrap_arm_summaries": summaries,
        "separated_pairs": separated,
        "pair_counts": {metric: len(pairs) for metric, pairs in separated.items()},
        "outcome_access": {"rung450_loaded": True, "reliability_rows_loaded": False,
                           "reliability_model_run": False, "sealed_opened": False},
        "pred_a_exact_old_sources": True, "pred_b_metric_reconstruction": True,
        "pred_c_deterministic_pair_freeze": True, "pred_d_new_outcomes_closed": True,
        "strong_null_specification_invalid": False,
        "next_step": "preregister_and_run_independent_mlp_pca_reliability",
    }
    write_create_only(payload)
    print(json.dumps({"status": "complete", "rung": 452, "output": str(OUT),
                      "sha256": sha256(OUT), "pair_counts": payload["pair_counts"],
                      "new_outcomes_opened": False}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
