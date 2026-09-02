#!/usr/bin/env python3
"""Plot the corrected rung-457 equality-term identification result."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
SOURCE = ROOT / "equality_term_subset_factorial_stage1_analysis_correction.json"
OUTPUT = ROOT.parent / "polynomial_causal/explanations/assets/equality_factorial_stage1_2026-09-02.png"
SOURCE_SHA256 = "4fc0ed40b2007b8140c75f1e3aa85dac15e7078ee41a8d61840462af581f43ae"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if sha256(SOURCE) != SOURCE_SHA256:
        raise RuntimeError("corrected rung-457 result changed")
    result = json.loads(SOURCE.read_text())["corrected_analysis"]
    recovery = result["extraction_recovery_nat"]
    pooled = recovery["all_positive"]["1111"]

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.25), constrained_layout=True)
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]

    names = ["L5H5", "L7H3", "L8H3", "L8H4"]
    singleton = [100 * recovery["all_positive"][bits] / pooled
                 for bits in ("0001", "0010", "0100", "1000")]
    axes[0].bar(names, singleton, color=colors)
    axes[0].axhline(100, color="#555555", linewidth=1, linestyle="--")
    axes[0].set_ylabel("Recovery (% of all-four recovery)")
    axes[0].set_title("Each term alone")
    axes[0].text(.02, .97, "Bars sum to 186% → overlap", transform=axes[0].transAxes,
                 va="top", fontsize=9)

    pair = result["primary_pair"]
    block = result["early_vs_layer8_block"]
    interactions = np.array([pair["point_nat"], block["point_nat"]]) * 100 / pooled
    lows = np.array([pair["simultaneous_low"], block["simultaneous_low"]]) * 100 / pooled
    highs = np.array([pair["simultaneous_high"], block["simultaneous_high"]]) * 100 / pooled
    axes[1].bar(["L8H3 + L8H4", "early block + L8 block"], interactions,
                color=["#72B7B2", "#B279A2"])
    axes[1].errorbar(np.arange(2), interactions,
                     yerr=np.vstack([interactions - lows, highs - interactions]),
                     fmt="none", color="black", capsize=5, linewidth=1.5)
    axes[1].axhline(0, color="#333333", linewidth=1)
    axes[1].set_ylabel("Interaction (% of all-four recovery)")
    axes[1].set_title("Negative means overlapping benefit")
    axes[1].tick_params(axis="x", rotation=12)

    context_names = ["near", "far", "one prior", "multiple priors"]
    context_cells = ["near_positive", "far_positive", "one_predecessor_positive",
                     "multiple_predecessor_positive"]
    context = [100 * recovery[cell]["1111"] / pooled for cell in context_cells]
    axes[2].bar(context_names, context, color=["#9D755D", "#59A14F", "#EDC948", "#AF7AA1"])
    axes[2].axhline(100, color="#555555", linewidth=1, linestyle="--")
    axes[2].set_ylabel("All-four recovery (% of pooled positives)")
    axes[2].set_title("Context changes effect size")
    axes[2].tick_params(axis="x", rotation=12)

    fig.suptitle("Equality-term subset factorial on 192 natural-text documents", fontsize=14)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=180)
    print(OUTPUT)


if __name__ == "__main__":
    main()
