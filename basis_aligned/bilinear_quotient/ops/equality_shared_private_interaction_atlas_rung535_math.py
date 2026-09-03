#!/usr/bin/env python3
"""Rung 535 CPU atlas of the exact S-by-R causal interaction saved by rung 534."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import torch


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
for search_path in (ROOT, ROOT / "ops", ROOT.parent / "polynomial_causal"):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))

import equality_product_shared_private_rung534 as rung534


RESULT = ROOT / "equality_product_shared_private_rung534_results.json"
BUNDLE = ROOT / "equality_product_shared_private_rung534_bundle.pt"
OUT = ROOT / "equality_shared_private_interaction_atlas_rung535_math.json"
EXPECTED = {
    RESULT: "8804dca2cbd0203a6ef9517a15ec7a4186ed5e69ec8c284b854967c8e13197a7",
    BUNDLE: "77ca551a19004abade5ec5dcc79023a01f3d9c5d97ca693c012ca74f512cef80",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def vector_summary(interaction, native, marginal, private):
    def rms(x):
        return float(x.square().mean().sqrt())

    native_rms = rms(native)
    marginal_rms = rms(marginal)
    interaction_rms = rms(interaction)
    return {
        "documents": int(interaction.numel()),
        "interaction_mean_nat": float(interaction.mean()),
        "interaction_rms_nat": interaction_rms,
        "native_effect_mean_nat": float(native.mean()),
        "native_effect_rms_nat": native_rms,
        "private_standalone_mean_nat": float(private.mean()),
        "private_marginal_mean_nat": float(marginal.mean()),
        "interaction_over_native_rms": interaction_rms / max(native_rms, 1e-30),
        "interaction_over_private_marginal_rms": interaction_rms / max(marginal_rms, 1e-30),
        "exact_factorial_closure_max_abs": float(
            (marginal - private - interaction).abs().max()),
    }


def build_atlas(collection):
    atlas = {}
    for role in rung534.ROLES:
        counts = collection["document_counts"][role]
        ce = collection["document_sums"][role] / counts[None, None].clamp_min(1)
        for background_index, background in enumerate(rung534.BACKGROUNDS):
            for half in range(2):
                start = half * rung534.DOCUMENT_SPLIT
                stop = (half + 1) * rung534.DOCUMENT_SPLIT
                for cell_index, cell in enumerate(rung534.TASK_CELLS):
                    live = counts[cell_index, start:stop] > 0
                    values = ce[background_index, :, cell_index, start:stop][:, live]
                    absent = values[rung534.ARMS.index("absent")]
                    native = absent - values[rung534.ARMS.index("native")]
                    shared = absent - values[rung534.ARMS.index("shared")]
                    private = absent - values[rung534.ARMS.index("private")]
                    marginal = native - shared
                    interaction = native - shared - private
                    key = f"{role}/{background}/half{half}/{cell}"
                    atlas[key] = vector_summary(
                        interaction, native, marginal, private)
    return atlas


def summarize(atlas):
    summary = {}
    for role in rung534.ROLES:
        for background in rung534.BACKGROUNDS:
            for cell in rung534.TASK_CELLS:
                rows = [atlas[f"{role}/{background}/half{half}/{cell}"] for half in range(2)]
                signs = [0 if row["interaction_mean_nat"] == 0 else
                         (1 if row["interaction_mean_nat"] > 0 else -1) for row in rows]
                summary[f"{role}/{background}/{cell}"] = {
                    "same_mean_sign_across_halves": signs[0] == signs[1] and signs[0] != 0,
                    "mean_signs": signs,
                    "interaction_over_native_rms_by_half": [
                        row["interaction_over_native_rms"] for row in rows],
                    "interaction_over_private_marginal_rms_by_half": [
                        row["interaction_over_private_marginal_rms"] for row in rows],
                    "interaction_mean_nat_by_half": [row["interaction_mean_nat"] for row in rows],
                }
    return summary


def main():
    observed = {str(path): sha256(path) for path in EXPECTED}
    if any(observed[str(path)] != expected for path, expected in EXPECTED.items()):
        raise RuntimeError("rung534 input hashes changed")
    result = json.loads(RESULT.read_text())
    if result.get("interaction_only_correction_strong_null") is not True:
        raise RuntimeError("rung534 is not the registered interaction-only authority")
    bundle = torch.load(BUNDLE, map_location="cpu", weights_only=False)
    atlas = build_atlas(bundle["collection"])
    if max(row["exact_factorial_closure_max_abs"] for row in atlas.values()) != 0.0:
        raise RuntimeError("factorial interaction identity did not close exactly")
    payload = {
        "status": "cpu_atlas_complete",
        "rung": 535,
        "definition": "interaction = E_native - E_shared - E_private",
        "interpretation": "behavioral effect present only in the joint S-plus-R composition",
        "atlas": atlas,
        "half_summary": summarize(atlas),
        "input_hashes": observed,
        "model_loaded": False,
        "new_model_forwards": 0,
        "new_scientific_outcomes_opened": False,
    }
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": payload["status"],
        "cells": len(atlas),
        "same_sign_cells": sum(
            row["same_mean_sign_across_halves"] for row in payload["half_summary"].values()),
        "median_interaction_over_native_rms": float(torch.tensor([
            row["interaction_over_native_rms"] for row in atlas.values()]).median()),
        "maximum_factorial_closure_error": max(
            row["exact_factorial_closure_max_abs"] for row in atlas.values()),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
