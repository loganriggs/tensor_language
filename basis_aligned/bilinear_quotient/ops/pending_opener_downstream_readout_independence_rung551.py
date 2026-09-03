#!/usr/bin/env python3
"""CPU-only pre-outcome guard for readout redundancy in the R549 selected response."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "pending_opener_downstream_response_atlas_rung549_results.json"
BUNDLE = ROOT / "pending_opener_downstream_response_atlas_rung549_vectors.pt"
OUT = ROOT / "pending_opener_downstream_readout_independence_rung551_results.json"
WEIGHTS = Path(
    "/workspace/.hf_home/hub/models--Elriggs--gpt2-bilinear-sqrd-attn-18l-9h-1152embd/"
    "snapshots/ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240/pytorch_model.bin"
)
WEIGHTS_SHA256 = "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"
CLOSERS = (1, 8, 60)
READOUT_SPAN_FRACTION_MAX = 0.50
TOL = 2e-7


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gather(buffers: dict, site: str, family: str) -> dict:
    cells = [buffers[site]["FIT"][family][direction] for direction in ("base_to_donor", "donor_to_base")]
    return {
        "patch": torch.cat([cell["patch"] for cell in cells]),
        "transition": sum((cell["transition"] for cell in cells), []),
    }


def pooled_templates(buffers: dict, site: str) -> tuple[list[str], torch.Tensor]:
    records = [
        gather(buffers, site, "direct_three_value_type_substitution"),
        gather(buffers, site, "completed_then_reopened_three_value_order"),
    ]
    patch = torch.cat([record["patch"] for record in records])
    transition = records[0]["transition"] + records[1]["transition"]
    labels = sorted(set(transition))
    templates = torch.stack([
        patch[[label == wanted for label in transition]].mean(0) for wanted in labels
    ])
    return labels, templates


def candidate_readouts(state: dict, site: str) -> torch.Tensor:
    unembedding = state["lm_head.weight"].float()
    contrasts = torch.stack([
        unembedding[left] - unembedding[right]
        for index, left in enumerate(CLOSERS) for right in CLOSERS[index + 1:]
    ])
    if site.startswith("mlp"):
        return contrasts
    layer = int(site.removeprefix("attn").split("h")[0])
    head = int(site.split("h")[1].split("_")[0])
    weight = state[f"transformer.h.{layer}.attn.c_proj.weight"].float()
    return contrasts @ weight[:, head * 128:(head + 1) * 128]


def normalized_cosines(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return (
        left / left.norm(dim=1, keepdim=True).clamp_min(1e-12)
    ) @ (
        right / right.norm(dim=1, keepdim=True).clamp_min(1e-12)
    ).T


def main() -> None:
    if not RESULT.is_file() or not BUNDLE.is_file():
        raise RuntimeError("R549 result and vector bundle must exist before R551")
    result = json.loads(RESULT.read_text())
    if result["bundle_sha256"] != sha256(BUNDLE):
        raise RuntimeError("R549 result does not bind its vector bundle")
    bundle = torch.load(BUNDLE, map_location="cpu", weights_only=False)
    selected = result["selected_candidate"]
    if selected is not None and selected not in bundle["candidate_order"]:
        raise RuntimeError("R549 selected candidate is absent from its frozen candidate order")
    if sha256(WEIGHTS) != WEIGHTS_SHA256:
        raise RuntimeError("checkpoint weights changed")

    if selected is None:
        labels, fractions, pairwise = [], [], None
        independent = False
    else:
        labels, templates = pooled_templates(bundle["buffers"], selected)
        state = torch.load(WEIGHTS, map_location="cpu", weights_only=True, mmap=True)
        readouts = candidate_readouts(state, selected)
        del state
        _, singular_values, vh = torch.linalg.svd(readouts, full_matrices=False)
        numerical_rank = int((singular_values > singular_values[0] * 1e-6).sum())
        basis = vh[:numerical_rank].T
        fractions = (
            (templates @ basis).norm(dim=1) / templates.norm(dim=1).clamp_min(1e-12)
        ).tolist()
        pairwise = float(normalized_cosines(templates, readouts).abs().max(1).values.median())
        reported = result["metrics"][selected]["readout_alignment_diagnostic"][
            "median_template_max_absolute_cosine_with_closer_contrasts"
        ]
        if not np.isclose(pairwise, reported, rtol=0.0, atol=TOL):
            raise AssertionError(f"R549 readout diagnostic mismatch: {reported} versus {pairwise}")
        independent = bool(
            result["pred_c_selected_candidate_validates"]
            and float(np.median(fractions)) <= READOUT_SPAN_FRACTION_MAX
        )

    output = {
        "rung": 551,
        "stage": "pending_opener_downstream_readout_independence_guard",
        "selected_candidate": selected,
        "r549_candidate_validates": result["pred_c_selected_candidate_validates"],
        "transition_labels": labels,
        "readout_span_fraction_by_transition": dict(zip(labels, fractions)),
        "median_readout_span_fraction": float(np.median(fractions)) if fractions else None,
        "maximum_readout_span_fraction": float(np.max(fractions)) if fractions else None,
        "recomputed_median_max_pairwise_readout_cosine": pairwise,
        "bar_median_readout_span_fraction_max": READOUT_SPAN_FRACTION_MAX,
        "pred_a_result_and_bundle_bound": True,
        "pred_b_r549_pairwise_readout_diagnostic_reproduced": bool(selected is None or pairwise is not None),
        "pred_c_distinct_downstream_target": independent,
        "strong_null": not independent,
        "model_forwards": 0,
        "model_backwards": 0,
        "model_weights_updated": False,
        "checkpoint_weights_sha256": WEIGHTS_SHA256,
        "input_sha256": {str(RESULT): sha256(RESULT), str(BUNDLE): sha256(BUNDLE)},
        "evaluated_splits": ["FIT"],
        "forbidden_splits_opened": [],
        "decision": (
            "The R549 response is sufficiently outside the direct closer-readout span to serve as an independent "
            "second target for a multi-output interchange."
            if independent else
            "Do not use the R549 response as an independent second target; no candidate validated or its response "
            "is too concentrated in the direct closer-readout span."
        ),
    }
    OUT.write_text(json.dumps(output, indent=1) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
