#!/usr/bin/env python3
"""Map later-module responses to the confirmed pending-opener L13H8 intervention."""

# BQGATE: EXPERIMENT

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch


os.environ["BQLIB_NO_MODEL"] = "1"
ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
POLY = ROOT.parent / "polynomial_causal"
for search_path in (ROOT, ROOT / "ops", POLY):
    if str(search_path) not in sys.path:
        sys.path.insert(0, str(search_path))
import bilin18_observed_model_facade as facade  # noqa: E402
import pending_opener_common_site_rung538 as core  # noqa: E402

ROWS = ROOT / "pending_opener_three_value_fresh_rows_rung545.json"
RECEIPT = ROOT / "pending_opener_three_value_fresh_rows_rung545_receipt.json"
CONFIRMATION = ROOT / "pending_opener_three_value_confirmation_rung546_results.json"
AUDIT = ROOT / "pending_opener_three_value_confirmation_rung548_audit.json"
PREREG = POLY / "PENDING_OPENER_DOWNSTREAM_RESPONSE_ATLAS_RUNG549_PREREGISTRATION.md"
OUT = ROOT / "pending_opener_downstream_response_atlas_rung549_results.json"
BUNDLE = ROOT / "pending_opener_downstream_response_atlas_rung549_vectors.pt"
HASHES = {
    ROWS: "07b64d2e48a6ca67685c81d3475a064daba612d6fe7ff233efd5b6c157b940a9",
    RECEIPT: "a6b3e7468f510277b247cb78148b619625ecdde07f9ba264e5358f7bb5138609",
    CONFIRMATION: "209b9bfcc20bff13bb37d822137003d6878506e66b0d9321ba0a0f7e9d8f2c5c",
    AUDIT: "25acb35355f457163c1ed1183aeb55aea0c08a224992d688250ba5e272564875",
    PREREG: "90aab91025476e364608efc3f012e467c5284f3ba677003ed2daf1b6daf6bdeb",
}
SPLITS = ("FIT", "SELECT")
TARGETS = (
    "direct_three_value_type_substitution",
    "completed_then_reopened_three_value_order",
)
CONTROLS = (
    "pending_type_preserved_surface_rewrite",
    "pending_type_preserved_distance_extension",
    "pending_type_preserved_nonopener_punctuation",
)
CANDIDATES = ("mlp13_write",) + tuple(
    name
    for layer in range(14, 18)
    for name in tuple(f"attn{layer}h{head}_output" for head in range(9)) + (f"mlp{layer}_write",)
)
CLOSERS = (1, 8, 60)
BATCH = 8
EXPECTED_ROWS = 540
EXPECTED_FORWARDS = math.ceil(EXPECTED_ROWS / BATCH) * 3
FIT_ACCURACY_BAR = 0.50
FIT_ANTIPODAL_BAR = 0.30
FIT_CONTROL_COSINE_BAR = 0.40
SELECT_ACCURACY_BAR = 0.50
SELECT_CONTROL_COSINE_BAR = 0.35
SELECT_RESPONSE_RATIO_BAR = 0.05


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows() -> list[dict]:
    for path, expected in HASHES.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen input mismatch: {path}")
    confirmation = json.loads(CONFIRMATION.read_text())
    audit = json.loads(AUDIT.read_text())
    if confirmation["all_gates_pass"] is not True or audit["all_gates_held"] is not True:
        raise RuntimeError("R546/R548 did not authorize the downstream-response atlas")
    rows = [row for row in json.loads(ROWS.read_text())["rows"] if row["split"] in SPLITS]
    if len(rows) != EXPECTED_ROWS or len({row["row_id"] for row in rows}) != EXPECTED_ROWS:
        raise RuntimeError("R545 FIT/SELECT identity changed")
    return rows


def pad(rows: list[dict]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    length = max(max(len(row["base_ids"]), len(row["donor_ids"])) for row in rows)
    base = torch.full((len(rows), length), 50256, dtype=torch.long, device="cuda")
    donor = base.clone()
    base_finals, donor_finals = [], []
    for index, row in enumerate(rows):
        base[index, :len(row["base_ids"])] = torch.tensor(row["base_ids"], device="cuda")
        donor[index, :len(row["donor_ids"])] = torch.tensor(row["donor_ids"], device="cuda")
        base_finals.append(len(row["base_ids"]) - 1)
        donor_finals.append(len(row["donor_ids"]) - 1)
    return base, donor, torch.tensor(base_finals, device="cuda"), torch.tensor(donor_finals, device="cuda")


def _pick(value: torch.Tensor, finals: torch.Tensor) -> torch.Tensor:
    return value[torch.arange(value.shape[0], device=value.device), finals]


@torch.no_grad()
def forward_capture(
    model,
    tokens: torch.Tensor,
    finals: torch.Tensor,
    l13h8_replacement: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Run the real model, optionally patch L13H8, and return all frozen later component writes."""
    captured: dict[str, torch.Tensor] = {}
    handles = []

    def l13_hook(_module, args):
        value = args[0]
        captured["l13h8_source"] = _pick(value, finals)[:, 8 * 128:9 * 128].detach().clone()
        if l13h8_replacement is None:
            return None
        changed = value.clone()
        changed[torch.arange(value.shape[0], device=value.device), finals, 8 * 128:9 * 128] = (
            l13h8_replacement.to(value.dtype)
        )
        return (changed,) + tuple(args[1:])

    def mlp_hook(name: str):
        def hook(_module, _args, output):
            captured[name] = _pick(output, finals).detach().clone()
        return hook

    def attn_hook(layer: int):
        def hook(_module, args):
            heads = _pick(args[0], finals).reshape(-1, 9, 128)
            for head in range(9):
                captured[f"attn{layer}h{head}_output"] = heads[:, head].detach().clone()
            return None
        return hook

    try:
        handles.append(model.transformer.h[13].attn.c_proj.register_forward_pre_hook(l13_hook))
        handles.append(model.transformer.h[13].mlp.Down.register_forward_hook(mlp_hook("mlp13_write")))
        for layer in range(14, 18):
            handles.append(model.transformer.h[layer].attn.c_proj.register_forward_pre_hook(attn_hook(layer)))
            handles.append(model.transformer.h[layer].mlp.Down.register_forward_hook(mlp_hook(f"mlp{layer}_write")))
        logits = core.forward_native(model, tokens)
    finally:
        for handle in handles:
            handle.remove()
    expected = set(CANDIDATES) | {"l13h8_source"}
    if set(captured) != expected:
        raise RuntimeError(f"capture mismatch: missing={expected-set(captured)}, extra={set(captured)-expected}")
    source = captured.pop("l13h8_source")
    return source, {name: captured[name].float().cpu() for name in CANDIDATES}


def empty_buffers() -> dict:
    return {
        site: {
            split: {
                family: {
                    direction: {"patch": [], "natural": [], "transition": [], "row_id": []}
                    for direction in ("base_to_donor", "donor_to_base")
                }
                for family in TARGETS + CONTROLS
            }
            for split in SPLITS
        }
        for site in CANDIDATES
    }


def append_vector(cell: dict, patch: torch.Tensor, natural: torch.Tensor, transition: str, row_id: str) -> None:
    cell["patch"].append(patch)
    cell["natural"].append(natural)
    cell["transition"].append(transition)
    cell["row_id"].append(row_id)


def finalize_buffers(buffers: dict) -> dict:
    for site in CANDIDATES:
        for split in SPLITS:
            for family in TARGETS + CONTROLS:
                for direction in ("base_to_donor", "donor_to_base"):
                    cell = buffers[site][split][family][direction]
                    cell["patch"] = torch.stack(cell["patch"])
                    cell["natural"] = torch.stack(cell["natural"])
    return buffers


def gather(buffers: dict, site: str, split: str, family: str) -> dict:
    cells = [buffers[site][split][family][direction] for direction in ("base_to_donor", "donor_to_base")]
    return {
        "patch": torch.cat([cell["patch"] for cell in cells]),
        "natural": torch.cat([cell["natural"] for cell in cells]),
        "transition": sum((cell["transition"] for cell in cells), []),
        "row_id": sum((cell["row_id"] for cell in cells), []),
    }


def make_templates(records: dict) -> tuple[list[str], torch.Tensor]:
    labels = sorted(set(records["transition"]))
    if labels != ["1->60", "1->8", "60->1", "60->8", "8->1", "8->60"]:
        raise RuntimeError(f"ordered-transition coverage mismatch: {labels}")
    return labels, torch.stack([
        records["patch"][[label == wanted for label in records["transition"]]].mean(0)
        for wanted in labels
    ])


def cosine_matrix(rows: torch.Tensor, templates: torch.Tensor) -> torch.Tensor:
    rows = rows / rows.norm(dim=1, keepdim=True).clamp_min(1e-12)
    templates = templates / templates.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return rows @ templates.T


def accuracy(records: dict, labels: list[str], templates: torch.Tensor) -> float:
    predicted = cosine_matrix(records["patch"], templates).argmax(1)
    truth = torch.tensor([labels.index(label) for label in records["transition"]])
    return float((predicted == truth).float().mean())


def antipodal(labels: list[str], templates: torch.Tensor) -> float:
    values = []
    for index, label in enumerate(labels):
        left, right = label.split("->")
        reverse = labels.index(f"{right}->{left}")
        values.append(float(torch.nn.functional.cosine_similarity(
            templates[index:index+1], -templates[reverse:reverse+1]
        )))
    return float(np.median(values))


def control_template_cosine(buffers: dict, site: str, split: str, templates: torch.Tensor) -> float:
    rows = torch.cat([gather(buffers, site, split, family)["patch"] for family in CONTROLS])
    return float(cosine_matrix(rows, templates).abs().max(1).values.median())


def response_ratio(buffers: dict, site: str, split: str) -> float:
    records = [gather(buffers, site, split, family) for family in TARGETS]
    patch = torch.cat([record["patch"] for record in records])
    natural = torch.cat([record["natural"] for record in records])
    return float((patch.norm(dim=1) / natural.norm(dim=1).clamp_min(1e-12)).median())


def readout_vectors(model, site: str) -> torch.Tensor:
    unembedding = model.lm_head.weight.detach().float().cpu()
    contrasts = torch.stack([
        unembedding[left] - unembedding[right]
        for i, left in enumerate(CLOSERS) for right in CLOSERS[i+1:]
    ])
    if site.startswith("mlp"):
        return contrasts
    layer = int(site.removeprefix("attn").split("h")[0])
    head = int(site.split("h")[1].split("_")[0])
    weight = model.transformer.h[layer].attn.c_proj.weight.detach().float().cpu()
    return contrasts @ weight[:, head * 128:(head + 1) * 128]


def score_candidate(buffers: dict, model, site: str) -> dict:
    direct_fit = gather(buffers, site, "FIT", TARGETS[0])
    order_fit = gather(buffers, site, "FIT", TARGETS[1])
    direct_labels, direct_templates = make_templates(direct_fit)
    order_labels, order_templates = make_templates(order_fit)
    if direct_labels != order_labels:
        raise RuntimeError("target families do not share transition labels")
    pooled_fit = {
        "patch": torch.cat((direct_fit["patch"], order_fit["patch"])),
        "transition": direct_fit["transition"] + order_fit["transition"],
    }
    labels, pooled_templates = make_templates(pooled_fit)
    fit_a = accuracy(order_fit, direct_labels, direct_templates)
    fit_b = accuracy(direct_fit, order_labels, order_templates)
    fit_control = control_template_cosine(buffers, site, "FIT", pooled_templates)
    anti = antipodal(labels, pooled_templates)
    eligible = bool(
        min(fit_a, fit_b) >= FIT_ACCURACY_BAR
        and anti >= FIT_ANTIPODAL_BAR
        and fit_control <= FIT_CONTROL_COSINE_BAR
    )

    direct_select = gather(buffers, site, "SELECT", TARGETS[0])
    order_select = gather(buffers, site, "SELECT", TARGETS[1])
    select_a = accuracy(order_select, direct_labels, direct_templates)
    select_b = accuracy(direct_select, order_labels, order_templates)
    select_control = control_template_cosine(buffers, site, "SELECT", pooled_templates)
    select_ratio = response_ratio(buffers, site, "SELECT")
    readouts = readout_vectors(model, site)
    readout_alignment = float(cosine_matrix(pooled_templates, readouts).abs().max(1).values.median())
    return {
        "dimension": int(pooled_templates.shape[1]),
        "fit": {
            "direct_templates_classify_order_accuracy": fit_a,
            "order_templates_classify_direct_accuracy": fit_b,
            "median_antipodal_transition_cosine": anti,
            "control_median_max_absolute_template_cosine": fit_control,
            "eligible": eligible,
            "selection_score": min(fit_a, fit_b) - fit_control,
        },
        "select": {
            "direct_templates_classify_order_accuracy": select_a,
            "order_templates_classify_direct_accuracy": select_b,
            "control_median_max_absolute_template_cosine": select_control,
            "median_patch_to_natural_response_norm_ratio": select_ratio,
        },
        "readout_alignment_diagnostic": {
            "median_template_max_absolute_cosine_with_closer_contrasts": readout_alignment,
            "used_for_selection": False,
        },
    }


def main() -> None:
    started = time.time()
    rows = load_rows()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print(json.dumps({
            "status": "dryrun_passed", "rows": len(rows), "candidates": list(CANDIDATES),
            "expected_forwards": EXPECTED_FORWARDS, "evaluated_splits": list(SPLITS),
            "final_or_ood_opened": False,
        }, indent=2))
        return

    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.float32, verify_weights_sha256=True)
    buffers = empty_buffers()
    calls, minimum_edit_rms = 0, float("inf")
    for start in range(0, len(rows), BATCH):
        chunk = rows[start:start+BATCH]
        base, donor, base_finals, donor_finals = pad(chunk)
        both = torch.cat((base, donor))
        finals = torch.cat((base_finals, donor_finals))
        native_source, native = forward_capture(model, both, finals)
        calls += 1
        base_source, donor_source = native_source.chunk(2)
        minimum_edit_rms = min(minimum_edit_rms, float(
            (donor_source - base_source).square().mean(-1).sqrt().min()
        ))
        _, base_patched = forward_capture(model, base, base_finals, donor_source)
        _, donor_patched = forward_capture(model, donor, donor_finals, base_source)
        calls += 2
        for site in CANDIDATES:
            base_native, donor_native = native[site].chunk(2)
            for index, row in enumerate(chunk):
                forward_transition = f"{row['base_answer_id']}->{row['donor_answer_id']}"
                reverse_transition = f"{row['donor_answer_id']}->{row['base_answer_id']}"
                family, split = row["family_id"], row["split"]
                append_vector(
                    buffers[site][split][family]["base_to_donor"],
                    base_patched[site][index] - base_native[index],
                    donor_native[index] - base_native[index],
                    forward_transition, row["row_id"],
                )
                append_vector(
                    buffers[site][split][family]["donor_to_base"],
                    donor_patched[site][index] - donor_native[index],
                    base_native[index] - donor_native[index],
                    reverse_transition, row["row_id"],
                )
        del base, donor, both, native_source, native, base_patched, donor_patched

    buffers = finalize_buffers(buffers)
    metrics = {site: score_candidate(buffers, model, site) for site in CANDIDATES}
    eligible = [site for site in CANDIDATES if metrics[site]["fit"]["eligible"]]
    selected = max(eligible, key=lambda site: (metrics[site]["fit"]["selection_score"], -CANDIDATES.index(site))) \
        if eligible else None
    selected_pass = False
    if selected is not None:
        report = metrics[selected]["select"]
        selected_pass = bool(
            min(report["direct_templates_classify_order_accuracy"],
                report["order_templates_classify_direct_accuracy"]) >= SELECT_ACCURACY_BAR
            and report["control_median_max_absolute_template_cosine"] <= SELECT_CONTROL_COSINE_BAR
            and report["median_patch_to_natural_response_norm_ratio"] >= SELECT_RESPONSE_RATIO_BAR
        )

    bundle = {
        "schema": "pending_opener_downstream_response_atlas_rung549_vectors_v1",
        "candidate_order": list(CANDIDATES),
        "evaluated_splits": list(SPLITS),
        "buffers": buffers,
    }
    torch.save(bundle, BUNDLE)
    result = {
        "rung": 549,
        "stage": "pending_opener_downstream_response_atlas",
        "pred_a_exact_instrument": bool(
            calls == EXPECTED_FORWARDS and minimum_edit_rms > 0
            and checkpoint.weights_sha256 == facade.WEIGHTS_SHA256
        ),
        "pred_b_fit_selects_candidate": selected is not None,
        "pred_c_selected_candidate_validates": selected_pass,
        "strong_null": bool(selected is None or not selected_pass),
        "candidate_order": list(CANDIDATES),
        "fit_eligible_candidates": eligible,
        "selected_candidate": selected,
        "metrics": metrics,
        "bars": {
            "fit_cross_family_accuracy_min": FIT_ACCURACY_BAR,
            "fit_antipodal_cosine_min": FIT_ANTIPODAL_BAR,
            "fit_control_template_cosine_max": FIT_CONTROL_COSINE_BAR,
            "select_cross_family_accuracy_min": SELECT_ACCURACY_BAR,
            "select_control_template_cosine_max": SELECT_CONTROL_COSINE_BAR,
            "select_response_norm_ratio_min": SELECT_RESPONSE_RATIO_BAR,
        },
        "model_forwards": calls,
        "model_backwards": 0,
        "model_weights_updated": False,
        "minimum_l13h8_source_target_rms": minimum_edit_rms,
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "input_sha256": {str(path): sha256(path) for path in HASHES},
        "bundle_path": str(BUNDLE),
        "bundle_sha256": sha256(BUNDLE),
        "evaluated_splits": list(SPLITS),
        "forbidden_splits_opened": [],
        "elapsed_seconds": time.time() - started,
        "next_step": (
            "preregister_multi_output_selective_interchange_using_selected_reader"
            if selected_pass else
            "retain_endpoint_plus_invariance_objective_without_claiming_a_downstream_reader"
        ),
    }
    OUT.write_text(json.dumps(result, indent=1) + "\n")
    summary = {key: value for key, value in result.items() if key.startswith("pred_")}
    summary.update({
        "strong_null": result["strong_null"], "fit_eligible_candidates": eligible,
        "selected_candidate": selected, "selected_metrics": metrics.get(selected),
        "model_forwards": calls, "next_step": result["next_step"],
    })
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
