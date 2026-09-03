#!/usr/bin/env python3
"""CPU-only independent audit of R549's saved later-module response vectors."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "pending_opener_downstream_response_atlas_rung549_results.json"
BUNDLE = ROOT / "pending_opener_downstream_response_atlas_rung549_vectors.pt"
ROWS = ROOT / "pending_opener_three_value_fresh_rows_rung545.json"
OUT = ROOT / "pending_opener_downstream_response_atlas_rung550_audit.json"
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
FIT_ACCURACY_BAR = 0.50
FIT_ANTIPODAL_BAR = 0.30
FIT_CONTROL_COSINE_BAR = 0.40
SELECT_ACCURACY_BAR = 0.50
SELECT_CONTROL_COSINE_BAR = 0.35
SELECT_RESPONSE_RATIO_BAR = 0.05
TOL = 2e-7


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(actual: float, expected: float, label: str) -> None:
    if not np.isclose(actual, expected, rtol=0.0, atol=TOL):
        raise AssertionError(f"{label}: reported={actual!r}, recomputed={expected!r}")


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
    templates = torch.stack([
        records["patch"][[label == wanted for label in records["transition"]]].mean(0)
        for wanted in labels
    ])
    return labels, templates


def cosine_matrix(rows: torch.Tensor, templates: torch.Tensor) -> torch.Tensor:
    return (
        rows / rows.norm(dim=1, keepdim=True).clamp_min(1e-12)
    ) @ (
        templates / templates.norm(dim=1, keepdim=True).clamp_min(1e-12)
    ).T


def accuracy(records: dict, labels: list[str], templates: torch.Tensor) -> float:
    predictions = cosine_matrix(records["patch"], templates).argmax(1)
    truth = torch.tensor([labels.index(label) for label in records["transition"]])
    return float((predictions == truth).float().mean())


def antipodal(labels: list[str], templates: torch.Tensor) -> float:
    values = []
    for index, label in enumerate(labels):
        left, right = label.split("->")
        reverse = labels.index(f"{right}->{left}")
        values.append(float(torch.nn.functional.cosine_similarity(
            templates[index:index+1], -templates[reverse:reverse+1]
        )))
    return float(np.median(values))


def control_cosine(buffers: dict, site: str, split: str, templates: torch.Tensor) -> float:
    rows = torch.cat([gather(buffers, site, split, family)["patch"] for family in CONTROLS])
    return float(cosine_matrix(rows, templates).abs().max(1).values.median())


def response_ratio(buffers: dict, site: str) -> float:
    records = [gather(buffers, site, "SELECT", family) for family in TARGETS]
    patch = torch.cat([record["patch"] for record in records])
    natural = torch.cat([record["natural"] for record in records])
    return float((patch.norm(dim=1) / natural.norm(dim=1).clamp_min(1e-12)).median())


def recompute(buffers: dict, site: str) -> dict:
    direct_fit = gather(buffers, site, "FIT", TARGETS[0])
    order_fit = gather(buffers, site, "FIT", TARGETS[1])
    direct_labels, direct_templates = make_templates(direct_fit)
    order_labels, order_templates = make_templates(order_fit)
    if direct_labels != order_labels or len(direct_labels) != 6:
        raise AssertionError(f"transition template mismatch for {site}")
    pooled = {
        "patch": torch.cat((direct_fit["patch"], order_fit["patch"])),
        "transition": direct_fit["transition"] + order_fit["transition"],
    }
    labels, templates = make_templates(pooled)
    direct_select = gather(buffers, site, "SELECT", TARGETS[0])
    order_select = gather(buffers, site, "SELECT", TARGETS[1])
    fit_a = accuracy(order_fit, direct_labels, direct_templates)
    fit_b = accuracy(direct_fit, order_labels, order_templates)
    fit_control = control_cosine(buffers, site, "FIT", templates)
    anti = antipodal(labels, templates)
    return {
        "dimension": int(templates.shape[1]),
        "fit": {
            "direct_templates_classify_order_accuracy": fit_a,
            "order_templates_classify_direct_accuracy": fit_b,
            "median_antipodal_transition_cosine": anti,
            "control_median_max_absolute_template_cosine": fit_control,
            "eligible": bool(
                min(fit_a, fit_b) >= FIT_ACCURACY_BAR
                and anti >= FIT_ANTIPODAL_BAR
                and fit_control <= FIT_CONTROL_COSINE_BAR
            ),
            "selection_score": min(fit_a, fit_b) - fit_control,
        },
        "select": {
            "direct_templates_classify_order_accuracy": accuracy(order_select, direct_labels, direct_templates),
            "order_templates_classify_direct_accuracy": accuracy(direct_select, order_labels, order_templates),
            "control_median_max_absolute_template_cosine": control_cosine(
                buffers, site, "SELECT", templates),
            "median_patch_to_natural_response_norm_ratio": response_ratio(buffers, site),
        },
    }


def verify_rows(buffers: dict) -> None:
    authority = json.loads(ROWS.read_text())["rows"]
    for site in CANDIDATES:
        for split in SPLITS:
            for family in TARGETS + CONTROLS:
                expected = {row["row_id"] for row in authority
                            if row["split"] == split and row["family_id"] == family}
                for direction in ("base_to_donor", "donor_to_base"):
                    cell = buffers[site][split][family][direction]
                    ids = cell["row_id"]
                    if len(ids) != len(set(ids)) or set(ids) != expected:
                        raise AssertionError(f"row identity mismatch: {site}/{split}/{family}/{direction}")
                    if cell["patch"].shape != cell["natural"].shape or cell["patch"].shape[0] != len(ids):
                        raise AssertionError(f"tensor shape mismatch: {site}/{split}/{family}/{direction}")
                    if not torch.isfinite(cell["patch"]).all() or not torch.isfinite(cell["natural"]).all():
                        raise AssertionError(f"non-finite response: {site}/{split}/{family}/{direction}")


def main() -> None:
    result = json.loads(RESULT.read_text())
    bundle = torch.load(BUNDLE, map_location="cpu", weights_only=False)
    if result["bundle_sha256"] != sha256(BUNDLE):
        raise AssertionError("result does not bind the saved response-vector bundle")
    if bundle["candidate_order"] != list(CANDIDATES) or result["candidate_order"] != list(CANDIDATES):
        raise AssertionError("candidate order changed")
    if bundle["evaluated_splits"] != list(SPLITS) or result["evaluated_splits"] != list(SPLITS):
        raise AssertionError("split authority changed")
    if result["forbidden_splits_opened"] != [] or result["model_forwards"] != 204:
        raise AssertionError("call budget or split authority mismatch")
    if result["model_backwards"] != 0 or result["model_weights_updated"] is not False:
        raise AssertionError("unexpected optimization in the response atlas")
    if result["input_sha256"][str(ROWS)] != sha256(ROWS):
        raise AssertionError("result does not bind R545 rows")

    buffers = bundle["buffers"]
    verify_rows(buffers)
    rebuilt = {site: recompute(buffers, site) for site in CANDIDATES}
    for site in CANDIDATES:
        saved = result["metrics"][site]
        for split in ("fit", "select"):
            for key, value in rebuilt[site][split].items():
                if isinstance(value, bool):
                    if saved[split][key] is not value:
                        raise AssertionError(f"verdict mismatch: {site}/{split}/{key}")
                else:
                    close(saved[split][key], value, f"{site}/{split}/{key}")
        if saved["dimension"] != rebuilt[site]["dimension"]:
            raise AssertionError(f"dimension mismatch for {site}")
        if saved["readout_alignment_diagnostic"]["used_for_selection"] is not False:
            raise AssertionError("readout alignment entered candidate selection")

    eligible = [site for site in CANDIDATES if rebuilt[site]["fit"]["eligible"]]
    selected = max(
        eligible,
        key=lambda site: (rebuilt[site]["fit"]["selection_score"], -CANDIDATES.index(site)),
    ) if eligible else None
    if result["fit_eligible_candidates"] != eligible or result["selected_candidate"] != selected:
        raise AssertionError("FIT-only candidate selection mismatch")
    selected_pass = False
    if selected is not None:
        report = rebuilt[selected]["select"]
        selected_pass = bool(
            min(report["direct_templates_classify_order_accuracy"],
                report["order_templates_classify_direct_accuracy"]) >= SELECT_ACCURACY_BAR
            and report["control_median_max_absolute_template_cosine"] <= SELECT_CONTROL_COSINE_BAR
            and report["median_patch_to_natural_response_norm_ratio"] >= SELECT_RESPONSE_RATIO_BAR
        )
    if result["pred_b_fit_selects_candidate"] is not bool(selected is not None):
        raise AssertionError("FIT-selection predicate mismatch")
    if result["pred_c_selected_candidate_validates"] is not selected_pass:
        raise AssertionError("SELECT-validation predicate mismatch")
    if result["strong_null"] is not bool(selected is None or not selected_pass):
        raise AssertionError("strong-null predicate mismatch")

    ranked = sorted(CANDIDATES, key=lambda site: rebuilt[site]["fit"]["selection_score"], reverse=True)
    audit = {
        "rung": 550,
        "audited_rung": 549,
        "status": "terminal_audit_complete",
        "result_sha256": sha256(RESULT),
        "bundle_sha256": sha256(BUNDLE),
        "rows_sha256": sha256(ROWS),
        "complete_row_identity_and_tensor_shape_check": True,
        "independent_fit_and_select_recomputation": True,
        "selection_depends_only_on_fit": True,
        "fit_eligible_candidates": eligible,
        "selected_candidate": selected,
        "selected_candidate_validates": selected_pass,
        "selected_metrics": rebuilt.get(selected),
        "top_five_fit_candidates": [
            {"candidate": site, **rebuilt[site]["fit"]} for site in ranked[:5]
        ],
        "readout_alignment_is_diagnostic_only": True,
        "decision": (
            "Use the validated later-module response as a second frozen output for the next selective interchange."
            if selected_pass else
            "No later-module response passed the frozen FIT-selection and SELECT-validation rule; do not claim an "
            "independent downstream reader from this atlas."
        ),
        "final_test_or_ood_opened": False,
    }
    OUT.write_text(json.dumps(audit, indent=1) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
