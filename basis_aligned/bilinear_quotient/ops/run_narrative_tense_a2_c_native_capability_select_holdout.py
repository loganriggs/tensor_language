#!/usr/bin/env python3
"""Native-only A2/C selection, lexical holdout, and causal capability license."""

# BQGATE: EXPERIMENT pred_a_fit_package_selected pred_b_holdout_all_cells_pass pred_c_license_issued

from __future__ import annotations

from collections import defaultdict
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
from typing import Mapping, Sequence

import circuit_fast_screen_candidate_narrative_tense_a2_c_native_capability as authority
import circuit_fast_screen_managed_runner as managed
import native_capability_license as licensing
import run_task14_head11_3_subject_attractor_score_payload_factorial as model_helpers


ROOT = Path(__file__).resolve().parent.parent
PRIOR_ART = ROOT / "circuits/prior_art/narrative_tense_a2_c_native_capability_select_holdout_v1.json"
RESULT = ROOT / "circuits/fast_screens/narrative_tense_a2_c_native_capability_select_holdout_v1_result.json"
CAPABILITY_RESULT = ROOT / "circuits/fast_screens/narrative_tense_a2_c_native_capability_selected_package_v1_result.json"
LICENSE = ROOT / "circuits/fast_screens/narrative_tense_a2_c_native_capability_selected_package_v1_license.json"
PRIOR_ART_SHA256 = "8acbb3418bcd967a3c5891dcb65309f444ee8dd01dbd5828fc2110ff1fa65292"
AUTHORITY_FILE_SHA256 = "2eb632e88cd27d599aa745308a2973a2bfde29321e4f9ebb18a9ad0b8e46bf1b"
AUTHORITY_BANK_SHA256 = "ee7b04735f5040ee0fc3128d47b587bef1841af9baace6596b7db734c762f64d"
MINIMUM_ACCURACY = .875
PACKAGE_SHA256 = {
    "record_coordination+explicit_period": "5776b228a4350da3421883fef8f90102ebd00f69658dd7f7051105d749ebf2e3",
    "record_coordination+years_nowadays": "d122e5bc3d0fb0ce4a1384948e74035e2dd1471f1193a5179372411198a4a8fb",
    "record_coordination+back_then_right_now": "1f8b676ea7271e36b0703cbf32db97110f871e51f301ba76049fc272811c86d9",
    "while_observers+explicit_period": "6f5058b30a92d2d4ac83c4e8f001c03a569f4e52e337ad173ee66b9da0ed75c8",
    "while_observers+years_nowadays": "af5b3859abc2531356acbed70aa5909469a9a1058c23e88c304560ff4c192a1e",
    "while_observers+back_then_right_now": "8f1ff2fc9ed4c54792f554bda3c5fd90e5c577538e43c3782c41e4ad518deb84",
    "reported_frame+explicit_period": "b427429fe88c082743efbcdf30b5857763f92bb26e73330af30ef32585ea5758",
    "reported_frame+years_nowadays": "241c06a262aa8019bfece40658a9be853847119328e44212356faa1807b493eb",
    "reported_frame+back_then_right_now": "ee450098829b8061c96f4953848f80849eec278bca96f0d0615df22958403d3a",
}


class SelectorError(ValueError):
    """The frozen native-only selection or licensing contract was violated."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _capability_id(a2: str, c: str) -> str:
    return f"narrative_tense.native_capability.{a2}.{c}.v1"


def causal_candidate_id(a2: str, c: str) -> str:
    return f"narrative_tense.attn11_head3_carrier.{a2}.{c}.v1"


def _cell_id(row: Mapping[str, object], side: str) -> str:
    return "/".join((str(row["phase"]), str(row["family"]),
                     str(row["template_id"]), str(row["direction_id"]), side))


def compile_plan():
    if _sha256(PRIOR_ART) != PRIOR_ART_SHA256 \
            or _sha256(Path(authority.__file__)) != AUTHORITY_FILE_SHA256 \
            or authority.authority_sha256() != AUTHORITY_BANK_SHA256:
        raise SelectorError("frozen receipt or authority changed")
    rows = authority.build_rows()
    fit = [row for row in rows if row["phase"] == "FIT"]
    return {
        "schema": "narrative_tense_a2_c_native_capability_select_holdout_plan_v1",
        "candidate_id": authority.CAPABILITY_ID,
        "model_loaded": False, "gpu_accessed": False, "queue_touched": False,
        "native_only": True, "causal_interventions": 0,
        "prior_art_sha256": PRIOR_ART_SHA256,
        "authority_file_sha256": AUTHORITY_FILE_SHA256,
        "authority_bank_sha256": AUTHORITY_BANK_SHA256,
        "fit_paired_rows": len(fit), "fit_endpoint_evaluations": 128,
        "maximum_holdout_endpoint_evaluations": 64,
        "A2_candidates": list(authority.A2_TEMPLATE_ORDER),
        "C_candidates": list(authority.C_TEMPLATE_ORDER),
        "maximum_price": {"model_forwards": 2, "example_evaluations": 192,
                          "backwards": 0, "parameter_updates": 0},
        "selection_rule": ["all fixed A1/P FIT cells pass",
                           "candidate eligibility requires every FIT cell pass",
                           "maximum worst-cell accuracy", "maximum worst-cell mean margin",
                           "minimum worst-cell mean full-vocabulary CE", "fixed order"],
        "holdout_policy": "Only the globally selected A2+C package may open HOLDOUT.",
    }


def _pad(rows, torch, device):
    endpoints = [(row, side) for row in rows for side in ("base", "donor")]
    length = max(len(row[f"{side}_ids"]) for row, side in endpoints)
    tokens = torch.full((len(endpoints), length), 50256, dtype=torch.long, device=device)
    finals = []
    for index, (row, side) in enumerate(endpoints):
        ids = row[f"{side}_ids"]
        tokens[index, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
        finals.append(len(ids) - 1)
    return endpoints, tokens, torch.tensor(finals, dtype=torch.long, device=device)


def evaluate_native(model, rows, torch, F):
    device = next(model.parameters()).device
    endpoints, tokens, finals = _pad(rows, torch, device)
    logits = model_helpers._native_logits(model, tokens, torch, F)
    evidence = []
    for index, (row, side) in enumerate(endpoints):
        query = int(finals[index])
        answer, foil = int(row[f"{side}_answer_id"]), int(row[f"{side}_foil_id"])
        margin = float(logits[index, query, answer] - logits[index, query, foil])
        ce = float(-torch.log_softmax(logits[index, query], dim=-1)[answer])
        evidence.append({
            "example_id": f"{row['row_id']}:{side}",
            "cell_id": _cell_id(row, side),
            "correct": bool(margin > 0),
            "full_vocab_CE": ce,
            "answer_minus_foil_margin": margin,
        })
    return evidence


def _summary(evidence):
    grouped = defaultdict(list)
    for item in evidence:
        grouped[str(item["cell_id"])].append(item)
    return {
        cell: {
            "count": len(items),
            "accuracy": statistics.fmean(item["correct"] for item in items),
            "mean_margin": statistics.fmean(item["answer_minus_foil_margin"] for item in items),
            "mean_full_vocab_CE": statistics.fmean(item["full_vocab_CE"] for item in items),
        }
        for cell, items in sorted(grouped.items())
    }


def _validate_fit_evidence(evidence):
    """Fail closed unless FIT is exactly the frozen all-candidate endpoint panel."""
    expected = defaultdict(int)
    expected_examples = set()
    for row in authority.build_rows():
        if row["phase"] != "FIT":
            continue
        for side in ("base", "donor"):
            expected[_cell_id(row, side)] += 1
            expected_examples.add(f"{row['row_id']}:{side}")
    if not isinstance(evidence, (list, tuple)) or len(evidence) != 128:
        raise SelectorError("FIT evidence must contain exactly 128 endpoints")
    observed = defaultdict(int)
    examples = set()
    required = {"example_id", "cell_id", "correct", "full_vocab_CE",
                "answer_minus_foil_margin"}
    for item in evidence:
        if set(item) != required or type(item["correct"]) is not bool:
            raise SelectorError("FIT evidence fields or correctness type changed")
        if not isinstance(item["example_id"], str) or item["example_id"] in examples:
            raise SelectorError("FIT evidence example IDs are invalid or duplicated")
        for metric in ("full_vocab_CE", "answer_minus_foil_margin"):
            value = item[metric]
            if type(value) not in (int, float) or not math.isfinite(float(value)):
                raise SelectorError("FIT evidence metric is not finite")
        examples.add(item["example_id"])
        observed[str(item["cell_id"])] += 1
    if examples != expected_examples or dict(observed) != dict(expected):
        raise SelectorError("FIT evidence differs from the frozen endpoint panel")


def _family_candidate(summary, family: str, template: str):
    prefix = f"FIT/{family}/{template}/"
    cells = [value for cell, value in summary.items() if cell.startswith(prefix)]
    expected_cells = 4 if family == "A2" else 8
    expected_count = 4 if family == "A2" else 2
    if len(cells) != expected_cells or any(value["count"] != expected_count for value in cells):
        raise SelectorError("candidate FIT cells or counts changed")
    eligible = all(value["accuracy"] >= MINIMUM_ACCURACY for value in cells)
    return {
        "eligible": eligible,
        "minimum_cell_accuracy": min(value["accuracy"] for value in cells),
        "minimum_cell_mean_margin": min(value["mean_margin"] for value in cells),
        "maximum_cell_mean_full_vocab_CE": max(value["mean_full_vocab_CE"] for value in cells),
    }


def select_fit_package(evidence):
    _validate_fit_evidence(evidence)
    summary = _summary(evidence)
    fixed_prefixes = ("FIT/A1/served_one_purpose/", "FIT/P/served_one_purpose_practical/")
    fixed_cells = [value for cell, value in summary.items()
                   if cell.startswith(fixed_prefixes)]
    fixed_passed = len(fixed_cells) == 12 \
        and all(value["accuracy"] >= MINIMUM_ACCURACY for value in fixed_cells)

    reports = {"A2": {}, "C": {}}
    for family, order in (("A2", authority.A2_TEMPLATE_ORDER),
                          ("C", authority.C_TEMPLATE_ORDER)):
        for template in order:
            reports[family][template] = _family_candidate(summary, family, template)

    def choose(family, order):
        eligible = [template for template in order if reports[family][template]["eligible"]]
        if not eligible:
            return None
        def key(template):
            item = reports[family][template]
            return (-item["minimum_cell_accuracy"], -item["minimum_cell_mean_margin"],
                    item["maximum_cell_mean_full_vocab_CE"], order.index(template))
        return min(eligible, key=key)

    selected_a2 = choose("A2", authority.A2_TEMPLATE_ORDER)
    selected_c = choose("C", authority.C_TEMPLATE_ORDER)
    return {
        "fit_summary": summary, "fixed_A1_P_passed": fixed_passed,
        "candidate_reports": reports, "selected_A2": selected_a2, "selected_C": selected_c,
        "fit_package_selected": fixed_passed and selected_a2 is not None and selected_c is not None,
    }


def selected_rows(a2: str, c: str, phase: str):
    if phase not in {"FIT", "HOLDOUT"}:
        raise SelectorError("phase is invalid")
    return [row for row in authority.build_package(a2, c) if row["phase"] == phase]


def build_selected_gate(a2: str, c: str):
    key = f"{a2}+{c}"
    recomputed = authority.package_sha256(a2, c)
    if recomputed != PACKAGE_SHA256.get(key):
        raise SelectorError("selected authority package hash changed")
    cells = []
    for phase in ("FIT", "HOLDOUT"):
        counts = defaultdict(int)
        for row in selected_rows(a2, c, phase):
            for side in ("base", "donor"):
                counts[_cell_id(row, side)] += 1
        for cell_id, count in sorted(counts.items()):
            cells.append(licensing.CapabilityCell(cell_id, count, MINIMUM_ACCURACY))
    gate = licensing.CapabilityGate(
        capability_id=_capability_id(a2, c),
        authority_path=Path(authority.__file__),
        expected_authority_file_sha256=AUTHORITY_FILE_SHA256,
        authority_logical_sha256=recomputed,
        cells=tuple(cells),
    )
    if len(gate.cells) != 48:
        raise SelectorError("selected package gate must contain exactly 48 cells")
    licensing.validate_gate(gate)
    return gate


def _write_selector_result(value):
    payload = managed.atomic_create_json(RESULT, value)
    return hashlib.sha256(payload).hexdigest()


def run_two_stage(model, torch, F):
    fit_rows = [row for row in authority.build_rows() if row["phase"] == "FIT"]
    fit_evidence = evaluate_native(model, fit_rows, torch, F)
    selection = select_fit_package(fit_evidence)
    if not selection["fit_package_selected"]:
        return selection, fit_evidence, [], None, None, None
    a2, c = selection["selected_A2"], selection["selected_C"]
    gate = build_selected_gate(a2, c)  # independently recomputes package_sha256
    holdout_evidence = evaluate_native(model, selected_rows(a2, c, "HOLDOUT"), torch, F)
    selected_fit_ids = {
        f"{row['row_id']}:{side}" for row in selected_rows(a2, c, "FIT")
        for side in ("base", "donor")
    }
    selected_fit = [item for item in fit_evidence if item["example_id"] in selected_fit_ids]
    capability_result, result_sha = licensing.finalize_native_capability(
        gate, selected_fit + holdout_evidence, CAPABILITY_RESULT)
    license_value = license_sha = None
    if capability_result["terminal"] == "pass":
        license_value, license_sha = licensing.issue_capability_license(
            gate, CAPABILITY_RESULT, LICENSE,
            causal_candidate_id=causal_candidate_id(a2, c))
        licensing.validate_causal_preflight(
            gate, CAPABILITY_RESULT, LICENSE, expected_license_sha256=license_sha,
            causal_candidate_id=causal_candidate_id(a2, c))
    return (selection, fit_evidence, holdout_evidence, capability_result,
            result_sha, {"value": license_value, "sha256": license_sha})


def main(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    for name in ("BQLIB_DRYRUN", "BQLIB_NO_MODEL"):
        if os.environ.get(name) not in {None, "1"}:
            raise SelectorError(f"{name} must be absent or exactly 1")
    plan = compile_plan()
    if args.dry_run or os.environ.get("BQLIB_DRYRUN") == "1" \
            or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(plan, sort_keys=True)); return
    if any(path.exists() for path in (RESULT, CAPABILITY_RESULT, LICENSE)):
        raise SelectorError("refusing to overwrite a selector, capability, or license artifact")
    torch, F, facade = model_helpers._dependencies()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.float32, verify_weights_sha256=True)
    with torch.no_grad():
        selection, fit, holdout, capability, capability_sha, license_record = \
            run_two_stage(model, torch, F)
    if not selection["fit_package_selected"]:
        terminal, forwards, evaluations = "fit_no_capable_package", 1, 128
    elif capability["terminal"] != "pass":
        terminal, forwards, evaluations = "holdout_failed", 2, 192
    else:
        terminal, forwards, evaluations = "licensed", 2, 192
    result = {
        "schema": "narrative_tense_a2_c_native_capability_select_holdout_result_v1",
        "candidate_id": authority.CAPABILITY_ID, "terminal": terminal,
        "plan": plan, "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "selection": selection, "fit_evidence": fit, "holdout_evidence": holdout,
        "selected_capability_result_sha256": capability_sha,
        "license": license_record,
        "predictions": {
            "pred_a_fit_package_selected": bool(selection["fit_package_selected"]),
            "pred_b_holdout_all_cells_pass": bool(
                capability and capability["terminal"] == "pass"),
            "pred_c_license_issued": bool(license_record and license_record["sha256"]),
        },
        "active_price": {"model_forwards": forwards, "example_evaluations": evaluations,
                         "backwards": 0, "parameter_updates": 0},
        "causal_interventions": 0,
    }
    digest = _write_selector_result(result)
    print(json.dumps({"terminal": terminal, "result_sha256": digest,
                      "selected_A2": selection["selected_A2"],
                      "selected_C": selection["selected_C"],
                      "license_sha256": license_record["sha256"] if license_record else None},
                     sort_keys=True))


if __name__ == "__main__":
    main()
