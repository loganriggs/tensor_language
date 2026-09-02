#!/usr/bin/env python3
"""RUNG468 -- frozen code-selected MLP product terms on natural text.

Registered before opening any natural product-term intervention:
  pred_a: exact hashes/replay/factors/no-op/census and closed SEALED role.
  pred_b: fixed proposed union transfers the complete-MLP correction direction.
  pred_c: fixed proposal beats both matched-count controls in both halves.
  pred_d: at least two individual MLP groups transfer under both sources.
  pred_e: the code-identified non-additive interaction transfers.
Strong null: invalid, tiny/source-opposed, loses both controls, or no positive member.
Literal deployed price: zero parameters saved and zero added.
"""

# BQGATE: EXPERIMENT

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time

import torch

from receipt import dump


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
POLY = ROOT.parent / "polynomial_causal"
for path in (POLY, ROOT, ROOT / "ops"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import bilin18_observed_model_facade as facade
import equality_mlp_product_term_group_rung467 as parent
import equality_term_subset_factorial_stage1 as natural_parent


PREREG = POLY / "EQUALITY_MLP_PRODUCT_TERM_NATURAL_TRANSFER_RUNG468_PREREGISTRATION.md"
PARENT_RESULT = ROOT / "equality_mlp_product_term_group_rung467_results.json"
PARENT_SOURCE = ROOT / "ops/equality_mlp_product_term_group_rung467.py"
SCALE_RESULT = ROOT / "equality_term_score_payload_rung459_results.json"
ROWS = ROOT / ".rowcache_induction_equality_tensor_final_ood_v2/final_natural.pt"
ROW_RECEIPT = ROOT / "induction_equality_tensor_final_ood_v2_rows_receipt.json"
OUT = ROOT / "equality_mlp_product_term_natural_transfer_rung468_results.json"
DOCUMENTS = 192
BATCH = 4
SOURCES = parent.SOURCES
ALL_SOURCES = parent.ALL_SOURCES
SITES = parent.SITES
SUBSETS = parent.SUBSETS
CONTROL_TYPES = parent.CONTROL_TYPES
CONTROL_MASKS = parent.CONTROL_MASKS
CELLS = parent.CELLS
CONTEXT_CELLS = parent.CONTEXT_CELLS
HIDDEN = parent.HIDDEN
EXPECTED_FORWARDS = (DOCUMENTS // BATCH) * (
    2 + len(ALL_SOURCES) + len(SOURCES) * (
        len(SUBSETS) + len(SUBSETS) + len(CONTROL_TYPES) * len(CONTROL_MASKS)
    )
)
HASHES = {
    PREREG: "76e100d8d11fd4b67ddffe76adbd9b3b95001459c9c487d3b1301c02c7aaa5f8",
    PARENT_RESULT: "cc0480fc260c81b0fe512ec694413178de181b767f1dbfec43c56804b1ee5015",
    PARENT_SOURCE: "3665fc1b33ebb7bff78f78a9548d75219a43e3a0593e79bed6075a42a821bc8b",
    SCALE_RESULT: "f157681ced170cbf8664db5710414a38d4f928f8d15dc0dd2b4d8cea9288aefa",
    ROWS: "5f2813eacc3ec66162c2ce695b978264137c66126fdc25e3d49b4efd44a9d759",
    ROW_RECEIPT: "755c456db9384420d3b2a2d5d27f0201739592b65b55eefa5871a75851dc702e",
    POLY / "bilin18_observed_model_facade.py":
        "b62947f772c807259890a9d09dfcbe5e91ad339a0bffa867ab99177fde4c728c",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_inputs():
    for path, expected in HASHES.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen hash mismatch: {path}")
    receipt = json.loads(ROW_RECEIPT.read_text())
    entry = receipt["entries"]["final_natural"]
    if receipt.get("status") != "frozen_before_any_v2_model_forward" \
            or entry.get("file_sha256") != HASHES[ROWS]:
        raise RuntimeError("natural row authority changed")
    payload = torch.load(ROWS, map_location="cpu", weights_only=True)
    if payload.get("schema") != "induction_equality_tensor_final_ood_v2_role" \
            or payload.get("role") != "final_natural" \
            or list(payload["rows"].shape) != [DOCUMENTS, 257]:
        raise RuntimeError("natural row payload changed")
    masks = natural_parent.build_masks(payload["rows"], payload["copy_cells"])
    support = {}
    for cell, expected in natural_parent.EXPECTED_SUPPORT.items():
        observed = (int(masks[cell].sum()), int(masks[cell].any(1).sum()))
        if observed != expected:
            raise RuntimeError(f"natural support changed at {cell}: {observed}")
        support[cell] = {"tokens": observed[0], "documents": observed[1]}
    result = json.loads(PARENT_RESULT.read_text())
    if result.get("rung") != 467 or not all(result.get(key) is True for key in (
        "pred_a_instrument", "pred_b_stable_split", "pred_c_exact_heldout_correction",
        "pred_d_beats_matched_controls", "pred_e_cross_module_composition",
    )) or result.get("strong_null") is not False:
        raise RuntimeError("rung467 full-pass identity changed")
    groups, controls = {}, {name: {} for name in CONTROL_TYPES}
    expected_counts = {"m8": 450, "m9": 426, "m12": 482}
    for site in SITES:
        row = result["selection"][site]
        groups[site] = row["selected_indices"]
        controls["amplitude"][site] = row["amplitude_control_indices"]
        controls["random"][site] = row["random_control_indices"]
        n = expected_counts[site]
        if row["selected_count"] != n or any(len(x) != n for x in (
            groups[site], controls["amplitude"][site], controls["random"][site],
        )):
            raise RuntimeError(f"frozen selection count changed at {site}")
        if any(len(set(x)) != len(x) or min(x) < 0 or max(x) >= HIDDEN for x in (
            groups[site], controls["amplitude"][site], controls["random"][site],
        )):
            raise RuntimeError(f"frozen selection indices malformed at {site}")
    scale_result = json.loads(SCALE_RESULT.read_text())
    scale = scale_result["frozen_fit_scales"]["L5H5->L8H4"]
    metadata = {
        "role": "final_natural", "row_sha256": sha256(ROWS),
        "row_receipt_sha256": sha256(ROW_RECEIPT), "support": support,
        "rung467_result_sha256": sha256(PARENT_RESULT),
        "rung467_source_sha256": sha256(PARENT_SOURCE),
        "selection_counts": expected_counts, "scale_result_sha256": sha256(SCALE_RESULT),
        "waves": [[0, 96], [96, 192]],
    }
    return payload, masks, scale, groups, controls, metadata, result


def _record(audit_totals, key, audit):
    row = audit_totals.setdefault(key, {"forwards": 0, "product_captures": 0,
                                        "product_patches": 0})
    row["forwards"] += 1
    row["product_captures"] += audit["product_captures"]
    row["product_patches"] += audit["product_patches"]


@torch.no_grad()
def collect(model, payload, masks, scale, groups, controls):
    rows = payload["rows"]
    full = torch.zeros(len(ALL_SOURCES), DOCUMENTS, len(CELLS), dtype=torch.float64)
    proposed = torch.zeros(len(SOURCES), len(SUBSETS), DOCUMENTS, len(CELLS), dtype=torch.float64)
    complete = torch.zeros_like(proposed)
    control = torch.zeros(len(SOURCES), len(CONTROL_TYPES), len(CONTROL_MASKS),
                          DOCUMENTS, len(CELLS), dtype=torch.float64)
    counts = torch.zeros(DOCUMENTS, len(CELLS), dtype=torch.float64)
    audit_totals = {}
    replay = {"max_abs": 0.0, "relative_squared": 0.0}
    reconstruction = 0.0
    device = next(model.parameters()).device
    for start in range(0, DOCUMENTS, BATCH):
        batch_rows = rows[start:start + BATCH]
        tokens = batch_rows[:, :-1].to(device)
        native, _, audit, _ = parent.source_parent.run_forward(model, tokens, arm="native")
        parent.source_parent.path_parent.parent._record_audit(
            audit_totals, "transfer:native", audit, analytical=False, captures=0, patches=0,
        )
        replay_logits, _, audit, error = parent.source_parent.run_forward(model, tokens, arm="replay")
        parent.source_parent.path_parent.parent._record_audit(
            audit_totals, "transfer:replay", audit, analytical=True, captures=0, patches=0,
        )
        difference = replay_logits - native
        replay["max_abs"] = max(replay["max_abs"], float(difference.abs().max()))
        replay["relative_squared"] = max(
            replay["relative_squared"],
            float(difference.square().sum()) / max(float(native.square().sum()), 1e-30),
        )
        reconstruction = max(reconstruction, error)
        products_by_source = {}
        for ai, source in enumerate(ALL_SOURCES):
            logits, products, _, audit, error = parent.run_term_forward(
                model, tokens, arm=parent.source_parent.SOURCE_ARMS[source], scale=scale,
                capture_products=True,
            )
            _record(audit_totals, f"transfer:capture:{source}", audit)
            reconstruction = max(reconstruction, error)
            sums, observed = parent._ce_sums(logits, batch_rows, masks, start)
            full[ai, start:start + BATCH] = sums
            if ai == 0:
                counts[start:start + BATCH] = observed
            elif not torch.equal(observed, counts[start:start + BATCH]):
                raise RuntimeError("natural transfer support changed")
            products_by_source[source] = products
        absent = products_by_source["0"]
        for si, source in enumerate(SOURCES):
            arm = parent.source_parent.SOURCE_ARMS[source]
            for mask in SUBSETS:
                proposed_groups = {site: groups[site] for bit, site in enumerate(SITES)
                                   if mask & (1 << bit)}
                baselines = {site: absent[site] for site in proposed_groups}
                logits, _, _, audit, error = parent.run_term_forward(
                    model, tokens, arm=arm, scale=scale,
                    baseline_products=baselines, term_groups=proposed_groups,
                )
                _record(audit_totals, f"transfer:proposed:{source}:{mask}", audit)
                reconstruction = max(reconstruction, error)
                proposed[si, mask, start:start + BATCH], observed = parent._ce_sums(
                    logits, batch_rows, masks, start,
                )
                complete_groups = {site: range(HIDDEN) for bit, site in enumerate(SITES)
                                   if mask & (1 << bit)}
                baselines = {site: absent[site] for site in complete_groups}
                logits, _, _, audit, error = parent.run_term_forward(
                    model, tokens, arm=arm, scale=scale,
                    baseline_products=baselines, term_groups=complete_groups,
                )
                _record(audit_totals, f"transfer:complete:{source}:{mask}", audit)
                reconstruction = max(reconstruction, error)
                complete[si, mask, start:start + BATCH], observed = parent._ce_sums(
                    logits, batch_rows, masks, start,
                )
            for ti, name in enumerate(CONTROL_TYPES):
                for ci, mask in enumerate(CONTROL_MASKS):
                    control_groups = {site: controls[name][site]
                                      for bit, site in enumerate(SITES) if mask & (1 << bit)}
                    baselines = {site: absent[site] for site in control_groups}
                    logits, _, _, audit, error = parent.run_term_forward(
                        model, tokens, arm=arm, scale=scale,
                        baseline_products=baselines, term_groups=control_groups,
                    )
                    _record(audit_totals, f"transfer:{name}:{source}:{mask}", audit)
                    reconstruction = max(reconstruction, error)
                    control[si, ti, ci, start:start + BATCH], observed = parent._ce_sums(
                        logits, batch_rows, masks, start,
                    )
    return full, proposed, complete, control, counts, audit_totals, replay, reconstruction


def analyze_transfer(full, proposed, complete, control, counts, selection_report, code_result):
    base = parent.analyze(full, proposed, complete, control, counts, selection_report)
    pooled, halves = base["pooled"], base["halves"]
    pred_b = bool(
        all(parent._sign_pattern(pooled["proposed_vectors"][source][7])
            and base["union_metrics"][source]["cosine"] >= .75
            and .15 <= base["union_metrics"][source]["projection_on_parent"] <= 1.50
            and base["union_metrics"][source]["vector_norm"] >= .01
            and base["union_metrics"][source]["vector_norm"]
            >= 2 * abs(pooled["proposed_off_target"][source]) for source in SOURCES)
        and base["union_source_agreement"]["cosine"] >= .75
        and all(parent._metrics(half["complete_vectors"][source][7],
                                half["proposed_vectors"][source][7])["cosine"] > 0
                for half in halves for source in SOURCES)
        and all(parent._cosine(half["proposed_vectors"]["N"][7],
                               half["proposed_vectors"]["H"][7]) > 0 for half in halves)
    )
    comparison = {}
    pred_c = True
    for source in SOURCES:
        selected = base["union_metrics"][source]
        controls = {name: parent._metrics(
            pooled["complete_vectors"][source][7], pooled["control_vectors"][source][name][7]
        ) for name in CONTROL_TYPES}
        max_cos = max(row["cosine"] for row in controls.values())
        max_proj = max(0.0, *(row["projection_on_parent"] for row in controls.values()))
        branch = ("cosine" if selected["cosine"] >= max_cos + .10 else
                  "projection" if selected["cosine"] >= .70
                  and selected["projection_on_parent"] >= 2 * max_proj else "failed")
        half_wins = True
        for half in halves:
            sm = parent._metrics(half["complete_vectors"][source][7],
                                 half["proposed_vectors"][source][7])
            cm = [parent._metrics(half["complete_vectors"][source][7],
                                  half["control_vectors"][source][name][7])
                  for name in CONTROL_TYPES]
            if branch == "cosine":
                half_wins &= sm["cosine"] > max(x["cosine"] for x in cm)
            elif branch == "projection":
                half_wins &= sm["projection_on_parent"] > max(
                    0.0, *(x["projection_on_parent"] for x in cm)
                )
            else:
                half_wins = False
        pred_c &= branch != "failed" and half_wins
        comparison[source] = {"selected": selected, "controls": controls,
                              "winning_branch": branch, "half_wins": half_wins}
    qualifying = []
    individual = {source: {} for source in SOURCES}
    for bit, site in enumerate(SITES):
        mask = 1 << bit
        okay, vectors = True, []
        for source in SOURCES:
            row = parent._metrics(pooled["complete_vectors"][source][mask],
                                  pooled["proposed_vectors"][source][mask])
            individual[source][site] = row
            vectors.append(pooled["proposed_vectors"][source][mask])
            okay &= row["cosine"] >= .50
            okay &= all(parent._metrics(half["complete_vectors"][source][mask],
                                        half["proposed_vectors"][source][mask])["cosine"] > 0
                          for half in halves)
        okay &= parent._cosine(*vectors) >= .60
        if okay:
            qualifying.append(site)
    pred_d = len(qualifying) >= 2
    interactions = base["composition"]["source_interactions"]
    source_cos = base["composition"]["source_interaction_cosine"]
    code_interactions = code_result["analysis"]["composition"]["source_interactions"]
    code_cosines = {source: parent._cosine(interactions[source]["vector"],
                                            code_interactions[source]["vector"])
                    for source in SOURCES}
    pred_e = bool(all(interactions[source]["norm"] >= .005 for source in SOURCES)
                  and source_cos >= .70 and all(value >= .50 for value in code_cosines.values()))
    loses_both = all(comparison[source]["winning_branch"] == "failed" for source in SOURCES)
    no_positive = not any(
        individual[source][site]["cosine"] > 0 for site in SITES for source in SOURCES
    )
    strong_null = bool(
        all(base["union_metrics"][source]["vector_norm"] < .005 for source in SOURCES)
        or base["union_source_agreement"]["cosine"] <= 0 or loses_both or no_positive
    )
    return {
        **base, "transfer_control_comparison": comparison,
        "transfer_individual_metrics": individual,
        "transfer_qualifying_modules": qualifying,
        "code_interaction_cosines": code_cosines,
        "pred_b_union_transfer": pred_b, "pred_c_control_separation": bool(pred_c),
        "pred_d_cross_module_transfer": pred_d, "pred_e_interaction_transfer": pred_e,
        "transfer_strong_science_null": strong_null,
    }


def main():
    started = time.time()
    payload, masks, scale, groups, controls, metadata, code_result = validate_inputs()
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print(json.dumps({
            "status": "dry_run_passed", "rung": 468, "model_loaded": False,
            "natural_product_interventions_opened": False, "sealed_opened": False,
            "role": "final_natural", "selection_counts": metadata["selection_counts"],
            "expected_forwards": EXPECTED_FORWARDS,
        }, indent=2, sort_keys=True))
        return
    if OUT.exists():
        raise RuntimeError("rung468 result namespace already exists")
    torch.cuda.reset_peak_memory_stats()
    model, checkpoint = facade.load_bilin18(
        device="cuda", dtype=torch.bfloat16, verify_weights_sha256=True,
    )
    full, proposed, complete, control, counts, audit, replay, reconstruction = collect(
        model, payload, masks, scale, groups, controls,
    )
    analysis = analyze_transfer(
        full, proposed, complete, control, counts, code_result["selection"], code_result,
    )
    forwards = sum(row["forwards"] for row in audit.values())
    empty_error = max(abs(analysis["pooled"]["proposed_vectors"][source][0][ci])
                      for source in SOURCES for ci in range(len(CONTEXT_CELLS)))
    pred_a = bool(checkpoint.weights_sha256 == facade.WEIGHTS_SHA256
                  and replay["relative_squared"] <= 1e-12 and reconstruction <= 1e-10
                  and empty_error <= 1e-12 and forwards == EXPECTED_FORWARDS)
    strong_null = bool(not pred_a or analysis["transfer_strong_science_null"])
    result = {
        "status": "complete", "rung": 468,
        "claim_level": "fixed_code_selected_product_terms_natural_transfer",
        "input_identity": metadata,
        "source_hashes": {str(path): digest for path, digest in HASHES.items()},
        "checkpoint_weights_sha256": checkpoint.weights_sha256,
        "sealed_attention0_confirmation_opened": False,
        "frozen_groups": groups, "frozen_controls": controls,
        "analysis": analysis, "native_replay": replay,
        "factor_reconstruction_relative_squared_max": reconstruction,
        "empty_group_max_abs_effect_nat": empty_error, "audit_totals": audit,
        "execution_price": {"outer_forwards": forwards,
                            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
                            "deployed_parameters_saved": 0, "deployed_parameters_added": 0},
        'pred_a_instrument': pred_a,
        'pred_b_union_transfer': analysis["pred_b_union_transfer"],
        'pred_c_control_separation': analysis["pred_c_control_separation"],
        'pred_d_cross_module_transfer': analysis["pred_d_cross_module_transfer"],
        'pred_e_interaction_transfer': analysis["pred_e_interaction_transfer"],
        "strong_null": strong_null, "runtime_s": time.time() - started,
        "next_step": (
            "explain_and_compile_fixed_1358_term_read_write_structure_without_count_tuning"
            if pred_a and all(analysis[key] for key in (
                "pred_b_union_transfer", "pred_c_control_separation",
                "pred_d_cross_module_transfer", "pred_e_interaction_transfer",
            )) else "retain_code_split_and_move_to_full_bilinear_form_or_state_quotient"
        ),
    }
    dump(result, OUT)
    print(json.dumps({
        "status": "complete", "rung": 468,
        "predictions": {key: value for key, value in result.items() if key.startswith("pred_")},
        "strong_null": strong_null, "union": analysis["union_metrics"],
        "source_agreement": analysis["union_source_agreement"],
        "controls": analysis["transfer_control_comparison"],
        "qualifying": analysis["transfer_qualifying_modules"],
        "composition": analysis["composition"],
        "code_interaction_cosines": analysis["code_interaction_cosines"],
        "instrument": {"replay": replay, "factor": reconstruction, "empty": empty_error},
        "execution_price": result["execution_price"], "runtime_s": result["runtime_s"],
        "next_step": result["next_step"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
