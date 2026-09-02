#!/usr/bin/env python3
"""CPU-only diagnosis of rung498's broad-equality versus copy-task masks."""

# BQGATE: EXPERIMENT

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

import torch

from receipt import dump


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
POLY = ROOT.parent / "polynomial_causal"
for path in (ROOT, ROOT / "ops", POLY):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import circuit_induction_tensor as induction
import equality_matcher_causal_action_quotient_rung498 as parent


PLAN = POLY / "RUNG498_COPY_TASK_PORTABILITY_DIAGNOSIS_PLAN.md"
PARENT_SOURCE = ROOT / "ops/equality_matcher_causal_action_quotient_rung498.py"
PARENT_RESULT = ROOT / "equality_matcher_causal_action_quotient_rung498_results.json"
PARENT_BUNDLE = ROOT / "equality_matcher_causal_action_quotient_rung498_bundle.pt"
OLD_ROWS = ROOT / ".rowcache_induction_equality_tensor_final_ood_v2/final_natural.pt"
CENSUS = ROOT / "census_state_diverse.pt"
OUT = ROOT / "rung498_copy_task_portability_diagnosis_results.json"
HASHES = {
    PLAN: "3b777d5623b20ae385e0d22e928d19cffb4b27941e273963f928b9255f73bc6d",
    PARENT_SOURCE: "3186d610b77e1684849a54af79e83ce3d7a6a4338e36b3ec27ce2d7cc8696e59",
    PARENT_RESULT: "206ab207fc8698016c76a611cc2dbc353428fc54772c10cd173e45c9cd774a55",
    PARENT_BUNDLE: "dcdee47c84649e4bd01e124d7a7758ca63acf461a7d04d09beb4e7bffe625588",
    OLD_ROWS: "5f2813eacc3ec66162c2ce695b978264137c66126fdc25e3d49b4efd44a9d759",
    CENSUS: "c785f3d938091253535aa4f613ab2b4107bf297c8d615da4f7eab4f8282f5e0b",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_masks(rows):
    inputs = rows[:, :-1]
    broad = induction.induction_fetch_mask(inputs).any(-1)
    broad[:, :64] = False
    copy = torch.zeros_like(broad)
    near = torch.zeros_like(broad)
    far = torch.zeros_like(broad)
    one = torch.zeros_like(broad)
    multiple = torch.zeros_like(broad)
    for row_index, row in enumerate(rows):
        for query in range(64, 256):
            predecessors = torch.nonzero(row[:query] == row[query], as_tuple=False).flatten()
            if not len(predecessors):
                continue
            nearest = int(predecessors[-1])
            if int(row[nearest + 1]) != int(row[query + 1]):
                continue
            copy[row_index, query] = True
            (near if query - nearest <= 16 else far)[row_index, query] = True
            (one if len(predecessors) == 1 else multiple)[row_index, query] = True
    if bool((copy & ~broad).any()) or not torch.equal(near | far, copy) \
            or not torch.equal(one | multiple, copy):
        raise RuntimeError("copy-task mask identities failed")
    return {"broad_equality": broad, "copy_positive": copy,
            "noncopy_equality": broad & ~copy, "copy_near": near, "copy_far": far,
            "copy_one_predecessor": one, "copy_multiple_predecessors": multiple}


def metric(nll, donor_index, background_index, state, mask, lo, hi):
    absent = nll[donor_index, background_index, parent.STATES.index("late_absent"), lo:hi]
    native = nll[donor_index, background_index, parent.STATES.index("late_native"), lo:hi]
    hybrid = nll[donor_index, background_index, parent.STATES.index(state), lo:hi]
    selected = mask[lo:hi]
    native_effect, hybrid_effect = absent - native, absent - hybrid
    native_sum = float(native_effect[selected].sum())
    hybrid_sum = float(hybrid_effect[selected].sum())
    rows_native, rows_hybrid = [], []
    for row in range(hi - lo):
        if not bool(selected[row].any()):
            continue
        rows_native.append(native_effect[row, selected[row]].mean())
        rows_hybrid.append(hybrid_effect[row, selected[row]].mean())
    fit = parent._fit_report(torch.stack(rows_native), torch.stack(rows_hybrid))
    count = int(selected.sum())
    return {
        "tokens": count, "documents": len(rows_native),
        "native_recipient_effect_nat": native_sum / count,
        "hybrid_restored_effect_nat": hybrid_sum / count,
        "recovery": hybrid_sum / native_sum if abs(native_sum) > 1e-30 else None,
        **fit,
    }


def main():
    if os.environ.get("BQLIB_DRYRUN") == "1":
        exact = parent._fit_report(torch.tensor([1.0, -2.0]), torch.tensor([1.0, -2.0]))
        assert exact["cosine"] > .999 and exact["scaled_residual"] < 1e-12
        print(json.dumps({"status": "dry_run_passed", "model_loaded": False,
                          "gpu_used": False, "validation_nll_opened": False,
                          "predictions": ["pred_a", "pred_b", "pred_c"]}, sort_keys=True))
        return
    if OUT.exists():
        raise RuntimeError("diagnostic output namespace already exists")
    for path, expected in HASHES.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen hash mismatch: {path}")
    result = json.loads(PARENT_RESULT.read_text())
    if result.get("pred_a_exact_lawful_live_instrument") is not True \
            or result.get("pred_b_known_positive_recovered") is not False \
            or result.get("validation_licensed_and_opened") is not False:
        raise RuntimeError("rung498 route changed")
    bundle = torch.load(PARENT_BUNDLE, map_location="cpu", weights_only=True)
    if bundle.get("validation_nll") is not None:
        raise RuntimeError("rung498 validation outcomes unexpectedly present")
    nll = bundle["discovery_nll"].double()
    rows, *_ = parent.validate_inputs()
    masks = build_masks(rows)
    reports = {}
    for mask_name, mask in masks.items():
        reports[mask_name] = {}
        for donor_index, donor in enumerate(parent.DONORS):
            states = ("score_donor", "payload_donor") if donor == "L5H5" else ("score_donor",)
            reports[mask_name][donor] = {}
            for background_index, background in enumerate(parent.BACKGROUNDS):
                reports[mask_name][donor][background] = {
                    state: [metric(nll, donor_index, background_index, state, mask, lo, hi)
                            for lo, hi in ((0, 250), (250, 500))]
                    for state in states
                }
    copy_cells = [reports["copy_positive"]["L5H5"][background]["score_donor"][half]
                  for background in parent.BACKGROUNDS for half in range(2)]
    noncopy_cells = [reports["noncopy_equality"]["L5H5"][background]["score_donor"][half]
                     for background in parent.BACKGROUNDS for half in range(2)]
    def b_holds(row):
        return bool(.75 <= row["recovery"] <= 1.30 and row["cosine"] >= .75
                    and row["scaled_residual"] <= .70)
    pred_b = bool(all(map(b_holds, copy_cells)) and not all(map(b_holds, noncopy_cells)))
    service_differences = [
        abs(reports["copy_positive"]["L5H5"]["early_present"]["score_donor"][half]["recovery"]
            - reports["copy_positive"]["L5H5"]["early_absent"]["score_donor"][half]["recovery"])
        for half in range(2)]
    pred_c = max(service_differences) >= .20
    old_rows = torch.load(OLD_ROWS, map_location="cpu", weights_only=True)["rows"][:, :256]
    census_rows = rows[:, :256]
    census_set = {bytes(row.numpy()) for row in census_rows}
    overlap = sum(bytes(row.numpy()) in census_set for row in old_rows)
    output = {
        "status": "complete", "claim_level": "posthoc_cpu_diagnosis_not_circuit_evidence",
        "source_hashes": {str(path): digest for path, digest in HASHES.items()},
        "mask_definitions": {
            "broad_equality": "any earlier equal-token successor edge",
            "copy_positive": "nearest earlier equal token is followed by current next-token target",
        },
        "supports_by_discovery_half": {
            name: [int(mask[:250].sum()), int(mask[250:500].sum())]
            for name, mask in masks.items()},
        "reports": reports,
        "old_rung459_rows": len(old_rows), "census_rows": len(census_rows),
        "exact_256_token_prefix_overlap": overlap,
        "copy_recovery_early_service_absolute_differences": service_differences,
        'pred_a_hashes_masks_bundle_and_no_validation': True,
        'pred_b_task_mask_mismatch': pred_b,
        'pred_c_earlier_service_interaction': pred_c,
        "classification": "task_mask_mismatch" if pred_b else "corpus_or_action_shift",
        "gpu_used": False, "model_loaded": False, "validation_nll_opened": False,
        "deployed_parameters_added": 0, "deployed_parameters_saved": 0,
        "next_step": ("preregister_unopened_copy_task_calibration" if pred_b else
                      "follow_rung498_corpus_action_failure_route"),
    }
    dump(output, OUT)
    print(json.dumps({key: output[key] for key in (
        "status", "classification", "supports_by_discovery_half",
        "copy_recovery_early_service_absolute_differences",
        "pred_a_hashes_masks_bundle_and_no_validation", "pred_b_task_mask_mismatch",
        "pred_c_earlier_service_interaction", "next_step")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
