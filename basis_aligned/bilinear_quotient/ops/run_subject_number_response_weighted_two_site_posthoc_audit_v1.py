#!/usr/bin/env python3
# BQLANE: gpu
# BQGATE: EXPERIMENT pred_a_replays_v1_coefficients pred_b_reference_panel_available pred_c_open_slot_comparison_complete
"""Post-hoc audit of the failed two-site response-weighted scalar generator."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import numpy as np

import circuit_fast_screen_managed_runner as managed
import run_subject_number_response_weighted_two_site_composition_v1 as parent


RUNNER = Path(__file__).resolve()
ROOT = RUNNER.parents[3]
POLY = ROOT / "basis_aligned/polynomial_causal"
ARTIFACT = POLY / "SUBJECT_NUMBER_RESPONSE_WEIGHTED_PROTOTYPE_FROZEN_V1_ARTIFACT.json"
ROWS = POLY / "SUBJECT_NUMBER_TWO_SITE_COMPOSITION_V2_ROWS.json"
PARENT_RESULT = POLY / "SUBJECT_NUMBER_RESPONSE_WEIGHTED_TWO_SITE_COMPOSITION_V1_RESULT.json"
FRESH_RESULT = ROOT / "basis_aligned/bilinear_quotient/circuits/fast_screens/subject_number_response_weighted_fresh_causal_v1_result.json"
OUT = POLY / "SUBJECT_NUMBER_RESPONSE_WEIGHTED_TWO_SITE_COMPOSITION_V1_POSTHOC_AUDIT.json"
PRICE = {"physical_model_forwards": 2, "sequences": 32, "fits": 0,
         "backwards": 0, "updates": 0}
PREDICTION_REGISTRY = {"pred_a_replays_v1_coefficients": None,
                       "pred_b_reference_panel_available": None,
                       "pred_c_open_slot_comparison_complete": None}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return {"minimum": float(values.min()), "q25": float(np.quantile(values, .25)),
            "median": float(np.median(values)), "q75": float(np.quantile(values, .75)),
            "maximum": float(values.max()), "mean": float(values.mean()),
            "standard_deviation": float(values.std())}


def plan():
    return {"schema": "subject_number_response_weighted_two_site_posthoc_audit_v1_plan",
            "posthoc": True, "model_loaded": False, "gpu_accessed": False,
            "queue_touched": False, "site_positions": list(parent.SITE_POSITIONS),
            "price": PRICE,
            "bound_sha256": {"artifact": sha(ARTIFACT), "rows": sha(ROWS),
                             "parent_result": sha(PARENT_RESULT),
                             "fresh_result": sha(FRESH_RESULT)}}


def main():
    planned = plan()
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(planned, sort_keys=True)); return
    if OUT.exists():
        raise FileExistsError(OUT)
    artifact = json.loads(ARTIFACT.read_text())
    frozen = json.loads(ROWS.read_text())
    prior = json.loads(PARENT_RESULT.read_text())
    fresh = json.loads(FRESH_RESULT.read_text())
    if prior["terminal"] != "response_weighted_generator_two_site_null":
        raise ValueError("parent status changed")

    torch, F, facade = parent.tangent.parent.factors._dependencies()
    model, checkpoint = facade.load_bilin18(device="cuda", dtype=torch.float32,
                                             verify_weights_sha256=True)
    device = next(model.parameters()).device
    rows = frozen["rows"]
    tokens = torch.tensor([row["token_ids"] for row in rows], dtype=torch.long, device=device)
    axis = torch.tensor(artifact["native_axis"], dtype=torch.float32, device=device)
    prototype = torch.tensor(artifact["prototypes"]["singular_to_plural"],
                             dtype=torch.float32, device=device)
    beta = torch.tensor(artifact["interaction_beta"], dtype=torch.float64, device=device)
    attention = model.transformer.h[parent.tangent.parent.LAYER].attn
    observed = []
    head_errors = []
    for site_index, position in enumerate(parent.SITE_POSITIONS):
        finals = torch.full((len(rows),), position, dtype=torch.long, device=device)
        with torch.no_grad():
            _, captured, projection, _, inputs = parent.tangent.parent._decomposed_forward(
                model, tokens, finals, torch, F, facade)
            function = parent.head_function_at(model, captured, projection, position,
                                               attention, torch, F)
            x = inputs["raw_state"][:, position]
            h0, hp = function(x), function(x + prototype)
            z = (h0 @ axis).double()
            s = ((hp - h0) @ axis).double()
            alpha = torch.stack([torch.ones_like(z), z, s, z * s], dim=1) @ beta
            head_errors.append(float((h0 - captured["head"]).abs().max()))
        for row_index, row in enumerate(rows):
            observed.append({"row_id": row["row_id"], "site": site_index + 1,
                             "position": position, "z": float(z[row_index]),
                             "s": float(s[row_index]), "alpha": float(alpha[row_index])})

    replay = []
    for site in (1, 2):
        expected = np.array([record[f"site{site}_alpha"] for record in prior["records"]])
        actual = np.array([record["alpha"] for record in observed if record["site"] == site])
        replay.append(float(np.max(np.abs(expected - actual))))

    reference = [record for record in fresh["scalar_evidence"]
                 if record["direction"] == "singular_to_plural" and record["cardinality"] == 4]
    ref = np.array([[record["z"], record["s"], record["candidate_alpha"]]
                    for record in reference], dtype=np.float64)
    reports = {}
    for site in (1, 2):
        cur = np.array([[record["z"], record["s"], record["alpha"]]
                        for record in observed if record["site"] == site], dtype=np.float64)
        report = {}
        for index, name in enumerate(("z", "s", "alpha")):
            lower, upper = float(ref[:, index].min()), float(ref[:, index].max())
            report[name] = {"reference": summarize(ref[:, index]),
                            "two_site": summarize(cur[:, index]),
                            "outside_reference_range_fraction": float(np.mean(
                                (cur[:, index] < lower) | (cur[:, index] > upper)))}
        center = ref[:, :2].mean(axis=0)
        scale = np.maximum(ref[:, :2].std(axis=0), 1e-30)
        standardized = (cur[:, :2] - center) / scale
        report["open_slot_standardized_distance"] = summarize(
            np.sqrt(np.sum(standardized ** 2, axis=1)))
        reports[f"site{site}"] = report

    instrument = {"head_replay_max_absolute_error": max(head_errors),
                  "coefficient_replay_max_absolute_error": max(replay),
                  "reference_examples": len(reference), "observed_examples": len(observed),
                  "counts": PRICE}
    predictions = {
        "pred_a_replays_v1_coefficients": instrument["head_replay_max_absolute_error"] <= 5e-5
            and instrument["coefficient_replay_max_absolute_error"] <= 1e-10,
        "pred_b_reference_panel_available": len(reference) > 0,
        "pred_c_open_slot_comparison_complete": len(reports) == 2
            and all(set(report) == {"z", "s", "alpha", "open_slot_standardized_distance"}
                    for report in reports.values()),
    }
    valid = all(predictions.values())
    result = {"schema": "subject_number_response_weighted_two_site_posthoc_audit_v1",
              "terminal": "two_site_generator_open_slot_audit" if valid else "invalid",
              "posthoc": True, "predictions": predictions, "instrument": instrument,
              "distribution_comparison": reports, "records": observed,
              "bound_sha256": planned["bound_sha256"],
              "checkpoint_weights_sha256": checkpoint.weights_sha256,
              "runner_sha256": sha(RUNNER),
              "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
              "scope": "Post-hoc diagnosis only: compares already-open two-site open-slot coordinates with already-open fresh single-clause cardinality-four coordinates; makes no prospective OOD claim."}
    managed.atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("terminal", "instrument",
                                                    "distribution_comparison")}, indent=2))
    assert valid


if __name__ == "__main__":
    main()
