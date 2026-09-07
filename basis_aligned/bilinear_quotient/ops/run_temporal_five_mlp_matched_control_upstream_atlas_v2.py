#!/usr/bin/env python3
"""Complete split-half endpoint-matched control atlas for upstream writers."""

# BQGATE: EXPERIMENT pred_a_authority_population_self_patch_finiteness_and_price pred_b_mlp1_is_generic_transport pred_c_control_incidence_generalizes pred_d_selective_pool_contains_known_writers pred_e_selective_pool_improves_validation
from datetime import datetime, timezone
import hashlib, json, math, os, time
from pathlib import Path
import numpy as np

import circuit_candidate_temporal_auxiliary_fresh_cues_v13 as temporal
import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v12 as iswas
import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import run_temporal_five_mlp_upstream_input_tensor_incidence_atlas_v1 as atlasrun
import run_temporal_five_mlp_upstream_tensor_ranked_greedy_program_v1 as greedy

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_five_mlp_matched_control_upstream_atlas_v2.json"
GENERIC = ROOT / "circuits/followups/temporal_five_mlp_upstream_tensor_ranked_greedy_program_v1_result.json"
ATLAS = ROOT / "circuits/followups/temporal_five_mlp_upstream_input_tensor_incidence_atlas_v1_result.json"
ATLAS_RUNNER = ROOT / "ops/run_temporal_five_mlp_upstream_input_tensor_incidence_atlas_v1.py"
TEMPORAL_BUILDER = ROOT / "ops/circuit_candidate_temporal_auxiliary_fresh_cues_v13.py"
ISWAS_BUILDER = ROOT / "ops/circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v12.py"
WEIGHTS = ROOT / "circuits/followups/temporal_iswas_two_mode_weight_pullback_v3_result.json"
OUT = ROOT / "circuits/followups/temporal_five_mlp_matched_control_upstream_atlas_v2_result.json"
CANDIDATE_ID = "temporal_auxiliary.five_mlp_matched_control_upstream_atlas_v2"
KNOWN = ("L9H1", "L9H4", "MLP7", "MLP9")
EXPECTED = {
    "prior": "dae0b89e71b6934fa3f6b96d52478f6a027ef259d46f50a4318dd0b72266e76e",
    "generic": "57402478b86e88237bb745824e7aa8e6d56c17336753cee3d1b4e9359b5febe3",
    "atlas": "0cc9909dcab7a17b93820300da56a07f4cd9a2610f71a1de1c7008710d064467",
    "atlas_runner": "6e28d38ec1446eafb3518c1bfe603a5e3469ceadb6f80266e2c695a274692366",
    "temporal_builder": "3f738bf2fb2d4a5425dba85eaf948d7ed888e4b891666ff77a51c8638acd2509",
    "iswas_builder": "2734cbeceb4e6979dab22fe5b24870386874ac2f905ae426e6e689548e43e8a2",
    "weights": "c8ab608fa116342f9cbc8af4955e6087faa0f1eee9dd74dacb5c0ec168c5bf4d",
}

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")

def rank(values):
    order = np.argsort(np.asarray(values), kind="stable")
    out = np.empty(len(values)); out[order] = np.arange(len(values)); return out

def spearman(a, b): return float(np.corrcoef(rank(a), rank(b))[0, 1])

def control_rows():
    task_rows = {}
    for task, builder in (("temporal", temporal), ("iswas", iswas)):
        source = [row for row in builder.build_rows()
                  if row["transform_id"] == "C" and row["base_semantic_position"] in (12, 14)]
        matched = []
        for endpoint in (12, 14):
            group = [row for row in source if row["base_semantic_position"] == endpoint]
            for index, row in enumerate(group):
                donor = group[(index + 1) % len(group)]
                changed = dict(row)
                for key in ("text", "ids", "answer", "answer_id", "foil", "foil_id",
                            "prediction_position", "semantic_position"):
                    changed[f"donor_{key}"] = donor[f"base_{key}"]
                changed["row_id"] = f"matched_control_{task}_{endpoint}_{index}"
                matched.append(changed)
        discovery, validation = [], []
        for endpoint in (12, 14):
            group = [row for row in matched if row["base_semantic_position"] == endpoint]
            discovery += group[::2]; validation += group[1::2]
        task_rows[task] = {"discovery": discovery, "validation": validation}
    return {split: task_rows["temporal"][split] + task_rows["iswas"][split]
            for split in ("discovery", "validation")}

def evaluate_split(backend, rows, sites, reader, target_scales):
    torch = backend.torch
    base_batch, donor_batch = das._batch(backend, rows, side="base"), das._batch(backend, rows, side="donor")
    base_output, base_cache = atlasrun.capture_native(backend, base_batch)
    donor_output, donor_cache = atlasrun.capture_native(backend, donor_batch)
    self_output, _ = atlasrun.run_patch(backend, base_batch, base_cache, sites)
    base_state = atlasrun.states(torch, backend, base_output, rows)
    donor_state = atlasrun.states(torch, backend, donor_output, rows)
    self_state = atlasrun.states(torch, backend, self_output, rows)
    answer = torch.as_tensor([row["donor_answer_id"] for row in rows], device=backend.device)
    foil = torch.as_tensor([row["donor_foil_id"] for row in rows], device=backend.device)
    ix = torch.arange(len(rows), device=backend.device)
    def margin(state):
        logits = das.head_logits(backend, state)
        return logits[ix, answer] - logits[ix, foil]
    base_margin = margin(base_state)
    records = {"temporal": {}, "iswas": {}}
    temporal_n = sum(row["task_id"] == temporal.TASK_ID for row in rows)
    task_indices = {"temporal": slice(0, temporal_n), "iswas": slice(temporal_n, len(rows))}
    for site in sites:
        output, _ = atlasrun.run_patch(backend, base_batch, donor_cache, (site,))
        state = atlasrun.states(torch, backend, output, rows)
        delta = margin(state) - base_margin
        mode_delta = (state - base_state) @ reader
        for task, ids in task_indices.items():
            rms = float(delta[ids].square().mean().sqrt())
            records[task][site] = {
                "rms_margin_effect": rms,
                "fraction_of_target_scale": rms / max(target_scales[task], 1e-30),
                "mode_rms": [float(mode_delta[ids, j].square().mean().sqrt()) for j in range(2)],
            }
    return records, float((self_state - base_state).abs().max()), float((donor_state - base_state).abs().max())

def main():
    paths = {"prior": PRIOR, "generic": GENERIC, "atlas": ATLAS, "atlas_runner": ATLAS_RUNNER,
             "temporal_builder": TEMPORAL_BUILDER, "iswas_builder": ISWAS_BUILDER, "weights": WEIGHTS}
    if {k: sha(v) for k, v in paths.items()} != EXPECTED: raise RuntimeError("control-atlas authority changed")
    prior, generic, atlas, weights = map(lambda p: json.loads(p.read_text()), (PRIOR, GENERIC, ATLAS, WEIGHTS))
    sites = atlasrun.UPSTREAM_SITES
    dryrun = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
              "model_loaded": False, "queue_touched": False, "sites": len(sites), "rows": 42,
              "model_forwards_max": 366, "example_evaluations_max": 7644,
              "fit_updates": 0, "model_updates": 0, "transformer_backwards": 0}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True)); return
    rows = control_rows()
    authority_ok = bool(prior["candidate_id"] == CANDIDATE_ID and generic["terminal"] == "generic_transport_null"
                        and atlas["terminal"] == "invalid" and len(sites) == 179
                        and sorted(map(len, rows.values())) == [20, 22]
                        and all(r["base_answer_id"] == r["donor_answer_id"]
                                and r["base_foil_id"] == r["donor_foil_id"]
                                and len(r["base_ids"]) == len(r["donor_ids"])
                                and r["base_semantic_position"] == r["donor_semantic_position"]
                                and r["base_ids"] != r["donor_ids"]
                                for group in rows.values() for r in group))
    if not authority_ok: raise RuntimeError("control population changed")
    if OUT.exists(): raise FileExistsError(OUT)
    started, tic = now(), time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    reader, orientation, reader_hash_ok = greedy.physical_reader(backend, weights)
    scales = generic["target_behavior_rms_scales"]
    records, self_errors, donor_changes = {}, [], []
    for split, group in rows.items():
        rec, self_error, donor_change = evaluate_split(backend, group, sites, reader, scales)
        for task in ("temporal", "iswas"):
            records[f"{task}_{split}"] = rec[task]
        self_errors.append(self_error); donor_changes.append(donor_change)
    discovery_worst = {s: max(records[f"{task}_discovery"][s]["fraction_of_target_scale"]
                              for task in ("temporal", "iswas")) for s in sites}
    validation_worst = {s: max(records[f"{task}_validation"][s]["fraction_of_target_scale"]
                               for task in ("temporal", "iswas")) for s in sites}
    target = {s: atlas["site_metrics"][s]["final_causal_mode_magnitude"] for s in sites}
    selective = sorted(sites, key=lambda s: (-(target[s] / (0.05 + discovery_worst[s])), -target[s], sites.index(s)))
    old = atlas["tensor_ranking"][:20]; new = selective[:20]
    control_corr = spearman([discovery_worst[s] for s in sites], [validation_worst[s] for s in sites])
    mlp1_top = all("MLP1" in sorted(sites, key=lambda s: records[f"{task}_discovery"][s]["fraction_of_target_scale"], reverse=True)[:18]
                   for task in ("temporal", "iswas"))
    pred_a = bool(authority_ok and reader_hash_ok and orientation <= 1e-6 and max(self_errors) <= 1e-4
                  and all(math.isfinite(v) for rec in records.values() for x in rec.values() for v in x["mode_rms"] + [x["rms_margin_effect"], x["fraction_of_target_scale"]]))
    pred_b = mlp1_top and validation_worst["MLP1"] >= 0.20
    pred_c = control_corr >= 0.70
    pred_d = len(set(new) & set(KNOWN)) >= 2 and any((s.startswith("L8") or s.startswith("L9") or s.startswith("L10") or s.startswith("L11")) for s in new if not s.startswith("MLP"))
    old_med_c, new_med_c = float(np.median([validation_worst[s] for s in old])), float(np.median([validation_worst[s] for s in new]))
    old_med_t, new_med_t = float(np.median([target[s] for s in old])), float(np.median([target[s] for s in new]))
    pred_e = new_med_c <= 0.5 * old_med_c and new_med_t >= 0.5 * old_med_t
    preds = {"pred_a_authority_population_self_patch_finiteness_and_price": pred_a,
             "pred_b_mlp1_is_generic_transport": bool(pred_b), "pred_c_control_incidence_generalizes": bool(pred_c),
             "pred_d_selective_pool_contains_known_writers": bool(pred_d), "pred_e_selective_pool_improves_validation": bool(pred_e)}
    terminal = "invalid" if not pred_a else "selective_writer_screen" if all(preds.values()) else "control_instability_null" if not pred_c else "transport_explanation_null" if not pred_b else "generic_transport_confirmed_no_selective_pool"
    result = {"schema": "temporal_five_mlp_control_conditioned_upstream_atlas_result_v1", "candidate_id": CANDIDATE_ID,
              "started_utc": started, "finished_utc": now(), "serial_seconds": time.perf_counter()-tic,
              "authority_sha256": EXPECTED, "dryrun": dryrun, "instrument": {"authority_ok": authority_ok,
              "physical_reader_hash_ok": reader_hash_ok, "orientation_max_abs": orientation,
              "max_self_patch_abs": max(self_errors), "min_full_donor_state_change": min(donor_changes)},
              "records": records, "discovery_worst": discovery_worst, "validation_worst": validation_worst,
              "selective_ranking": selective, "selective_top20": new, "original_top20": old,
              "summary": {"control_rank_spearman": control_corr, "mlp1_discovery_top_decile_both": mlp1_top,
              "mlp1_validation_worst": validation_worst["MLP1"], "known_in_selective_top20": sorted(set(new)&set(KNOWN)),
              "old_median_validation_control": old_med_c, "new_median_validation_control": new_med_c,
              "old_median_target_magnitude": old_med_t, "new_median_target_magnitude": new_med_t},
              "predictions": preds, "terminal": terminal, "price": {"model_forwards": 2*(3+len(sites)),
              "example_evaluations": (3+len(sites))*sum(map(len, rows.values())), "fit_updates": 0, "model_updates": 0, "transformer_backwards": 0}}
    atomic_create_json(OUT, result)
    print(json.dumps({k: result[k] for k in ("summary","selective_top20","predictions","terminal","price")}, sort_keys=True))

if __name__ == "__main__": main()
