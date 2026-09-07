#!/usr/bin/env python3
"""Exact attention OV weight-function usage within the joint causal projector."""
# BQGATE: EXPERIMENT pred_a_authority_weight_identity_finiteness_and_price pred_b_two_modes_capture_task_head_response pred_c_shared_attention_ov_usage_exists pred_d_shared_sites_agree_as_value_readers_and_residual_writers pred_e_attention_weight_usage_is_crossfit_stable
from datetime import datetime, timezone
import hashlib, json, math, os, time
from pathlib import Path

import circuit_candidate_temporal_auxiliary_fresh_cues_v12 as ood_t
import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v11 as ood_i
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import compiled_response_graph as graph
import quadratic_reader_metric as qmetric
import run_temporal_iswas_joint_projector_task_usage_v1 as usage
import run_temporal_five_mlp_crossfit_response_weight_interface_v1 as interface

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_attention_ov_task_usage_v1.json"
OLD = ROOT / "circuits/followups/temporal_iswas_joint_projector_task_usage_v1_result.json"
WEIGHT = ROOT / "circuits/followups/temporal_five_mlp_crossfit_response_weight_interface_v1_result.json"
TCAP = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_v12_capability_v1_result.json"
ICAP = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v11_capability_v1_result.json"
OUT = ROOT / "circuits/followups/temporal_iswas_attention_ov_task_usage_v1_result.json"
EXPECTED = {"old": "c446f90beeb2f86856b062faf85fe8d9218985cabee0670cd2f1002a2985934c",
            "weight": "578fa690f2f3962409bb50bd40a1c4c6f0b5ce7f88828e2e889a3a47bb20f013",
            "tcap": "4758b02cd026c85289dc3eaf352cc496d238057c6f8b52dfc6fe49ae17893324",
            "icap": "6dd757b066304d1f81ea1e52e0db601fea05adeac516a49cc84ab42bc73a86a2"}
SITES = ("L8H1", "L9H1", "L9H4", "L11H3")


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def principal(torch, left, right):
    return qmetric.principal_cosines(torch, left.T @ left, right.T @ right, left.T @ right)


def main():
    observed = {key: sha(path) for key, path in {"old": OLD, "weight": WEIGHT, "tcap": TCAP, "icap": ICAP}.items()}
    if observed != EXPECTED: raise RuntimeError(f"attention task-usage authority changed: {observed}")
    dry = {"candidate_id": "temporal_auxiliary.iswas_attention_ov_task_usage_v1", "dryrun": True,
           "gpu_accessed": False, "model_loaded": False, "queue_touched": False, "sites": SITES,
           "model_forwards_exact": 12, "fit_updates": 0, "model_updates": 0, "transformer_backwards": 0}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dry, sort_keys=True)); return
    if OUT.exists(): raise FileExistsError(OUT)
    started, tic = now(), time.perf_counter(); backend = producer.Bilin18TorchBackend.load("cuda"); torch = backend.torch
    replay = graph.reconstruct_projectors(backend)
    trows, tbatch, tbase, tdonor = usage.task_capture(backend, ood_t, TCAP)
    irows, ibatch, ibase, idonor = usage.task_capture(backend, ood_i, ICAP)
    width = backend.model.config.n_embd // backend.model.config.n_head
    records = {}; finite = []
    for site in SITES:
        _prefix, rest = site[0], site[1:]; layer_text, head_text = rest.split("H"); layer, head = int(layer_text), int(head_text)
        sl = slice(head * width, (head + 1) * width)
        td = interface.valid_delta(tbatch, tbase["attention"][layer], tdonor["attention"][layer], sl).to(backend.device).float()
        idelta = interface.valid_delta(ibatch, ibase["attention"][layer], idonor["attention"][layer], sl).to(backend.device).float()
        module = backend.model.transformer.h[layer].attn
        wv = module.c_v.weight.detach().float()[sl]
        wo = module.c_proj.weight.detach().float()[:, sl]
        records[site] = {}
        for label, qs in replay["projectors"].items():
            q = qs[site]; zt, zi = td @ q, idelta @ q
            vt, et = usage.top_usage(torch, zt); vi, ei = usage.top_usage(torch, zi)
            qt, qi = q @ vt, q @ vi
            tin, iin = wv.T @ qt, wv.T @ qi
            tout, iout = wo @ qt, wo @ qi
            input_cos = principal(torch, tin, iin); output_cos = principal(torch, tout, iout)
            gtt = (tout.T @ tout) * (tin.T @ tin); gii = (iout.T @ iout) * (iin.T @ iin)
            gti = (tout.T @ iout) * (tin.T @ iin)
            ov_cos = qmetric.principal_cosines(torch, gtt, gii, gti)
            activation = torch.linalg.svdvals(vt.T @ vi).clamp(0, 1)
            cross = {"temporal_in_iswas": float((zt @ vi).square().sum() / zt.square().sum().clamp_min(1e-30)),
                     "iswas_in_temporal": float((zi @ vt).square().sum() / zi.square().sum().clamp_min(1e-30))}
            record = {"temporal_top2_energy": et, "iswas_top2_energy": ei,
                      "activation_principal_cosines": [float(x) for x in activation],
                      "value_input_principal_cosines": [float(x) for x in input_cos],
                      "residual_output_principal_cosines": [float(x) for x in output_cos],
                      "ov_operator_principal_cosines": [float(x) for x in ov_cos],
                      "ov_mean": float(ov_cos.mean()), "ov_shared_count_080": int((ov_cos >= .8).sum()),
                      "cross_task_coordinate_energy": cross}
            records[site][label] = record
            finite += [et, ei, *record["activation_principal_cosines"], *record["value_input_principal_cosines"],
                       *record["residual_output_principal_cosines"], *record["ov_operator_principal_cosines"], *cross.values()]
    weight = json.loads(WEIGHT.read_text())
    pa = (replay["hashes_ok"] and weight["summary"]["max_attention_closure_rse"] <= 1e-10
          and len(trows) > 0 and len(irows) > 0 and all(math.isfinite(float(value)) for value in finite))
    pb = all(records[site][label][task + "_top2_energy"] >= .5 for site in SITES for label in replay["projectors"] for task in ("temporal", "iswas"))
    shared = [site for site in SITES if all(max(records[site][label]["ov_operator_principal_cosines"]) >= .8 for label in replay["projectors"])]
    pc = len(shared) >= 2
    pd = all(all(max(records[site][label][family]) >= .8 for label in replay["projectors"] for family in ("value_input_principal_cosines", "residual_output_principal_cosines")) for site in shared)
    pe = all(abs(records[site]["even_fit"]["ov_mean"] - records[site]["odd_fit"]["ov_mean"]) <= .1
             and abs(records[site]["even_fit"]["ov_shared_count_080"] - records[site]["odd_fit"]["ov_shared_count_080"]) <= 1 for site in SITES)
    predictions = {"pred_a_authority_weight_identity_finiteness_and_price": bool(pa),
                   "pred_b_two_modes_capture_task_head_response": bool(pb),
                   "pred_c_shared_attention_ov_usage_exists": bool(pc),
                   "pred_d_shared_sites_agree_as_value_readers_and_residual_writers": bool(pd),
                   "pred_e_attention_weight_usage_is_crossfit_stable": bool(pe)}
    terminal = "invalid" if not pa else "shared_attention_ov_usage" if all(predictions.values()) else "task_typed_attention_usage"
    result = {"schema": "temporal_iswas_attention_ov_task_usage_result_v1", "started_utc": started, "finished_utc": now(),
              "serial_seconds": time.perf_counter()-tic, "authority_sha256": EXPECTED,
              "row_counts": {"temporal": len(trows), "iswas": len(irows)}, "shared_sites": shared,
              "records": records, "predictions": predictions, "terminal": terminal,
              "price": {"model_forwards": 12, "fit_updates": 0, "model_updates": 0, "transformer_backwards": 0}}
    atomic_create_json(OUT, result)
    print(json.dumps({"shared_sites": shared, "records": records, "predictions": predictions,
                      "terminal": terminal, "price": result["price"]}, sort_keys=True))


if __name__ == "__main__": main()
