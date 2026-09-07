#!/usr/bin/env python3
"""Task usage directions inside frozen joint MLP response projectors."""
# BQGATE: EXPERIMENT pred_a_authority_joint_hashes_finiteness_and_price pred_b_two_usage_modes_capture_task_response pred_c_shared_quadratic_usage_exists pred_d_task_typed_complements_also_exist pred_e_joint_usage_overlap_is_crossfit_stable
from datetime import datetime, timezone
import hashlib, json, math, os, time
from pathlib import Path

import circuit_candidate_temporal_auxiliary_fresh_cues_v12 as ood_t
import circuit_candidate_tense_auxiliary_is_was_fresh_lexicon_v11 as ood_i
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import compiled_response_graph as graph
import quadratic_reader_metric as qmetric
import run_temporal_iswas_three_mlp_response_program_fresh_v12_v1 as population
import run_temporal_five_mlp_crossfit_response_weight_interface_v1 as interface
import run_temporal_five_mlp_upstream_input_tensor_incidence_atlas_v1 as atlasrun

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_joint_projector_task_usage_v1.json"
OLD = ROOT / "circuits/followups/temporal_iswas_mlp_quadratic_reader_overlap_v1_result.json"
INTERFACE = ROOT / "circuits/followups/temporal_five_mlp_crossfit_response_weight_interface_v1_result.json"
TCAP = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_v12_capability_v1_result.json"
ICAP = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v11_capability_v1_result.json"
METRIC = ROOT / "ops/quadratic_reader_metric.py"
OUT = ROOT / "circuits/followups/temporal_iswas_joint_projector_task_usage_v1_result.json"
EXPECTED = {"old": "d40b589653845f50488c7d960467577f138ac3c5d5285f433e970a735838982f",
            "interface": "578fa690f2f3962409bb50bd40a1c4c6f0b5ce7f88828e2e889a3a47bb20f013",
            "temporal_capability": "4758b02cd026c85289dc3eaf352cc496d238057c6f8b52dfc6fe49ae17893324",
            "iswas_capability": "6dd757b066304d1f81ea1e52e0db601fea05adeac516a49cc84ab42bc73a86a2",
            "metric": "1f12fd4d08bdbd46b099500ec05ceab5d96225f4dd3ffc9d8a91bb4a694bacf7"}
SITES = ("MLP1", "MLP3", "MLP4", "MLP6")


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def task_capture(backend, module, capability):
    cap = json.loads(capability.read_text()); rows = sum((population.capable_rows(module, cap, panel, 12) for panel in ("A1", "A2")), [])
    batch, _, base, _, donor, _, _ = interface.cap_inputs(backend, rows)
    return rows, batch, base, donor


def valid_delta(torch, batch, base, donor, layer):
    return torch.cat([(donor["mlp"][layer][i, :int(pos)+1]-base["mlp"][layer][i, :int(pos)+1]).to(base["mlp"][layer])
                      for i, pos in enumerate(batch.semantic_positions)]).float()


def top_usage(torch, coordinates):
    _u, singular, vh = torch.linalg.svd(coordinates, full_matrices=False); modes = vh[:2].T.contiguous()
    energy = float(singular[:2].square().sum()/singular.square().sum().clamp_min(1e-30))
    return modes, energy


def main():
    paths = {"old": OLD, "interface": INTERFACE, "temporal_capability": TCAP, "iswas_capability": ICAP, "metric": METRIC}
    observed = {k: sha(v) for k, v in paths.items()}
    if observed != EXPECTED: raise RuntimeError(f"joint-usage authority changed: {observed}")
    dry = {"candidate_id": "temporal_auxiliary.iswas_joint_projector_task_usage_v1", "dryrun": True,
           "gpu_accessed": False, "model_loaded": False, "queue_touched": False, "sites": SITES,
           "model_forwards_max": 12, "fit_updates": 0, "model_updates": 0, "transformer_backwards": 0}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1": print(json.dumps(dry, sort_keys=True)); return
    if OUT.exists(): raise FileExistsError(OUT)
    started, tic = now(), time.perf_counter(); backend = producer.Bilin18TorchBackend.load("cuda"); torch = backend.torch
    replay = graph.reconstruct_projectors(backend)
    trows, tbatch, tbase, tdonor = task_capture(backend, ood_t, TCAP)
    irows, ibatch, ibase, idonor = task_capture(backend, ood_i, ICAP)
    deltas = {site: {"temporal": valid_delta(torch, tbatch, tbase, tdonor, int(site[3:])),
                     "iswas": valid_delta(torch, ibatch, ibase, idonor, int(site[3:]))} for site in SITES}
    records = {}; finite = []
    for site in SITES:
        layer = int(site[3:]); module = backend.model.transformer.h[layer].mlp
        left, right, down = module.Left.weight.detach().float(), module.Right.weight.detach().float(), module.Down.weight.detach().float()
        hidden_metric = qmetric.hidden_metric(left, right); records[site] = {}
        for label, qs in replay["projectors"].items():
            q = qs[site]; zt, zi = deltas[site]["temporal"].to(backend.device) @ q, deltas[site]["iswas"].to(backend.device) @ q
            vt, et = top_usage(torch, zt); vi, ei = top_usage(torch, zi)
            qt, qi = q @ vt, q @ vi; at, ai = down.T @ qt, down.T @ qi
            gtt = qmetric.gram_from_hidden_metric(at, hidden_metric); gii = qmetric.gram_from_hidden_metric(ai, hidden_metric)
            gti = qmetric.gram_from_hidden_metric(at, hidden_metric, ai)
            quadratic = qmetric.principal_cosines(torch, gtt, gii, gti); activation = torch.linalg.svdvals(vt.T @ vi).clamp(0, 1)
            cross_energy = {"temporal_in_iswas": float((zt@vi).square().sum()/zt.square().sum().clamp_min(1e-30)),
                            "iswas_in_temporal": float((zi@vt).square().sum()/zi.square().sum().clamp_min(1e-30))}
            records[site][label] = {"temporal_top2_energy": et, "iswas_top2_energy": ei,
                                    "activation_principal_cosines": [float(x) for x in activation],
                                    "quadratic_principal_cosines": [float(x) for x in quadratic],
                                    "quadratic_mean": float(quadratic.mean()), "quadratic_shared_count_080": int((quadratic >= .8).sum()),
                                    "cross_task_coordinate_energy": cross_energy}
            finite += [et, ei, *records[site][label]["activation_principal_cosines"], *records[site][label]["quadratic_principal_cosines"], *cross_energy.values()]
        del hidden_metric
    old = json.loads(OLD.read_text())
    pa = replay["hashes_ok"] and old["metric_identity_max_relative_error"] <= 2e-4 and len(trows) > 0 and len(irows) > 0 and all(math.isfinite(float(v)) for v in finite)
    pb = all(records[site][label]["temporal_top2_energy"] >= .5 and records[site][label]["iswas_top2_energy"] >= .5 for site in SITES for label in replay["projectors"])
    pc = sum(all(max(records[site][label]["quadratic_principal_cosines"]) >= .8 for label in replay["projectors"]) for site in SITES) >= 2
    pd = sum(all(min(records[site][label]["quadratic_principal_cosines"]) <= .5 for label in replay["projectors"]) for site in SITES) >= 2
    pe = all(abs(records[site]["even_fit"]["quadratic_mean"]-records[site]["odd_fit"]["quadratic_mean"]) <= .1 and
             abs(records[site]["even_fit"]["quadratic_shared_count_080"]-records[site]["odd_fit"]["quadratic_shared_count_080"]) <= 1 for site in SITES)
    predictions = {"pred_a_authority_joint_hashes_finiteness_and_price": bool(pa),
                   "pred_b_two_usage_modes_capture_task_response": bool(pb), "pred_c_shared_quadratic_usage_exists": bool(pc),
                   "pred_d_task_typed_complements_also_exist": bool(pd), "pred_e_joint_usage_overlap_is_crossfit_stable": bool(pe)}
    terminal = "invalid" if not pa else "shared_joint_quadratic_usage" if all(predictions.values()) else "task_typed_joint_projector_usage"
    result = {"schema": "temporal_iswas_joint_projector_task_usage_result_v1", "started_utc": started, "finished_utc": now(),
              "serial_seconds": time.perf_counter()-tic, "authority_sha256": EXPECTED, "row_counts": {"temporal": len(trows), "iswas": len(irows)},
              "records": records, "predictions": predictions, "terminal": terminal,
              "price": {"model_forwards": 12, "fit_updates": 0, "model_updates": 0, "transformer_backwards": 0}}
    atomic_create_json(OUT, result); print(json.dumps({k: result[k] for k in ("records", "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__": main()
