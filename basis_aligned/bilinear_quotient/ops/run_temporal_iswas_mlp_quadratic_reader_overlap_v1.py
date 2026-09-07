#!/usr/bin/env python3
"""Task-specific MLP quadratic-reader overlap through exact native weights."""
# BQGATE: EXPERIMENT pred_a_authority_metric_identity_finiteness_and_price pred_b_every_mlp_has_a_shared_weight_function_direction pred_c_multiple_shared_directions_exist pred_d_task_specific_quadratic_complements_exist pred_e_quadratic_overlap_is_crossfit_stable
from datetime import datetime, timezone
import hashlib, json, math, os, time
from pathlib import Path

import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import quadratic_reader_metric as qmetric
import run_temporal_five_mlp_generic_subspace_complement_program_v1 as comp
import run_temporal_five_mlp_matched_control_upstream_atlas_v2 as matched
import run_temporal_five_mlp_upstream_tensor_ranked_greedy_program_v1 as greedy
import run_temporal_five_mlp_crossfit_response_weight_interface_v1 as interface
import run_temporal_five_mlp_target_contrast_response_basis_v1 as target

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_mlp_quadratic_reader_overlap_v1.json"
INTERFACE = ROOT / "circuits/followups/temporal_five_mlp_crossfit_response_weight_interface_v1_result.json"
TCAP = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_v13_capability_v1_result.json"
ICAP = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v12_capability_v1_result.json"
METRIC = ROOT / "ops/quadratic_reader_metric.py"
OUT = ROOT / "circuits/followups/temporal_iswas_mlp_quadratic_reader_overlap_v1_result.json"
EXPECTED = {"interface": "578fa690f2f3962409bb50bd40a1c4c6f0b5ce7f88828e2e889a3a47bb20f013",
            "temporal_capability": "e053d3381680ce5a933356a060448466d7567e3079c4a2b9a5bff262bd98b9c1",
            "iswas_capability": "67cb3efbd1ea86f98f94a826922928229a7c7b0a247f218778fc4960a6e8c6f4",
            "metric": "1f12fd4d08bdbd46b099500ec05ceab5d96225f4dd3ffc9d8a91bb4a694bacf7"}
SITES = ("MLP1", "MLP3", "MLP4", "MLP6")


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def explicit_inner(torch, left, right, a, b):
    ba = left.T @ (a[:, None] * right); bb = left.T @ (b[:, None] * right)
    sa, sb = .5*(ba+ba.T), .5*(bb+bb.T)
    return (sa*sb).sum()


def main():
    paths = {"interface": INTERFACE, "temporal_capability": TCAP, "iswas_capability": ICAP, "metric": METRIC}
    observed = {k: sha(v) for k, v in paths.items()}
    if observed != EXPECTED: raise RuntimeError(f"quadratic-reader authority changed: {observed}")
    dry = {"candidate_id": "temporal_auxiliary.iswas_mlp_quadratic_reader_overlap_v1", "dryrun": True,
           "gpu_accessed": False, "model_loaded": False, "queue_touched": False, "sites": SITES,
           "model_forwards_max": 12, "fit_updates": 0, "model_updates": 0, "transformer_backwards": 0}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1": print(json.dumps(dry, sort_keys=True)); return
    if OUT.exists(): raise FileExistsError(OUT)
    started, tic = now(), time.perf_counter(); backend = producer.Bilin18TorchBackend.load("cuda"); torch = backend.torch
    tcap, icap = json.loads(TCAP.read_text()), json.loads(ICAP.read_text()); tr, ir, *_ = greedy.rows_and_controls(tcap, icap)
    rows = {"even": {"temporal": tr[::2], "iswas": ir[::2]}, "odd": {"temporal": tr[1::2], "iswas": ir[1::2]}}
    controls = matched.control_rows(); control_rows = {"even": controls["discovery"], "odd": controls["validation"]}
    projectors = {}; spectra = {}; captures = {}
    for label in ("even", "odd"):
        cb, _, c0, _, c1, _, _ = interface.cap_inputs(backend, control_rows[label]); control_basis, _ = comp.fit_bases(backend, cb, c0, c1)
        projectors[label], spectra[label], captures[label] = {}, {}, {}
        for task_name in ("temporal", "iswas"):
            batch, _, base, _, donor, _, _ = interface.cap_inputs(backend, rows[label][task_name])
            q, singular = target.fit_target(backend, batch, base, donor, control_basis)
            projectors[label][task_name] = q; spectra[label][task_name] = singular; captures[label][task_name] = len(rows[label][task_name])
    records = {}; finite = []; identity_errors = []; orthogonality = []
    for site in SITES:
        layer = int(site[3:]); module = backend.model.transformer.h[layer].mlp
        left = module.Left.weight.detach().float(); right = module.Right.weight.detach().float(); down = module.Down.weight.detach().float()
        hidden_metric = qmetric.hidden_metric(left, right); records[site] = {}
        for label in ("even", "odd"):
            qt, qi = projectors[label]["temporal"][site], projectors[label]["iswas"][site]
            orthogonality += [float((qt.T@qt-torch.eye(8, device=qt.device)).abs().max()), float((qi.T@qi-torch.eye(8, device=qi.device)).abs().max())]
            at, ai = down.T @ qt, down.T @ qi
            gtt = qmetric.gram_from_hidden_metric(at, hidden_metric); gii = qmetric.gram_from_hidden_metric(ai, hidden_metric)
            gti = qmetric.gram_from_hidden_metric(at, hidden_metric, ai)
            quadratic = qmetric.principal_cosines(torch, gtt, gii, gti)
            activation = torch.linalg.svdvals(qt.T @ qi).clamp(0, 1)
            explicit = explicit_inner(torch, left, right, at[:, 0], ai[:, 0]); implicit = gti[0, 0]
            scale = (gtt[0, 0].abs()*gii[0, 0].abs()).sqrt().clamp_min(1e-30)
            identity = float((implicit-explicit).abs()/scale); identity_errors.append(identity)
            record = {"activation_principal_cosines": [float(x) for x in activation],
                      "quadratic_principal_cosines": [float(x) for x in quadratic],
                      "activation_mean": float(activation.mean()), "quadratic_mean": float(quadratic.mean()),
                      "quadratic_shared_count_080": int((quadratic >= .8).sum()),
                      "explicit_implicit_inner_relative_error": identity,
                      "quadratic_gram_min_eigenvalue": min(float(torch.linalg.eigvalsh(.5*(g+g.T)).min()) for g in (gtt, gii))}
            records[site][label] = record
            finite += record["activation_principal_cosines"] + record["quadratic_principal_cosines"] + [identity, record["quadratic_gram_min_eigenvalue"]]
        del hidden_metric
    pa = len(tr) > 0 and len(ir) > 0 and max(orthogonality) <= 1e-4 and max(identity_errors) <= 2e-4 and all(math.isfinite(float(v)) for v in finite)
    pb = all(max(records[site][label]["quadratic_principal_cosines"]) >= .8 for site in SITES for label in ("even", "odd"))
    pc = sum(all(records[site][label]["quadratic_shared_count_080"] >= 2 for label in ("even", "odd")) for site in SITES) >= 2
    pd = sum(all(min(records[site][label]["quadratic_principal_cosines"]) <= .5 for label in ("even", "odd")) for site in SITES) >= 2
    pe = all(abs(records[site]["even"]["quadratic_mean"]-records[site]["odd"]["quadratic_mean"]) <= .1 and
             abs(records[site]["even"]["quadratic_shared_count_080"]-records[site]["odd"]["quadratic_shared_count_080"]) <= 1 for site in SITES)
    predictions = {"pred_a_authority_metric_identity_finiteness_and_price": bool(pa),
                   "pred_b_every_mlp_has_a_shared_weight_function_direction": bool(pb),
                   "pred_c_multiple_shared_directions_exist": bool(pc),
                   "pred_d_task_specific_quadratic_complements_exist": bool(pd),
                   "pred_e_quadratic_overlap_is_crossfit_stable": bool(pe)}
    terminal = "invalid" if not pa else "shared_and_task_specific_quadratic_readers" if all(predictions.values()) else "quadratic_reader_overlap_screen"
    result = {"schema": "temporal_iswas_mlp_quadratic_reader_overlap_result_v1", "started_utc": started, "finished_utc": now(),
              "serial_seconds": time.perf_counter()-tic, "authority_sha256": EXPECTED, "row_counts": captures, "records": records,
              "orthogonality_max_abs": max(orthogonality), "metric_identity_max_relative_error": max(identity_errors),
              "predictions": predictions, "terminal": terminal,
              "price": {"model_forwards": 12, "fit_updates": 0, "model_updates": 0, "transformer_backwards": 0}}
    atomic_create_json(OUT, result); print(json.dumps({k: result[k] for k in ("records", "orthogonality_max_abs", "metric_identity_max_relative_error", "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__": main()
