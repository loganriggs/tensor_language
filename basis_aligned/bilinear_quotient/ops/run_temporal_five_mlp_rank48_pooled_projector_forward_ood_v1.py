#!/usr/bin/env python3
"""Forward OOD execution of the complete-family pooled rank48 response projector."""
# BQGATE: EXPERIMENT pred_a_authority_projector_replay_disjointness_finiteness_and_price pred_b_pooled_forward_coordinates_transfer pred_c_pooled_forward_behavior_transfers pred_d_pooled_forward_controls_are_selective pred_e_one_pooled_interface_is_bidirectional
from datetime import datetime, timezone
import hashlib, json, math, os, time
from pathlib import Path

import circuit_das_subspace as das
import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import compiled_response_ood as oodctx
import pooled_response_projector as pooled
import run_temporal_five_mlp_upstream_input_tensor_incidence_atlas_v1 as atlasrun
import run_temporal_five_mlp_upstream_tensor_ranked_greedy_program_v1 as greedy
import run_temporal_five_mlp_crossfit_response_weight_interface_v1 as interface
import run_temporal_five_mlp_compiled_coordinate_upstream_atlas_v1 as atlas
import run_temporal_five_mlp_source_clamped_compiled_graph_v1 as source
import run_temporal_five_mlp_source_clamped_layer_band_addback_v1 as addback
import run_temporal_five_mlp_generic_subspace_complement_program_v1 as comp

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_five_mlp_rank48_pooled_projector_forward_ood_v1.json"
REVERSE = ROOT / "circuits/followups/temporal_five_mlp_rank48_pooled_projector_reverse_ood_v2_result.json"
FORWARD = ROOT / "circuits/followups/temporal_five_mlp_rank48_lexical_ood_v1_result.json"
SUPPORT = ROOT / "circuits/followups/temporal_five_mlp_rank49_joint_single_deletion_v1_result.json"
TCAP = ROOT / "circuits/followups/temporal_auxiliary_will_had_fresh_v12_capability_v1_result.json"
ICAP = ROOT / "circuits/followups/tense_auxiliary_is_was_fresh_lexicon_v11_capability_v1_result.json"
OUT = ROOT / "circuits/followups/temporal_five_mlp_rank48_pooled_projector_forward_ood_v1_result.json"
EXPECTED = {"reverse": "2791135e5abf88ceb5b76a3e0da44c1bb5625a5031ae58b0aba351f1232fb9b2",
            "forward": "a19d8f07f1233e7ee607ee18d15d114afda429b396c405f17330681294e7b08b",
            "support": "92384a4496c7e3ceb5dbccecfd8ef65a61e7f5c219e52b5cf919b91ef1ba5fe2"}


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def now(): return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def main():
    observed = {key: sha(path) for key, path in {"reverse": REVERSE, "forward": FORWARD, "support": SUPPORT}.items()}
    if observed != EXPECTED: raise RuntimeError(f"pooled forward authority changed: {observed}")
    dry = {"candidate_id": "temporal_auxiliary.five_mlp_rank48_pooled_projector_forward_ood_v1",
           "dryrun": True, "gpu_accessed": False, "model_loaded": False, "queue_touched": False,
           "model_forwards_exact": 16, "fit_updates": 0, "model_updates": 0, "transformer_backwards": 0}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dry, sort_keys=True)); return
    if OUT.exists(): raise FileExistsError(OUT)
    started, tic = now(), time.perf_counter(); backend = producer.Bilin18TorchBackend.load("cuda"); torch = backend.torch
    fitted = pooled.fit(backend); qs = fitted["projector"]; training, mlp_bases = pooled.fit_mlp_input_bases(backend, qs)
    fresh = oodctx.capture(backend, TCAP, ICAP); support = json.loads(SUPPORT.read_text())["selected_support"]
    _go, gai, gmi, counts = source.restricted_graph(backend, fresh["batch"], fresh["donor_batch"], fresh["base_full"], support, source=True)
    _co, cai, cmi, ccounts = source.restricted_graph(backend, fresh["control_batch"], fresh["control_donor_batch"], fresh["control_base_full"], support, source=True)
    generated = atlas.compiled_coordinates(backend, fresh["rows"], qs, fresh["base"][2], fresh["base"][5], gai,
                                           fresh["base"][3], gmi, mlp_bases)
    cgenerated = atlas.compiled_coordinates(backend, fresh["controls"], qs, fresh["control_base"][2], fresh["control_base"][5], cai,
                                            fresh["control_base"][3], cmi, mlp_bases)
    reader, orientation, reader_ok = greedy.physical_reader(backend, json.loads(comp.WEIGHTS.read_text()))
    ctx = {"fresh": fresh, "reader": reader}; report = addback.fit_arm(backend, ctx, support, "pooled", qs, mlp_bases, generated, cgenerated)
    reverse_result = json.loads(REVERSE.read_text()); hashes = {site: interface.thash(q) for site, q in qs.items()}
    fit_ids = {row["row_id"] for row in fitted["target_rows"] + fitted["control_rows"]}
    eval_ids = {row["row_id"] for row in fresh["rows"] + fresh["controls"]}
    finite = [orientation, training["capture_error"]] + list(report["coordinate"]["signed_projection"].values()) + list(report["coordinate"]["residual"].values())
    finite += list(report["target"]["cells"].values()) + list(report["target"]["behavior_signed_projection"].values())
    finite += list(report["control"]["margin_rms_fraction"].values()) + [report["control"]["median_kl"], report["control"]["max_kl"], report["control"]["top1_flip_fraction"]]
    pa = (reader_ok and orientation <= 1e-6 and hashes == reverse_result["pooled_projector_sha256"] and not (fit_ids & eval_ids)
          and len(support) == 48 and set(counts) == {1} and set(ccounts) == {1} and all(math.isfinite(float(value)) for value in finite))
    pb = min(report["coordinate"]["signed_projection"].values()) >= .75 and report["coordinate"]["mean_residual"] <= .2 and report["coordinate"]["worst_residual"] <= .2
    pc = min(report["target"]["behavior_signed_projection"].values()) >= .8 and report["target"]["worst_target_residual"] <= .15
    pd = max(report["control"]["margin_rms_fraction"].values()) <= .1 and report["control"]["median_kl"] <= .02 and report["control"]["top1_flip_fraction"] <= .05
    pe = reverse_result["terminal"] == "pooled_reverse_ood_rank48_program" and all(reverse_result["predictions"].values()) and pb and pc and pd
    predictions = {"pred_a_authority_projector_replay_disjointness_finiteness_and_price": bool(pa),
                   "pred_b_pooled_forward_coordinates_transfer": bool(pb), "pred_c_pooled_forward_behavior_transfers": bool(pc),
                   "pred_d_pooled_forward_controls_are_selective": bool(pd), "pred_e_one_pooled_interface_is_bidirectional": bool(pe)}
    terminal = "invalid" if not pa else "bidirectional_pooled_rank48_ood_program" if all(predictions.values()) else "pooled_forward_ood_failure"
    result = {"schema": "temporal_five_mlp_rank48_pooled_projector_forward_ood_result_v1",
              "started_utc": started, "finished_utc": now(), "serial_seconds": time.perf_counter()-tic,
              "authority_sha256": EXPECTED, "pooled_projector_sha256": hashes, "report": report,
              "predictions": predictions, "terminal": terminal,
              "price": {"model_forwards": 16, "fit_updates": 0, "model_updates": 0, "transformer_backwards": 0}}
    atomic_create_json(OUT, result)
    print(json.dumps({"report": report, "predictions": predictions, "terminal": terminal, "price": result["price"]}, sort_keys=True))


if __name__ == "__main__": main()
