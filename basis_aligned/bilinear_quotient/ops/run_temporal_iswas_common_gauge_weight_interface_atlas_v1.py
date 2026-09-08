#!/usr/bin/env python3
"""Complete exact-weight atlas for the stored temporal/is-was physical bases."""

# BQGATE: EXPERIMENT pred_a_authority_basis_config_finiteness_inventory_and_exact_price pred_b_iswas_top_weight_interfaces_are_fold_stable pred_c_at_least_one_fold_stable_iswas_reader_and_writer_are_material pred_d_at_least_one_weight_interface_is_material_for_both_tasks
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import common_gauge_weight_interface_contract as contract

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_common_gauge_weight_interface_atlas_v1.json"
BASIS = ROOT / "circuits/followups/temporal_iswas_common_final_gauge_basis_capture_v1_result.json"
HELPER = ROOT / "ops/common_gauge_weight_interface_contract.py"
PRODUCER = ROOT / "ops/circuit_fast_screen_producer.py"
OUT = ROOT / "circuits/followups/temporal_iswas_common_gauge_weight_interface_atlas_v1_result.json"
EXPECTED = {
    "prior": "0ed18f6f94d4306bc8db61a222b2d268095e96e12e86922f971997f1caa6ee41",
    "basis": "01b18fe723e49676ade214c1aa05dbb0509476e11ddbcdae42b9d7070a626471",
    "helper": "0aa6bbd050da0925a0cb46206e1996669cae222fb59aef64c6d7a557efe2f9a4",
    "producer": "14624b9959fe4bf0b43841a9e349bab50cd564a417595d1c1a048252c6c3b498",
}
PRICE = {"checkpoint_loads": 1, "model_forwards": 0, "transformer_backwards": 0,
         "model_updates": 0, "weight_interfaces_scored": 1026,
         "basis_weight_contractions": 3078, "fit_parameters": 0}
EXPECTED_CONFIG = {"vocab_size": 50304, "n_layer": 18, "n_head": 9,
                   "n_embd": 1152, "squared_mlp": False, "bilinear": True,
                   "expansion_factor": 4, "gated": False,
                   "squared_attn": True, "bilinear_attn": True}
PREDICTION_KEYS = (
    "pred_a_authority_basis_config_finiteness_inventory_and_exact_price",
    "pred_b_iswas_top_weight_interfaces_are_fold_stable",
    "pred_c_at_least_one_fold_stable_iswas_reader_and_writer_are_material",
    "pred_d_at_least_one_weight_interface_is_material_for_both_tasks",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def restore_basis(torch, stored, device):
    shape = stored["shape"]
    if shape[0] != 1152 or stored["layout"] != "columns_then_rows_float32":
        raise ValueError("stored basis has wrong physical layout")
    cpu = torch.tensor(stored["values"], dtype=torch.float32).reshape(shape[1], shape[0])
    if hashlib.sha256(cpu.numpy().tobytes()).hexdigest() != stored["sha256"]:
        raise ValueError("stored basis byte hash failed")
    return cpu.T.contiguous().to(device)


def finite(value):
    if isinstance(value, dict): return all(finite(item) for item in value.values())
    if isinstance(value, list): return all(finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def main():
    paths = {"prior": PRIOR, "basis": BASIS, "helper": HELPER, "producer": PRODUCER}
    observed = {name: sha(path) for name, path in paths.items()}
    prior, basis_result = json.loads(PRIOR.read_text()), json.loads(BASIS.read_text())
    authority = (observed == EXPECTED
                 and prior.get("candidate_id") == "cross_task.temporal_iswas_common_gauge_weight_interface_atlas_v1"
                 and basis_result.get("terminal") == "task_typed_direct_sum")
    dryrun = {"candidate_id": prior.get("candidate_id"), "dryrun": True,
              "authority_ok": authority, "gpu_accessed": False, "model_loaded": False,
              "queue_touched": False, "writer_interfaces": 180, "reader_interfaces": 846,
              "price": PRICE}
    if not authority:
        raise RuntimeError(f"weight atlas authority changed: {observed}")
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True)); return
    if OUT.exists(): raise FileExistsError(f"refusing overwrite: {OUT}")

    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch = backend.torch
    bases = {name: restore_basis(torch, basis_result["basis_storage"][stored], backend.device)
             for name, stored in (("temporal", "temporal_q8"),
                                  ("iswas_fold0", "v15_fold0"),
                                  ("iswas_fold1", "v15_fold1"))}
    records = contract.score_interfaces(torch, backend.model, bases)
    top20, stability = {}, {}
    for role in ("reader", "writer"):
        fold0 = contract.top_labels(records, role, "iswas_fold0_enrichment")
        fold1 = contract.top_labels(records, role, "iswas_fold1_enrichment")
        top20[role] = {"iswas_fold0": fold0, "iswas_fold1": fold1,
                       "temporal": contract.top_labels(records, role, "temporal_enrichment")}
        stability[role] = contract.jaccard(fold0, fold1)
    stable_iswas = [row for row in records if row["iswas_fold0_enrichment"] >= 4
                    and row["iswas_fold1_enrichment"] >= 4]
    shared = [row for row in stable_iswas if row["temporal_enrichment"] >= 4]
    config = {key: getattr(backend.model.config, key) for key in EXPECTED_CONFIG}
    labels_ok = (len(records) == 1026 and len({(row["role"], row["label"]) for row in records}) == 1026
                 and sum(row["role"] == "writer" for row in records) == 180
                 and sum(row["role"] == "reader" for row in records) == 846)
    fractions_ok = all(0 <= row[f"{name}_fraction"] <= 1 for row in records for name in bases)
    A = bool(authority and config == EXPECTED_CONFIG and labels_ok and fractions_ok
             and finite(records) and PRICE == prior["frozen_design"]["price"])
    B = all(stability[role] >= .60 for role in ("reader", "writer"))
    C = all(any(row["role"] == role for row in stable_iswas) for role in ("reader", "writer"))
    D = bool(shared)
    predictions = dict(zip(PREDICTION_KEYS, (A, B, C, D)))
    terminal = ("invalid" if not A else "shared_weight_interface_candidates" if B and C and D
                else "task_typed_weight_interfaces" if B and C else "unstable_weight_interface_atlas")
    result = {"schema": "temporal_iswas_common_gauge_weight_interface_atlas_result_v1",
              "candidate_id": prior["candidate_id"], "started_utc": started_utc,
              "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
              "serial_seconds": time.perf_counter() - started, "authority_sha256": observed,
              "model_config": config, "basis_terminal": basis_result["terminal"],
              "top20": top20, "v15_fold_top20_jaccard": stability,
              "fold_stable_iswas_material_interfaces": stable_iswas,
              "shared_material_interfaces": shared, "interface_records": records,
              "predictions": predictions, "terminal": terminal, "price": PRICE}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in
                      ("v15_fold_top20_jaccard", "fold_stable_iswas_material_interfaces",
                       "shared_material_interfaces", "predictions", "terminal", "price")},
                     sort_keys=True))


if __name__ == "__main__": main()
