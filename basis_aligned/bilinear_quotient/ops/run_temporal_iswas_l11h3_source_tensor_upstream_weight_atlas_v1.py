#!/usr/bin/env python3
"""Translate extracted L11H3 source tensors into exact upstream weight maps."""

# BQGATE: EXPERIMENT pred_a_authority_tensor_hash_config_inventory_finiteness_and_exact_price pred_b_task_conditioned_weight_rankings_are_original_ood_stable pred_c_each_task_has_a_material_fold_stable_upstream_writer_candidate pred_d_at_least_one_upstream_weight_writer_is_shared_across_tasks
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
import source_tensor_upstream_weight_contract as contract


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_l11h3_source_tensor_upstream_weight_atlas_v1.json"
SOURCE_RESULT = ROOT / "circuits/followups/temporal_iswas_l11h3_native_routing_source_term_extraction_v1_result.json"
CONTRACT = ROOT / "ops/source_tensor_upstream_weight_contract.py"
PRODUCER = ROOT / "ops/circuit_fast_screen_producer.py"
OUT = ROOT / "circuits/followups/temporal_iswas_l11h3_source_tensor_upstream_weight_atlas_v1_result.json"
EXPECTED = {
    "prior": "a8ba48df91af9931e87d4502fb0ff485eb9648d11d9f679b5b9a00b12c38946a",
    "source_result": "d55e448e5c950f9a6e316cc43e8d1ca7c95fd625b31021b2572a8e20ba4f245a",
    "contract": "a2d10980cb1443419dafad5680eb727836f4d091c242c6623ac7841e58acd71b",
    "producer": "14624b9959fe4bf0b43841a9e349bab50cd564a417595d1c1a048252c6c3b498",
}
PRICE = {"checkpoint_loads": 1, "model_forwards": 0, "transformer_backwards": 0,
         "model_updates": 0, "fit_parameters": 0, "writer_interfaces_scored": 111,
         "weight_compositions": 111, "task_tensor_contractions": 444}
EXPECTED_CONFIG = {"n_layer": 18, "n_head": 9, "n_embd": 1152}
PREDICTION_KEYS = (
    "pred_a_authority_tensor_hash_config_inventory_finiteness_and_exact_price",
    "pred_b_task_conditioned_weight_rankings_are_original_ood_stable",
    "pred_c_each_task_has_a_material_fold_stable_upstream_writer_candidate",
    "pred_d_at_least_one_upstream_weight_writer_is_shared_across_tasks",
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value):
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def restore_tensor(torch, stored, device):
    value = torch.tensor(stored["query_tensor"], dtype=torch.float32)
    if list(value.shape) != stored["shape"] or stored["shape"] != [128, 128]:
        raise ValueError("stored query tensor shape changed")
    if hashlib.sha256(value.numpy().tobytes()).hexdigest() != stored["sha256_fp32_le"]:
        raise ValueError("stored query tensor hash failed")
    return value.to(device)


def propagation_coefficients(model):
    scalars = [float(block.lambdas[0].detach().float()) for block in model.transformer.h]
    coefficients = {"embedding": None}
    embedded = 1.0
    for layer in range(12):
        embedded = scalars[layer] * embedded + float(
            model.transformer.h[layer].lambdas[1].detach().float())
    coefficients["embedding"] = embedded
    for layer in range(11):
        coefficient = 1.0
        for later in range(layer + 1, 12):
            coefficient *= scalars[later]
        for head in range(9):
            coefficients[f"L{layer:02d}H{head:02d}:attn_out"] = coefficient
        coefficients[f"MLP{layer:02d}:down"] = coefficient
    return coefficients


def main():
    paths = {"prior": PRIOR, "source_result": SOURCE_RESULT, "contract": CONTRACT,
             "producer": PRODUCER}
    observed = {name: sha(path) for name, path in paths.items()}
    source = json.loads(SOURCE_RESULT.read_text())
    authority_ok = bool(observed == EXPECTED
        and source.get("terminal") == "native_routing_source_term_extracted"
        and all(source.get("predictions", {}).values()))
    dry = {"candidate_id": json.loads(PRIOR.read_text())["candidate_id"], "dryrun": True,
           "authority_ok": authority_ok, "gpu_accessed": False, "model_loaded": False,
           "queue_touched": False, "writer_interfaces": 111, "price": PRICE}
    if not authority_ok:
        raise RuntimeError(f"upstream atlas authority changed: {observed}")
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dry, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch, model = backend.torch, backend.model
    config = {key: getattr(model.config, key) for key in EXPECTED_CONFIG}
    banks, covariances = {}, {}
    for population in ("original", "ood"):
        for role in ("temporal", "iswas"):
            key = f"{population}_{role}"
            banks[key] = restore_tensor(torch, source["panels"][population]["tensors"][role],
                                        backend.device)
            covariances[key] = contract.trace_one_covariance(torch, banks[key])
    width = model.config.n_embd // model.config.n_head
    reader = model.transformer.h[11].attn.c_v.weight[3 * width:4 * width].detach().float()
    coefficients = propagation_coefficients(model)
    records = []
    for label, writer in contract.writer_interfaces(model):
        composed = reader @ writer.detach().float()
        row = {"label": label, "writer_shape": list(writer.shape),
               "composed_shape": list(composed.shape),
               "composed_frobenius_norm": float(composed.norm()),
               "residual_propagation_coefficient": coefficients[label]}
        for key, covariance in covariances.items():
            row[f"{key}_enrichment"] = contract.enrichment(torch, composed, covariance)
        records.append(row)
    top10 = {role: {population: contract.top_labels(records,
        f"{population}_{role}_enrichment", 10) for population in ("original", "ood")}
        for role in ("temporal", "iswas")}
    stability = {role: contract.jaccard(top10[role]["original"], top10[role]["ood"])
                 for role in top10}
    top5 = {}
    for role in ("temporal", "iswas"):
        ranked = sorted(records, key=lambda row: (-min(row[f"original_{role}_enrichment"],
            row[f"ood_{role}_enrichment"]), row["label"]))
        top5[role] = [row["label"] for row in ranked[:5]]
    by_label = {row["label"]: row for row in records}
    material = {role: [label for label in top5[role]
        if min(by_label[label][f"original_{role}_enrichment"],
               by_label[label][f"ood_{role}_enrichment"]) >= 2.0]
        for role in top5}
    shared = [label for label in sorted(set(top5["temporal"]) & set(top5["iswas"]))
        if min(*(by_label[label][f"{population}_{role}_enrichment"]
                 for population in ("original", "ood") for role in ("temporal", "iswas"))) >= 2.0]
    labels_ok = (len(records) == 111 and len({row["label"] for row in records}) == 111
        and sum(row["label"].endswith(":attn_out") for row in records) == 99
        and sum(row["label"].startswith("MLP") for row in records) == 11
        and sum(row["label"] == "embedding" for row in records) == 1)
    A = bool(authority_ok and config == EXPECTED_CONFIG and labels_ok and finite(records)
             and PRICE == json.loads(PRIOR.read_text())["frozen_design"]["price"])
    B = bool(all(value >= .60 for value in stability.values()))
    C = bool(all(material[role] for role in material))
    D = bool(shared)
    predictions = dict(zip(PREDICTION_KEYS, map(bool, (A, B, C, D))))
    terminal = "invalid" if not A else ("shared_upstream_weight_candidates" if B and C and D
        else "task_typed_upstream_weight_candidates" if B and C
        else "unstable_upstream_weight_atlas")
    result = {"schema": "temporal_iswas_l11h3_source_tensor_upstream_weight_atlas_result_v1",
        "candidate_id": json.loads(PRIOR.read_text())["candidate_id"],
        "started_utc": started_utc,
        "finished_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "serial_seconds": time.perf_counter() - started, "authority_sha256": observed,
        "model_config": config, "reader_shape": list(reader.shape),
        "top10": top10, "original_ood_top10_jaccard": stability,
        "frozen_top5": top5, "material_top5": material,
        "shared_material_top5": shared, "interface_records": records,
        "predictions": predictions, "terminal": terminal, "price": PRICE}
    atomic_create_json(OUT, result)
    print(json.dumps({"original_ood_top10_jaccard": stability, "frozen_top5": top5,
        "material_top5": material, "shared_material_top5": shared,
        "predictions": predictions, "terminal": terminal, "price": PRICE}, sort_keys=True))


if __name__ == "__main__":
    main()
