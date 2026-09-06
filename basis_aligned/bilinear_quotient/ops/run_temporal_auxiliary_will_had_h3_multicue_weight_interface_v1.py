#!/usr/bin/env python3
"""Translate the multi-cue H3 subspace into exact model-weight interfaces."""

# BQGATE: EXPERIMENT pred_a_authority_exact_shapes_gauge_and_price pred_b_known_writer_is_weight_enriched pred_c_rank2_strengthens_writer_interface pred_d_known_late_attention_readers_are_enriched pred_e_complete_weight_interface_is_reproducible
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import circuit_fast_screen_producer as producer
from circuit_fast_screen_managed_runner import atomic_create_json
import subspace_weight_atlas as atlas

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_auxiliary_will_had_h3_multicue_weight_interface_v1.json"
SUBSPACE = ROOT / "circuits/followups/temporal_auxiliary_will_had_block11h3_multicue_subspace_v2_result.json"
WEIGHTS = ROOT / "ops/subspace_weight_atlas.py"
PRODUCER = ROOT / "ops/circuit_fast_screen_producer.py"
WRITER_AUTH = ROOT / "circuits/followups/temporal_auxiliary_will_had_writer_subspace_weight_reader_atlas_v2_result.json"
DOWNSTREAM_AUTH = ROOT / "circuits/followups/temporal_auxiliary_will_had_original_writer_downstream_module_atlas_v1_result.json"
OUT = ROOT / "circuits/followups/temporal_auxiliary_will_had_h3_multicue_weight_interface_v1_result.json"
CANDIDATE_ID = "temporal_auxiliary.will_vs_had.h3_multicue_weight_interface_v1"
EXPECTED = {
    "prior": "3e1e6d83749cefec3ae7a0d96ffa36e661e0d569799ab2b7dfdc30c93a089219",
    "subspace": "d84c72d9d3c87a159fb453efc9ce9000fb8bfb7f3d2c34c96a3f7238914879c9",
    "weights": "2e7d3a546813a6029eca6fae455ad5abd03b429fcee92432ca6fe06e835e83f5",
    "producer": "14624b9959fe4bf0b43841a9e349bab50cd564a417595d1c1a048252c6c3b498",
    "writer_auth": "bf231eb67127b6378fbeb9c72621607cbf12526dba1d68fd034d3cdc594684c4",
    "downstream_auth": "f8517f43f41444b966b95ff0d8da9449f25bd95d32b726c1082066605cfd076a",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tensor_sha(tensor):
    value = tensor.detach().float().contiguous().cpu().numpy()
    return hashlib.sha256(value.tobytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def normalized_contraction(torch, read, write):
    denominator = float(torch.linalg.matrix_norm(read)) * float(torch.linalg.matrix_norm(write))
    return float(torch.linalg.matrix_norm(read @ write)) / denominator if denominator > 0 else 0.0


def percentile_rows(rows, metric):
    ranked = sorted(rows, key=lambda row: (row[metric], row["label"]))
    denominator = max(1, len(ranked) - 1)
    for index, row in enumerate(ranked):
        row[metric + "_percentile"] = index / denominator
    return sorted(rows, key=lambda row: (-row[metric], row["label"]))


def main():
    paths = {"prior": PRIOR, "subspace": SUBSPACE, "weights": WEIGHTS,
             "producer": PRODUCER, "writer_auth": WRITER_AUTH,
             "downstream_auth": DOWNSTREAM_AUTH}
    if {name: sha(path) for name, path in paths.items()} != EXPECTED:
        raise RuntimeError("authority or implementation hash changed")
    prior, subspace, writer_auth, downstream_auth = [json.loads(path.read_text()) for path in
        (PRIOR, SUBSPACE, WRITER_AUTH, DOWNSTREAM_AUTH)]
    if (prior.get("candidate_id") != CANDIDATE_ID or subspace.get("terminal") != "task_conditioned"
            or writer_auth.get("terminal") != "screen" or downstream_auth.get("terminal") != "screen"):
        raise RuntimeError("authority terminal changed")
    dryrun = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
              "model_loaded": False, "queue_touched": False,
              "model_forwards": 0, "example_evaluations": 0,
              "fit_updates": 0, "model_updates": 0,
              "upstream_heads": 99, "upstream_mlps": 11,
              "downstream_heads": 54, "downstream_mlps": 6}
    if os.environ.get("BQLIB_DRYRUN") == "1" or os.environ.get("BQLIB_NO_MODEL") == "1":
        print(json.dumps(dryrun, sort_keys=True)); return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    started_utc, started = utc_now(), time.perf_counter()
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch, model = backend.torch, backend.model
    q1 = atlas.orthonormal_basis(torch.tensor(
        subspace["axis_artifacts"]["pooled_aligned_rank1"], device=backend.device).unsqueeze(1))
    q2 = atlas.orthonormal_basis(torch.tensor(
        subspace["axis_artifacts"]["two_task_dim_union_rank2"], device=backend.device))
    head = model.transformer.h[11].attn
    width = int(head.head_dim)
    value_rows = head.c_v.weight.detach().float()[3 * width:4 * width]
    output_slice = head.c_proj.weight.detach().float()[:, 3 * width:4 * width]

    def interface(q):
        read = q.T @ value_rows
        write = output_slice @ q
        residual_basis, singular = atlas.map_head_subspace_to_residual(head, 3, q)
        return {"q": q, "read": read, "write": write,
                "residual_basis": residual_basis, "singular": singular}

    interfaces = {"rank1": interface(q1), "rank2": interface(q2)}
    upstream = {}
    for name, item in interfaces.items():
        rows = []
        for layer in range(11):
            block = model.transformer.h[layer]
            for h in range(int(block.attn.n_head)):
                start = h * int(block.attn.head_dim)
                write = block.attn.c_proj.weight.detach().float()[:, start:start + int(block.attn.head_dim)]
                rows.append({"label": f"L{layer}H{h}", "kind": "attention",
                             "layer": layer, "head": h,
                             "incidence": normalized_contraction(torch, item["read"], write)})
            down = block.mlp.Down.weight.detach().float()
            rows.append({"label": f"MLP{layer}", "kind": "mlp", "layer": layer,
                         "incidence": normalized_contraction(torch, item["read"], down)})
        upstream[name] = percentile_rows(rows, "incidence")

    downstream = {}
    for name, item in interfaces.items():
        heads, mlps = [], []
        basis = item["residual_basis"]
        for layer in range(12, 18):
            block = model.transformer.h[layer]
            for h in range(int(block.attn.n_head)):
                scores = {}
                for weight_name in ("c_q", "c_k", "c_q2", "c_k2", "c_v"):
                    if hasattr(block.attn, weight_name):
                        weight = getattr(block.attn, weight_name).weight.detach().float()
                        section = weight[h * width:(h + 1) * width]
                        denominator = float(torch.linalg.matrix_norm(section))
                        scores[weight_name.removeprefix("c_")] = (
                            float(torch.linalg.matrix_norm(section @ basis)) / denominator
                            if denominator > 0 else 0.0)
                heads.append({"label": f"L{layer}H{h}", "layer": layer, "head": h,
                              "max_read": max(scores.values()), "read_factors": scores})
            mlp = block.mlp
            left = mlp.Left.weight.detach().float()
            right = mlp.Right.weight.detach().float()
            left_score = float(torch.linalg.matrix_norm(left @ basis)) / float(torch.linalg.matrix_norm(left))
            right_score = float(torch.linalg.matrix_norm(right @ basis)) / float(torch.linalg.matrix_norm(right))
            mlps.append({"label": f"MLP{layer}", "layer": layer,
                         "max_read": max(left_score, right_score),
                         "left_read": left_score, "right_read": right_score})
        downstream[name] = {"attention": percentile_rows(heads, "max_read"),
                            "mlp": percentile_rows(mlps, "max_read")}

    generator = torch.Generator(device="cpu").manual_seed(20261028)
    random = torch.randn(2, 2, generator=generator).to(backend.device)
    gauge = torch.linalg.qr(random).Q
    gauged_read = gauge.T @ interfaces["rank2"]["read"]
    gauge_errors = []
    for row in upstream["rank2"]:
        if row["kind"] == "attention":
            block = model.transformer.h[row["layer"]].attn
            start = row["head"] * width
            write = block.c_proj.weight.detach().float()[:, start:start + width]
        else:
            write = model.transformer.h[row["layer"]].mlp.Down.weight.detach().float()
        gauge_errors.append(abs(row["incidence"]
                                - normalized_contraction(torch, gauged_read, write)))
    by_upstream = {name: {row["label"]: row for row in rows} for name, rows in upstream.items()}
    by_downstream = {name: {row["label"]: row for row in value["attention"]}
                     for name, value in downstream.items()}
    writer_rank2 = by_upstream["rank2"]["L8H1"]
    writer_rank1 = by_upstream["rank1"]["L8H1"]
    known_late = {label: by_downstream["rank2"][label]["max_read_percentile"]
                  for label in ("L15H1", "L15H5")}
    hashes = {name: {key: tensor_sha(value) for key, value in item.items()
                     if key in {"q", "read", "write", "residual_basis"}}
              for name, item in interfaces.items()}
    pred_a = bool(q1.shape == (128, 1) and q2.shape == (128, 2)
                  and all(item["read"].shape == (item["q"].shape[1], 1152)
                          and item["write"].shape == (1152, item["q"].shape[1])
                          for item in interfaces.values()) and max(gauge_errors) <= 1e-5)
    pred_b = writer_rank2["incidence_percentile"] >= 0.75
    pred_c = writer_rank2["incidence"] > writer_rank1["incidence"]
    pred_d = all(value >= 0.50 for value in known_late.values())
    inventory_ok = (all(len(rows) == 110 and len({r["label"] for r in rows}) == 110
                        for rows in upstream.values())
                    and all(len(value["attention"]) == 54 and len(value["mlp"]) == 6
                            for value in downstream.values()))
    finite = all(math.isfinite(row["incidence"]) for rows in upstream.values() for row in rows)
    pred_e = bool(inventory_ok and finite and len(hashes) == 2
                  and all(len(value) == 4 for value in hashes.values()))
    predictions = {"pred_a_authority_exact_shapes_gauge_and_price": pred_a,
        "pred_b_known_writer_is_weight_enriched": pred_b,
        "pred_c_rank2_strengthens_writer_interface": pred_c,
        "pred_d_known_late_attention_readers_are_enriched": pred_d,
        "pred_e_complete_weight_interface_is_reproducible": pred_e}
    terminal = ("invalid" if not pred_a or not pred_e else "screen" if all(predictions.values())
                else "partial" if pred_b and pred_d else "null")
    result = {"schema": "temporal_auxiliary_h3_multicue_weight_interface_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_queue_only",
        "started_utc": started_utc, "finished_utc": utc_now(),
        "serial_seconds": time.perf_counter() - started, "authority_sha256": EXPECTED,
        "dryrun": dryrun, "interface_hashes": hashes,
        "interface_singular_values": {name: [float(v) for v in item["singular"]]
                                      for name, item in interfaces.items()},
        "gauge_max_abs_score_error": max(gauge_errors),
        "known_writer": {"rank1": writer_rank1, "rank2": writer_rank2},
        "known_late_attention_percentiles": known_late,
        "upstream_rankings": upstream, "downstream_rankings": downstream,
        "predictions": predictions, "terminal": terminal,
        "price": {"model_forwards": 0, "example_evaluations": 0,
                  "fit_updates": 0, "model_updates": 0}}
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in ("candidate_id", "interface_singular_values",
          "gauge_max_abs_score_error", "known_writer", "known_late_attention_percentiles",
          "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__":
    main()
