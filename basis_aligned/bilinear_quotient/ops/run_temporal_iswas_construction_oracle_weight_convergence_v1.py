#!/usr/bin/env python3
"""Exact-weight convergence test for construction-specific causal axes."""

# BQLANE: cpu
# BQGATE: EXPERIMENT pred_a_authority_relative_exact_map_gauge_finiteness_and_price pred_b_local_geometry_replays_causal_result pred_c_construction_axes_converge_as_residual_writes pred_d_l15h5_reads_a_shared_construction_response pred_e_sources_converge_on_one_shared_reader_interface pred_f_value_pullbacks_share_an_upstream_covector
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time

import fastload
from circuit_fast_screen_managed_runner import atomic_create_json
import run_temporal_iswas_v15_selected_head_projector_weight_interfaces_v1 as weight


ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "circuits/prior_art/temporal_iswas_construction_oracle_weight_convergence_v1.json"
ORACLES = ROOT / "circuits/followups/temporal_iswas_v15_construction_oracle_projective_bisector_v1_result.json"
FASTLOAD = ROOT / "ops/fastload.py"
WEIGHT_HELPER = ROOT / "ops/run_temporal_iswas_v15_selected_head_projector_weight_interfaces_v1.py"
OUT = ROOT / "circuits/followups/temporal_iswas_construction_oracle_weight_convergence_v1_result.json"
CANDIDATE_ID = "temporal_iswas.construction_oracle_weight_convergence_v1"
EXPECTED = {
    "prior": "5797a3cf3eee170fa4c3a61ebe13d7482b9bf411249dfa31b474cb8483c36d89",
    "oracles": "dde8cc793f2ba2d1f0bf7439dd9c62cb53ad979ea4ab3363928015b8cbe7e224",
    "fastload": "5803de7f127d1f556470107b559c06daecf7fbc2bccf4574aeb1c347b6225d90",
    "weight_helper": "99d71ea96c1ad2c888094143737db42c5661699a1377089296b298b9ac251b88",
    "checkpoint": "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3",
}
SOURCES = ((8, 1), (9, 1), (9, 4), (11, 3))
PRICE = {"model_forwards": 0, "example_evaluations": 0, "fit_updates": 0,
         "model_updates": 0, "transformer_backwards": 0, "checkpoint_loads": 1}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def absolute_cosine(torch, left, right):
    denominator = float(left.norm()) * float(right.norm())
    return abs(float(left.reshape(-1) @ right.reshape(-1))) / denominator if denominator else 0.0


def reader_rows(torch, model, source_layer, left_write, right_write):
    rows = []
    for layer in range(source_layer, 18):
        block = model.transformer.h[layer]
        if layer > source_layer:
            for head in range(9):
                for attr in ("c_q", "c_k", "c_q2", "c_k2", "c_v"):
                    matrix = getattr(block.attn, attr).weight.detach().float()[head * 128:(head + 1) * 128]
                    rows.append(reader_row(
                        torch, f"L{layer}H{head}:{attr[2:]}", matrix, left_write, right_write))
        for attr in ("Left", "Right"):
            matrix = getattr(block.mlp, attr).weight.detach().float()
            rows.append(reader_row(
                torch, f"MLP{layer}:{attr.lower()}", matrix, left_write, right_write))
    return weight.ranked(rows, key="joint_score")


def reader_row(torch, label, matrix, left_write, right_write):
    left, right = matrix @ left_write, matrix @ right_write
    alignment = absolute_cosine(torch, left, right)
    left_strength = weight.normalized_score(torch, matrix, left_write)
    right_strength = weight.normalized_score(torch, matrix, right_write)
    return {"label": label, "response_alignment": alignment,
            "left_strength": left_strength, "right_strength": right_strength,
            "joint_score": alignment * min(left_strength, right_strength)}


def main():
    config, checkpoint, _module = fastload._paths()
    paths = {"prior": PRIOR, "oracles": ORACLES, "fastload": FASTLOAD,
             "weight_helper": WEIGHT_HELPER}
    observed = {name: sha(path) for name, path in paths.items()}
    if observed != {name: EXPECTED[name] for name in observed}:
        raise RuntimeError(f"authority changed: {observed}")
    prior, oracles = json.loads(PRIOR.read_text()), json.loads(ORACLES.read_text())
    if (prior.get("candidate_id") != CANDIDATE_ID
            or oracles.get("terminal") != "construction_conditioned_coordinate"):
        raise RuntimeError("causal disposition changed")
    dryrun = {"candidate_id": CANDIDATE_ID, "dryrun": True, "gpu_accessed": False,
              "model_loaded": False, "queue_touched": False, "sources": len(SOURCES),
              "folds": 2, **PRICE}
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(dryrun, sort_keys=True))
        return
    if OUT.exists():
        raise FileExistsError(f"refusing overwrite: {OUT}")
    if sha(checkpoint) != EXPECTED["checkpoint"]:
        raise RuntimeError("checkpoint changed")
    if config != {"vocab_size": 50304, "n_layer": 18, "n_head": 9, "n_embd": 1152,
                  "squared_mlp": False, "bilinear": True, "expansion_factor": 4,
                  "gated": False, "squared_attn": True, "bilinear_attn": True}:
        raise RuntimeError("model configuration changed")

    started_utc, started = utc_now(), time.perf_counter()
    model = fastload.load_model_fast().eval()
    import torch
    records, mapping_relative, local_replay_error, gauge_error = {}, 0.0, 0.0, 0.0
    for fold in ("0", "1"):
        records[fold] = {}
        for layer, head in SOURCES:
            site = f"L{layer}H{head}"
            units = {panel: torch.tensor(
                oracles["projectors"][panel][fold][site], dtype=torch.float32)[:, 0]
                     for panel in ("A1", "A2")}
            output = model.transformer.h[layer].attn.c_proj.weight.detach().float()
            writes = {panel: output[:, head * 128:(head + 1) * 128] @ unit
                      for panel, unit in units.items()}
            for panel, unit in units.items():
                embedded = torch.zeros(1152)
                embedded[head * 128:(head + 1) * 128] = unit
                error = float((writes[panel] - output @ embedded).abs().max())
                mapping_relative = max(mapping_relative, error / max(float(writes[panel].norm()), 1e-30))
            local = absolute_cosine(torch, units["A1"], units["A2"])
            expected_local = oracles["between_construction_geometry"][fold][site]["absolute_axis_cosine"]
            local_replay_error = max(local_replay_error, abs(local - expected_local))
            value = model.transformer.h[layer].attn.c_v.weight.detach().float()[head * 128:(head + 1) * 128]
            pullbacks = {panel: value.T @ unit for panel, unit in units.items()}
            readers = reader_rows(torch, model, layer, writes["A1"], writes["A2"])
            neg_readers = reader_rows(torch, model, layer, -writes["A1"], writes["A2"])
            gauge_error = max(gauge_error, max(
                abs(left["joint_score"] - right["joint_score"])
                for left, right in zip(readers, neg_readers)))
            records[fold][site] = {
                "local_axis_cosine": local,
                "residual_write_cosine": absolute_cosine(torch, writes["A1"], writes["A2"]),
                "write_cosine_gain": absolute_cosine(torch, writes["A1"], writes["A2"]) - local,
                "value_pullback_cosine": absolute_cosine(
                    torch, pullbacks["A1"], pullbacks["A2"]),
                "write_norms": {panel: float(value.norm()) for panel, value in writes.items()},
                "reader_interfaces": readers,
            }

    l15h5, shared = {}, {}
    for fold in ("0", "1"):
        l15h5[fold], tops = {}, []
        for site, record in records[fold].items():
            selected = [row for row in record["reader_interfaces"] if row["label"].startswith("L15H5:")]
            l15h5[fold][site] = max((row["percentile"] for row in selected), default=0.0)
            tops.extend(row["label"] for row in record["reader_interfaces"][:10])
        counts = Counter(tops)
        shared[fold] = sorted(
            ({"label": label, "source_top10_count": count} for label, count in counts.items() if count >= 2),
            key=lambda row: (-row["source_top10_count"], row["label"]))

    values = [mapping_relative, local_replay_error, gauge_error]
    for fold in records.values():
        for record in fold.values():
            values.extend((record["local_axis_cosine"], record["residual_write_cosine"],
                           record["write_cosine_gain"], record["value_pullback_cosine"]))
            for row in record["reader_interfaces"]:
                values.extend((row["response_alignment"], row["left_strength"],
                               row["right_strength"], row["joint_score"], row["percentile"]))
    finite = all(math.isfinite(value) for value in values)
    pred_a = bool(mapping_relative <= 8 * torch.finfo(torch.float32).eps
                  and gauge_error <= 1e-12 and finite and PRICE["checkpoint_loads"] == 1)
    pred_b = local_replay_error <= 1e-6
    pred_c = all(sum(record["residual_write_cosine"] >= .75
                     for record in records[fold].values()) >= 3 for fold in ("0", "1"))
    pred_d = all(sum(value >= .75 for value in l15h5[fold].values()) >= 3
                 for fold in ("0", "1"))
    pred_e = all(any(row["source_top10_count"] >= 3 for row in shared[fold])
                 for fold in ("0", "1"))
    pred_f = all(sum(record["value_pullback_cosine"] >= .75
                     for record in records[fold].values()) >= 3 for fold in ("0", "1"))
    predictions = {
        "pred_a_authority_relative_exact_map_gauge_finiteness_and_price": pred_a,
        "pred_b_local_geometry_replays_causal_result": pred_b,
        "pred_c_construction_axes_converge_as_residual_writes": pred_c,
        "pred_d_l15h5_reads_a_shared_construction_response": pred_d,
        "pred_e_sources_converge_on_one_shared_reader_interface": pred_e,
        "pred_f_value_pullbacks_share_an_upstream_covector": pred_f,
    }
    terminal = ("invalid" if not pred_a or not pred_b else
                "shared_residual_and_reader_interfaces" if pred_c and (pred_d or pred_e) else
                "reader_equivalent_distinct_writes" if pred_d or pred_e else
                "construction_specific_linear_interfaces")
    result = {
        "schema": "temporal_iswas_construction_oracle_weight_convergence_result_v1",
        "candidate_id": CANDIDATE_ID, "execution_policy": "managed_cpu_queue_only",
        "started_utc": started_utc, "finished_utc": utc_now(),
        "serial_seconds": time.perf_counter() - started,
        "authority_sha256": EXPECTED, "dryrun": dryrun,
        "instrument": {"wo_embedding_relative_max_abs_error": mapping_relative,
                       "local_geometry_replay_max_abs_error": local_replay_error,
                       "sign_gauge_score_max_abs_error": gauge_error, "finite": finite},
        "records": records, "l15h5_joint_score_percentiles": l15h5,
        "shared_top10_joint_interfaces": shared,
        "scope": "diagnostic_weight_geometry_not_causal_identification",
        "predictions": predictions, "terminal": terminal, "price": PRICE,
    }
    atomic_create_json(OUT, result)
    print(json.dumps({key: result[key] for key in (
        "instrument", "l15h5_joint_score_percentiles", "shared_top10_joint_interfaces",
        "predictions", "terminal", "price")}, sort_keys=True))


if __name__ == "__main__":
    main()
